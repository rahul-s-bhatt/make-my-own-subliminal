# ==========================================
# Pro Subliminal Audio Editor V2
# ==========================================
import streamlit as st

# --- Early Config ---
# Set page config early, especially layout
st.set_page_config(layout="wide", page_title="Pro Subliminal Editor")

# --- Imports ---
import logging
import logging.handlers
import os
import queue  # For async logging setup
import tempfile
import time  # For potential delays
import traceback  # For detailed error logging if needed
import uuid
from io import BytesIO

# Configure logging (do this early before other imports might log)
log_queue = queue.Queue(-1)  # Infinite queue size
log_file = "editor.log"
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# File Handler (Rotating)
file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=1 * 1024 * 1024, backupCount=3)  # 1MB per file, 3 backups
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)  # Log everything DEBUG and above to file

# Queue Handler (for async) - attached to root logger
queue_handler = logging.handlers.QueueHandler(log_queue)

# Root logger setup
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Capture everything from DEBUG level
root_logger.addHandler(queue_handler)  # Send logs to the queue

# Queue Listener (processes the queue and sends to actual handlers)
# The listener runs in its own thread
listener = logging.handlers.QueueListener(log_queue, file_handler, respect_handler_level=True)
listener.start()
# Note: Ideally, call listener.stop() on clean exit, but Streamlit makes this hard.

# --- Log application start ---
logging.info("Application starting up.")

# --- Core Libraries ---
try:
    import librosa
    import librosa.effects  # Explicitly import effects
    import numpy as np
    import pyttsx3
    import soundfile as sf
    from scipy import signal  # For filtering
    from streamlit.runtime.uploaded_file_manager import UploadedFile
    from streamlit_advanced_audio import WaveSurferOptions, audix
except ImportError as e:
    logging.exception("Failed to import core libraries. Please ensure all dependencies in requirements.txt are installed.")
    st.error(f"Core library import failed: {e}. Please check installation and restart.")
    st.stop()  # Stop execution if core libs are missing


# --- Constants ---
GLOBAL_SR = 44100  # Fixed sample rate for all tracks (Hz)
logging.debug(f"Global Sample Rate set to: {GLOBAL_SR} Hz")

# --- Data Types ---
AudioData = np.ndarray  # Audio data as NumPy array (float32, shape: (samples, 2))
SampleRate = int  # Sample rate in Hz (e.g., 44100)
TrackID = str  # Unique identifier for each track

# --- Session State Initialization ---
if "tracks" not in st.session_state:
    logging.debug("Initializing 'tracks' in session state.")
    st.session_state.tracks = {}  # Dict[TrackID, dict] storing track data

# --- Utility Functions ---


def load_audio(file_source: UploadedFile | BytesIO | str, target_sr: SampleRate = GLOBAL_SR) -> tuple[AudioData, SampleRate]:
    """
    Load audio file (from UploadedFile, BytesIO, or path string),
    ensure stereo, resample to target_sr.
    Returns (audio_data, sample_rate)
    """
    logging.info(f"Loading audio from source type: {type(file_source)}")
    try:
        source = file_source
        audio, sr = librosa.load(source, sr=None, mono=False)
        logging.debug(f"Loaded audio with original SR: {sr}, shape: {audio.shape}")

        # Ensure audio is 2D (samples, channels)
        if audio.ndim == 1:
            logging.debug("Audio is mono, duplicating to stereo.")
            audio = np.stack([audio, audio], axis=-1)
        elif audio.shape[0] == 2:
            logging.debug("Audio channels are first dim, transposing.")
            audio = audio.T

        # Ensure shape is exactly (samples, 2)
        if audio.shape[1] > 2:
            logging.warning(f"Audio has {audio.shape[1]} channels. Using only first two.")
            st.warning(f"Audio has more than 2 channels ({audio.shape[1]}). Using only the first two.")
            audio = audio[:, :2]
        elif audio.shape[1] == 1:
            logging.warning("Audio loaded as mono unexpectedly. Duplicating channel.")
            st.warning("Audio loaded as mono unexpectedly. Duplicating channel.")
            audio = np.concatenate([audio, audio], axis=1)

        # Resample if necessary
        if sr != target_sr:
            logging.info(f"Resampling audio from {sr} Hz to {target_sr} Hz.")
            if audio.size > 0:
                audio_resampled = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr)
                audio = audio_resampled.T
                logging.debug(f"Resampled audio shape: {audio.shape}")
            else:
                logging.debug("Audio is empty, skipping resampling, setting SR.")
                sr = target_sr
        else:
            logging.debug("Audio already at target sample rate.")

        # Ensure float32 type for processing
        return audio.astype(np.float32), target_sr

    except Exception as e:
        logging.exception("Error loading audio.")
        st.error(f"Error loading audio: {e}")
        return np.zeros((0, 2), dtype=np.float32), target_sr


def save_audio(audio: AudioData, sr: SampleRate) -> BytesIO:
    """Save NumPy audio array to WAV file format in an in-memory BytesIO buffer."""
    buffer = BytesIO()
    try:
        sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        logging.debug(f"Saved audio ({audio.shape}, {sr}Hz) to BytesIO buffer (size: {buffer.getbuffer().nbytes} bytes).")
    except Exception as e:
        logging.exception("Error saving audio to buffer.")
        st.error(f"Error saving audio to buffer: {e}")
        buffer = BytesIO()  # Return empty buffer on failure
    return buffer


def save_audio_to_temp(audio: AudioData, sr: SampleRate) -> str | None:
    """Save NumPy audio array to a temporary WAV file on disk. Returns the file path or None on error."""
    temp_file_path = None
    try:
        # delete=False because audix needs the path to remain valid after the 'with' block
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as tmp:
            sf.write(tmp, audio, sr, format="WAV", subtype="PCM_16")
            temp_file_path = tmp.name
        logging.debug(f"Saved audio ({audio.shape}, {sr}Hz) to temporary file: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        logging.exception("Failed to save temporary audio file.")
        st.error(f"Failed to save temporary audio file: {e}")
        # Clean up if file was created but write failed
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logging.debug(f"Cleaned up partially created temp file: {temp_file_path}")
            except OSError as e_os:
                logging.warning(f"Failed to clean up temp file {temp_file_path}: {e_os}")
        return None


def generate_binaural_beats(duration: float, freq_left: float, freq_right: float, sr: SampleRate, volume: float) -> AudioData:
    """Generate stereo binaural beats."""
    logging.info(f"Generating binaural beats: dur={duration}s, L={freq_left}Hz, R={freq_right}Hz, vol={volume}")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = volume * np.sin(2 * np.pi * freq_left * t)
    right = volume * np.sin(2 * np.pi * freq_right * t)
    audio = np.stack([left, right], axis=1).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def generate_solfeggio_frequency(duration: float, freq: float, sr: SampleRate, volume: float) -> AudioData:
    """Generate a stereo sine wave at the solfeggio frequency."""
    logging.info(f"Generating Solfeggio frequency: dur={duration}s, F={freq}Hz, vol={volume}")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = volume * np.sin(2 * np.pi * freq * t)
    audio = np.stack([sine_wave, sine_wave], axis=1).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


# --- Effect Application Functions ---


# Using time_stretch version as requested by user to fix audix playback
def apply_speed_change(audio: AudioData, sr: SampleRate, speed_factor: float) -> AudioData:
    """Apply speed change using time stretching (preserves pitch better). Returns modified audio."""
    logging.debug(f"Applying speed change with factor: {speed_factor} (time_stretch method)")
    # Add the NaN/Inf check here for safety as well
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        logging.error("Audio data contains NaN or Inf values before time stretch. Skipping effect.")
        st.error("Audio data contains NaN or Inf values before time stretch. Skipping effect.")
        return audio

    if not np.isclose(speed_factor, 1.0):
        if speed_factor <= 0:
            logging.warning(f"Invalid speed factor: {speed_factor}. Must be positive.")
            st.warning("Speed factor must be positive.")
            return audio
        try:
            if audio.size == 0:
                logging.debug("Audio size is zero, skipping speed change.")
                return audio
            logging.info(f"Applying speed change using time stretch (factor: {speed_factor})...")
            # librosa.effects.time_stretch works on (channels, samples) or (samples,)
            audio_stretched = librosa.effects.time_stretch(audio.T, rate=speed_factor)
            logging.debug("Time stretch applied successfully.")
            return audio_stretched.T.astype(np.float32)
        except Exception as e:
            logging.exception("Error applying speed change (time stretch method).")
            st.error(f"Error applying speed change (time stretch method): {e}")
            # st.error(traceback.format_exc()) # Show full traceback if needed
            return audio
    logging.debug("Speed factor is close to 1.0, skipping speed change.")
    return audio


def apply_pitch_shift(audio: AudioData, sr: SampleRate, n_steps: float) -> AudioData:
    """Apply pitch shift in semitones. Returns modified audio."""
    logging.debug(f"Applying pitch shift: {n_steps} semitones")
    if not np.isclose(n_steps, 0.0):
        try:
            if audio.size == 0:
                logging.debug("Audio size is zero, skipping pitch shift.")
                return audio
            logging.info(f"Applying pitch shift ({n_steps} semitones)...")
            audio_shifted = librosa.effects.pitch_shift(audio.T, sr=sr, n_steps=n_steps)
            logging.debug("Pitch shift applied successfully.")
            return audio_shifted.T.astype(np.float32)
        except Exception as e:
            logging.exception("Error applying pitch shift.")
            st.error(f"Error applying pitch shift: {e}")
            return audio
    logging.debug("Pitch shift is zero, skipping.")
    return audio


def apply_filter(audio: AudioData, sr: SampleRate, filter_type: str, cutoff: float) -> AudioData:
    """Apply low-pass or high-pass filter. Returns modified audio."""
    logging.debug(f"Applying filter: type={filter_type}, cutoff={cutoff} Hz")
    if filter_type == "off" or cutoff <= 0:
        logging.debug("Filter type is 'off' or cutoff <= 0, skipping filter.")
        return audio

    try:
        if audio.size == 0:
            logging.debug("Audio size is zero, skipping filter.")
            return audio

        nyquist = 0.5 * sr
        normalized_cutoff = cutoff / nyquist

        if normalized_cutoff >= 1.0:
            msg = f"Filter cutoff ({cutoff} Hz) is at or above Nyquist frequency ({nyquist} Hz). Disabling filter."
            logging.warning(msg)
            st.warning(msg)
            return audio
        if normalized_cutoff <= 0:  # Should be caught earlier, but double check
            msg = f"Filter cutoff ({cutoff} Hz) must be positive. Disabling filter."
            logging.warning(msg)
            st.warning(msg)
            return audio

        filter_order = 4
        logging.info(f"Applying {filter_type} filter with cutoff {cutoff} Hz (order {filter_order}).")
        if filter_type == "lowpass":
            b, a = signal.butter(filter_order, normalized_cutoff, btype="low", analog=False)
        elif filter_type == "highpass":
            b, a = signal.butter(filter_order, normalized_cutoff, btype="high", analog=False)
        else:
            logging.warning(f"Unknown filter type: {filter_type}")
            st.warning(f"Unknown filter type: {filter_type}")
            return audio

        # Check data length for filtfilt
        min_len_filtfilt = signal.filtfilt_coeffs(b, a)[-1].size * 3  # Rule of thumb
        if audio.shape[0] <= min_len_filtfilt:
            msg = f"Track is too short ({audio.shape[0]} samples, need > {min_len_filtfilt}) to apply filter reliably. Skipping filter."
            logging.warning(msg)
            st.warning(msg)
            return audio

        audio_filtered = signal.filtfilt(b, a, audio, axis=0)
        logging.debug("Filter applied successfully.")
        return audio_filtered.astype(np.float32)

    except Exception as e:
        logging.exception("Error applying filter.")
        st.error(f"Error applying filter: {e}")
        return audio


def apply_all_effects(track_data: dict) -> AudioData:
    """
    Applies speed, pitch, and filter effects sequentially to the original audio.
    Returns the fully processed audio.
    """
    track_name = track_data.get("name", "Unnamed")
    logging.info(f"Applying all effects to track: '{track_name}'")
    audio = track_data.get("original_audio")
    if audio is None:
        logging.error(f"Track '{track_name}' missing 'original_audio' data.")
        return np.zeros((0, 2), dtype=np.float32)

    audio = audio.copy()  # Start with a fresh copy
    sr = track_data.get("sr", GLOBAL_SR)

    if audio.size == 0:
        logging.warning(f"Track '{track_name}' has no original audio data to process.")
        st.warning(f"Track '{track_name}' has no original audio data to process.")
        return audio

    # --- Apply effects in order ---
    logging.debug(f"Track '{track_name}' - Applying speed change...")
    audio = apply_speed_change(audio, sr, track_data.get("speed_factor", 1.0))

    logging.debug(f"Track '{track_name}' - Applying pitch shift...")
    audio = apply_pitch_shift(audio, sr, track_data.get("pitch_shift", 0))

    logging.debug(f"Track '{track_name}' - Applying filter...")
    audio = apply_filter(audio, sr, track_data.get("filter_type", "off"), track_data.get("filter_cutoff", 8000.0))

    logging.info(f"Finished applying effects to track: '{track_name}'")
    return audio


# --- Mixing Function ---


def mix_tracks(tracks: dict, preview: bool = False) -> AudioData:
    """Mix all tracks based on their settings (volume, pan, mute, solo)."""
    logging.info(f"Starting track mixing. Preview mode: {preview}")
    if not tracks:
        logging.warning("Mix command called but track dictionary is empty.")
        st.info("Track dictionary is empty.")
        return np.zeros((0, 2), dtype=np.float32)

    solo_active = any(t.get("solo", False) for t in tracks.values())
    logging.debug(f"Solo active: {solo_active}")
    if solo_active:
        active_tracks_data = [t for t in tracks.values() if t.get("solo", False)]
    else:
        active_tracks_data = list(tracks.values())

    valid_tracks_to_mix = [t for t in active_tracks_data if not t.get("mute", False) and t.get("processed_audio", np.array([])).size > 0]
    logging.info(f"Found {len(valid_tracks_to_mix)} valid tracks to mix (considering mute/solo/content).")

    if not valid_tracks_to_mix:
        logging.warning("No valid tracks found to mix.")
        st.warning("No valid tracks to mix (check mute/solo states and track content).")
        return np.zeros((0, 2), dtype=np.float32)

    try:
        max_len = max(len(t["processed_audio"]) for t in valid_tracks_to_mix)
        logging.debug(f"Maximum track length for mixing: {max_len} samples.")
    except (ValueError, KeyError) as e:
        logging.exception("Could not determine maximum track length.")
        st.warning(f"Could not determine maximum track length: {e}")
        return np.zeros((0, 2), dtype=np.float32)

    if preview:
        preview_len = int(GLOBAL_SR * 10)  # 10 seconds for preview
        original_max_len = max_len
        max_len = min(max_len, preview_len) if max_len > 0 else preview_len
        logging.debug(f"Preview mode: Limiting mix length from {original_max_len} to {max_len} samples.")

    if max_len <= 0:
        logging.warning("Resulting mix length is zero or negative.")
        st.info("Resulting mix length is zero.")
        return np.zeros((0, 2), dtype=np.float32)

    mix = np.zeros((max_len, 2), dtype=np.float32)
    logging.info(f"Mixing {len(valid_tracks_to_mix)} tracks. Target length: {max_len / GLOBAL_SR:.2f}s")

    for i, t_data in enumerate(valid_tracks_to_mix):
        track_name = t_data.get("name", f"Track_{i}")
        logging.debug(f"Mixing track '{track_name}'...")
        audio = t_data["processed_audio"]
        current_len = len(audio)

        if current_len < max_len:
            pad_width = ((0, max_len - current_len), (0, 0))
            audio_adjusted = np.pad(audio, pad_width, mode="constant", constant_values=0)
            logging.debug(f"Track '{track_name}' padded from {current_len} to {max_len} samples.")
        elif current_len > max_len:
            audio_adjusted = audio[:max_len, :]
            logging.debug(f"Track '{track_name}' truncated from {current_len} to {max_len} samples.")
        else:
            audio_adjusted = audio.copy()

        pan = t_data.get("pan", 0.0)
        vol = t_data.get("volume", 1.0)
        logging.debug(f"Track '{track_name}': vol={vol:.2f}, pan={pan:.2f}")

        # Constant Power Panning
        pan_rad = (pan + 1) * np.pi / 4
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)

        panned_audio = np.zeros_like(audio_adjusted)
        panned_audio[:, 0] = audio_adjusted[:, 0] * left_gain
        panned_audio[:, 1] = audio_adjusted[:, 1] * right_gain

        mix += panned_audio
        logging.debug(f"Added track '{track_name}' to mix.")

    logging.info("Clipping final mix.")
    final_mix = np.clip(mix, -1.0, 1.0)
    logging.info("Mixing complete.")
    return final_mix.astype(np.float32)


# === Streamlit App ===

# App title and markdown moved after imports and logging setup
st.title("🎧 Pro Subliminal Audio Editor V2")
st.markdown("Create layered audio tracks with affirmations, frequencies, and effects.")
st.divider()


# --- Track State Structure Helper ---
def get_default_track_params():
    """Returns a dictionary with default parameters for a new track."""
    return {
        "original_audio": np.zeros((0, 2), dtype=np.float32),
        "processed_audio": np.zeros((0, 2), dtype=np.float32),
        "sr": GLOBAL_SR,
        "name": "New Track",
        "volume": 1.0,
        "mute": False,
        "solo": False,
        "speed_factor": 1.0,
        "pitch_shift": 0,  # Semitones
        "pan": 0.0,  # -1 (L) to 1 (R)
        "filter_type": "off",  # 'off', 'lowpass', 'highpass'
        "filter_cutoff": 8000.0,  # Hz
        "temp_file_path": None,  # Potential future use for cleanup
        "update_counter": 0,  # For forcing audix refresh
    }


# --- Sidebar for Adding Tracks ---
with st.sidebar:
    st.header("➕ Add Tracks")

    # --- File Upload ---
    st.subheader("📁 Upload Audio")
    uploaded_files = st.file_uploader(
        "Upload WAV, MP3, OGG, FLAC files",
        type=["wav", "mp3", "ogg", "flac"],
        accept_multiple_files=True,
        key="upload_files_key",  # Use a distinct key
    )

    loaded_track_names = [t.get("name") for t in st.session_state.tracks.values()]

    if uploaded_files is not None:
        files_processed_this_run = False
        for file in uploaded_files:
            if file.name not in loaded_track_names:
                files_processed_this_run = True
                logging.info(f"Processing uploaded file: {file.name}")
                with st.spinner(f"Loading {file.name}..."):
                    audio, sr = load_audio(file, target_sr=GLOBAL_SR)

                if audio.size == 0:
                    logging.warning(f"Skipped empty or failed track from upload: {file.name}")
                    st.warning(f"Skipped empty or failed track: {file.name}")
                    continue

                track_id = str(uuid.uuid4())
                logging.debug(f"Creating new track ID: {track_id} for file: {file.name}")
                track_params = get_default_track_params()
                track_params.update(
                    {
                        "original_audio": audio,
                        "processed_audio": audio.copy(),
                        "sr": sr,
                        "name": file.name,
                    }
                )
                st.session_state.tracks[track_id] = track_params
                logging.info(f"Successfully loaded and added track '{file.name}' with ID {track_id}")
                st.success(f"Loaded '{file.name}'")
                loaded_track_names.append(file.name)

    # --- Generate Affirmations (TTS) ---
    st.divider()
    st.subheader("🗣️ Generate Affirmations (TTS)")
    affirmation_text = st.text_area("Enter Affirmations (one per line)", height=100, key="affirmation_text")
    tts_button_key = "generate_tts_key"

    if st.button("Generate TTS Audio", key=tts_button_key):
        if affirmation_text:
            logging.info("Generate TTS button clicked.")
            tts_filename = None
            engine = None
            try:
                with st.spinner("Generating TTS audio file..."):
                    logging.debug("Initializing pyttsx3 engine.")
                    engine = pyttsx3.init()
                    # engine.setProperty('rate', 150) # Example config
                    logging.debug("Creating temporary file for TTS output.")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_tts_file:
                        tts_filename = tmp_tts_file.name
                    logging.debug(f"Saving TTS to temporary file: {tts_filename}")
                    engine.save_to_file(affirmation_text, tts_filename)
                    engine.runAndWait()
                    logging.debug("pyttsx3 engine finished.")
                    # engine.stop() # Consider stopping engine

                with st.spinner(f"Loading generated TTS audio..."):
                    logging.debug(f"Loading audio back from temp file: {tts_filename}")
                    with open(tts_filename, "rb") as f:
                        tts_bytes_io = BytesIO(f.read())
                    audio, sr = load_audio(tts_bytes_io, target_sr=GLOBAL_SR)

                if audio.size == 0:
                    logging.error("TTS process resulted in an empty audio file.")
                    raise ValueError("TTS generated an empty audio file.")

                track_id = str(uuid.uuid4())
                logging.debug(f"Creating new track ID: {track_id} for TTS")
                track_params = get_default_track_params()
                track_params.update(
                    {
                        "original_audio": audio,
                        "processed_audio": audio.copy(),
                        "sr": sr,
                        "name": "Affirmations",
                    }
                )
                st.session_state.tracks[track_id] = track_params
                logging.info(f"Successfully generated and added TTS track with ID {track_id}")
                st.success("Affirmations track generated!")

            except Exception as e:
                logging.exception("TTS Generation or Loading Failed.")
                st.error(f"TTS Generation or Loading Failed: {e}")

            finally:
                if tts_filename and os.path.exists(tts_filename):
                    try:
                        os.remove(tts_filename)
                        logging.debug(f"Cleaned up temporary TTS file: {tts_filename}")
                    except OSError as e_os:
                        logging.warning(f"Could not delete temporary TTS file {tts_filename}: {e_os}")
                # Ensure engine is stopped if initialized
                # try:
                #     if engine is not None: engine.stop()
                # except Exception: pass # Ignore stop errors

        else:
            logging.warning("Generate TTS clicked but no text was entered.")
            st.warning("Please enter some text for affirmations.")

    # --- Generate Binaural Beats ---
    # (Similar logging can be added for Binaural and Solfeggio generation)
    st.divider()
    st.subheader("🧠 Generate Binaural Beats")
    bb_cols = st.columns(2)
    bb_duration = bb_cols[0].number_input("Duration (s)", min_value=1, value=60, step=1, key="bb_duration")
    bb_vol = bb_cols[1].slider("Volume##BB", 0.0, 1.0, 0.3, step=0.05, key="bb_volume")
    bb_freq_cols = st.columns(2)
    bb_freq_left = bb_freq_cols[0].number_input("Left Freq (Hz)", min_value=20, max_value=1000, value=200, step=1, key="bb_freq_left")
    bb_freq_right = bb_freq_cols[1].number_input("Right Freq (Hz)", min_value=20, max_value=1000, value=210, step=1, key="bb_freq_right")

    if st.button("Generate Binaural Beats", key="generate_bb"):
        logging.info("Generate Binaural Beats button clicked.")
        with st.spinner("Generating Binaural Beats..."):
            audio = generate_binaural_beats(bb_duration, bb_freq_left, bb_freq_right, GLOBAL_SR, bb_vol)
            track_id = str(uuid.uuid4())
            logging.debug(f"Creating new track ID: {track_id} for Binaural Beats")
            track_params = get_default_track_params()
            track_params.update(
                {
                    "original_audio": audio,
                    "processed_audio": audio.copy(),
                    "sr": GLOBAL_SR,
                    "name": f"Binaural {bb_freq_left}/{bb_freq_right}Hz",
                }
            )
            st.session_state.tracks[track_id] = track_params
            logging.info(f"Successfully generated and added Binaural Beats track with ID {track_id}")
            st.success("Binaural Beats track generated!")

    # --- Generate Solfeggio Frequency ---
    st.divider()
    st.subheader("✨ Generate Solfeggio Frequency")
    solfeggio_freqs = [174, 285, 396, 417, 528, 639, 741, 852, 963]
    solf_cols = st.columns(2)
    solf_freq = solf_cols[0].selectbox("Frequency (Hz)", solfeggio_freqs, index=4, key="solf_freq")
    solf_duration = solf_cols[1].number_input("Duration (s)##Solf", min_value=1, value=60, step=1, key="solf_duration")
    solf_volume = st.slider("Volume##Solf", 0.0, 1.0, 0.3, step=0.05, key="solf_volume")

    if st.button("Generate Solfeggio Track", key="generate_solf"):
        logging.info("Generate Solfeggio Track button clicked.")
        with st.spinner("Generating Solfeggio Track..."):
            audio = generate_solfeggio_frequency(solf_duration, solf_freq, GLOBAL_SR, solf_volume)
            track_id = str(uuid.uuid4())
            logging.debug(f"Creating new track ID: {track_id} for Solfeggio")
            track_params = get_default_track_params()
            track_params.update(
                {
                    "original_audio": audio,
                    "processed_audio": audio.copy(),
                    "sr": GLOBAL_SR,
                    "name": f"Solfeggio {solf_freq}Hz",
                }
            )
            st.session_state.tracks[track_id] = track_params
            logging.info(f"Successfully generated and added Solfeggio track with ID {track_id}")
            st.success(f"Solfeggio {solf_freq}Hz track generated!")


# --- Main Interface for Tracks ---
st.header("🎚️ Tracks Editor")
st.markdown("Adjust settings for each track below. Volume and Pan changes are live. Speed, Pitch, and Filter require 'Apply Effects'.")
st.divider()

track_ids_to_delete = []

if not st.session_state.tracks:
    st.info("No tracks loaded. Add tracks using the sidebar.")
else:
    logging.debug(f"Displaying editor for {len(st.session_state.tracks)} tracks.")
    for track_id in list(st.session_state.tracks.keys()):
        if track_id not in st.session_state.tracks:
            logging.warning(f"Track ID {track_id} listed but not found in session state during UI render. Skipping.")
            continue
        track = st.session_state.tracks[track_id]
        track_name_safe = track.get("name", "Unnamed")  # Safe access for logging

        with st.expander(f"Track: **{track_name_safe}** (`{track_id[:6]}`)", expanded=True):
            logging.debug(f"Rendering expander for track: '{track_name_safe}' ({track_id})")
            col_main, col_controls = st.columns([3, 1])

            # --- Main Column (Waveform & Effects) ---
            with col_main:
                try:  # Add try block for safety during rendering complex tracks
                    processed_audio_len = len(track.get("processed_audio", []))
                    track_sr = track.get("sr", GLOBAL_SR)
                    track_len_sec = processed_audio_len / track_sr if track_sr > 0 else 0
                    st.caption(f"Sample Rate: {track_sr} Hz | Length: {track_len_sec:.2f}s")

                    # --- Waveform Visualization ---
                    audio_to_display = track.get("processed_audio")
                    if audio_to_display is not None and audio_to_display.size > 0:
                        logging.debug(f"Track '{track_name_safe}': Preparing waveform display.")
                        temp_wav_path = save_audio_to_temp(audio_to_display, track_sr)
                        if temp_wav_path:
                            # Log details about the temp file being passed to audix
                            try:
                                with sf.SoundFile(temp_wav_path) as f_debug:
                                    logging.debug(
                                        f"Audix Input ({track_id[:6]}): Temp File='{temp_wav_path}', Header SR={f_debug.samplerate}, Samples={len(f_debug)}, Channels={f_debug.channels}, SR Param={track_sr}"
                                    )
                            except Exception as e_debug_read:
                                logging.error(f"Debug error reading back temp file '{temp_wav_path}': {e_debug_read}")

                            ws_options = WaveSurferOptions(
                                height=100,
                                normalize=True,
                                wave_color="#A020F0",
                                progress_color="#800080",
                                cursor_color="#333333",
                                cursor_width=1,
                                bar_width=2,
                                bar_gap=1,
                            )
                            # Use update_counter in the key to force refresh
                            update_count = track.get("update_counter", 0)
                            audix_key = f"audix_{track_id}_{update_count}"
                            logging.debug(f"Calling audix for track '{track_name_safe}' with key: {audix_key}")
                            audix_result = audix(
                                data=temp_wav_path,
                                sample_rate=track_sr,  # Pass original SR for playback interpretation
                                wavesurfer_options=ws_options,
                                key=audix_key,
                            )
                            # Consider managing temp file cleanup here or centrally if needed
                        else:
                            logging.error(f"Track '{track_name_safe}': Could not save temp file for waveform.")
                            st.error("Could not display waveform (failed to save temp file).")
                    else:
                        logging.debug(f"Track '{track_name_safe}': No processed audio to display.")
                        st.info("Track has no audio data to display.")

                    st.markdown("**Effects** (Require 'Apply Effects' Button)")
                    fx_col1, fx_col2, fx_col3 = st.columns(3)

                    # Speed Factor
                    track["speed_factor"] = fx_col1.slider(
                        "Speed", 0.25, 4.0, track.get("speed_factor", 1.0), step=0.05, key=f"speed_{track_id}", help="Changes speed using time stretching (>1 faster, <1 slower)."
                    )
                    # Pitch Shift
                    track["pitch_shift"] = fx_col2.slider(
                        "Pitch Shift (semitones)", -12, 12, track.get("pitch_shift", 0), step=1, key=f"pitch_{track_id}", help="Changes pitch independently of speed."
                    )
                    # Filter Type
                    current_filter_index = ["off", "lowpass", "highpass"].index(track.get("filter_type", "off"))
                    filter_type = fx_col3.selectbox("Filter", ["off", "lowpass", "highpass"], index=current_filter_index, key=f"filter_type_{track_id}")
                    track["filter_type"] = filter_type
                    # Filter Cutoff
                    filter_enabled = track["filter_type"] != "off"
                    max_cutoff = track_sr / 2 - 1
                    track["filter_cutoff"] = fx_col3.number_input(
                        f"Cutoff ({'Hz' if filter_enabled else 'Disabled'})",
                        min_value=20.0,
                        max_value=max_cutoff if max_cutoff > 20 else 20.0,
                        value=float(track.get("filter_cutoff", 8000.0)),
                        step=100.0,
                        key=f"filter_cutoff_{track_id}",
                        disabled=not filter_enabled,
                        help="Cutoff frequency for low/high pass filter.",
                    )

                    # Apply Effects Button
                    if st.button("⚙️ Apply Effects", key=f"apply_fx_{track_id}", help="Apply Speed, Pitch, and Filter changes to this track's audio."):
                        logging.info(f"Apply Effects button clicked for track: '{track_name_safe}' ({track_id})")
                        if track.get("original_audio") is not None and track["original_audio"].size > 0:
                            with st.spinner(f"Applying effects to '{track_name_safe}'..."):
                                processed_audio = apply_all_effects(track)
                                st.session_state.tracks[track_id]["processed_audio"] = processed_audio
                                # Increment update counter to force audix refresh
                                current_counter = st.session_state.tracks[track_id].get("update_counter", 0)
                                st.session_state.tracks[track_id]["update_counter"] = current_counter + 1
                                logging.debug(f"Track '{track_name_safe}' ({track_id}): Incremented update_counter to {current_counter + 1}")

                            st.success(f"Effects applied to '{track_name_safe}'")
                            logging.info(f"Effects applied successfully for track '{track_name_safe}'. Triggering rerun.")
                            # Debug print for verification (can be removed later)
                            try:
                                debug_orig_len = len(track["original_audio"])
                                debug_proc_len = len(st.session_state.tracks[track_id]["processed_audio"])
                                debug_speed = track["speed_factor"]
                                debug_expected_len = int(debug_orig_len / debug_speed) if debug_speed > 0 else -1  # Approx for resampling, not time_stretch
                                logging.debug(f"--- Apply Effects Post-Processing ({track_id[:6]}) ---")
                                logging.debug(f"- Original samples: {debug_orig_len}")
                                logging.debug(f"- Processed samples: {debug_proc_len}")  # Length might not change much with time_stretch
                                logging.debug(f"- Speed Factor: {debug_speed}")
                                # logging.debug(f"- Expected samples (approx): {debug_expected_len}") # Less relevant for time_stretch
                                logging.debug(f"-------------------------------------------")
                            except Exception as e_debug_print:
                                logging.error(f"Debug print error (Apply Effects): {e_debug_print}")

                            time.sleep(0.1)  # Small delay before rerun
                            st.rerun()
                        else:
                            logging.warning(f"Cannot apply effects: Track '{track_name_safe}' has no original audio data.")
                            st.warning(f"Cannot apply effects: Track '{track_name_safe}' has no original audio data.")

                except Exception as e_render_main:
                    logging.exception(f"Error rendering main column for track '{track_name_safe}' ({track_id})")
                    st.error(f"An error occurred displaying effects/waveform for this track: {e_render_main}")

            # --- Controls Column (Name, Volume, Pan, Mute/Solo, Delete) ---
            with col_controls:
                try:  # Add try block for safety during rendering controls
                    st.markdown("**Track Controls**")
                    track["name"] = st.text_input("Name", value=track.get("name", "Unnamed"), key=f"name_{track_id}")

                    vol_pan_col1, vol_pan_col2 = st.columns(2)
                    track["volume"] = vol_pan_col1.slider(
                        "Volume", 0.0, 2.0, track.get("volume", 1.0), step=0.05, key=f"vol_{track_id}", help="Track volume multiplier (applied live)."
                    )
                    track["pan"] = vol_pan_col2.slider(
                        "Pan", -1.0, 1.0, track.get("pan", 0.0), step=0.1, key=f"pan_{track_id}", help="Stereo balance: -1=Left, 0=Center, 1=Right (applied live)."
                    )

                    mute_solo_col1, mute_solo_col2 = st.columns(2)
                    track["mute"] = mute_solo_col1.checkbox("Mute", value=track.get("mute", False), key=f"mute_{track_id}", help="Silence this track in the mix.")
                    track["solo"] = mute_solo_col2.checkbox(
                        "Solo", value=track.get("solo", False), key=f"solo_{track_id}", help="Only play this track (and other solo'd tracks) in the mix."
                    )

                    st.markdown("---")

                    if st.button("🗑️ Delete Track", key=f"delete_{track_id}", help="Permanently delete this track."):
                        logging.info(f"Delete button clicked for track: '{track_name_safe}' ({track_id})")
                        if track_id not in track_ids_to_delete:
                            track_ids_to_delete.append(track_id)
                        st.warning(f"Track '{track_name_safe}' marked for deletion.")

                except Exception as e_render_ctrl:
                    logging.exception(f"Error rendering controls for track '{track_name_safe}' ({track_id})")
                    st.error(f"An error occurred displaying controls for this track: {e_render_ctrl}")


# --- Process Track Deletions ---
if track_ids_to_delete:
    logging.info(f"Processing deletions for track IDs: {track_ids_to_delete}")
    delete_count = 0
    for track_id_del in track_ids_to_delete:
        if track_id_del in st.session_state.tracks:
            track_name_deleted = st.session_state.tracks[track_id_del].get("name", "Unnamed")
            # Optional: Cleanup associated temp files if tracked via temp_file_path state
            del st.session_state.tracks[track_id_del]
            logging.info(f"Deleted track '{track_name_deleted}' ({track_id_del}) from session state.")
            delete_count += 1
    if delete_count > 0:
        st.toast(f"Deleted {delete_count} track(s).")
        st.rerun()  # Rerun to update the UI fully


# --- Master Controls (Preview & Export) ---
st.divider()
st.header("🔊 Master Output")
master_cols = st.columns(2)

with master_cols[0]:
    if st.button("🎧 Preview Mix (10s)", key="preview_mix", use_container_width=True, help="Generate and play the first 10 seconds of the mixed audio."):
        logging.info("Preview Mix button clicked.")
        if not st.session_state.tracks:
            logging.warning("Preview Mix clicked but no tracks loaded.")
            st.warning("No tracks loaded to generate a preview.")
        else:
            with st.spinner("Generating preview mix..."):
                mix_preview = mix_tracks(st.session_state.tracks, preview=True)
                if mix_preview.size > 0:
                    preview_buffer = save_audio(mix_preview, GLOBAL_SR)
                    st.audio(preview_buffer, format="audio/wav")
                    logging.info("Preview mix generated and displayed.")
                else:
                    logging.warning("Preview mix generated but was empty.")
                    # Warning already shown in mix_tracks if applicable

with master_cols[1]:
    if st.button("💾 Export Full Mix (.wav)", key="export_mix", use_container_width=True, help="Generate the complete mixed audio file for download."):
        logging.info("Export Full Mix button clicked.")
        if not st.session_state.tracks:
            logging.warning("Export Mix clicked but no tracks loaded.")
            st.warning("No tracks loaded to generate an export.")
        else:
            with st.spinner("Generating full mix... This may take a moment."):
                full_mix = mix_tracks(st.session_state.tracks, preview=False)
                if full_mix.size > 0:
                    export_buffer = save_audio(full_mix, GLOBAL_SR)
                    st.download_button(
                        label="⬇️ Download Full Mix (.wav)",
                        data=export_buffer,
                        file_name="pro_subliminal_mix.wav",
                        mime="audio/wav",
                        key="download_full_mix_key",
                        use_container_width=True,
                    )
                    logging.info("Full mix generated and download button displayed.")
                else:
                    logging.warning("Full mix generated but was empty.")
                    # Warning already shown in mix_tracks if applicable


# --- Instructions Expander ---
st.divider()
with st.expander("📖 Show Instructions & Notes", expanded=False):
    # (Instructions remain the same as before)
    st.markdown("""
    ### How to Use:
    1.  **Add Tracks**: Use the **sidebar** (+) to:
        * `Upload Audio`: Load your own files (WAV, MP3, OGG, FLAC).
        * `Generate Affirmations (TTS)`: Create audio from text.
        * `Generate Binaural Beats` or `Generate Solfeggio Frequency`: Add specific tones.
    2.  **Edit Tracks**: In the main **Tracks Editor** area, each track has an expandable section:
        * **Waveform**: Visual preview of the processed audio. Should update after effects are applied.
        * **Effects Section**: Adjust `Speed`, `Pitch Shift`, and select a `Filter` (Lowpass/Highpass) with its `Cutoff` frequency. **You MUST click `⚙️ Apply Effects`** for these changes to take effect on the audio data and be reflected in the waveform/mix/playback.
        * **Track Controls Section**:
            * Rename the track.
            * Adjust `Volume` (loudness) and `Pan` (left/right stereo balance) - these changes are applied *live* during mixing/preview/export.
            * `Mute` or `Solo` the track. Mute silences it. Solo plays *only* this track (and any other solo'd tracks).
            * `🗑️ Delete Track`: Permanently removes the track.
    3.  **Mix & Export**: Use the **Master Output** section at the bottom:
        * `🎧 Preview Mix (10s)`: Listen to the first 10 seconds of how the tracks combine with current Volume/Pan/Mute/Solo settings and *applied* effects.
        * `💾 Export Full Mix (.wav)`: Generate the complete mixed audio based on current settings and download it as a WAV file.

    ### Important Notes:
    - **Apply Effects Button**: Remember that Speed, Pitch, and Filter settings **only affect the audio after you click `⚙️ Apply Effects`** for that specific track. The waveform and individual track playback should update after clicking this button. Volume and Pan are applied dynamically.
    - **Processing Time**: Applying effects and exporting the full mix can take time.
    - **Audio Format**: Processed internally as stereo WAV at $44100$ Hz (32-bit float). Export is 16-bit WAV.
    - **Clipping**: Final mix is clipped to prevent levels exceeding -1.0 to 1.0. Manage track volumes.
    - **Dependencies**: Ensure `requirements.txt` is installed correctly.
    - **Logging**: Debug and informational messages are written to `editor.log` in the script's directory.
    """)

# --- Footer ---
st.divider()
st.caption("Pro Subliminal Audio Editor - Built with Streamlit")
logging.info("Reached end of Streamlit script execution.")

# --- Cleanup (Optional but Recommended) ---
# Ideally, stop the queue listener on script exit/stop
# However, Streamlit doesn't provide a reliable hook for this.
# For long-running servers, manual stop or process termination is needed.
# For typical Streamlit runs, the thread will exit when the main process does.
# Consider adding cleanup for temporary WAV files created for audix if they accumulate.
