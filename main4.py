# ==========================================
# Pro Subliminal Audio Editor V2
# ==========================================
import os
import tempfile
import traceback  # For detailed error logging if needed
import uuid
from io import BytesIO

import librosa
import librosa.effects  # Explicitly import effects
import numpy as np
import pyttsx3
import soundfile as sf
import streamlit as st
from scipy import signal  # For filtering
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_advanced_audio import WaveSurferOptions, audix

# --- Constants ---
GLOBAL_SR = 44100  # Fixed sample rate for all tracks (Hz)

# --- Data Types ---
AudioData = np.ndarray  # Audio data as NumPy array (float32, shape: (samples, 2))
SampleRate = int  # Sample rate in Hz (e.g., 44100)
TrackID = str  # Unique identifier for each track

# --- Session State Initialization ---
if "tracks" not in st.session_state:
    st.session_state.tracks = {}  # Dict[TrackID, dict] storing track data
# Removed play_position as it wasn't actively used in the core logic provided
# if "play_position" not in st.session_state:
#     st.session_state.play_position = 0.0

# --- Utility Functions ---


def load_audio(file_source: UploadedFile | BytesIO | str, target_sr: SampleRate = GLOBAL_SR) -> tuple[AudioData, SampleRate]:
    """
    Load audio file (from UploadedFile, BytesIO, or path string),
    ensure stereo, resample to target_sr.
    Returns (audio_data, sample_rate)
    """
    try:
        # Use a consistent variable name 'source' internally
        source = file_source
        audio, sr = librosa.load(source, sr=None, mono=False)  # Load as stereo if possible

        # Ensure audio is 2D (samples, channels)
        if audio.ndim == 1:
            # If mono after load, duplicate to stereo
            audio = np.stack([audio, audio], axis=-1)
        elif audio.shape[0] == 2:
            # If channels were loaded as the first dimension, transpose
            audio = audio.T

        # Ensure shape is exactly (samples, 2)
        if audio.shape[1] > 2:
            st.warning(f"Audio has more than 2 channels ({audio.shape[1]}). Using only the first two.")
            audio = audio[:, :2]
        elif audio.shape[1] == 1:  # Should be rare after mono=False and ndim check, but handle just in case
            st.warning("Audio loaded as mono unexpectedly. Duplicating channel.")
            audio = np.concatenate([audio, audio], axis=1)

        # Resample if necessary
        if sr != target_sr:
            if audio.size > 0:  # Only resample if there's audio data
                # Librosa resample works on (channels, samples) or (samples,)
                # Need to transpose, resample, transpose back
                audio_resampled = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr)
                audio = audio_resampled.T
            else:
                sr = target_sr  # If empty audio, just update the sr variable

        # Ensure float32 type for processing
        return audio.astype(np.float32), target_sr

    except Exception as e:
        st.error(f"Error loading audio: {e}")
        # st.error(traceback.format_exc()) # Uncomment for detailed debug info
        # Return empty data and target sample rate on failure
        return np.zeros((0, 2), dtype=np.float32), target_sr


def save_audio(audio: AudioData, sr: SampleRate) -> BytesIO:
    """Save NumPy audio array to WAV file format in an in-memory BytesIO buffer."""
    buffer = BytesIO()
    try:
        sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")  # Specify WAV format and subtype
        buffer.seek(0)
    except Exception as e:
        st.error(f"Error saving audio to buffer: {e}")
        # Return empty buffer on failure
        buffer = BytesIO()
    return buffer


def save_audio_to_temp(audio: AudioData, sr: SampleRate) -> str | None:
    """Save NumPy audio array to a temporary WAV file on disk. Returns the file path or None on error."""
    temp_file = None
    try:
        # Use NamedTemporaryFile ensuring it's deleted automatically when closed if 'delete=True'
        # But audix needs the file path, so we use delete=False and manage cleanup manually (or let OS do it)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio, sr, format="WAV", subtype="PCM_16")  # Specify WAV format
            temp_file = tmp.name
        return temp_file
    except Exception as e:
        st.error(f"Failed to save temporary audio file: {e}")
        # Clean up if file was created but write failed
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass
        return None


def generate_binaural_beats(duration: float, freq_left: float, freq_right: float, sr: SampleRate, volume: float) -> AudioData:
    """Generate stereo binaural beats."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = volume * np.sin(2 * np.pi * freq_left * t)
    right = volume * np.sin(2 * np.pi * freq_right * t)
    audio = np.stack([left, right], axis=1).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)  # Clip generated audio


def generate_solfeggio_frequency(duration: float, freq: float, sr: SampleRate, volume: float) -> AudioData:
    """Generate a stereo sine wave at the solfeggio frequency."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = volume * np.sin(2 * np.pi * freq * t)
    audio = np.stack([sine_wave, sine_wave], axis=1).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)  # Clip generated audio


# --- Effect Application Functions ---


def apply_speed_change(audio: AudioData, sr: SampleRate, speed_factor: float) -> AudioData:
    """Apply speed change effect by RESAMPLING (Links Speed & Pitch)."""
    if np.isclose(speed_factor, 1.0):
        return audio
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        st.error("Audio data contains NaN or Inf values before time stretch. Skipping effect.")
        return audio
    if speed_factor <= 0:
        st.warning("Speed factor must be positive.")
        return audio

    st.warning("Using resampling for speed change (also affects pitch).")  # Add warning

    try:
        if audio.size == 0:
            return audio  # Return empty if input is empty

        # Calculate the target sample rate for resampling
        # Using float for precision before ensuring it's valid for librosa if needed
        new_sr_float = float(sr) / speed_factor

        # Add checks for validity, although librosa might handle some cases
        if new_sr_float <= 0:
            st.error(f"Invalid calculated target sample rate: {new_sr_float:.2f} Hz. Skipping speed change.")
            return audio
        # Librosa might need target_sr > 0, ensure float doesn't cause issues, though unlikely here
        target_sr_resample = max(1, new_sr_float)  # Ensure > 0, practically will be much higher

        # Librosa resample works on (channels, samples) or (samples,)
        audio_resampled = librosa.resample(y=audio.T, orig_sr=sr, target_sr=target_sr_resample)

        return audio_resampled.T.astype(np.float32)

    except Exception as e:
        st.error(f"Error applying speed change (resampling method): {e}")
        # st.error(traceback.format_exc()) # Uncomment for detailed trace
        return audio


def apply_speed_change_old(audio: AudioData, sr: SampleRate, speed_factor: float) -> AudioData:
    """Apply speed change using time stretching. Returns modified audio."""
    if not np.isclose(speed_factor, 1.0):
        if speed_factor <= 0:
            st.warning("Speed factor must be positive.")
            return audio
        try:
            # Ensure audio is non-empty before stretching
            if audio.size == 0:
                return audio
            # librosa.effects.time_stretch works on (channels, samples) or (samples,)
            audio_stretched = librosa.effects.time_stretch(audio.T, rate=speed_factor)
            return audio_stretched.T.astype(np.float32)
        except Exception as e:
            st.error(f"Error applying speed change: {e}")
            return audio
    return audio


def apply_pitch_shift(audio: AudioData, sr: SampleRate, n_steps: float) -> AudioData:
    """Apply pitch shift in semitones. Returns modified audio."""
    if not np.isclose(n_steps, 0.0):
        try:
            # Ensure audio is non-empty before shifting
            if audio.size == 0:
                return audio
            # librosa.effects.pitch_shift works on (channels, samples) or (samples,)
            audio_shifted = librosa.effects.pitch_shift(audio.T, sr=sr, n_steps=n_steps)
            return audio_shifted.T.astype(np.float32)
        except Exception as e:
            st.error(f"Error applying pitch shift: {e}")
            return audio
    return audio


def apply_filter(audio: AudioData, sr: SampleRate, filter_type: str, cutoff: float) -> AudioData:
    """Apply low-pass or high-pass filter. Returns modified audio."""
    if filter_type == "off" or cutoff <= 0:
        return audio

    try:
        if audio.size == 0:
            return audio  # Ensure audio is non-empty

        nyquist = 0.5 * sr
        normalized_cutoff = cutoff / nyquist

        # Validate cutoff relative to Nyquist frequency
        if normalized_cutoff >= 1.0:
            st.warning(f"Filter cutoff ({cutoff} Hz) is at or above Nyquist frequency ({nyquist} Hz). Disabling filter.")
            return audio
        if normalized_cutoff <= 0:
            st.warning(f"Filter cutoff ({cutoff} Hz) must be positive. Disabling filter.")
            return audio

        # Design the filter (Butterworth filter, order 4 - common choice)
        filter_order = 4
        if filter_type == "lowpass":
            b, a = signal.butter(filter_order, normalized_cutoff, btype="low", analog=False)
        elif filter_type == "highpass":
            b, a = signal.butter(filter_order, normalized_cutoff, btype="high", analog=False)
        else:
            st.warning(f"Unknown filter type: {filter_type}")
            return audio  # Should not happen if UI is correct

        # Apply the filter using filtfilt for zero phase distortion
        # filtfilt needs sufficient data length, check shape[0] (samples)
        if audio.shape[0] <= signal.filtfilt_coeffs(b, a)[-1].size * 3:  # Rule of thumb from scipy docs
            st.warning(f"Track is too short ({audio.shape[0]} samples) to apply filter reliably. Skipping filter.")
            return audio

        audio_filtered = signal.filtfilt(b, a, audio, axis=0)  # Apply along sample axis
        return audio_filtered.astype(np.float32)

    except Exception as e:
        st.error(f"Error applying filter: {e}")
        # st.error(traceback.format_exc()) # Uncomment for detailed debug info
        return audio


def apply_all_effects(track_data: dict) -> AudioData:
    """
    Applies speed, pitch, and filter effects sequentially to the original audio.
    Returns the fully processed audio ready for mixing (volume/pan applied later).
    """
    # Start with a fresh copy of the original audio
    audio = track_data["original_audio"].copy()
    sr = track_data["sr"]

    # Check if audio data exists
    if audio.size == 0:
        st.warning(f"Track '{track_data['name']}' has no audio data to process.")
        return audio  # Return empty array

    # --- Apply effects in order ---
    # 1. Speed Change (Time Stretching)
    audio = apply_speed_change(audio, sr, track_data["speed_factor"])

    # 2. Pitch Shift (applied to the speed-adjusted audio)
    # Note: Pitch shifting after time stretching maintains the new duration
    audio = apply_pitch_shift(audio, sr, track_data["pitch_shift"])

    # 3. Filter (applied to the speed/pitch adjusted audio)
    audio = apply_filter(audio, sr, track_data["filter_type"], track_data["filter_cutoff"])

    return audio  # Return the final processed audio


# --- Mixing Function ---


def mix_tracks(tracks: dict, preview: bool = False) -> AudioData:
    """Mix all tracks based on their settings (volume, pan, mute, solo)."""
    if not tracks:
        st.info("Track dictionary is empty.")
        return np.zeros((0, 2), dtype=np.float32)

    # Determine active tracks based on Solo status
    solo_active = any(t["solo"] for t in tracks.values())
    if solo_active:
        active_tracks_data = [t for t in tracks.values() if t["solo"]]
    else:
        active_tracks_data = list(tracks.values())  # Mix all non-solo'd tracks

    # Filter out muted tracks and tracks with no audio *before* calculating length
    valid_tracks_to_mix = [t for t in active_tracks_data if not t["mute"] and t["processed_audio"].size > 0]

    if not valid_tracks_to_mix:
        st.warning("No valid tracks to mix (check mute/solo states and track content).")
        return np.zeros((0, 2), dtype=np.float32)

    # Find max length among the tracks that will actually be mixed
    try:
        max_len = max(len(t["processed_audio"]) for t in valid_tracks_to_mix)
    except ValueError:  # Handles case where valid_tracks_to_mix is empty after filtering
        st.warning("Could not determine maximum track length.")
        return np.zeros((0, 2), dtype=np.float32)

    if preview:
        preview_len = int(GLOBAL_SR * 10)  # 10 seconds for preview
        max_len = min(max_len, preview_len) if max_len > 0 else preview_len

    if max_len <= 0:
        st.info("Resulting mix length is zero.")
        return np.zeros((0, 2), dtype=np.float32)

    # Initialize the mix buffer
    mix = np.zeros((max_len, 2), dtype=np.float32)
    st.write(f"Mixing {len(valid_tracks_to_mix)} tracks. Target length: {max_len / GLOBAL_SR:.2f}s")  # Debug info

    # --- Mix the valid tracks ---
    for t_data in valid_tracks_to_mix:
        audio = t_data["processed_audio"]
        current_len = len(audio)

        # Pad or truncate audio to match max_len for mixing
        if current_len < max_len:
            pad_width = ((0, max_len - current_len), (0, 0))  # Pad samples dim, not channels
            audio_adjusted = np.pad(audio, pad_width, mode="constant", constant_values=0)
        elif current_len > max_len:
            audio_adjusted = audio[:max_len, :]
        else:
            audio_adjusted = audio.copy()  # Use a copy to avoid modifying track data directly here

        # Apply Panning and Volume for this track during mixing
        pan = t_data["pan"]  # Pan value from -1.0 (L) to 1.0 (R)
        vol = t_data["volume"]  # Volume multiplier

        # Constant Power Panning Law
        # Angle derived from pan: 0 at center, pi/4 at hard left/right
        pan_rad = (pan + 1) * np.pi / 4
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)

        # Apply gain to left and right channels
        panned_audio = np.zeros_like(audio_adjusted)
        panned_audio[:, 0] = audio_adjusted[:, 0] * left_gain  # Apply gain to Left channel
        panned_audio[:, 1] = audio_adjusted[:, 1] * right_gain  # Apply gain to Right channel

        # Add the processed audio of this track to the main mix buffer
        mix += panned_audio

    # Final Clipping to prevent audio distortion (-1.0 to 1.0 range)
    final_mix = np.clip(mix, -1.0, 1.0)
    return final_mix.astype(np.float32)


# === Streamlit App ===

st.set_page_config(layout="wide", page_title="Pro Subliminal Editor")
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
        "filter_cutoff": 8000.0,  # Hz (default sensible cutoff)
        "temp_file_path": None,  # To potentially track temp files for audix
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

    # Get names of tracks already loaded to prevent duplicates
    loaded_track_names = [t["name"] for t in st.session_state.tracks.values()]

    if uploaded_files is not None:
        files_processed_this_run = False
        for file in uploaded_files:
            if file.name not in loaded_track_names:
                files_processed_this_run = True
                with st.spinner(f"Loading {file.name}..."):
                    audio, sr = load_audio(file, target_sr=GLOBAL_SR)

                if audio.size == 0:  # Check if loading resulted in empty audio
                    st.warning(f"Skipped empty or failed track: {file.name}")
                    continue

                track_id = str(uuid.uuid4())
                track_params = get_default_track_params()  # Start with defaults
                track_params.update(
                    {
                        "original_audio": audio,
                        "processed_audio": audio.copy(),  # Start processed as a copy
                        "sr": sr,
                        "name": file.name,
                    }
                )
                st.session_state.tracks[track_id] = track_params
                st.success(f"Loaded '{file.name}'")
                loaded_track_names.append(file.name)  # Update list for current run

        # Optional: Rerun might help ensure UI consistency after uploads.
        # if files_processed_this_run:
        #     st.rerun()

    # --- Generate Affirmations (TTS) ---
    st.divider()
    st.subheader("🗣️ Generate Affirmations (TTS)")
    affirmation_text = st.text_area("Enter Affirmations (one per line)", height=100, key="affirmation_text")
    tts_button_key = "generate_tts_key"

    if st.button("Generate TTS Audio", key=tts_button_key):
        if affirmation_text:
            tts_filename = None  # Initialize filename variable
            engine = None  # Initialize engine variable
            try:
                with st.spinner("Generating TTS audio file..."):
                    engine = pyttsx3.init()
                    # --- Configure pyttsx3 engine ---
                    # engine.setProperty('rate', 150) # Example: Set speed
                    # voices = engine.getProperty('voices')
                    # if voices: engine.setProperty('voice', voices[0].id) # Example: Set voice
                    # ---------------------------------

                    # 1. Create a temporary file and get its name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_tts_file:
                        tts_filename = tmp_tts_file.name

                    # 2. Save TTS audio to the temporary filename
                    engine.save_to_file(affirmation_text, tts_filename)
                    engine.runAndWait()
                    # engine.stop() # Stop the engine explicitly

                # 3. Load the audio from the generated temporary file
                with st.spinner(f"Loading generated TTS audio..."):
                    with open(tts_filename, "rb") as f:
                        tts_bytes_io = BytesIO(f.read())
                    audio, sr = load_audio(tts_bytes_io, target_sr=GLOBAL_SR)

                if audio.size == 0:
                    raise ValueError("TTS generated an empty audio file.")

                # --- Add the loaded audio as a new track ---
                track_id = str(uuid.uuid4())
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
                st.success("Affirmations track generated!")

            except Exception as e:
                st.error(f"TTS Generation or Loading Failed: {e}")
                # st.error(traceback.format_exc()) # Uncomment for detailed errors

            finally:
                # 4. Clean up the temporary file
                if tts_filename and os.path.exists(tts_filename):
                    try:
                        os.remove(tts_filename)
                    except OSError as e_os:
                        st.warning(f"Could not delete temporary TTS file {tts_filename}: {e_os}")
                # 5. Ensure engine is stopped
                # try:
                #     if engine is not None: # Check if engine was initialized
                #         engine.stop()
                # except Exception as e_eng:
                #      st.warning(f"Error stopping TTS engine: {e_eng}")

        else:
            st.warning("Please enter some text for affirmations.")

    # --- Generate Binaural Beats ---
    st.divider()
    st.subheader("🧠 Generate Binaural Beats")
    bb_cols = st.columns(2)
    bb_duration = bb_cols[0].number_input("Duration (s)", min_value=1, value=60, step=1, key="bb_duration")
    bb_vol = bb_cols[1].slider("Volume##BB", 0.0, 1.0, 0.3, step=0.05, key="bb_volume")  # Lower default volume
    bb_freq_cols = st.columns(2)
    bb_freq_left = bb_freq_cols[0].number_input("Left Freq (Hz)", min_value=20, max_value=1000, value=200, step=1, key="bb_freq_left")
    bb_freq_right = bb_freq_cols[1].number_input("Right Freq (Hz)", min_value=20, max_value=1000, value=210, step=1, key="bb_freq_right")

    if st.button("Generate Binaural Beats", key="generate_bb"):
        with st.spinner("Generating Binaural Beats..."):
            audio = generate_binaural_beats(bb_duration, bb_freq_left, bb_freq_right, GLOBAL_SR, bb_vol)
            track_id = str(uuid.uuid4())
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
            st.success("Binaural Beats track generated!")

    # --- Generate Solfeggio Frequency ---
    st.divider()
    st.subheader("✨ Generate Solfeggio Frequency")
    solfeggio_freqs = [174, 285, 396, 417, 528, 639, 741, 852, 963]
    solf_cols = st.columns(2)
    solf_freq = solf_cols[0].selectbox("Frequency (Hz)", solfeggio_freqs, index=4, key="solf_freq")  # Default to 528 Hz
    solf_duration = solf_cols[1].number_input("Duration (s)##Solf", min_value=1, value=60, step=1, key="solf_duration")
    solf_volume = st.slider("Volume##Solf", 0.0, 1.0, 0.3, step=0.05, key="solf_volume")  # Lower default volume

    if st.button("Generate Solfeggio Track", key="generate_solf"):
        with st.spinner("Generating Solfeggio Track..."):
            audio = generate_solfeggio_frequency(solf_duration, solf_freq, GLOBAL_SR, solf_volume)
            track_id = str(uuid.uuid4())
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
            st.success(f"Solfeggio {solf_freq}Hz track generated!")


# --- Main Interface for Tracks ---
st.header("🎚️ Tracks Editor")
st.markdown("Adjust settings for each track below. Volume and Pan changes are live. Speed, Pitch, and Filter require 'Apply Effects'.")
st.divider()

track_ids_to_delete = []  # Store IDs to delete after iteration
temp_files_to_clean = []  # List to potentially track temp files from audix

if not st.session_state.tracks:
    st.info("No tracks loaded. Add tracks using the sidebar.")
else:
    # Iterate safely over a copy of keys in case of deletion
    for track_id in list(st.session_state.tracks.keys()):
        # Check if track still exists (might be deleted in a previous iteration by error?)
        if track_id not in st.session_state.tracks:
            continue
        track = st.session_state.tracks[track_id]

        # Use track name and ID in the expander title
        with st.expander(f"Track: **{track.get('name', 'Unnamed')}** (`{track_id[:6]}`)", expanded=True):
            col_main, col_controls = st.columns([3, 1])  # Main area (waveform, effects) and controls column

            # --- Main Column (Waveform & Effects) ---
            with col_main:
                st.caption(f"Sample Rate: {track.get('sr', GLOBAL_SR)} Hz | Length: {len(track.get('processed_audio', [])) / track.get('sr', GLOBAL_SR):.2f}s")

                # --- Waveform Visualization ---
                audio_to_display = track.get("processed_audio")
                if audio_to_display is not None and audio_to_display.size > 0:
                    temp_wav_path = save_audio_to_temp(audio_to_display, track["sr"])
                    if temp_wav_path:
                        # Add path to list for potential cleanup later if needed
                        # temp_files_to_clean.append(temp_wav_path)
                        ws_options = WaveSurferOptions(
                            height=100,
                            normalize=True,
                            wave_color="#A020F0",
                            progress_color="#800080",
                            cursor_color="#333333",
                            cursor_width=1,
                            bar_width=2,
                            bar_gap=1,
                            # Consider adding more options if desired, e.g., interaction=True
                        )

                        # ---> MODIFY DEBUG BLOCK BELOW (use print) <---
                        try:
                            with sf.SoundFile(temp_wav_path) as f_debug:
                                debug_wav_sr = f_debug.samplerate
                                debug_wav_samples = len(f_debug)
                                debug_wav_channels = f_debug.channels
                            print(f"--- Audix Input Debug ({track_id[:6]}) ---")  # Header
                            print(f"- Temp File Path: {temp_wav_path}")
                            print(f"- File Header SR: {debug_wav_sr}")
                            print(f"- File Samples: {debug_wav_samples}")
                            print(f"- File Channels: {debug_wav_channels}")
                            print(f"- SR Passed to Audix: {track['sr']}")  # Confirm what we intend to pass
                            print(f"---------------------------------------")  # Footer
                        except Exception as e_debug_read:
                            print(f"*** Debug error reading back temp file: {e_debug_read} ***")
                        # ---> END OF MODIFIED DEBUG BLOCK <---

                        audix_result = audix(data=temp_wav_path, sample_rate=track["sr"], wavesurfer_options=ws_options, key=f"audix_{track_id}")
                        # Note: Streamlit usually handles temp files created this way unless delete=False is used aggressively
                        # Manual cleanup might be needed in long-running apps or if files aren't closed properly.
                        # Example cleanup (run cautiously):
                        # if os.path.exists(temp_wav_path):
                        #     try: os.remove(temp_wav_path)
                        #     except OSError: pass
                    else:
                        st.error("Could not display waveform (failed to save temp file).")
                else:
                    st.info("Track has no audio data to display.")

                st.markdown("**Effects** (Require 'Apply Effects' Button)")
                fx_col1, fx_col2, fx_col3 = st.columns(3)

                # Speed Factor
                track["speed_factor"] = fx_col1.slider(
                    "Speed", 0.25, 4.0, track.get("speed_factor", 1.0), step=0.05, key=f"speed_{track_id}", help="Changes speed and pitch together. >1 faster, <1 slower."
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
                max_cutoff = track.get("sr", GLOBAL_SR) / 2 - 1  # Nyquist limit
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

                # Apply Effects Button - Triggers processing
                if st.button("⚙️ Apply Effects", key=f"apply_fx_{track_id}", help="Apply Speed, Pitch, and Filter changes to this track's audio."):
                    if track.get("original_audio") is not None and track["original_audio"].size > 0:
                        with st.spinner(f"Applying effects to '{track['name']}'..."):
                            processed_audio = apply_all_effects(track)  # Calculate new audio
                            # Update the 'processed_audio' in session state
                            st.session_state.tracks[track_id]["processed_audio"] = processed_audio
                        st.success(f"Effects applied to '{track['name']}'")
                        # ---> MODIFY DEBUG LINES BELOW (use print) <---
                        try:
                            debug_orig_len = len(track["original_audio"])
                            debug_proc_len = len(st.session_state.tracks[track_id]["processed_audio"])
                            debug_speed = track["speed_factor"]
                            debug_expected_len = int(debug_orig_len / debug_speed) if debug_speed > 0 else -1
                            print(f"--- Apply Effects Debug ({track_id[:6]}) ---")  # Header for clarity
                            print(f"- Original samples: {debug_orig_len}")
                            print(f"- Processed samples: {debug_proc_len}")
                            print(f"- Speed Factor: {debug_speed}")
                            print(f"- Expected samples (approx): {debug_expected_len}")
                            print(f"-------------------------------------------")  # Footer
                        except Exception as e_debug_print:
                            print(f"*** Debug print error (Apply Effects): {e_debug_print} ***")  # Print errors too
                        # ---> END OF MODIFIED DEBUG LINES <---

                        # Optional small delay before rerun, might help sometimes with state consistency
                        import time

                        time.sleep(0.2)
                        st.rerun()  # Rerun to update waveform and length display immediately
                    else:
                        st.warning(f"Cannot apply effects: Track '{track['name']}' has no original audio data.")

            # --- Controls Column (Name, Volume, Pan, Mute/Solo, Delete) ---
            with col_controls:
                st.markdown("**Track Controls**")
                track["name"] = st.text_input("Name", value=track.get("name", "Unnamed"), key=f"name_{track_id}")

                # Volume and Pan side-by-side
                vol_pan_col1, vol_pan_col2 = st.columns(2)
                track["volume"] = vol_pan_col1.slider(
                    "Volume", 0.0, 2.0, track.get("volume", 1.0), step=0.05, key=f"vol_{track_id}", help="Track volume multiplier (applied live)."
                )
                track["pan"] = vol_pan_col2.slider(
                    "Pan", -1.0, 1.0, track.get("pan", 0.0), step=0.1, key=f"pan_{track_id}", help="Stereo balance: -1=Left, 0=Center, 1=Right (applied live)."
                )

                # Mute/Solo Checkboxes side-by-side
                mute_solo_col1, mute_solo_col2 = st.columns(2)
                track["mute"] = mute_solo_col1.checkbox("Mute", value=track.get("mute", False), key=f"mute_{track_id}", help="Silence this track in the mix.")
                track["solo"] = mute_solo_col2.checkbox(
                    "Solo", value=track.get("solo", False), key=f"solo_{track_id}", help="Only play this track (and other solo'd tracks) in the mix."
                )

                st.markdown("---")  # Separator

                # --- Delete Button ---
                if st.button("🗑️ Delete Track", key=f"delete_{track_id}", help="Permanently delete this track."):
                    if track_id not in track_ids_to_delete:  # Avoid duplicates if clicked fast
                        track_ids_to_delete.append(track_id)
                    # Optionally trigger immediate UI feedback before rerun
                    st.warning(f"Track '{track['name']}' marked for deletion.")


# --- Process Track Deletions ---
if track_ids_to_delete:
    delete_count = 0
    for track_id_del in track_ids_to_delete:
        if track_id_del in st.session_state.tracks:
            # --- Optional: Cleanup associated resources like temp files ---
            # track_to_del = st.session_state.tracks[track_id_del]
            # if track_to_del.get("temp_file_path") and os.path.exists(track_to_del["temp_file_path"]):
            #     try: os.remove(track_to_del["temp_file_path"])
            #     except OSError: pass
            # -------------------------------------------------------------
            del st.session_state.tracks[track_id_del]
            delete_count += 1
    if delete_count > 0:
        st.toast(f"Deleted {delete_count} track(s).")  # Show brief confirmation
        st.rerun()  # Rerun Streamlit to update the UI fully


# --- Master Controls (Preview & Export) ---
st.divider()
st.header("🔊 Master Output")
master_cols = st.columns(2)

with master_cols[0]:
    if st.button("🎧 Preview Mix (10s)", key="preview_mix", use_container_width=True, help="Generate and play the first 10 seconds of the mixed audio."):
        if not st.session_state.tracks:
            st.warning("No tracks loaded to generate a preview.")
        else:
            with st.spinner("Generating preview mix..."):
                mix_preview = mix_tracks(st.session_state.tracks, preview=True)
                if mix_preview.size > 0:
                    preview_buffer = save_audio(mix_preview, GLOBAL_SR)
                    st.audio(preview_buffer, format="audio/wav")
                else:
                    # Warning already shown in mix_tracks if applicable
                    pass  # Avoid duplicate warnings

with master_cols[1]:
    if st.button("💾 Export Full Mix (.wav)", key="export_mix", use_container_width=True, help="Generate the complete mixed audio file for download."):
        if not st.session_state.tracks:
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
                        key="download_full_mix_key",  # Use distinct key
                        use_container_width=True,
                    )
                else:
                    # Warning already shown in mix_tracks if applicable
                    pass  # Avoid duplicate warnings


# --- Instructions Expander ---
st.divider()
with st.expander("📖 Show Instructions & Notes", expanded=False):
    st.markdown("""
    ### How to Use:
    1.  **Add Tracks**: Use the **sidebar** (+) to:
        * `Upload Audio`: Load your own files (WAV, MP3, OGG, FLAC).
        * `Generate Affirmations (TTS)`: Create audio from text.
        * `Generate Binaural Beats` or `Generate Solfeggio Frequency`: Add specific tones.
    2.  **Edit Tracks**: In the main **Tracks Editor** area, each track has an expandable section:
        * **Waveform**: Visual preview of the processed audio.
        * **Effects Section**: Adjust `Speed`, `Pitch Shift`, and select a `Filter` (Lowpass/Highpass) with its `Cutoff` frequency. **You MUST click `⚙️ Apply Effects`** for these changes to take effect on the audio data and be reflected in the waveform/mix.
        * **Track Controls Section**:
            * Rename the track.
            * Adjust `Volume` (loudness) and `Pan` (left/right stereo balance) - these changes are applied *live* during mixing/preview/export.
            * `Mute` or `Solo` the track. Mute silences it. Solo plays *only* this track (and any other solo'd tracks).
            * `🗑️ Delete Track`: Permanently removes the track.
    3.  **Mix & Export**: Use the **Master Output** section at the bottom:
        * `🎧 Preview Mix (10s)`: Listen to the first 10 seconds of how the tracks combine with current Volume/Pan/Mute/Solo settings and *applied* effects.
        * `💾 Export Full Mix (.wav)`: Generate the complete mixed audio based on current settings and download it as a WAV file.

    ### Important Notes:
    - **Apply Effects Button**: Remember that Speed, Pitch, and Filter settings **only affect the audio after you click `⚙️ Apply Effects`** for that specific track. Volume and Pan are applied dynamically without needing the button.
    - **Processing Time**: Applying effects and exporting the full mix can take time, especially for long tracks or complex effects. Please be patient.
    - **Audio Format**: All audio is processed internally as stereo WAV at $44100$ Hz using 32-bit float precision. The final export is a 16-bit WAV file.
    - **Clipping**: The final mix is clipped to prevent levels exceeding the maximum ($-1.0$ to $1.0$), but individual track volumes should still be managed carefully to avoid excessive loudness or distortion *before* the final clipping stage.
    - **Dependencies**: Ensure you have installed all required libraries (`streamlit`, `numpy`, `librosa`, `soundfile`, `pyttsx3`, `scipy`, `streamlit-advanced-audio`).
    """)

# --- Footer ---
st.divider()
st.caption("Pro Subliminal Audio Editor - Built with Streamlit")
