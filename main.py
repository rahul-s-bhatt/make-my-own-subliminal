# ==========================================
# Pro Subliminal Audio Editor V2 (OOP Refactor)
# ==========================================
import streamlit as st

# --- Early Config ---
st.set_page_config(layout="wide", page_title="Pro Subliminal Editor OOP")

# --- Imports ---
import logging
import logging.handlers
import os
import queue
import tempfile
import time
import traceback
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

# --- Logging Setup ---
log_queue = queue.Queue(-1)
log_file = "editor_oop.log"
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s")
file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=1 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)
queue_handler = logging.handlers.QueueHandler(log_queue)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(queue_handler)
listener = logging.handlers.QueueListener(log_queue, file_handler, respect_handler_level=True)
listener.start()
logger = logging.getLogger(__name__)  # Use a specific logger for the app module
logger.info("Application starting up.")

# --- Core Libraries ---
try:
    import librosa
    import librosa.effects
    import numpy as np
    import pyttsx3
    import soundfile as sf
    from scipy import signal
    from streamlit.runtime.uploaded_file_manager import UploadedFile
    from streamlit_advanced_audio import WaveSurferOptions, audix
except ImportError as e:
    logger.exception("Failed to import core libraries.")
    st.error(f"Core library import failed: {e}. Please check installation and restart.")
    st.stop()

# --- Constants ---
GLOBAL_SR = 44100
logger.debug(f"Global Sample Rate set to: {GLOBAL_SR} Hz")

# --- Data Types ---
AudioData = np.ndarray
SampleRate = int
TrackID = str
TrackData = Dict[str, Any]  # Type hint for track dictionaries

# ==========================================
# 1. Audio Processing Utilities (Functions)
# ==========================================
# These functions are mostly stateless operations on audio data.
# Keeping them as functions for now, but they could be grouped into a class later.


def load_audio(file_source: UploadedFile | BytesIO | str, target_sr: SampleRate = GLOBAL_SR) -> tuple[AudioData, SampleRate]:
    """Loads, ensures stereo, and resamples audio."""
    # (Implementation is the same as the previous version)
    logger.info(f"Loading audio from source type: {type(file_source)}")
    try:
        source = file_source
        audio, sr = librosa.load(source, sr=None, mono=False)
        logger.debug(f"Loaded audio with original SR: {sr}, shape: {audio.shape}")

        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)
        elif audio.shape[0] == 2:
            audio = audio.T

        if audio.shape[1] > 2:
            logger.warning(f"Audio has {audio.shape[1]} channels. Using first two.")
            audio = audio[:, :2]
        elif audio.shape[1] == 1:
            logger.warning("Audio loaded as mono unexpectedly. Duplicating channel.")
            audio = np.concatenate([audio, audio], axis=1)

        if sr != target_sr:
            logger.info(f"Resampling audio from {sr} Hz to {target_sr} Hz.")
            if audio.size > 0:
                audio_resampled = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr)
                audio = audio_resampled.T
            else:
                sr = target_sr
        else:
            logger.debug("Audio already at target sample rate.")

        return audio.astype(np.float32), target_sr
    except Exception as e:
        logger.exception("Error loading audio.")
        st.error(f"Error loading audio: {e}")
        return np.zeros((0, 2), dtype=np.float32), target_sr


def save_audio(audio: AudioData, sr: SampleRate) -> BytesIO:
    """Saves audio to an in-memory BytesIO buffer (WAV format)."""
    # (Implementation is the same as the previous version)
    buffer = BytesIO()
    try:
        sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        logger.debug(f"Saved audio ({audio.shape}, {sr}Hz) to BytesIO buffer.")
    except Exception as e:
        logger.exception("Error saving audio to buffer.")
        st.error(f"Error saving audio to buffer: {e}")
        buffer = BytesIO()
    return buffer


def save_audio_to_temp(audio: AudioData, sr: SampleRate) -> str | None:
    """Saves audio to a temporary WAV file."""
    # (Implementation is the same as the previous version)
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as tmp:
            sf.write(tmp, audio, sr, format="WAV", subtype="PCM_16")
            temp_file_path = tmp.name
        logger.debug(f"Saved audio ({audio.shape}, {sr}Hz) to temporary file: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        logger.exception("Failed to save temporary audio file.")
        st.error(f"Failed to save temporary audio file: {e}")
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e_os:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e_os}")
        return None


def generate_binaural_beats(duration: float, freq_left: float, freq_right: float, sr: SampleRate, volume: float) -> AudioData:
    """Generates stereo binaural beats."""
    # (Implementation is the same as the previous version)
    logger.info(f"Generating binaural beats: dur={duration}s, L={freq_left}Hz, R={freq_right}Hz, vol={volume}")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = volume * np.sin(2 * np.pi * freq_left * t)
    right = volume * np.sin(2 * np.pi * freq_right * t)
    audio = np.stack([left, right], axis=1).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def generate_solfeggio_frequency(duration: float, freq: float, sr: SampleRate, volume: float) -> AudioData:
    """Generates a stereo sine wave at the solfeggio frequency."""
    # (Implementation is the same as the previous version)
    logger.info(f"Generating Solfeggio frequency: dur={duration}s, F={freq}Hz, vol={volume}")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = volume * np.sin(2 * np.pi * freq * t)
    audio = np.stack([sine_wave, sine_wave], axis=1).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


# --- Effect Application Functions ---
def apply_speed_change(audio: AudioData, sr: SampleRate, speed_factor: float) -> AudioData:
    """Applies speed change using time stretching."""
    # (Implementation is the same as the previous version - time_stretch)
    logger.debug(f"Applying speed change with factor: {speed_factor} (time_stretch method)")
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
                return audio
            logging.info(f"Applying speed change using time stretch (factor: {speed_factor})...")
            audio_stretched = librosa.effects.time_stretch(audio.T, rate=speed_factor)
            logging.debug("Time stretch applied successfully.")
            return audio_stretched.T.astype(np.float32)
        except Exception as e:
            logging.exception("Error applying speed change (time stretch method).")
            st.error(f"Error applying speed change (time stretch method): {e}")
            return audio
    return audio


def apply_pitch_shift(audio: AudioData, sr: SampleRate, n_steps: float) -> AudioData:
    """Applies pitch shift."""
    # (Implementation is the same as the previous version)
    logger.debug(f"Applying pitch shift: {n_steps} semitones")
    if not np.isclose(n_steps, 0.0):
        try:
            if audio.size == 0:
                return audio
            logging.info(f"Applying pitch shift ({n_steps} semitones)...")
            audio_shifted = librosa.effects.pitch_shift(audio.T, sr=sr, n_steps=n_steps)
            logging.debug("Pitch shift applied successfully.")
            return audio_shifted.T.astype(np.float32)
        except Exception as e:
            logging.exception("Error applying pitch shift.")
            st.error(f"Error applying pitch shift: {e}")
            return audio
    return audio


def apply_filter(audio: AudioData, sr: SampleRate, filter_type: str, cutoff: float) -> AudioData:
    """Applies low/high-pass filter."""
    # (Implementation is the same as the previous version)
    logger.debug(f"Applying filter: type={filter_type}, cutoff={cutoff} Hz")
    if filter_type == "off" or cutoff <= 0:
        return audio
    try:
        if audio.size == 0:
            return audio
        nyquist = 0.5 * sr
        normalized_cutoff = cutoff / nyquist
        if normalized_cutoff >= 1.0:
            msg = f"Filter cutoff ({cutoff} Hz) >= Nyquist ({nyquist} Hz)."
            logging.warning(msg)
            st.warning(msg)
            return audio
        if normalized_cutoff <= 0:
            msg = f"Filter cutoff ({cutoff} Hz) must be positive."
            logging.warning(msg)
            st.warning(msg)
            return audio
        filter_order = 4
        logging.info(f"Applying {filter_type} filter, cutoff {cutoff} Hz, order {filter_order}.")
        if filter_type == "lowpass":
            b, a = signal.butter(filter_order, normalized_cutoff, btype="low")
        elif filter_type == "highpass":
            b, a = signal.butter(filter_order, normalized_cutoff, btype="high")
        else:
            logging.warning(f"Unknown filter type: {filter_type}")
            st.warning(f"Unknown filter type: {filter_type}")
            return audio
        min_len_filtfilt = signal.filtfilt_coeffs(b, a)[-1].size * 3
        if audio.shape[0] <= min_len_filtfilt:
            msg = f"Track too short ({audio.shape[0]} samples) for filter. Skipping."
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


def apply_all_effects(track_data: TrackData) -> AudioData:
    """Applies all configured effects sequentially to original audio."""
    # (Implementation is the same as the previous version)
    track_name = track_data.get("name", "Unnamed")
    logger.info(f"Applying all effects to track: '{track_name}'")
    audio = track_data.get("original_audio")
    if audio is None:
        logging.error(f"Track '{track_name}' missing 'original_audio'.")
        return np.zeros((0, 2), dtype=np.float32)
    audio = audio.copy()
    sr = track_data.get("sr", GLOBAL_SR)
    if audio.size == 0:
        logging.warning(f"Track '{track_name}' has no original audio data.")
        return audio
    audio = apply_speed_change(audio, sr, track_data.get("speed_factor", 1.0))
    audio = apply_pitch_shift(audio, sr, track_data.get("pitch_shift", 0))
    audio = apply_filter(audio, sr, track_data.get("filter_type", "off"), track_data.get("filter_cutoff", 8000.0))
    logging.info(f"Finished applying effects to track: '{track_name}'")
    return audio


def mix_tracks(tracks_dict: Dict[TrackID, TrackData], preview: bool = False) -> AudioData:
    """Mixes tracks based on current settings."""
    # (Implementation is the same as the previous version)
    logger.info(f"Starting track mixing. Preview mode: {preview}")
    if not tracks_dict:
        logging.warning("Mix command called but track dict is empty.")
        return np.zeros((0, 2), dtype=np.float32)
    tracks = tracks_dict.values()
    solo_active = any(t.get("solo", False) for t in tracks)
    active_tracks_data = [t for t in tracks if t.get("solo", False)] if solo_active else list(tracks)
    valid_tracks_to_mix = [t for t in active_tracks_data if not t.get("mute", False) and t.get("processed_audio", np.array([])).size > 0]
    if not valid_tracks_to_mix:
        logging.warning("No valid tracks found to mix.")
        return np.zeros((0, 2), dtype=np.float32)
    try:
        max_len = max(len(t["processed_audio"]) for t in valid_tracks_to_mix)
    except (ValueError, KeyError) as e:
        logging.exception("Could not determine max length.")
        return np.zeros((0, 2), dtype=np.float32)
    if preview:
        max_len = min(max_len, int(GLOBAL_SR * 10)) if max_len > 0 else int(GLOBAL_SR * 10)
    if max_len <= 0:
        logging.warning("Resulting mix length <= 0.")
        return np.zeros((0, 2), dtype=np.float32)
    mix = np.zeros((max_len, 2), dtype=np.float32)
    logger.info(f"Mixing {len(valid_tracks_to_mix)} tracks. Target length: {max_len / GLOBAL_SR:.2f}s")
    for t_data in valid_tracks_to_mix:
        audio = t_data["processed_audio"]
        current_len = len(audio)
        if current_len < max_len:
            audio_adjusted = np.pad(audio, ((0, max_len - current_len), (0, 0)), mode="constant")
        elif current_len > max_len:
            audio_adjusted = audio[:max_len, :]
        else:
            audio_adjusted = audio.copy()
        pan = t_data.get("pan", 0.0)
        vol = t_data.get("volume", 1.0)
        pan_rad = (pan + 1) * np.pi / 4
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)
        panned_audio = np.zeros_like(audio_adjusted)
        panned_audio[:, 0] = audio_adjusted[:, 0] * left_gain
        panned_audio[:, 1] = audio_adjusted[:, 1] * right_gain
        mix += panned_audio
    final_mix = np.clip(mix, -1.0, 1.0)
    logger.info("Mixing complete.")
    return final_mix.astype(np.float32)


# ==========================================
# 2. Application State Management
# ==========================================


class AppState:
    """Manages the application state, primarily the tracks dictionary stored in Streamlit session state."""

    STATE_KEY = "tracks"  # Key used in st.session_state

    def __init__(self):
        """Initializes the state, ensuring the track dictionary exists in session state."""
        if self.STATE_KEY not in st.session_state:
            logger.info(f"Initializing '{self.STATE_KEY}' in session state.")
            st.session_state[self.STATE_KEY] = {}
        # Ensure existing tracks have the update_counter if loading from old state
        for track_id, track_data in self.get_all_tracks().items():
            if "update_counter" not in track_data:
                st.session_state[self.STATE_KEY][track_id]["update_counter"] = 0

    @staticmethod
    def get_default_track_params() -> TrackData:
        """Returns a dictionary with default parameters for a new track."""
        # --- MOVED HERE and made static ---
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
            "temp_file_path": None,  # Potential future use
            "update_counter": 0,  # For forcing audix refresh
        }

    def _get_tracks_dict(self) -> Dict[TrackID, TrackData]:
        """Internal helper to safely get the tracks dictionary."""
        return st.session_state.get(self.STATE_KEY, {})

    def get_all_tracks(self) -> Dict[TrackID, TrackData]:
        """Returns the dictionary of all current tracks."""
        return self._get_tracks_dict()

    def get_track(self, track_id: TrackID) -> Optional[TrackData]:
        """Gets data for a specific track by ID."""
        return self._get_tracks_dict().get(track_id)

    def add_track(self, track_id: TrackID, track_data: TrackData):
        """Adds a new track or updates an existing one."""
        if not isinstance(track_data, dict):
            logger.error(f"Attempted to add non-dict track data for ID {track_id}")
            return
        # Ensure default parameters are present if missing
        default_params = AppState.get_default_track_params()  # Use static method
        for key, default_value in default_params.items():
            if key not in track_data:
                track_data[key] = default_value

        st.session_state[self.STATE_KEY][track_id] = track_data
        logger.info(f"Added/Updated track ID: {track_id}, Name: '{track_data.get('name', 'N/A')}'")

    def delete_track(self, track_id: TrackID):
        """Deletes a track by ID."""
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            track_name = tracks[track_id].get("name", "N/A")
            del st.session_state[self.STATE_KEY][track_id]
            logger.info(f"Deleted track ID: {track_id}, Name: '{track_name}'")
            return True
        else:
            logger.warning(f"Attempted to delete non-existent track ID: {track_id}")
            return False

    def update_track_param(self, track_id: TrackID, param_name: str, value: Any):
        """Updates a specific parameter for a given track."""
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            # Check if param_name exists in the default structure for safety
            if param_name in AppState.get_default_track_params():
                st.session_state[self.STATE_KEY][track_id][param_name] = value
                # Avoid overly verbose logging for slider drags, etc.
                # logger.debug(f"Updated param '{param_name}' for track ID {track_id} to: {value}")
            else:
                logger.warning(f"Attempted to update potentially invalid param '{param_name}' for track ID: {track_id}")
        else:
            logger.warning(f"Attempted to update param for non-existent track ID: {track_id}")

    def increment_update_counter(self, track_id: TrackID):
        """Increments the update counter for a track to force UI refresh."""
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            current_counter = tracks[track_id].get("update_counter", 0)
            new_counter = current_counter + 1
            st.session_state[self.STATE_KEY][track_id]["update_counter"] = new_counter
            logger.debug(f"Incremented update_counter for track ID {track_id} to {new_counter}")
        else:
            logger.warning(f"Attempted to increment counter for non-existent track ID: {track_id}")

    def get_loaded_track_names(self) -> List[str]:
        """Gets a list of names of currently loaded tracks."""
        return [t.get("name") for t in self.get_all_tracks().values() if t.get("name")]


# ==========================================
# 3. TTS Generation (Class Wrapper)
# ==========================================


class TTSGenerator:
    """Handles Text-to-Speech generation using pyttsx3."""

    def __init__(self):
        self.engine = None
        logger.debug("TTSGenerator initialized.")

    def _init_engine(self):
        """Initializes the pyttsx3 engine if not already done."""
        if self.engine is None:
            try:
                logger.debug("Initializing pyttsx3 engine.")
                self.engine = pyttsx3.init()
            except Exception as e:
                logger.exception("Failed to initialize pyttsx3 engine.")
                self.engine = None
                raise

    def generate(self, text: str) -> Tuple[Optional[AudioData], Optional[SampleRate]]:
        """Generates audio from text, saves to temp file, loads, and cleans up."""
        logger.info("Starting TTS generation.")
        if not text:
            logger.warning("TTS generation requested with empty text.")
            return None, None
        tts_filename = None
        try:
            self._init_engine()
            if self.engine is None:
                st.error("TTS Engine failed to initialize.")
                return None, None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tts_filename = tmp_file.name
            logger.debug(f"Saving TTS to temporary file: {tts_filename}")
            self.engine.save_to_file(text, tts_filename)
            self.engine.runAndWait()
            logger.debug(f"Loading generated TTS audio from {tts_filename}")
            with open(tts_filename, "rb") as f:
                tts_bytes_io = BytesIO(f.read())
            audio, sr = load_audio(tts_bytes_io, target_sr=GLOBAL_SR)
            if audio.size == 0:
                logger.error("TTS empty audio file.")
                raise ValueError("TTS generated empty file.")
            logger.info("TTS generation successful.")
            return audio, sr
        except Exception as e:
            logger.exception("TTS Generation or Loading Failed.")
            st.error(f"TTS Failed: {e}")
            return None, None
        finally:
            if tts_filename and os.path.exists(tts_filename):
                try:
                    os.remove(tts_filename)
                    logger.debug(f"Cleaned up temp TTS file: {tts_filename}")
                except OSError as e_os:
                    logger.warning(f"Could not delete temp TTS file {tts_filename}: {e_os}")


# ==========================================
# 4. UI Management (Class)
# ==========================================


class UIManager:
    """Handles rendering Streamlit UI components."""

    def __init__(self, app_state: AppState, tts_generator: TTSGenerator):
        self.app_state = app_state
        self.tts_generator = tts_generator
        logger.debug("UIManager initialized.")

    def render_sidebar(self):
        """Renders the sidebar content for adding tracks."""
        with st.sidebar:
            st.header("➕ Add Tracks")
            self._render_uploader()
            st.divider()
            self._render_tts_generator()
            st.divider()
            self._render_binaural_generator()
            st.divider()
            self._render_solfeggio_generator()

    def _render_uploader(self):
        """Renders the file uploader section."""
        st.subheader("📁 Upload Audio")
        uploaded_files = st.file_uploader("Upload WAV, MP3, OGG, FLAC files", type=["wav", "mp3", "ogg", "flac"], accept_multiple_files=True, key="upload_files_key")
        loaded_track_names = self.app_state.get_loaded_track_names()
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in loaded_track_names:
                    logger.info(f"Processing uploaded file: {file.name}")
                    with st.spinner(f"Loading {file.name}..."):
                        audio, sr = load_audio(file, target_sr=GLOBAL_SR)
                    if audio.size > 0:
                        track_id = str(uuid.uuid4())
                        # Use AppState static method to get defaults
                        track_params = AppState.get_default_track_params()
                        track_params.update({"original_audio": audio, "processed_audio": audio.copy(), "sr": sr, "name": file.name})
                        self.app_state.add_track(track_id, track_params)
                        st.success(f"Loaded '{file.name}'")
                        loaded_track_names.append(file.name)
                    else:
                        logger.warning(f"Skipped empty/failed track from upload: {file.name}")
                        st.warning(f"Skipped empty or failed track: {file.name}")

    def _render_tts_generator(self):
        """Renders the TTS generation section."""
        st.subheader("🗣️ Generate Affirmations (TTS)")
        affirmation_text = st.text_area("Enter Affirmations (one per line)", height=100, key="affirmation_text")
        if st.button("Generate TTS Audio", key="generate_tts_key"):
            if affirmation_text:
                with st.spinner("Generating and loading TTS audio..."):
                    audio, sr = self.tts_generator.generate(affirmation_text)
                    if audio is not None and sr is not None:
                        track_id = str(uuid.uuid4())
                        # Use AppState static method to get defaults
                        track_params = AppState.get_default_track_params()
                        track_params.update({"original_audio": audio, "processed_audio": audio.copy(), "sr": sr, "name": "Affirmations"})
                        self.app_state.add_track(track_id, track_params)
                        st.success("Affirmations track generated!")
            else:
                st.warning("Please enter some text for affirmations.")

    def _render_binaural_generator(self):
        """Renders the Binaural Beats generation section."""
        st.subheader("🧠 Generate Binaural Beats")
        bb_cols = st.columns(2)
        bb_duration = bb_cols[0].number_input("Duration (s)", min_value=1, value=60, step=1, key="bb_duration")
        bb_vol = bb_cols[1].slider("Volume##BB", 0.0, 1.0, 0.3, step=0.05, key="bb_volume")
        bb_freq_cols = st.columns(2)
        bb_freq_left = bb_freq_cols[0].number_input("Left Freq (Hz)", min_value=20, max_value=1000, value=200, step=1, key="bb_freq_left")
        bb_freq_right = bb_freq_cols[1].number_input("Right Freq (Hz)", min_value=20, max_value=1000, value=210, step=1, key="bb_freq_right")
        if st.button("Generate Binaural Beats", key="generate_bb"):
            with st.spinner("Generating Binaural Beats..."):
                audio = generate_binaural_beats(bb_duration, bb_freq_left, bb_freq_right, GLOBAL_SR, bb_vol)
                track_id = str(uuid.uuid4())
                # Use AppState static method to get defaults
                track_params = AppState.get_default_track_params()
                track_params.update({"original_audio": audio, "processed_audio": audio.copy(), "sr": GLOBAL_SR, "name": f"Binaural {bb_freq_left}/{bb_freq_right}Hz"})
                self.app_state.add_track(track_id, track_params)
                st.success("Binaural Beats track generated!")

    def _render_solfeggio_generator(self):
        """Renders the Solfeggio Frequency generation section."""
        st.subheader("✨ Generate Solfeggio Frequency")
        solfeggio_freqs = [174, 285, 396, 417, 528, 639, 741, 852, 963]
        solf_cols = st.columns(2)
        solf_freq = solf_cols[0].selectbox("Frequency (Hz)", solfeggio_freqs, index=4, key="solf_freq")
        solf_duration = solf_cols[1].number_input("Duration (s)##Solf", min_value=1, value=60, step=1, key="solf_duration")
        solf_volume = st.slider("Volume##Solf", 0.0, 1.0, 0.3, step=0.05, key="solf_volume")
        if st.button("Generate Solfeggio Track", key="generate_solf"):
            with st.spinner("Generating Solfeggio Track..."):
                audio = generate_solfeggio_frequency(solf_duration, solf_freq, GLOBAL_SR, solf_volume)
                track_id = str(uuid.uuid4())
                # Use AppState static method to get defaults
                track_params = AppState.get_default_track_params()
                track_params.update({"original_audio": audio, "processed_audio": audio.copy(), "sr": GLOBAL_SR, "name": f"Solfeggio {solf_freq}Hz"})
                self.app_state.add_track(track_id, track_params)
                st.success(f"Solfeggio {solf_freq}Hz track generated!")

    def render_tracks_editor(self):
        """Renders the main editor area with all tracks."""
        st.header("🎚️ Tracks Editor")
        st.markdown("Adjust settings below. Volume/Pan are live. Speed/Pitch/Filter require 'Apply Effects'.")
        st.divider()
        tracks = self.app_state.get_all_tracks()
        if not tracks:
            st.info("No tracks loaded.")
            return
        track_ids_to_delete = []
        logger.debug(f"Rendering editor for {len(tracks)} tracks.")
        for track_id, track_data in list(tracks.items()):
            if track_id not in self.app_state.get_all_tracks():
                continue
            track_name_safe = track_data.get("name", "Unnamed")
            with st.expander(f"Track: **{track_name_safe}** (`{track_id[:6]}`)", expanded=True):
                logger.debug(f"Rendering expander for track: '{track_name_safe}' ({track_id})")
                col_main, col_controls = st.columns([3, 1])
                self._render_track_main_col(track_id, track_data, col_main)
                deleted = self._render_track_controls_col(track_id, track_data, col_controls)
                if deleted:
                    track_ids_to_delete.append(track_id)
        if track_ids_to_delete:
            deleted_count = 0
            for tid in track_ids_to_delete:
                if self.app_state.delete_track(tid):
                    deleted_count += 1
            if deleted_count > 0:
                st.toast(f"Deleted {deleted_count} track(s).")
                st.rerun()

    def _render_track_main_col(self, track_id: TrackID, track_data: TrackData, column: st.delta_generator.DeltaGenerator):
        """Renders the waveform and effects controls for a single track."""
        with column:
            try:
                processed_audio_len = len(track_data.get("processed_audio", []))
                track_sr = track_data.get("sr", GLOBAL_SR)
                track_len_sec = processed_audio_len / track_sr if track_sr > 0 else 0
                st.caption(f"Sample Rate: {track_sr} Hz | Length: {track_len_sec:.2f}s")
                # Waveform Visualization
                audio_to_display = track_data.get("processed_audio")
                if audio_to_display is not None and audio_to_display.size > 0:
                    temp_wav_path = save_audio_to_temp(audio_to_display, track_sr)
                    if temp_wav_path:
                        ws_options = WaveSurferOptions(
                            height=100, normalize=True, wave_color="#A020F0", progress_color="#800080", cursor_color="#333333", cursor_width=1, bar_width=2, bar_gap=1
                        )
                        update_count = track_data.get("update_counter", 0)
                        audix_key = f"audix_{track_id}_{update_count}"
                        logger.debug(f"Calling audix for track '{track_data.get('name', 'N/A')}' with key: {audix_key}")
                        audix(data=temp_wav_path, sample_rate=track_sr, wavesurfer_options=ws_options, key=audix_key)
                    else:
                        st.error("Could not display waveform (failed to save temp file).")
                else:
                    st.info("Track has no audio data to display.")
                st.markdown("**Effects** (Require 'Apply Effects' Button)")
                fx_col1, fx_col2, fx_col3 = st.columns(3)
                # Effect Sliders - Update state directly via AppState
                speed = fx_col1.slider("Speed", 0.25, 4.0, track_data.get("speed_factor", 1.0), step=0.05, key=f"speed_{track_id}", help="Changes speed (>1 faster, <1 slower).")
                if speed != track_data.get("speed_factor"):
                    self.app_state.update_track_param(track_id, "speed_factor", speed)
                pitch = fx_col2.slider("Pitch Shift (semitones)", -12, 12, track_data.get("pitch_shift", 0), step=1, key=f"pitch_{track_id}", help="Changes pitch.")
                if pitch != track_data.get("pitch_shift"):
                    self.app_state.update_track_param(track_id, "pitch_shift", pitch)
                f_type = fx_col3.selectbox(
                    "Filter", ["off", "lowpass", "highpass"], index=["off", "lowpass", "highpass"].index(track_data.get("filter_type", "off")), key=f"filter_type_{track_id}"
                )
                if f_type != track_data.get("filter_type"):
                    self.app_state.update_track_param(track_id, "filter_type", f_type)
                f_enabled = track_data["filter_type"] != "off"
                max_cutoff = track_sr / 2 - 1
                f_cutoff = fx_col3.number_input(
                    f"Cutoff ({'Hz' if f_enabled else 'Disabled'})",
                    min_value=20.0,
                    max_value=max_cutoff if max_cutoff > 20 else 20.0,
                    value=float(track_data.get("filter_cutoff", 8000.0)),
                    step=100.0,
                    key=f"filter_cutoff_{track_id}",
                    disabled=not f_enabled,
                    help="Filter cutoff frequency.",
                )
                if f_cutoff != track_data.get("filter_cutoff"):
                    self.app_state.update_track_param(track_id, "filter_cutoff", f_cutoff)
                # Apply Effects Button
                if st.button("⚙️ Apply Effects", key=f"apply_fx_{track_id}", help="Apply Speed, Pitch, Filter changes."):
                    logger.info(f"Apply Effects button clicked for track: '{track_data.get('name', 'N/A')}' ({track_id})")
                    original_audio = track_data.get("original_audio")
                    if original_audio is not None and original_audio.size > 0:
                        with st.spinner(f"Applying effects..."):
                            processed_audio = apply_all_effects(track_data)
                            self.app_state.update_track_param(track_id, "processed_audio", processed_audio)
                            self.app_state.increment_update_counter(track_id)
                        st.success(f"Effects applied.")
                        st.rerun()
                    else:
                        st.warning(f"Cannot apply effects: Track has no original audio data.")
            except Exception as e:
                logger.exception(f"Error rendering main column for track {track_id}")
                st.error(f"Error displaying waveform/effects: {e}")

    def _render_track_controls_col(self, track_id: TrackID, track_data: TrackData, column: st.delta_generator.DeltaGenerator) -> bool:
        """Renders the controls (name, vol, pan, etc.) for a single track. Returns True if delete was clicked."""
        delete_clicked = False
        with column:
            try:
                st.markdown("**Track Controls**")
                # Update state directly via AppState
                name = st.text_input("Name", value=track_data.get("name", "Unnamed"), key=f"name_{track_id}")
                if name != track_data.get("name"):
                    self.app_state.update_track_param(track_id, "name", name)
                vp_col1, vp_col2 = st.columns(2)
                vol = vp_col1.slider("Volume", 0.0, 2.0, track_data.get("volume", 1.0), step=0.05, key=f"vol_{track_id}", help="Live volume adjustment.")
                if vol != track_data.get("volume"):
                    self.app_state.update_track_param(track_id, "volume", vol)
                pan = vp_col2.slider("Pan", -1.0, 1.0, track_data.get("pan", 0.0), step=0.1, key=f"pan_{track_id}", help="Live stereo balance.")
                if pan != track_data.get("pan"):
                    self.app_state.update_track_param(track_id, "pan", pan)
                ms_col1, ms_col2 = st.columns(2)
                mute = ms_col1.checkbox("Mute", value=track_data.get("mute", False), key=f"mute_{track_id}", help="Silence track.")
                if mute != track_data.get("mute"):
                    self.app_state.update_track_param(track_id, "mute", mute)
                solo = ms_col2.checkbox("Solo", value=track_data.get("solo", False), key=f"solo_{track_id}", help="Isolate track.")
                if solo != track_data.get("solo"):
                    self.app_state.update_track_param(track_id, "solo", solo)
                st.markdown("---")
                if st.button("🗑️ Delete Track", key=f"delete_{track_id}", help="Permanently delete."):
                    delete_clicked = True
                    st.warning(f"Track marked for deletion.")
            except Exception as e:
                logger.exception(f"Error rendering controls column for track {track_id}")
                st.error(f"Error displaying controls: {e}")
        return delete_clicked

    def render_master_controls(self):
        """Renders the master preview and export buttons."""
        st.divider()
        st.header("🔊 Master Output")
        master_cols = st.columns(2)
        tracks = self.app_state.get_all_tracks()
        with master_cols[0]:
            if st.button("🎧 Preview Mix (10s)", key="preview_mix", use_container_width=True):
                logger.info("Preview Mix button clicked.")
                if not tracks:
                    st.warning("No tracks loaded.")
                else:
                    with st.spinner("Generating preview mix..."):
                        mix_preview = mix_tracks(tracks, preview=True)
                        if mix_preview.size > 0:
                            st.audio(save_audio(mix_preview, GLOBAL_SR), format="audio/wav")
                            logger.info("Preview mix generated.")
                        else:
                            logger.warning("Preview mix empty.")
        with master_cols[1]:
            if st.button("💾 Export Full Mix (.wav)", key="export_mix", use_container_width=True):
                logger.info("Export Full Mix button clicked.")
                if not tracks:
                    st.warning("No tracks loaded.")
                else:
                    with st.spinner("Generating full mix..."):
                        full_mix = mix_tracks(tracks, preview=False)
                        if full_mix.size > 0:
                            export_buffer = save_audio(full_mix, GLOBAL_SR)
                            st.download_button(
                                label="⬇️ Download Full Mix (.wav)",
                                data=export_buffer,
                                file_name="pro_subliminal_mix_oop.wav",
                                mime="audio/wav",
                                key="download_full_mix_key",
                                use_container_width=True,
                            )
                            logger.info("Full mix generated.")
                        else:
                            logger.warning("Full mix empty.")

    def render_instructions(self):
        """Renders the instructions expander."""
        st.divider()
        with st.expander("📖 Show Instructions & Notes", expanded=False):
            st.markdown("""
              ### How to Use:
              1.  **Add Tracks**: Use sidebar (+) to Upload, Generate TTS, Binaural Beats, or Solfeggio.
              2.  **Edit Tracks**: In main editor, adjust effects (Speed, Pitch, Filter) then **click `⚙️ Apply Effects`**. Waveform/playback should update. Adjust Volume/Pan (live), Mute/Solo, Rename, or Delete.
              3.  **Mix & Export**: Use Master Output buttons for Preview or full WAV Export.
              ### Notes:
              - **Apply Effects**: Speed/Pitch/Filter require button click. Vol/Pan are live.
              - **Processing**: Can take time.
              - **Format**: Internal 44.1kHz/32f Stereo. Export 16-bit WAV.
              - **Clipping**: Final mix clipped to [-1, 1]. Manage track volumes.
              - **Logging**: Details in `editor_oop.log`.
              """)


# ==========================================
# 5. Benchmarking (Placeholder Class)
# ==========================================


class Benchmarker:
    """Placeholder for performance benchmarking."""

    def __init__(self, tts_generator: TTSGenerator):
        self.tts_generator = tts_generator
        logger.debug("Benchmarker initialized.")

    def benchmark_tts(self, word_count: int = 10000, repetitions: int = 1):
        """Benchmarks TTS generation for a given number of words."""
        logger.info(f"Starting TTS benchmark: {word_count} words, {repetitions} repetitions.")
        st.subheader(f"Benchmark: TTS Generation ({word_count} words)")
        if word_count <= 0:
            st.warning("Word count must be positive.")
            return
        placeholder_word = "benchmark "
        text = (placeholder_word * (word_count // len(placeholder_word) + 1))[: word_count * 6]
        st.text(f"Generating {len(text)} chars (approx {word_count} words)...")
        total_time = 0
        min_time = float("inf")
        max_time = 0
        success_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(repetitions):
            status_text.text(f"Repetition {i + 1}/{repetitions}...")
            start_time = time.time()
            try:
                audio, sr = self.tts_generator.generate(text)
                end_time = time.time()
                if audio is not None and audio.size > 0:
                    duration = end_time - start_time
                    total_time += duration
                    min_time = min(min_time, duration)
                    max_time = max(max_time, duration)
                    success_count += 1
                    logger.info(f"TTS Benchmark rep {i + 1} success in {duration:.2f}s")
                else:
                    logger.error(f"TTS Benchmark rep {i + 1} failed (no audio).")
                    st.error(f"Rep {i + 1} failed.")
            except Exception as e:
                logger.exception(f"TTS Benchmark rep {i + 1} failed.")
                st.error(f"Rep {i + 1} failed: {e}")
            progress_bar.progress((i + 1) / repetitions)
        status_text.text("Benchmark complete.")
        if success_count > 0:
            avg_time = total_time / success_count
            st.metric("Average Time per Generation", f"{avg_time:.2f} s")
            st.text(f"Min: {min_time:.2f} s | Max: {max_time:.2f} s | Success: {success_count}/{repetitions}")
        else:
            st.error("TTS Benchmark failed for all repetitions.")


# ==========================================
# 6. Main Application Logic
# ==========================================


def main():
    """Main function to run the Streamlit application."""
    logger.info("Starting main application function.")
    # Initialize state and components
    app_state = AppState()
    tts_generator = TTSGenerator()
    ui_manager = UIManager(app_state, tts_generator)
    # benchmarker = Benchmarker(tts_generator)
    # Render UI sections
    ui_manager.render_sidebar()
    ui_manager.render_tracks_editor()
    ui_manager.render_master_controls()
    # Benchmarking Section
    st.divider()
    with st.expander("⏱️ Run Benchmarks", expanded=False):
        st.info("Run performance tests. May take time.")
        bm_words = st.number_input("Words for TTS Benchmark", 100, 20000, 10000, 100)
        bm_reps = st.number_input("Repetitions", 1, 10, 1, 1)
        # if st.button("Run TTS Benchmark"):
        # benchmarker.benchmark_tts(bm_words, bm_reps)
    # Instructions
    ui_manager.render_instructions()
    # Footer
    st.divider()
    st.caption("Pro Subliminal Audio Editor (OOP) - Built with Streamlit")
    logger.info("Reached end of main application function render.")


if __name__ == "__main__":
    main()
    # Attempt to stop listener - might not execute reliably in Streamlit
    # logger.info("Attempting to stop logging listener.")
    # listener.stop()
