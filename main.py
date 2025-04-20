# ==========================================
# MindMorph - Pro Subliminal Audio Editor
# Version: 2.4 (OOP - Optimized & Clarified + Onboarding v4 - Visual Steps)
# ==========================================
# --- Early Config ---
import os

import streamlit as st
from PIL import Image

favicon_path = os.path.join("assets", "favico.png")
page_icon = None
try:
    page_icon = Image.open(favicon_path)
except FileNotFoundError:
    pass
st.set_page_config(layout="wide", page_title="MindMorph - Pro Subliminal Editor", page_icon=page_icon)

# --- Imports ---
import logging
import logging.handlers
import queue
import tempfile
import time
import traceback
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

# --- Logging Setup ---
# (Same setup as before - creates editor_oop.log)
log_queue = queue.Queue(-1)
log_file = "editor_oop.log"
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s")
file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=1 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)
queue_handler = logging.handlers.QueueHandler(log_queue)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
if not root_logger.hasHandlers() or not any(isinstance(h, logging.handlers.QueueHandler) for h in root_logger.handlers):
    root_logger.addHandler(queue_handler)
if "listener_started" not in st.session_state:
    listener = logging.handlers.QueueListener(log_queue, file_handler, respect_handler_level=True)
    listener.start()
    st.session_state.listener_started = True
logger = logging.getLogger(__name__)
logger.info("-----------------------------------------------------")
logger.info("Application starting up / rerunning.")
logger.info("-----------------------------------------------------")

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

    logger.debug("Core audio and UI libraries imported successfully.")
except ImportError as e:
    logger.exception("CRITICAL: Failed to import core libraries. App cannot continue.")
    st.error(f"Core library import failed: {e}. Please ensure all dependencies from requirements.txt are installed and restart.")
    st.error(f"Missing library details: {e.name}")
    st.code(traceback.format_exc())
    st.stop()

# --- Constants ---
GLOBAL_SR = 44100
logger.debug(f"Global Sample Rate set to: {GLOBAL_SR} Hz")

# --- Data Types ---
AudioData = np.ndarray
SampleRate = int
TrackID = str
TrackData = Dict[str, Any]


# ==========================================
# 1. Audio Processing Utilities (Functions)
# ==========================================
# (Functions load_audio, save_audio, save_audio_to_temp, generate*, apply*, mix_tracks remain unchanged)
# ... (Keep all audio utility functions from the previous version here) ...
def load_audio(file_source: UploadedFile | BytesIO | str, target_sr: SampleRate = GLOBAL_SR) -> tuple[AudioData, SampleRate]:
    """Loads, ensures stereo, and resamples audio."""
    logger.info(f"Loading audio from source type: {type(file_source)}")
    try:
        source = file_source
        audio, sr = librosa.load(source, sr=None, mono=False)
        logger.debug(f"Loaded audio original SR: {sr}, shape: {audio.shape}")
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)
        elif audio.shape[0] == 2 and audio.shape[1] > 2:
            audio = audio.T  # Check if channels are first dim and there are samples
        if audio.shape[1] > 2:
            logger.warning(f"Audio > 2 channels ({audio.shape[1]}). Using first two.")
            audio = audio[:, :2]
        elif audio.shape[1] == 1:
            logger.warning("Mono audio loaded unexpectedly. Duplicating.")
            audio = np.concatenate([audio, audio], axis=1)
        if sr != target_sr:
            logger.info(f"Resampling from {sr} Hz to {target_sr} Hz.")
            if audio.size > 0:
                audio = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr).T
            else:
                sr = target_sr
        return audio.astype(np.float32), target_sr
    except Exception as e:
        logger.exception("Error loading audio.")
        st.error(f"Error loading audio: {e}")
        return np.zeros((0, 2), dtype=np.float32), target_sr


def save_audio(audio: AudioData, sr: SampleRate) -> BytesIO:
    """Saves audio to an in-memory BytesIO buffer (WAV format, PCM16)."""
    buffer = BytesIO()
    logger.debug(f"Saving audio ({audio.shape}, {sr}Hz) to BytesIO.")
    try:
        audio_int16 = (audio * 32767).astype(np.int16)
        sf.write(buffer, audio_int16, sr, format="WAV", subtype="PCM_16")
        buffer.seek(0)
    except Exception as e:
        logger.exception("Error saving audio to buffer.")
        st.error(f"Error saving: {e}")
        buffer = BytesIO()
    return buffer


def save_audio_to_temp(audio: AudioData, sr: SampleRate) -> str | None:
    """Saves audio to a temporary WAV file on disk (WAV format, PCM16). Returns file path or None."""
    temp_file_path = None
    logger.debug(f"Attempting to save audio ({audio.shape}, {sr}Hz) to temp file.")
    try:
        audio_int16 = (audio * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as tmp:
            sf.write(tmp, audio_int16, sr, format="WAV", subtype="PCM_16")
            temp_file_path = tmp.name
        logger.info(f"Audio saved successfully to temporary file: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        logger.exception("Failed to save temporary audio file.")
        st.error(f"Failed to save temp file: {e}")
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Cleaned up partial temp file: {temp_file_path}")
            except OSError as e_os:
                logger.warning(f"Failed cleanup {temp_file_path}: {e_os}")
        return None


def generate_binaural_beats(duration: float, freq_left: float, freq_right: float, sr: SampleRate, volume: float) -> AudioData:
    logger.info(f"Generating binaural beats: dur={duration}s, L={freq_left}Hz, R={freq_right}Hz, vol={volume}")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = volume * np.sin(2 * np.pi * freq_left * t)
    right = volume * np.sin(2 * np.pi * freq_right * t)
    audio = np.stack([left, right], axis=1).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def generate_solfeggio_frequency(duration: float, freq: float, sr: SampleRate, volume: float) -> AudioData:
    logger.info(f"Generating Solfeggio: dur={duration}s, F={freq}Hz, vol={volume}")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = volume * np.sin(2 * np.pi * freq * t)
    audio = np.stack([sine_wave, sine_wave], axis=1).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def apply_speed_change(audio: AudioData, sr: SampleRate, speed_factor: float) -> AudioData:
    logger.debug(f"Applying speed change factor: {speed_factor} (time_stretch)")
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        logging.error("NaN/Inf in audio data before time stretch.")
        st.error("Bad audio data.")
        return audio
    if not np.isclose(speed_factor, 1.0):
        if speed_factor <= 0:
            logging.warning(f"Invalid speed factor: {speed_factor}.")
            st.warning("Speed > 0 required.")
            return audio
        try:
            if audio.size == 0:
                return audio
            logging.info(f"Applying time stretch (factor: {speed_factor})...")
            audio_stretched = librosa.effects.time_stretch(audio.T, rate=speed_factor)
            return audio_stretched.T.astype(np.float32)
        except Exception as e:
            logging.exception("Error in time_stretch.")
            st.error(f"Speed change failed: {e}")
            return audio
    return audio


def apply_pitch_shift(audio: AudioData, sr: SampleRate, n_steps: float) -> AudioData:
    logger.debug(f"Applying pitch shift: {n_steps} semitones")
    if not np.isclose(n_steps, 0.0):
        try:
            if audio.size == 0:
                return audio
            logging.info(f"Applying pitch shift ({n_steps} semitones)...")
            audio_shifted = librosa.effects.pitch_shift(audio.T, sr=sr, n_steps=n_steps)
            return audio_shifted.T.astype(np.float32)
        except Exception as e:
            logging.exception("Error applying pitch shift.")
            st.error(f"Pitch shift failed: {e}")
            return audio
    return audio


def apply_filter(audio: AudioData, sr: SampleRate, filter_type: str, cutoff: float) -> AudioData:
    logger.debug(f"Applying filter: type={filter_type}, cutoff={cutoff} Hz")
    if filter_type == "off" or cutoff <= 0:
        return audio
    try:
        if audio.size == 0:
            return audio
        nyquist = 0.5 * sr
        normalized_cutoff = cutoff / nyquist
        if normalized_cutoff >= 1.0:
            msg = f"Filter cutoff ({cutoff}) >= Nyquist ({nyquist})."
            logging.warning(msg)
            st.warning(msg)
            return audio
        if normalized_cutoff <= 0:
            msg = f"Filter cutoff ({cutoff}) must be positive."
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
            st.warning(f"Unknown filter: {filter_type}")
            return audio
        min_len_filtfilt = signal.filtfilt_coeffs(b, a)[-1].size * 3
        if audio.shape[0] <= min_len_filtfilt:
            msg = f"Track too short ({audio.shape[0]}) for filter. Skipping."
            logging.warning(msg)
            st.warning(msg)
            return audio
        audio_filtered = signal.filtfilt(b, a, audio, axis=0)
        return audio_filtered.astype(np.float32)
    except Exception as e:
        logging.exception("Error applying filter.")
        st.error(f"Filter failed: {e}")
        return audio


def apply_all_effects(track_data: TrackData) -> AudioData:
    track_name = track_data.get("name", "Unnamed")
    logger.info(f"Applying all effects to track: '{track_name}'")
    audio = track_data.get("original_audio")
    if audio is None:
        logging.error(f"'{track_name}' missing 'original_audio'.")
        return np.zeros((0, 2), dtype=np.float32)
    audio = audio.copy()
    sr = track_data.get("sr", GLOBAL_SR)
    if audio.size == 0:
        logging.warning(f"'{track_name}' has no original audio.")
        return audio
    audio = apply_speed_change(audio, sr, track_data.get("speed_factor", 1.0))
    audio = apply_pitch_shift(audio, sr, track_data.get("pitch_shift", 0))
    audio = apply_filter(audio, sr, track_data.get("filter_type", "off"), track_data.get("filter_cutoff", 8000.0))
    logger.info(f"Finished applying effects to '{track_name}'")
    return audio


def mix_tracks(tracks_dict: Dict[TrackID, TrackData], preview: bool = False) -> AudioData:
    logger.info(f"Starting track mixing. Preview: {preview}")
    if not tracks_dict:
        logging.warning("Mix called but no tracks.")
        return np.zeros((0, 2), dtype=np.float32)
    tracks = tracks_dict.values()
    solo_active = any(t.get("solo", False) for t in tracks)
    active_tracks_data = [t for t in tracks if t.get("solo", False)] if solo_active else list(tracks)
    valid_tracks_to_mix = [t for t in active_tracks_data if not t.get("mute", False) and t.get("processed_audio", np.array([])).size > 0]
    if not valid_tracks_to_mix:
        logging.warning("No valid tracks to mix.")
        return np.zeros((0, 2), dtype=np.float32)
    try:
        max_len = max(len(t["processed_audio"]) for t in valid_tracks_to_mix)
    except (ValueError, KeyError):
        logging.exception("Could not get max length.")
        return np.zeros((0, 2), dtype=np.float32)
    if preview:
        max_len = min(max_len, int(GLOBAL_SR * 10)) if max_len > 0 else int(GLOBAL_SR * 10)
    if max_len <= 0:
        logging.warning("Mix length <= 0.")
        return np.zeros((0, 2), dtype=np.float32)
    mix = np.zeros((max_len, 2), dtype=np.float32)
    logger.info(f"Mixing {len(valid_tracks_to_mix)} tracks. Len: {max_len / GLOBAL_SR:.2f}s")
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
# (AppState class remains the same as previous 'optimized' version)
class AppState:
    """Manages the application state, primarily the tracks dictionary stored in Streamlit session state."""

    STATE_KEY = "tracks"

    def __init__(self):
        if self.STATE_KEY not in st.session_state:
            logger.info(f"Initializing '{self.STATE_KEY}'.")
            st.session_state[self.STATE_KEY] = {}
        default_keys = AppState.get_default_track_params().keys()
        for track_id, track_data in self.get_all_tracks().items():
            for key in default_keys:
                if key not in track_data:
                    logger.warning(f"Track {track_id} missing '{key}', adding default.")
                    st.session_state[self.STATE_KEY][track_id][key] = AppState.get_default_track_params()[key]

    @staticmethod
    def get_default_track_params() -> TrackData:
        return {
            "original_audio": np.zeros((0, 2), dtype=np.float32),
            "processed_audio": np.zeros((0, 2), dtype=np.float32),
            "sr": GLOBAL_SR,
            "name": "New Track",
            "volume": 1.0,
            "mute": False,
            "solo": False,
            "speed_factor": 1.0,
            "pitch_shift": 0,
            "pan": 0.0,
            "filter_type": "off",
            "filter_cutoff": 8000.0,
            "temp_file_path": None,
            "update_counter": 0,
        }

    def _get_tracks_dict(self) -> Dict[TrackID, TrackData]:
        return st.session_state.get(self.STATE_KEY, {})

    def get_all_tracks(self) -> Dict[TrackID, TrackData]:
        return self._get_tracks_dict()

    def get_track(self, track_id: TrackID) -> Optional[TrackData]:
        return self._get_tracks_dict().get(track_id)

    def add_track(self, track_id: TrackID, track_data: TrackData):
        if not isinstance(track_data, dict):
            logger.error(f"Invalid track data type {track_id}")
            return
        default_params = AppState.get_default_track_params()
        final_track_data = {**default_params, **track_data}
        initial_audio = final_track_data.get("processed_audio")
        initial_sr = final_track_data.get("sr", GLOBAL_SR)
        if initial_audio is not None and initial_audio.size > 0:
            initial_temp_path = save_audio_to_temp(initial_audio, initial_sr)
            if initial_temp_path:
                final_track_data["temp_file_path"] = initial_temp_path
            else:
                logger.error(f"Failed initial temp file {track_id}")
                final_track_data["temp_file_path"] = None
        else:
            final_track_data["temp_file_path"] = None
        st.session_state[self.STATE_KEY][track_id] = final_track_data
        logger.info(f"Added track ID: {track_id}, Name: '{final_track_data.get('name', 'N/A')}'")

    def delete_track(self, track_id: TrackID):
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            track_data = tracks[track_id]
            track_name = track_data.get("name", "N/A")
            temp_file_to_delete = track_data.get("temp_file_path")
            if temp_file_to_delete and os.path.exists(temp_file_to_delete):
                try:
                    os.remove(temp_file_to_delete)
                    logger.info(f"Deleted temp file '{temp_file_to_delete}' for track {track_id}")
                except OSError as e:
                    logger.warning(f"Failed delete temp file '{temp_file_to_delete}': {e}")
            del st.session_state[self.STATE_KEY][track_id]
            logger.info(f"Deleted track ID: {track_id}, Name: '{track_name}'")
            return True
        else:
            logger.warning(f"Attempted delete non-existent track {track_id}")
            return False

    def update_track_param(self, track_id: TrackID, param_name: str, value: Any):
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            if param_name in AppState.get_default_track_params():
                st.session_state[self.STATE_KEY][track_id][param_name] = value
            else:
                logger.warning(f"Attempted update invalid param '{param_name}' for {track_id}")
        else:
            logger.warning(f"Attempted update param for non-existent track {track_id}")

    def increment_update_counter(self, track_id: TrackID):
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            current = tracks[track_id].get("update_counter", 0)
            new = current + 1
            st.session_state[self.STATE_KEY][track_id]["update_counter"] = new
            logger.debug(f"Incr update_counter for {track_id} to {new}")
        else:
            logger.warning(f"Attempted incr counter for non-existent track {track_id}")

    def get_loaded_track_names(self) -> List[str]:
        return [t.get("name") for t in self.get_all_tracks().values() if t.get("name")]


# ==========================================
# 3. TTS Generation (Class Wrapper)
# ==========================================
# (TTSGenerator class remains the same)
class TTSGenerator:
    """Handles Text-to-Speech generation using pyttsx3."""

    def __init__(self):
        self.engine = None
        logger.debug("TTSGenerator initialized.")

    def _init_engine(self):
        if self.engine is None:
            try:
                logger.debug("Initializing pyttsx3 engine.")
                self.engine = pyttsx3.init()
            except Exception as e:
                logger.exception("Failed init pyttsx3.")
                self.engine = None
                raise

    def generate(self, text: str) -> Tuple[Optional[AudioData], Optional[SampleRate]]:
        logger.info("Starting TTS generation.")
        tts_filename = None
        if not text:
            logger.warning("Empty TTS text.")
            return None, None
        try:
            self._init_engine()
            if self.engine is None:
                st.error("TTS Engine failed.")
                return None, None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tts_filename = tmp.name
            logger.debug(f"Saving TTS to temp: {tts_filename}")
            self.engine.save_to_file(text, tts_filename)
            self.engine.runAndWait()
            logger.debug(f"Loading generated TTS from {tts_filename}")
            with open(tts_filename, "rb") as f:
                tts_bytes_io = BytesIO(f.read())
            audio, sr = load_audio(tts_bytes_io, target_sr=GLOBAL_SR)
            if audio.size == 0:
                logger.error("TTS empty audio.")
                raise ValueError("TTS generated empty file.")
            logger.info("TTS generation successful.")
            return audio, sr
        except Exception as e:
            logger.exception("TTS Failed.")
            st.error(f"TTS Failed: {e}")
            return None, None
        finally:
            if tts_filename and os.path.exists(tts_filename):
                try:
                    os.remove(tts_filename)
                    logger.debug(f"Cleaned temp TTS: {tts_filename}")
                except OSError as e:
                    logger.warning(f"Could not delete temp TTS {tts_filename}: {e}")


# ==========================================
# 4. UI Management (Class) - WITH ENHANCED ONBOARDING V4
# ==========================================
class UIManager:
    """Handles rendering Streamlit UI components with enhanced clarity."""

    def __init__(self, app_state: AppState, tts_generator: TTSGenerator):
        self.app_state = app_state
        self.tts_generator = tts_generator
        logger.debug("UIManager initialized.")

    def render_sidebar(self):
        """Renders the sidebar content for adding tracks."""
        with st.sidebar:
            logo_path = os.path.join("assets", "logo.png")
            if os.path.exists(logo_path):
                st.image(logo_path, width=200)
            else:
                st.header("MindMorph")
            st.caption("Subliminal Audio Editor")
            st.markdown("---")
            st.markdown("### STEP 1: Add Audio Layers")
            st.markdown("Use options below to add sounds to your project.")
            self._render_uploader()
            st.divider()
            self._render_tts_generator()
            st.divider()
            self._render_binaural_generator()
            st.divider()
            self._render_solfeggio_generator()
            st.markdown("---")
            st.info("Edit track details in the main panel after adding.")

    def _render_uploader(self):
        st.subheader("📁 Upload Audio File(s)")
        uploaded_files = st.file_uploader(
            "Upload background music or recordings",
            type=["wav", "mp3", "ogg", "flac"],
            accept_multiple_files=True,
            key="upload_files_key",
            help="Select audio files for separate tracks.",
        )
        loaded_track_names = self.app_state.get_loaded_track_names()
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in loaded_track_names:
                    logger.info(f"Processing upload: {file.name}")
                    with st.spinner(f"Loading {file.name}..."):
                        audio, sr = load_audio(file, target_sr=GLOBAL_SR)
                    if audio.size > 0:
                        track_id = str(uuid.uuid4())
                        track_params = AppState.get_default_track_params()
                        track_params.update({"original_audio": audio, "processed_audio": audio.copy(), "sr": sr, "name": file.name})
                        self.app_state.add_track(track_id, track_params)
                        st.success(f"Loaded '{file.name}'")
                        loaded_track_names.append(file.name)
                    else:
                        logger.warning(f"Skipped empty upload: {file.name}")
                        st.warning(f"Skipped empty: {file.name}")

    def _render_tts_generator(self):
        st.subheader("🗣️ Generate Affirmations")
        affirmation_text = st.text_area("Enter Affirmations (one per line)", height=100, key="affirmation_text", help="Type affirmations here.")
        if st.button("Generate Affirmation Track", key="generate_tts_key", help="Convert text above to spoken audio track."):
            if affirmation_text:
                with st.spinner("Generating TTS..."):
                    audio, sr = self.tts_generator.generate(affirmation_text)
                if audio is not None and sr is not None:
                    track_id = str(uuid.uuid4())
                    track_params = AppState.get_default_track_params()
                    track_params.update({"original_audio": audio, "processed_audio": audio.copy(), "sr": sr, "name": "Affirmations"})
                    self.app_state.add_track(track_id, track_params)
                    st.success("Affirmations track generated!")
            else:
                st.warning("Please enter text.")

    def _render_binaural_generator(self):
        st.subheader("🧠 Generate Binaural Beats")
        st.markdown("<small>Generates stereo tones potentially inducing brainwave states (requires headphones).</small>", unsafe_allow_html=True)
        bb_cols = st.columns(2)
        bb_duration = bb_cols[0].number_input("Duration (s)", 1, 3600, 60, 1, key="bb_duration", help="Length in seconds.")
        bb_vol = bb_cols[1].slider("Volume##BB", 0.0, 1.0, 0.3, 0.05, key="bb_volume", help="Loudness (0.0 to 1.0). Usually kept low.")
        bb_fcols = st.columns(2)
        bb_fleft = bb_fcols[0].number_input("Left Freq (Hz)", 20, 1000, 200, 1, key="bb_freq_left", help="Left ear frequency.")
        bb_fright = bb_fcols[1].number_input("Right Freq (Hz)", 20, 1000, 210, 1, key="bb_freq_right", help="Right ear frequency.")
        if st.button("Generate Binaural Track", key="generate_bb", help="Create track with these settings."):
            with st.spinner("Generating..."):
                audio = generate_binaural_beats(bb_duration, bb_fleft, bb_fright, GLOBAL_SR, bb_vol)
            track_id = str(uuid.uuid4())
            track_params = AppState.get_default_track_params()
            track_params.update({"original_audio": audio, "processed_audio": audio.copy(), "sr": GLOBAL_SR, "name": f"Binaural {bb_fleft}/{bb_fright}Hz"})
            self.app_state.add_track(track_id, track_params)
            st.success("Binaural Beats generated!")

    def _render_solfeggio_generator(self):
        st.subheader("✨ Generate Solfeggio Tone")
        st.markdown("<small>Generates pure tones based on historical Solfeggio frequencies.</small>", unsafe_allow_html=True)
        freqs = [174, 285, 396, 417, 528, 639, 741, 852, 963]
        cols = st.columns(2)
        freq = cols[0].selectbox("Frequency (Hz)", freqs, index=4, key="solf_freq", help="Select Solfeggio frequency.")
        duration = cols[1].number_input("Duration (s)##Solf", 1, 3600, 60, 1, key="solf_duration", help="Length in seconds.")
        vol = st.slider("Volume##Solf", 0.0, 1.0, 0.3, 0.05, key="solf_volume", help="Loudness (0.0 to 1.0). Usually kept low.")
        if st.button("Generate Solfeggio Track", key="generate_solf", help="Create track with this tone."):
            with st.spinner("Generating..."):
                audio = generate_solfeggio_frequency(duration, freq, GLOBAL_SR, vol)
            track_id = str(uuid.uuid4())
            track_params = AppState.get_default_track_params()
            track_params.update({"original_audio": audio, "processed_audio": audio.copy(), "sr": GLOBAL_SR, "name": f"Solfeggio {freq}Hz"})
            self.app_state.add_track(track_id, track_params)
            st.success(f"Solfeggio {freq}Hz generated!")

    def render_tracks_editor(self):
        """Renders the main editor area with all tracks."""
        st.header("🎚️ Tracks Editor")
        tracks = self.app_state.get_all_tracks()
        if not tracks:
            # --- Enhanced Empty State V3 ---
            # Displayed only when the welcome message has been dismissed
            if "welcome_message_shown" in st.session_state:
                with st.container(border=True):
                    st.markdown("#### ✨ Your Project is Empty!")
                    st.markdown(
                        """
                          Ready to start creating? Look at the **sidebar on the left** (👈 click the arrow if it's hidden).
                          That's where you can:
                          """
                    )
                    # Use columns for better visual separation of add options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("##### 📁 Upload Audio")
                        st.caption("Add music, background sounds, etc.")
                    with col2:
                        st.markdown("##### 🗣️ Generate Affirmations")
                        st.caption("Convert text to speech.")
                    with col3:
                        st.markdown("##### 🧠✨ Generate Tones")
                        st.caption("Add Binaural or Solfeggio.")

                    st.markdown("""
                          ---
                          Once you add a track, the editor controls will appear here in this main panel.
                          """)
            # ---------------------------
            return  # Stop rendering editor if no tracks

        st.markdown("Adjust settings below. **Volume & Pan** live. Others need **'Apply Effects'**.")
        st.divider()
        track_ids_to_delete = []
        logger.debug(f"Rendering editor for {len(tracks)} tracks.")
        for track_id, track_data in list(tracks.items()):
            if track_id not in self.app_state.get_all_tracks():
                continue
            track_name = track_data.get("name", "Unnamed")
            with st.expander(f"Track: **{track_name}** (`{track_id[:6]}`)", expanded=True):
                logger.debug(f"Rendering expander for: '{track_name}' ({track_id})")
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
                processed_audio = track_data.get("processed_audio")
                track_sr = track_data.get("sr", GLOBAL_SR)
                track_len_sec = len(processed_audio) / track_sr if processed_audio is not None and track_sr > 0 else 0
                st.caption(f"SR: {track_sr} Hz | Len: {track_len_sec:.2f}s")
                # Waveform Visualization (Optimized)
                st.markdown("**Waveform Preview**")
                current_temp_file = track_data.get("temp_file_path")
                display_path = None
                regenerate_temp_file = False
                if current_temp_file and os.path.exists(current_temp_file):
                    display_path = current_temp_file
                else:
                    regenerate_temp_file = True
                    logger.warning(f"Regenerating temp file for {track_id}")
                if regenerate_temp_file:
                    if processed_audio is not None and processed_audio.size > 0:
                        new_temp_path = save_audio_to_temp(processed_audio, track_sr)
                        if new_temp_path:
                            self.app_state.update_track_param(track_id, "temp_file_path", new_temp_path)
                            display_path = new_temp_path
                            if current_temp_file and current_temp_file != new_temp_path and os.path.exists(current_temp_file):
                                try:
                                    os.remove(current_temp_file)
                                    logger.info(f"Cleaned invalid old temp: {current_temp_file}")
                                except OSError as e:
                                    logger.warning(f"Could not remove old temp {current_temp_file}: {e}")
                        else:
                            st.error("Failed regen temp file.")
                    else:
                        st.info("No audio data.")
                if display_path:
                    ws_options = WaveSurferOptions(
                        height=100, normalize=True, wave_color="#A020F0", progress_color="#800080", cursor_color="#333333", cursor_width=1, bar_width=2, bar_gap=1
                    )
                    update_count = track_data.get("update_counter", 0)
                    audix_key = f"audix_{track_id}_{update_count}"
                    logger.debug(f"Calling audix '{track_data.get('name', 'N/A')}' key={audix_key} path={display_path}")
                    audix(data=display_path, sample_rate=track_sr, wavesurfer_options=ws_options, key=audix_key)
                # Effects Section
                st.markdown("---")
                st.markdown("**Audio Effects**")
                st.caption("Adjust settings, then click 'Apply Effects'.")
                fx_col1, fx_col2, fx_col3 = st.columns(3)
                speed = fx_col1.slider(
                    "Speed", 0.25, 4.0, track_data.get("speed_factor", 1.0), 0.05, key=f"speed_{track_id}", help="Playback speed (>1 faster, <1 slower). Uses time stretch."
                )
                if speed != track_data.get("speed_factor"):
                    self.app_state.update_track_param(track_id, "speed_factor", speed)
                pitch = fx_col2.slider("Pitch (semitones)", -12, 12, track_data.get("pitch_shift", 0), 1, key=f"pitch_{track_id}", help="Adjust pitch without changing speed.")
                if pitch != track_data.get("pitch_shift"):
                    self.app_state.update_track_param(track_id, "pitch_shift", pitch)
                f_type = fx_col3.selectbox(
                    "Filter",
                    ["off", "lowpass", "highpass"],
                    index=["off", "lowpass", "highpass"].index(track_data.get("filter_type", "off")),
                    key=f"filter_type_{track_id}",
                    help="Apply low/high pass filter.",
                )
                if f_type != track_data.get("filter_type"):
                    self.app_state.update_track_param(track_id, "filter_type", f_type)
                f_enabled = track_data["filter_type"] != "off"
                max_cutoff = track_sr / 2 - 1
                f_cutoff = fx_col3.number_input(
                    f"Cutoff ({'Hz' if f_enabled else 'Off'})",
                    20.0,
                    max_cutoff if max_cutoff > 20 else 20.0,
                    float(track_data.get("filter_cutoff", 8000.0)),
                    100.0,
                    key=f"filter_cutoff_{track_id}",
                    disabled=not f_enabled,
                    help="Filter cutoff frequency.",
                )
                if f_cutoff != track_data.get("filter_cutoff"):
                    self.app_state.update_track_param(track_id, "filter_cutoff", f_cutoff)
                # Apply Effects Button
                if st.button("⚙️ Apply Effects", key=f"apply_fx_{track_id}", help="Process audio with Speed, Pitch, Filter settings. Updates waveform/playback."):
                    logger.info(f"Apply Effects clicked for: '{track_data.get('name', 'N/A')}' ({track_id})")
                    original_audio = track_data.get("original_audio")
                    if original_audio is not None and original_audio.size > 0:
                        with st.spinner(f"Applying effects..."):
                            processed_audio = apply_all_effects(track_data)
                            old_temp_file = track_data.get("temp_file_path")
                            new_temp_file = save_audio_to_temp(processed_audio, track_sr)
                            if new_temp_file:
                                self.app_state.update_track_param(track_id, "processed_audio", processed_audio)
                                self.app_state.update_track_param(track_id, "temp_file_path", new_temp_file)
                                self.app_state.increment_update_counter(track_id)
                                if old_temp_file and os.path.exists(old_temp_file):
                                    try:
                                        os.remove(old_temp_file)
                                        logger.info(f"Deleted old temp file: {old_temp_file}")
                                    except OSError as e:
                                        logger.warning(f"Could not delete old temp file {old_temp_file}: {e}")
                                st.success(f"Effects applied.")
                                st.rerun()
                            else:
                                logger.error(f"Failed save new temp file for {track_id}")
                                st.error("Failed save updated audio.")
                    else:
                        st.warning(f"No original audio data.")
            except Exception as e:
                logger.exception(f"Error rendering main col for {track_id}")
                st.error(f"Error waveform/effects: {e}")

    def _render_track_controls_col(self, track_id: TrackID, track_data: TrackData, column: st.delta_generator.DeltaGenerator) -> bool:
        """Renders the controls (name, vol, pan, etc.). Returns True if delete clicked."""
        delete_clicked = False
        with column:
            try:
                st.markdown("**Track Mixing Controls**")
                st.caption("Changes affect final mix.")
                name = st.text_input("Track Name", value=track_data.get("name", "Unnamed"), key=f"name_{track_id}", help="Rename track.")
                if name != track_data.get("name"):
                    self.app_state.update_track_param(track_id, "name", name)
                vp_col1, vp_col2 = st.columns(2)
                vol = vp_col1.slider("Volume", 0.0, 2.0, track_data.get("volume", 1.0), 0.05, key=f"vol_{track_id}", help="Adjust loudness (live).")
                if vol != track_data.get("volume"):
                    self.app_state.update_track_param(track_id, "volume", vol)
                pan = vp_col2.slider("Pan", -1.0, 1.0, track_data.get("pan", 0.0), 0.1, key=f"pan_{track_id}", help="Adjust L/R balance (live).")
                if pan != track_data.get("pan"):
                    self.app_state.update_track_param(track_id, "pan", pan)
                ms_col1, ms_col2 = st.columns(2)
                mute = ms_col1.checkbox("Mute", value=track_data.get("mute", False), key=f"mute_{track_id}", help="Silence track in mix.")
                if mute != track_data.get("mute"):
                    self.app_state.update_track_param(track_id, "mute", mute)
                solo = ms_col2.checkbox("Solo", value=track_data.get("solo", False), key=f"solo_{track_id}", help="Isolate track(s) in mix.")
                if solo != track_data.get("solo"):
                    self.app_state.update_track_param(track_id, "solo", solo)
                st.markdown("---")
                if st.button("🗑️ Delete Track", key=f"delete_{track_id}", help="Permanently delete."):
                    delete_clicked = True
                    st.warning(f"Track marked for deletion.")
            except Exception as e:
                logger.exception(f"Error rendering controls for {track_id}")
                st.error(f"Error controls: {e}")
        return delete_clicked

    def render_master_controls(self):
        """Renders the master preview and export buttons."""
        st.divider()
        st.header("🔊 Master Output")
        master_cols = st.columns(2)
        with master_cols[0]:
            st.button("🎧 Preview Mix (10s)", key="preview_mix", use_container_width=True, help="Play first 10s of mix with current settings.", on_click=self._handle_preview_click)
        with master_cols[1]:
            st.button("💾 Export Full Mix (.wav)", key="export_mix", use_container_width=True, help="Generate complete mix for download.", on_click=self._handle_export_click)
            # Download button appears here after export generation
            if "export_buffer" in st.session_state and st.session_state.export_buffer:
                st.download_button(
                    label="⬇️ Download Full Mix (.wav)",
                    data=st.session_state.export_buffer,
                    file_name="mindmorph_mix.wav",
                    mime="audio/wav",
                    key="download_export_key",
                    use_container_width=True,
                )
                st.session_state.export_buffer = None  # Clear buffer after showing

    def _handle_preview_click(self):
        """Callback for Preview Mix button."""
        logger.info("Preview Mix button clicked.")
        tracks = self.app_state.get_all_tracks()
        if "preview_audio" in st.session_state:
            del st.session_state.preview_audio
        if not tracks:
            st.warning("No tracks loaded.")
            return
        with st.spinner("Generating preview mix..."):
            mix_preview = mix_tracks(tracks, preview=True)
            if mix_preview.size > 0:
                st.session_state.preview_audio = save_audio(mix_preview, GLOBAL_SR)
                logger.info("Preview generated.")
            else:
                logger.warning("Preview mix empty.")

    def _handle_export_click(self):
        """Callback for Export Mix button."""
        logger.info("Export Full Mix button clicked.")
        tracks = self.app_state.get_all_tracks()
        if "export_buffer" in st.session_state:
            del st.session_state.export_buffer
        if not tracks:
            st.warning("No tracks loaded.")
            return
        with st.spinner("Generating full mix..."):
            full_mix = mix_tracks(tracks, preview=False)
            if full_mix.size > 0:
                st.session_state.export_buffer = save_audio(full_mix, GLOBAL_SR)
                logger.info("Full mix generated.")
            else:
                logger.warning("Full mix empty.")

    def render_preview_audio_player(self):
        """Displays the preview audio player if preview data exists in state."""
        # This should be called in the main layout area *after* the master controls column
        if "preview_audio" in st.session_state and st.session_state.preview_audio:
            st.markdown("**Preview:**")
            st.audio(st.session_state.preview_audio, format="audio/wav")
            st.session_state.preview_audio = None  # Clear after displaying

    def render_instructions(self):
        """Renders the instructions expander with more detail."""
        st.divider()
        with st.expander("📖 Show Instructions & Notes", expanded=False):
            st.markdown("""
              **Welcome to MindMorph!** Create custom subliminal audio by layering affirmations, sounds, & frequencies.

              **Workflow:**

              1.  **➕ Add Tracks (Sidebar):**
                  * `Upload Audio`: Add background music, nature sounds, etc. (WAV, MP3...).
                  * `Generate Affirmations`: Convert typed affirmations to speech.
                  * `Generate Tones`: Create Binaural Beats (use headphones!) or Solfeggio frequencies.

              2.  **🎚️ Edit Tracks (Main Panel):**
                  * Find controls for each track in its expandable section.
                  * **Effects (Speed, Pitch, Filter):** Adjust these sliders/selectors. **Click `⚙️ Apply Effects`** to process the audio. The waveform/playback for *this track* will update.
                  * **Mixing Controls (Volume, Pan, Mute, Solo):** These affect the final mix directly (no 'Apply' needed). Use low volume for subliminal affirmations.
                  * `Track Name`: Rename tracks.
                  * `🗑️ Delete Track`: Remove unwanted tracks.

              3.  **🔊 Mix & Export (Bottom Panel):**
                  * `🎧 Preview Mix (10s)`: Hear the start of the combined audio with all current settings.
                  * `💾 Export Full Mix (.wav)`: Generate the final WAV file. Click the download button that appears below.

              **Tips:**
              * Keep affirmations clear but low volume (e.g., 0.1-0.3) under masking sounds.
              * High speed (e.g., 2x-4x) is common for affirmation tracks.
              * Layer multiple tracks creatively.
              """)


# ==========================================
# 5. Benchmarking (Placeholder Class)
# ==========================================
# (Benchmarker class remains the same)
class Benchmarker:
    """Placeholder for performance benchmarking."""

    def __init__(self, tts_generator: TTSGenerator):
        self.tts_generator = tts_generator
        logger.debug("Benchmarker initialized.")

    def benchmark_tts(self, word_count: int = 10000, repetitions: int = 1):
        logger.info(f"Starting TTS benchmark: {word_count} words, {repetitions} reps.")
        st.subheader(f"Benchmark: TTS Generation ({word_count} words)")
        if word_count <= 0:
            st.warning("Word count must be positive.")
            return
        placeholder_word = "benchmark "
        text = (placeholder_word * (word_count // len(placeholder_word) + 1))[: word_count * 6]
        st.text(f"Generating ~{word_count} words...")
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
# 6. Main Application Logic & Onboarding V4
# ==========================================


# --- Onboarding Helper ---
def show_welcome_message():
    """Displays the welcome message container using columns for clarity."""
    # Use session state to show only once per session
    if "welcome_message_shown" not in st.session_state:
        # --- Enhanced Welcome Message V4 ---
        with st.container(border=True):
            st.markdown("### 👋 Welcome to MindMorph!")
            st.markdown("Create custom subliminal audio by layering sounds and applying effects.")
            st.markdown("---")
            st.markdown("#### Quick Start:")

            col1, col2, col3 = st.columns(3)
            with col1:
                # Added stronger visual cue and more direct language
                st.markdown("#### 1. Add Tracks ➕")
                st.markdown("Look Left! **👈** Use the **sidebar** to add your first audio layer.")
                st.caption("Upload, Generate TTS, Tones")
            with col2:
                st.markdown("#### 2. Edit Tracks 🎚️")
                st.markdown("Adjust **effects** in the main panel (once tracks are added).")
                st.caption("Click 'Apply Effects' for Speed/Pitch/Filter!")
            with col3:
                st.markdown("#### 3. Mix & Export 🔊")
                st.markdown("Use **master controls** at the bottom.")
                st.caption("Preview or Download WAV")

            st.markdown("---")
            st.markdown("*(Click button below to hide this guide. Find details in Instructions at page bottom.)*")

            # --- Center the Dismiss Button ---
            button_cols = st.columns([1, 1.5, 1])  # Use columns for centering
            with button_cols[1]:
                if st.button("Got it! Let's Start Creating ✨", key="dismiss_welcome", type="primary", use_container_width=True):
                    st.session_state.welcome_message_shown = True
                    logger.info("Welcome message dismissed by user.")
                    st.rerun()
        # ---------------------------------


def main():
    """Main function to run the Streamlit application."""
    logger.info("Starting main application function.")

    # --- Page Header ---
    st.title("🧠 MindMorph - Subliminal Audio Editor")

    # --- Onboarding Welcome Message ---
    show_welcome_message()  # Function now handles the logic internally
    # ---------------------------------

    # Show intro text only if welcome message is dismissed
    if "welcome_message_shown" in st.session_state:
        st.markdown("""
        Create your own custom subliminal audio tracks by layering affirmations, background sounds, and therapeutic frequencies.
        Adjust speed, pitch, volume, and more, then export your creation.
        """)
        st.divider()

    # --- Initialize Core Components ---
    app_state = AppState()
    tts_generator = TTSGenerator()
    ui_manager = UIManager(app_state, tts_generator)
    # benchmarker = Benchmarker(tts_generator) # Keep benchmarker optional

    # --- Render UI Sections ---
    # Only render the main UI if the welcome message isn't showing OR if it has been dismissed
    if "welcome_message_shown" in st.session_state:
        ui_manager.render_sidebar()
        ui_manager.render_tracks_editor()  # Handles empty state with improved message
        ui_manager.render_master_controls()
        # Display preview audio player if generated
        ui_manager.render_preview_audio_player()
        # Instructions
        ui_manager.render_instructions()
        # --- Optional Benchmarking Section ---
        # (Commented out by default)
        # st.divider()
        # with st.expander("⏱️ Run Benchmarks", expanded=False):
        #      st.info("Run performance tests (can take time).")
        #      bm_words = st.number_input("Words for TTS Benchmark", 100, 20000, 10000, 100)
        #      bm_reps = st.number_input("Repetitions", 1, 10, 1, 1)
        #      if st.button("Run TTS Benchmark"): benchmarker.benchmark_tts(bm_words, bm_reps)
        # --- Footer ---
        st.divider()
        st.caption("MindMorph Subliminal Editor")
        logger.info("Reached end of main application function render.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("A critical error occurred in main execution.")
        st.error("A critical error occurred. Please check the logs (`editor_oop.log`) or restart.")
        st.code(traceback.format_exc())
    # Attempt to stop listener - might not execute reliably
    # logger.info("Attempting to stop logging listener.")
    # if 'listener' in locals() and listener is not None and listener.is_alive(): listener.stop()
