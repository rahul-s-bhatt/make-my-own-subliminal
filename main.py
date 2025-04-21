# ==========================================
# MindMorph - Pro Subliminal Audio Editor
# Version: 2.15 (OOP - Fixed Syntax Error in TTSGenerator)
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
import textwrap
import time
import traceback
import uuid
from io import BytesIO, StringIO
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
    import docx  # For reading .docx files
    import librosa
    import librosa.effects
    import numpy as np
    import pyttsx3
    import soundfile as sf
    from scipy import signal
    from streamlit.runtime.uploaded_file_manager import UploadedFile
    from streamlit_advanced_audio import WaveSurferOptions, audix

    logger.debug("Core audio, UI, and docx libraries imported successfully.")
except ImportError as e:
    logger.exception("CRITICAL: Failed to import core libraries. App cannot continue.")
    missing_lib = e.name
    install_cmd = f"pip install {missing_lib}"
    if missing_lib == "docx":
        install_cmd = "pip install python-docx"
    st.error(f"Core library import failed: {e}. Ensure dependencies installed.")
    st.error(f"Missing: {missing_lib}")
    st.error(f"Try: `{install_cmd}`")
    st.code(traceback.format_exc())
    st.stop()

# --- Constants ---
GLOBAL_SR = 44100
logger.debug(f"Global Sample Rate set to: {GLOBAL_SR} Hz")
TTS_CHUNK_SIZE = 1500  # For full generation chunking
MAX_WAVEFORM_DISPLAY_SAMPLES = GLOBAL_SR * 60 * 10  # ~10 minutes
logger.debug(f"Max samples for waveform display: {MAX_WAVEFORM_DISPLAY_SAMPLES}")

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
    """Applies speed, pitch, and filter effects sequentially to original audio."""
    track_name = track_data.get("name", "Unnamed")
    logger.info(f"Applying base effects (Speed/Pitch/Filter) to track: '{track_name}'")
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
    logger.info(f"Finished applying base effects to '{track_name}'")
    return audio


# --- UPDATED mix_tracks with more logging ---
def mix_tracks(tracks_dict: Dict[TrackID, TrackData], preview: bool = False) -> AudioData:
    """Mixes tracks based on current settings (volume, pan, mute, solo)."""
    logger.info(f"Starting track mixing. Preview mode: {preview}")
    if not tracks_dict:
        logging.warning("Mix command called but no tracks.")
        return np.zeros((0, 2), dtype=np.float32)
    tracks = tracks_dict.values()
    solo_active = any(t.get("solo", False) for t in tracks)
    active_tracks_data = [t for t in tracks if t.get("solo", False)] if solo_active else list(tracks)
    valid_tracks_to_mix = [t for t in active_tracks_data if not t.get("mute", False) and t.get("processed_audio", np.array([])).size > 0]
    if not valid_tracks_to_mix:
        logging.warning("No valid tracks to mix.")
        return np.zeros((0, 2), dtype=np.float32)
    try:
        max_len = 0
        for t in valid_tracks_to_mix:
            if t.get("processed_audio") is not None:
                max_len = max(max_len, len(t["processed_audio"]))
    except (ValueError, KeyError) as e:
        logging.exception("Could not get max length.")
        return np.zeros((0, 2), dtype=np.float32)

    logger.debug(f"Initial max_len calculated: {max_len} samples ({max_len / GLOBAL_SR:.2f}s)")  # DEBUG LOG

    if preview:
        preview_len = int(GLOBAL_SR * 10)  # 10 seconds for preview
        original_max_len = max_len
        max_len = min(max_len, preview_len) if max_len > 0 else preview_len
        logger.info(f"Preview mode: Limiting mix length from {original_max_len} samples to {max_len} samples ({max_len / GLOBAL_SR:.2f}s)")  # Log the change
    else:
        logger.info(f"Full mix mode: Target length is {max_len} samples ({max_len / GLOBAL_SR:.2f}s)")

    if max_len <= 0:
        logging.warning("Mix length <= 0.")
        return np.zeros((0, 2), dtype=np.float32)

    mix = np.zeros((max_len, 2), dtype=np.float32)
    logger.debug(f"Mix buffer created with shape: {(max_len, 2)}")  # DEBUG LOG

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
    logger.info(f"Mixing complete. Final mix shape: {final_mix.shape} ({len(final_mix) / GLOBAL_SR:.2f}s)")  # Log final shape/duration
    return final_mix.astype(np.float32)


# --- Helper to read text files ---
# (read_text_file function remains the same as v2.12 with enhanced checks)
def read_text_file(uploaded_file: UploadedFile) -> Optional[str]:
    """Reads content from uploaded txt or docx file with enhanced error handling."""
    if uploaded_file is None:
        logger.error("read_text_file called with None object.")
        return None
    file_name = uploaded_file.name
    logger.info(f"Attempting to read file: {file_name}, Type: {uploaded_file.type}")
    try:
        if file_name.lower().endswith(".txt"):
            logger.debug(f"Reading .txt file: {file_name}")
            try:
                return uploaded_file.read().decode("utf-8", errors="replace")
            except Exception as e_decode:
                logger.exception(f"Error decoding .txt file {file_name}")
                st.error(f"Error decoding: {e_decode}")
                return None
        elif file_name.lower().endswith(".docx"):
            logger.debug(f"Processing .docx file: {file_name}")
            try:
                uploaded_file.seek(0)
                document = docx.Document(uploaded_file)
                logger.debug(f"docx created. Found {len(document.paragraphs)} paragraphs.")
                para_texts = []
                for i, para in enumerate(document.paragraphs):
                    if para is None:
                        logger.warning(f"Para {i} in {file_name} is None.")
                        continue
                    para_text = getattr(para, "text", "")  # Safely get text
                    if para_text:
                        para_texts.append(para_text)
                logger.debug(f"Extracted {len(para_texts)} non-empty paragraphs.")
                return "\n".join(para_texts)
            except Exception as e_docx:
                logger.exception(f"Error processing docx: {file_name}")
                st.error(f"Error reading Word doc: {e_docx}")
                return None
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}.")
            logger.error(f"Unsupported type: {uploaded_file.type}")
            return None
    except Exception as e_outer:
        logger.exception(f"Outer error reading file: {uploaded_file.name}")
        st.error(f"Error reading file: {e_outer}")
        return None


# ==========================================
# 2. Application State Management
# ==========================================
# (AppState class remains the same as previous version with loop_to_fit)
class AppState:
    """Manages the application state, including tracks and their parameters."""

    STATE_KEY = "tracks"

    def __init__(self):
        if self.STATE_KEY not in st.session_state:
            logger.info(f"Initializing '{self.STATE_KEY}'.")
            st.session_state[self.STATE_KEY] = {}
        default_keys = AppState.get_default_track_params().keys()
        for track_id, track_data in list(self.get_all_tracks().items()):
            changed = False
            for key, default_value in AppState.get_default_track_params().items():
                if key not in track_data:
                    logger.warning(f"Track {track_id} missing key '{key}', adding default.")
                    st.session_state[self.STATE_KEY][track_id][key] = default_value
                    changed = True

    @staticmethod
    def get_default_track_params() -> TrackData:
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
            "pitch_shift": 0,
            "pan": 0.0,
            "filter_type": "off",
            "filter_cutoff": 8000.0,
            "loop_to_fit": False,  # Looping parameter
            "update_counter": 0,
        }

    def _get_tracks_dict(self) -> Dict[TrackID, TrackData]:
        return st.session_state.get(self.STATE_KEY, {})

    def get_all_tracks(self) -> Dict[TrackID, TrackData]:
        return self._get_tracks_dict()

    def get_track(self, track_id: TrackID) -> Optional[TrackData]:
        return self._get_tracks_dict().get(track_id)

    def add_track(self, track_id: TrackID, track_data: TrackData):
        """Adds a new track, ensuring defaults."""
        if not isinstance(track_data, dict):
            logger.error(f"Invalid track data type {track_id}")
            return
        default_params = AppState.get_default_track_params()
        final_track_data = {**default_params, **track_data}
        # Removed temp_file_path logic - generated on demand now
        st.session_state[self.STATE_KEY][track_id] = final_track_data
        logger.info(f"Added track ID: {track_id}, Name: '{final_track_data.get('name', 'N/A')}'")

    def delete_track(self, track_id: TrackID):
        """Deletes a track."""
        # Temp file cleanup for audix relies on OS or manual intervention now
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            track_name = tracks[track_id].get("name", "N/A")
            del st.session_state[self.STATE_KEY][track_id]
            logger.info(f"Deleted track ID: {track_id}, Name: '{track_name}'")
            return True
        else:
            logger.warning(f"Attempted delete non-existent track {track_id}")
            return False

    def update_track_param(self, track_id: TrackID, param_name: str, value: Any):
        """Updates a specific parameter for a given track."""
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            if param_name in AppState.get_default_track_params():
                st.session_state[self.STATE_KEY][track_id][param_name] = value
            else:
                logger.warning(f"Attempted update invalid param '{param_name}' for {track_id}")
        else:
            logger.warning(f"Attempted update param for non-existent track {track_id}")

    def increment_update_counter(self, track_id: TrackID):
        """Increments the update counter for a track."""
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
# 3. TTS Generation (Class Wrapper) - WITH CHUNKING & CORRECTED SPINNER
# ==========================================
class TTSGenerator:
    """Handles Text-to-Speech generation using pyttsx3 with chunking."""

    def __init__(self, chunk_size: int = TTS_CHUNK_SIZE):
        self.engine = None
        self.rate = 200
        self.volume = 1.0
        self.chunk_size = chunk_size
        logger.debug(f"TTSGenerator initialized chunk={chunk_size}, rate={self.rate}")

    def _init_engine(self):
        """Initializes or re-initializes the pyttsx3 engine."""
        try:
            logger.debug("Initializing pyttsx3 engine for generation.")
            self.engine = pyttsx3.init()
            if self.engine is None:
                raise RuntimeError("pyttsx3.init() returned None")
            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)
        except Exception as e:
            logger.exception("Failed init pyttsx3.")
            self.engine = None
            raise RuntimeError("TTS Engine init failed") from e

    def generate(self, text: str) -> Tuple[Optional[AudioData], Optional[SampleRate]]:
        """Generates audio from text using chunking, returns concatenated audio."""
        logger.info(f"Starting TTS generation for text length: {len(text)} chars.")
        if not text:
            logger.warning("Empty TTS text provided.")
            return None, None

        temp_chunk_files = []
        audio_chunks = []
        final_sr = None
        # --- Create placeholder for progress text ---
        progress_placeholder = st.empty()
        try:
            self._init_engine()
            if self.engine is None:
                st.error("TTS Engine could not be initialized.")
                return None, None

            logger.debug(f"Wrapping text into chunks of size: {self.chunk_size}")
            chunks = textwrap.wrap(text, self.chunk_size, break_long_words=True, replace_whitespace=False, drop_whitespace=True)
            num_chunks = len(chunks)
            logger.info(f"Split text into {num_chunks} chunks.")

            # Use a general spinner message
            with st.spinner(f"Synthesizing {num_chunks} audio chunks..."):
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        logger.debug(f"Skipping empty chunk {i + 1}/{num_chunks}")
                        continue

                    chunk_start_time = time.time()
                    # --- Update placeholder text ---
                    progress_placeholder.text(f"Synthesizing audio chunk {i + 1}/{num_chunks}...")
                    # ---------------------------------

                    temp_chunk_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk{i}.wav", mode="wb") as tmp:
                            temp_chunk_path = tmp.name
                        temp_chunk_files.append(temp_chunk_path)
                        logger.debug(f"Synthesizing chunk {i + 1}/{num_chunks} to {temp_chunk_path}...")
                        self.engine.save_to_file(chunk, temp_chunk_path)
                        self.engine.runAndWait()
                        chunk_end_time = time.time()
                        logger.debug(f"Chunk {i + 1} synthesized in {chunk_end_time - chunk_start_time:.2f}s.")
                        if not os.path.exists(temp_chunk_path) or os.path.getsize(temp_chunk_path) == 0:
                            logger.error(f"TTS engine failed create non-empty file chunk {i + 1}: {temp_chunk_path}")
                            st.error(f"TTS engine failed for chunk {i + 1}.")
                            if temp_chunk_path in temp_chunk_files:
                                temp_chunk_files.remove(temp_chunk_path)
                            temp_chunk_path = None
                            continue
                    # --- FIXED SYNTAX ERROR: Correct indentation for except block ---
                    except Exception as e_chunk:
                        logger.exception(f"Failed synthesize chunk {i + 1}")
                        st.error(f"Error processing chunk {i + 1}: {e_chunk}")
                        # Ensure problematic path isn't processed later if it was added
                        if temp_chunk_path in temp_chunk_files:
                            try:
                                temp_chunk_files.remove(temp_chunk_path)
                                logger.debug(f"Removed problematic chunk path {temp_chunk_path} from list.")
                            except ValueError:
                                logger.warning(f"Could not remove {temp_chunk_path} from temp_chunk_files list.")
                        continue  # Skip to next chunk
                    # --- END FIX ---

                # --- Update placeholder before concatenation ---
                progress_placeholder.text("Combining audio chunks...")
                logger.info("Synthesized all chunks. Concatenating.")
                # --- Concatenate chunks ---
                for i, file_path in enumerate(temp_chunk_files):
                    if not file_path or not os.path.exists(file_path):
                        logger.warning(f"Temp chunk file {file_path} missing, skipping.")
                        continue
                    try:
                        logger.debug(f"Loading chunk file: {file_path}")
                        audio_data, sr = sf.read(file_path, dtype="float32", always_2d=True)
                        if audio_data.shape[1] == 1:
                            audio_data = np.concatenate([audio_data, audio_data], axis=1)
                        if final_sr is None:
                            final_sr = sr
                            logger.debug(f"Detected SR: {final_sr} Hz")
                        elif sr != final_sr:
                            logger.warning(f"SR mismatch! Expected {final_sr}, got {sr}. Resampling chunk.")
                            if audio_data.size > 0:
                                audio_data = librosa.resample(audio_data.T, orig_sr=sr, target_sr=final_sr).T.astype(np.float32)
                            else:
                                continue
                        audio_chunks.append(audio_data)
                        logger.debug(f"Loaded chunk {i + 1} shape {audio_data.shape}")
                    except Exception as e_load:
                        logger.exception(f"Failed load chunk {file_path}")
                        st.error(f"Error loading chunk {i + 1}: {e_load}")

            # --- Final Assembly ---
            progress_placeholder.empty()  # Clear progress text
            if not audio_chunks:
                logger.error("No valid chunks loaded.")
                st.error("Failed process chunks.")
                return None, None
            logger.info(f"Concatenating {len(audio_chunks)} audio chunks...")
            final_audio = np.concatenate(audio_chunks, axis=0)
            if final_sr is None:
                logger.error("Could not determine SR.")
                st.error("Failed determine SR.")
                return None, None
            if final_sr != GLOBAL_SR:
                logger.info(f"Resampling final audio from {final_sr} Hz to {GLOBAL_SR} Hz.")
                if final_audio.size > 0:
                    final_audio = librosa.resample(final_audio.T, orig_sr=final_sr, target_sr=GLOBAL_SR).T.astype(np.float32)
                final_sr = GLOBAL_SR
            logger.info(f"TTS generation complete. Final shape: {final_audio.shape}, SR: {final_sr}")
            return final_audio, final_sr
        except Exception as e:
            progress_placeholder.empty()  # Clear progress on error
            logger.exception("TTS Gen Failed.")
            st.error(f"TTS Gen Failed: {e}")
            return None, None
        finally:
            progress_placeholder.empty()  # Ensure cleared
            logger.debug(f"Cleaning up {len(temp_chunk_files)} temp chunk files...")
            cleaned_count = 0
            for file_path in temp_chunk_files:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except OSError as e_os:
                        logger.warning(f"Could not delete temp TTS chunk {file_path}: {e_os}")
            logger.debug(f"Cleaned up {cleaned_count} temp files.")


# ==========================================
# 4. UI Management (Class) - REVERTED TTS PREVIEW
# ==========================================
# (UIManager class remains the same as v2.11)
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
            st.markdown("Use options below to add sounds.")
            self._render_uploader()
            st.divider()
            self._render_affirmation_inputs()
            st.divider()  # Uses new method
            self._render_binaural_generator()
            st.divider()
            self._render_solfeggio_generator()
            st.markdown("---")
            st.info("Edit track details in the main panel.")

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

    def _render_affirmation_inputs(self):
        """Renders options for adding affirmations (TTS, File Upload, Record Placeholder)."""
        st.subheader("🗣️ Add Affirmations")
        tab1, tab2, tab3 = st.tabs(["Type Text", "Upload File", "Record Audio"])
        with tab1:
            st.caption("Type or paste affirmations below.")
            affirmation_text = st.text_area(
                "Affirmations (one per line)", height=150, key="affirmation_text_area", label_visibility="collapsed", help="Enter each affirmation on a new line."
            )
            if st.button(
                "Generate Track from Text",
                key="generate_tts_text_key",
                use_container_width=True,
                type="primary",
                help="Convert the text above to a spoken audio track using Text-to-Speech.",
            ):
                if affirmation_text:
                    self._generate_tts_track(affirmation_text, "Affirmations (Text)")
                else:
                    st.warning("Please enter text in the text area first.")
        with tab2:
            st.caption("Upload a .txt or .docx file containing affirmations.")
            uploaded_file = st.file_uploader(
                "Upload Affirmation File", type=["txt", "docx"], key="affirmation_file_uploader", label_visibility="collapsed", help="Select text/Word document."
            )
            if st.button(
                "Generate Track from File",
                key="generate_tts_file_key",
                use_container_width=True,
                type="primary",
                help="Read the uploaded file and convert its text content to a spoken audio track.",
            ):
                if uploaded_file:
                    with st.spinner(f"Reading {uploaded_file.name}..."):
                        text = read_text_file(uploaded_file)
                    if text is not None:  # Check if reading succeeded
                        if text.strip():
                            self._generate_tts_track(text, f"Affirmations ({uploaded_file.name})")
                        else:
                            st.warning(f"File '{uploaded_file.name}' has no readable text.")
                            logger.warning(f"File '{uploaded_file.name}' read empty.")
                    else:
                        logger.error("Failed read uploaded file for TTS.")
                else:
                    st.warning("Please upload file.")
        with tab3:
            st.caption("Record your own voice for affirmations.")
            st.info("🎙️ Audio recording feature coming soon!")
            st.markdown("Use 'Upload Audio File(s)' for now.")
            st.button("Start Recording", key="start_record_key", disabled=True, use_container_width=True)

    def _generate_tts_track(self, text_content: str, track_name: str):
        """Helper to generate TTS (now uses chunking generator) and add track."""
        # Spinner is now handled inside the generate method with progress updates
        audio, sr = self.tts_generator.generate(text_content)
        if audio is not None and sr is not None:
            track_id = str(uuid.uuid4())
            track_params = AppState.get_default_track_params()
            # Store full original audio, remove processed
            track_params.update({"original_audio": audio, "processed_audio": audio.copy(), "sr": sr, "name": track_name})
            self.app_state.add_track(track_id, track_params)
            st.success(f"'{track_name}' track generated!")
            st.toast("Affirmation track added!", icon="✅")

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
            if "welcome_message_shown" in st.session_state:
                with st.container(border=True):
                    st.markdown("#### ✨ Your Project is Empty!")
                    st.markdown("Ready to start creating? Look at the **sidebar on the left** (👈 click the arrow if it's hidden).")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("##### 📁 Upload Audio")
                        st.caption("Add music, sounds...")
                    with col2:
                        st.markdown("##### 🗣️ Add Affirmations")
                        st.caption("From text, file, or recording.")
                    with col3:
                        st.markdown("##### 🧠✨ Generate Tones")
                        st.caption("Add Binaural or Solfeggio.")
                    st.markdown("---")
                    st.markdown("Once you add a track, the editor controls will appear here.")
            return
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
        """Renders the waveform and effects controls for a single track with downsampling for display."""
        # (Implementation remains the same as previous version with downsampling)
        with column:
            try:
                processed_audio = track_data.get("processed_audio")
                track_sr = track_data.get("sr", GLOBAL_SR)
                track_len_sec = len(processed_audio) / track_sr if processed_audio is not None and track_sr > 0 else 0
                st.caption(f"SR: {track_sr} Hz | Len: {track_len_sec:.2f}s")
                st.markdown("**Waveform Preview**")
                display_path = None
                temp_file_to_delete_later = None
                if processed_audio is not None and processed_audio.size > 0:
                    audio_for_display = processed_audio
                    num_samples = len(processed_audio)
                    if num_samples > MAX_WAVEFORM_DISPLAY_SAMPLES:
                        logger.warning(f"Track '{track_data.get('name', track_id)}' ({num_samples}) > max ({MAX_WAVEFORM_DISPLAY_SAMPLES}). Downsampling.")
                        st.caption(f"(Waveform downsampled from {track_len_sec:.1f}s original)")
                        try:
                            target_sr_display = (MAX_WAVEFORM_DISPLAY_SAMPLES / num_samples) * track_sr
                            target_sr_display = max(100, target_sr_display)
                            logger.debug(f"Downsampling waveform to effective SR: {target_sr_display:.2f} Hz")
                            with st.spinner("Generating downsampled preview..."):
                                audio_for_display = librosa.resample(processed_audio.T, orig_sr=track_sr, target_sr=target_sr_display).T.astype(np.float32)
                            logger.debug(f"Downsampled shape: {audio_for_display.shape}")
                        except Exception as e_resample:
                            logger.exception(f"Error downsampling {track_id}")
                            st.warning("Could not downsample waveform.", icon="⚠️")
                            audio_for_display = None
                    if audio_for_display is not None and audio_for_display.size > 0:
                        temp_wav_path = save_audio_to_temp(audio_for_display, track_sr)
                        if temp_wav_path:
                            display_path = temp_wav_path
                            temp_file_to_delete_later = temp_wav_path
                        else:
                            st.error("Could not save temp file for waveform.")
                    elif num_samples <= MAX_WAVEFORM_DISPLAY_SAMPLES:
                        st.info("Failed prepare audio for display.")
                else:
                    st.info("No audio data to display waveform.")
                if display_path:
                    ws_options = WaveSurferOptions(
                        height=100, normalize=True, wave_color="#A020F0", progress_color="#800080", cursor_color="#333333", cursor_width=1, bar_width=2, bar_gap=1
                    )
                    update_count = track_data.get("update_counter", 0)
                    audix_key = f"audix_{track_id}_{update_count}"
                    logger.debug(f"Calling audix '{track_data.get('name', 'N/A')}' key={audix_key} path={display_path}")
                    audix(data=display_path, sample_rate=track_sr, wavesurfer_options=ws_options, key=audix_key)
                st.markdown("---")
                st.markdown("**Audio Effects**")
                st.caption("Adjust settings, then click 'Apply Effects'.")
                loop_col, fx_col1, fx_col2, fx_col3 = st.columns([0.5, 1, 1, 1])
                with loop_col:
                    st.markdown("<br/>", unsafe_allow_html=True)
                    loop_value = st.checkbox(
                        "🔁 Loop",
                        key=f"loop_{track_id}",
                        value=track_data.get("loop_to_fit", False),
                        help="Repeat track to match longest track duration? Apply via 'Apply Effects'.",
                    )
                    if loop_value != track_data.get("loop_to_fit"):
                        self.app_state.update_track_param(track_id, "loop_to_fit", loop_value)
                with fx_col1:
                    speed = st.slider(
                        "Speed", 0.25, 4.0, track_data.get("speed_factor", 1.0), 0.05, key=f"speed_{track_id}", help="Playback speed (>1 faster, <1 slower). Uses time stretch."
                    )
                    if speed != track_data.get("speed_factor"):
                        self.app_state.update_track_param(track_id, "speed_factor", speed)
                with fx_col2:
                    pitch = st.slider("Pitch (semitones)", -12, 12, track_data.get("pitch_shift", 0), 1, key=f"pitch_{track_id}", help="Adjust pitch without changing speed.")
                    if pitch != track_data.get("pitch_shift"):
                        self.app_state.update_track_param(track_id, "pitch_shift", pitch)
                with fx_col3:
                    f_type = st.selectbox(
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
                    f_cutoff = st.number_input(
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
                if st.button("⚙️ Apply Effects", key=f"apply_fx_{track_id}", help="Process audio with Speed, Pitch, Filter, and Loop settings. Updates waveform/playback."):
                    logger.info(f"Apply Effects clicked for: '{track_data.get('name', 'N/A')}' ({track_id})")
                    original_audio = track_data.get("original_audio")
                    if original_audio is not None and original_audio.size > 0:
                        with st.spinner(f"Applying effects..."):
                            effects_applied_audio = apply_all_effects(track_data)
                            final_processed_audio = effects_applied_audio
                            should_loop = track_data.get("loop_to_fit", False)
                            if should_loop and effects_applied_audio.size > 0:
                                logger.info(f"Looping enabled for track {track_id}. Calc max len.")
                                all_tracks = self.app_state.get_all_tracks()
                                max_len = 0
                                for t_id, t_data in all_tracks.items():
                                    pa = t_data.get("processed_audio")
                                    if pa is not None and pa.size > 0:
                                        # Use length *after* base effects for current track comparison
                                        current_track_len = len(effects_applied_audio) if t_id == track_id else len(pa)
                                        max_len = max(max_len, current_track_len)
                                logger.debug(f"Max project length: {max_len} samples.")
                                current_len = len(effects_applied_audio)
                                if max_len > 0 and current_len < max_len:
                                    logger.info(f"Looping track '{track_data.get('name')}' from {current_len} to {max_len} samples.")
                                    n_repeats = max_len // current_len
                                    remainder = max_len % current_len
                                    looped_list = [effects_applied_audio] * n_repeats
                                    if remainder > 0:
                                        looped_list.append(effects_applied_audio[:remainder])
                                    final_processed_audio = np.concatenate(looped_list, axis=0)
                                    logger.debug(f"Looping complete. New length: {len(final_processed_audio)}")
                                else:
                                    logger.debug(f"Looping not needed for '{track_data.get('name')}' (curr: {current_len}, max: {max_len})")
                            else:
                                logger.debug(f"Looping disabled or audio empty for '{track_data.get('name')}'")
                            self.app_state.update_track_param(track_id, "processed_audio", final_processed_audio)
                            self.app_state.increment_update_counter(track_id)
                        st.success(f"Effects applied.")
                        st.rerun()
                    else:
                        st.warning(f"No original audio data.")
            except Exception as e:
                logger.exception(f"Error rendering main col for {track_id}")
                st.error(f"Error waveform/effects: {e}")

    def _render_track_controls_col(self, track_id: TrackID, track_data: TrackData, column: st.delta_generator.DeltaGenerator) -> bool:
        """Renders the controls (name, vol, pan, etc.). Returns True if delete clicked."""
        # (Implementation remains the same as previous version)
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
        # (Implementation remains the same as previous version)
        st.divider()
        st.header("🔊 Master Output")
        master_cols = st.columns(2)
        with master_cols[0]:
            st.button("🎧 Preview Mix (10s)", key="preview_mix", use_container_width=True, help="Play first 10s of mix with current settings.", on_click=self._handle_preview_click)
        with master_cols[1]:
            st.button("💾 Export Full Mix (.wav)", key="export_mix", use_container_width=True, help="Generate complete mix for download.", on_click=self._handle_export_click)
            if "export_buffer" in st.session_state and st.session_state.export_buffer:
                st.download_button(
                    label="⬇️ Download Full Mix (.wav)",
                    data=st.session_state.export_buffer,
                    file_name="mindmorph_mix.wav",
                    mime="audio/wav",
                    key="download_export_key",
                    use_container_width=True,
                )
                st.session_state.export_buffer = None

    def _handle_preview_click(self):
        """Callback for Preview Mix button."""
        # (Implementation remains the same)
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
        # (Implementation remains the same)
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
        # (Implementation remains the same)
        if "preview_audio" in st.session_state and st.session_state.preview_audio:
            st.markdown("**Preview:**")
            st.audio(st.session_state.preview_audio, format="audio/wav")
            st.session_state.preview_audio = None

    def render_instructions(self):
        """Renders the instructions expander, expanded if no tracks exist."""
        # (Implementation remains the same as previous version - expanded logic)
        tracks_exist = bool(self.app_state.get_all_tracks())
        st.divider()
        with st.expander("📖 Show Instructions & Notes", expanded=not tracks_exist):
            st.markdown("""
              **Welcome to MindMorph!** Create custom subliminal audio by layering affirmations, sounds, & frequencies.

              **Workflow:**

              1.  **➕ Add Tracks (Sidebar):**
                  * `Upload Audio`: Add background music, nature sounds, etc. (WAV, MP3...).
                  * `Generate Affirmations`: Convert typed affirmations to speech, upload from file, or (soon) record directly.
                  * `Generate Tones`: Create Binaural Beats (use headphones!) or Solfeggio frequencies.

              2.  **🎚️ Edit Tracks (Main Panel):**
                  * Find controls for each track in its expandable section.
                  * **Effects (Speed, Pitch, Filter, Loop):** Adjust these sliders/selectors/checkboxes. **Click `⚙️ Apply Effects`** to process the audio with *all* selected effects. The waveform/playback for *this track* will update.
                  * **Mixing Controls (Volume, Pan, Mute, Solo):** These affect the final mix directly (no 'Apply' needed). Use low volume for subliminal affirmations.
                  * `Track Name`: Rename tracks.
                  * `🗑️ Delete Track`: Remove unwanted tracks.

              3.  **🔊 Mix & Export (Bottom Panel):**
                  * `🎧 Preview Mix (10s)`: Hear the start of the combined audio with all current settings.
                  * `💾 Export Full Mix (.wav)`: Generate the final WAV file. Click the download button that appears below.

              **Tips:**
              * Keep affirmations clear but low volume (e.g., 0.1-0.3) under masking sounds.
              * High speed (e.g., 2x-4x) is common for affirmation tracks.
              * Use the `🔁 Loop` option for short affirmations/sounds you want repeated throughout a longer background track. Remember to 'Apply Effects' after checking/unchecking it.
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
# 6. Main Application Logic & Onboarding V5
# ==========================================
# (Onboarding Welcome Message remains the same as v2.4)
def show_welcome_message():
    """Displays the welcome message container using columns for clarity."""
    if "welcome_message_shown" not in st.session_state:
        with st.container(border=True):
            st.markdown("### 👋 Welcome to MindMorph!")
            st.markdown("Create custom subliminal audio by layering sounds and applying effects.")
            st.markdown("---")
            st.markdown("#### Quick Start:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### 1. Add Tracks ➕")
                st.markdown("Look Left! **👈** Use the **sidebar**.")
                st.caption("Upload, Generate TTS, Tones")
            with col2:
                st.markdown("#### 2. Edit Tracks 🎚️")
                st.markdown("Adjust **effects** in the main panel.")
                st.caption("Click 'Apply Effects' for Speed/Pitch/Filter!")
            with col3:
                st.markdown("#### 3. Mix & Export 🔊")
                st.markdown("Use **master controls** at the bottom.")
                st.caption("Preview or Download WAV")
            st.markdown("---")
            st.markdown("*(Click button below to hide this guide. Find details in Instructions at page bottom.)*")
            button_cols = st.columns([1, 1.5, 1])  # Centering columns
            with button_cols[1]:
                if st.button("Got it! Let's Start Creating ✨", key="dismiss_welcome", type="primary", use_container_width=True):
                    st.session_state.welcome_message_shown = True
                    logger.info("Welcome message dismissed.")
                    st.rerun()


def main():
    """Main function to run the Streamlit application."""
    logger.info("Starting main application function.")
    st.title("🧠 MindMorph - Subliminal Audio Editor")
    show_welcome_message()  # Show welcome message if needed
    if "welcome_message_shown" in st.session_state:
        st.markdown("Create custom subliminals by layering affirmations, sounds, & frequencies. Adjust effects & mix.")
        st.divider()
        app_state = AppState()
        tts_generator = TTSGenerator()  # Uses chunking version now
        ui_manager = UIManager(app_state, tts_generator)
        # benchmarker = Benchmarker(tts_generator) # Optional
        ui_manager.render_sidebar()
        ui_manager.render_tracks_editor()  # Handles empty state
        ui_manager.render_master_controls()
        ui_manager.render_preview_audio_player()  # Display preview if available
        ui_manager.render_instructions()  # Instructions at the end
        # --- Optional Benchmarking Section ---
        # (Commented out by default)
        # st.divider()
        # with st.expander("⏱️ Run Benchmarks", expanded=False):
        #      st.info("Run performance tests (can take time).")
        #      bm_words = st.number_input("Words for TTS Benchmark", 100, 20000, 10000, 100)
        #      bm_reps = st.number_input("Repetitions", 1, 10, 1, 1)
        #      if st.button("Run TTS Benchmark"): benchmarker.benchmark_tts(bm_words, bm_reps)
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
