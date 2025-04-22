# ==========================================
# MindMorph - Pro Subliminal Audio Editor
# Version: 4.3 (OOP - Fixed Filename Session State Error)
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
import math  # For fading
import queue
import re  # For sanitizing filename
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
file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=3)  # 10MB
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

    # Try importing pydub for MP3 export check
    try:
        from pydub import AudioSegment

        PYDUB_AVAILABLE = True
        logger.info("pydub library found. MP3 export might be available (if ffmpeg is also installed).")
    except ImportError:
        PYDUB_AVAILABLE = False
        logger.warning("pydub library not found. MP3 export will be disabled.")

    logger.debug("Core audio, UI, and docx libraries imported successfully.")
except ImportError as e:
    logger.exception("CRITICAL: Failed to import core libraries. App cannot continue.")
    missing_lib = e.name
    install_cmd = f"pip install {missing_lib}"
    if missing_lib == "docx":
        install_cmd = "pip install python-docx"
    elif missing_lib == "pydub":
        install_cmd = "pip install pydub"
    st.error(f"Core library import failed: {e}. Ensure dependencies installed.")
    st.error(f"Missing: {missing_lib}")
    st.error(f"Try: `{install_cmd}`")
    st.code(traceback.format_exc())
    st.stop()

# --- Constants ---
GLOBAL_SR = 44100
logger.debug(f"Global Sample Rate set to: {GLOBAL_SR} Hz")
TTS_CHUNK_SIZE = 1500  # For full generation chunking
PREVIEW_DURATION_S = 60  # Default duration for previews in editor
MIX_PREVIEW_DURATION_S = 10  # Duration for the master mix preview
MIX_PREVIEW_PROCESSING_BUFFER_S = 5  # Extra seconds to process for preview to handle speed changes
logger.debug(f"Editor Preview duration: {PREVIEW_DURATION_S}s")
logger.debug(f"Mix Preview duration: {MIX_PREVIEW_DURATION_S}s")

# --- Data Types ---
AudioData = np.ndarray
SampleRate = int
TrackID = str
TrackData = Dict[str, Any]
TrackType = str  # e.g., 'affirmation', 'background', 'frequency', 'voice', 'other'

# --- Constants for Track Types ---
TRACK_TYPE_AFFIRMATION = "🗣️ Affirmation"
TRACK_TYPE_BACKGROUND = "🎵 Background/Mask"
TRACK_TYPE_FREQUENCY = "🧠 Frequency"
TRACK_TYPE_VOICE = "🎤 Voice Recording"
TRACK_TYPE_OTHER = "⚪ Other"
TRACK_TYPES = [TRACK_TYPE_AFFIRMATION, TRACK_TYPE_BACKGROUND, TRACK_TYPE_FREQUENCY, TRACK_TYPE_VOICE, TRACK_TYPE_OTHER]


# ==========================================
# 1. Audio Processing Utilities (Functions)
# ==========================================
# (Functions load_audio, save_audio, save_audio_to_temp, generate*, apply*, mix_tracks remain unchanged from v4.0)
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


def generate_white_noise(duration: float, sr: SampleRate, volume: float) -> AudioData:
    """Generates stereo white noise."""
    logger.info(f"Generating white noise: dur={duration}s, vol={volume}")
    num_samples = int(sr * duration)
    noise = np.random.uniform(-1.0, 1.0, size=(num_samples, 2))
    audio = np.clip(noise * volume, -1.0, 1.0)
    return audio.astype(np.float32)


def apply_reverse(audio: AudioData) -> AudioData:
    """Reverses the audio data along the time axis."""
    logger.debug("Applying reverse effect.")
    if audio is None or audio.size == 0:
        return audio
    try:
        return audio[::-1].astype(np.float32)
    except Exception as e:
        logger.exception("Error applying reverse effect.")
        st.error(f"Reverse effect failed: {e}")
        return audio


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
            audio_contiguous = np.ascontiguousarray(audio.T)
            audio_stretched = librosa.effects.time_stretch(audio_contiguous, rate=speed_factor)
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
            audio_contiguous = np.ascontiguousarray(audio.T)
            audio_shifted = librosa.effects.pitch_shift(audio_contiguous, sr=sr, n_steps=n_steps)
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


def apply_all_effects(track_data: TrackData, audio_segment: Optional[AudioData] = None) -> AudioData:
    """Applies effects sequentially: Reverse (optional) -> Speed -> Pitch -> Filter."""
    track_name = track_data.get("name", "Unnamed")
    should_reverse = track_data.get("reverse_audio", False)
    if audio_segment is not None:
        audio = audio_segment.copy()
        logger.debug(f"Applying effects to segment for: '{track_name}' (Reverse={should_reverse})")
    else:
        logger.info(f"Applying effects to full original_audio for: '{track_name}' (Reverse={should_reverse})")
        audio = track_data.get("original_audio")
    if audio is None:
        logging.error(f"'{track_name}' missing 'original_audio'.")
        return np.zeros((0, 2), dtype=np.float32)
    audio = audio.copy()
    sr = track_data.get("sr", GLOBAL_SR)
    if audio.size == 0:
        logging.warning(f"Input audio for effects is empty for '{track_name}'.")
        return audio
    if should_reverse:
        audio = apply_reverse(audio)
    audio = apply_speed_change(audio, sr, track_data.get("speed_factor", 1.0))
    audio = apply_pitch_shift(audio, sr, track_data.get("pitch_shift", 0))
    audio = apply_filter(audio, sr, track_data.get("filter_type", "off"), track_data.get("filter_cutoff", 8000.0))
    logger.debug(f"Finished applying base effects for '{track_name}'")
    return audio


def get_preview_audio(track_data: TrackData, preview_duration_s: int = PREVIEW_DURATION_S) -> Optional[AudioData]:
    """Generates a preview (first N seconds) of the track with effects AND volume/pan applied."""
    track_name = track_data.get("name", "N/A")
    logger.info(f"Generating preview audio for track '{track_name}'")
    original_audio = track_data.get("original_audio")
    sr = track_data.get("sr", GLOBAL_SR)
    if original_audio is None or original_audio.size == 0:
        logger.warning(f"No original audio for '{track_name}'.")
        return None
    try:
        preview_samples = min(len(original_audio), int(sr * preview_duration_s))
        if preview_samples <= 0:
            logger.warning(f"Preview samples <= 0 for '{track_name}'.")
            return None
        preview_segment = original_audio[:preview_samples].copy()
        logger.debug(f"Applying effects to preview segment (len={preview_samples}) for '{track_name}'")
        processed_preview = apply_all_effects(track_data, audio_segment=preview_segment)
        vol = track_data.get("volume", 1.0)
        pan = track_data.get("pan", 0.0)
        logger.debug(f"Applying Vol ({vol:.2f}) / Pan ({pan:.2f}) to preview for '{track_name}'")
        pan_rad = (pan + 1) * np.pi / 4
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)
        if processed_preview.ndim == 2 and processed_preview.shape[1] == 2:
            processed_preview[:, 0] *= left_gain
            processed_preview[:, 1] *= right_gain
        else:
            logger.warning(f"Preview for '{track_name}' not stereo ({processed_preview.shape}). Cannot apply pan/vol.")
        processed_preview = np.clip(processed_preview, -1.0, 1.0)
        logger.debug(f"Preview generation complete for '{track_name}'. Shape: {processed_preview.shape}")
        return processed_preview
    except Exception as e:
        logger.exception(f"Error generating preview for '{track_name}'")
        st.error(f"Error previewing '{track_name}': {e}")
        return None


def mix_tracks(tracks_dict: Dict[TrackID, TrackData], preview: bool = False, fade_in_s: float = 0.0, fade_out_s: float = 0.0) -> Tuple[Optional[AudioData], Optional[int]]:
    """Mixes tracks. Applies full effects on-the-fly. Handles looping & fades."""
    logger.info(f"Starting track mixing (non-destructive). Preview: {preview}, FadeIn: {fade_in_s}s, FadeOut: {fade_out_s}s")
    if not tracks_dict:
        logging.warning("Mix called but no tracks.")
        return None, None

    processed_segments = {}  # Store processed audio (full or segment)
    max_len = 0
    valid_track_ids_for_mix = []
    solo_active = any(t.get("solo", False) for t in tracks_dict.values())
    logger.debug(f"Solo active: {solo_active}")

    # --- Determine target length and process segments ---
    target_mix_len_samples = 0  # Will be calculated or set for preview

    if preview:
        target_mix_len_samples = int(GLOBAL_SR * MIX_PREVIEW_DURATION_S)
        process_duration_s = MIX_PREVIEW_DURATION_S + MIX_PREVIEW_PROCESSING_BUFFER_S
        logger.info(f"Preview mode: Target mix length {target_mix_len_samples} samples ({MIX_PREVIEW_DURATION_S}s). Processing ~{process_duration_s}s per track.")
    else:
        logger.info("Full mix mode: Calculating full lengths.")
        temp_full_lengths = {}
        for track_id, t_data in tracks_dict.items():
            is_active = t_data.get("solo", False) if solo_active else not t_data.get("mute", False)
            original_audio = t_data.get("original_audio")
            if is_active and original_audio is not None and original_audio.size > 0:
                logger.debug(f"Processing FULL track '{t_data.get('name', track_id)}' for length.")
                full_processed = apply_all_effects(t_data)  # Includes potential reversal
                temp_full_lengths[track_id] = len(full_processed)
                processed_segments[track_id] = full_processed  # Store for later use
                valid_track_ids_for_mix.append(track_id)
            else:
                logger.debug(f"Skipping track '{t_data.get('name', track_id)}' (muted/soloed/no audio).")
        if not valid_track_ids_for_mix:
            logging.warning("No valid tracks for full mix.")
            return None, None
        target_mix_len_samples = max(temp_full_lengths.values()) if temp_full_lengths else 0
        logger.info(f"Full mix mode: Target length is {target_mix_len_samples} samples ({target_mix_len_samples / GLOBAL_SR:.2f}s)")

    if target_mix_len_samples <= 0:
        logging.warning("Mix length <= 0.")
        return None, None

    # --- Process required segments (only needed if preview, otherwise already done) ---
    if preview:
        for track_id, t_data in tracks_dict.items():
            is_active = t_data.get("solo", False) if solo_active else not t_data.get("mute", False)
            original_audio = t_data.get("original_audio")
            # Only process if it wasn't already processed in the full mix check (which shouldn't happen in preview mode)
            if track_id not in processed_segments and is_active and original_audio is not None and original_audio.size > 0:
                logger.debug(f"Processing PREVIEW segment for track '{t_data.get('name', track_id)}'.")
                process_samples = min(len(original_audio), int(GLOBAL_SR * process_duration_s))
                segment = original_audio[:process_samples].copy()
                processed_segment = apply_all_effects(t_data, audio_segment=segment)  # Includes potential reversal
                processed_segments[track_id] = processed_segment
                valid_track_ids_for_mix.append(track_id)  # Ensure it's in the list
        # Remove duplicates just in case
        valid_track_ids_for_mix = list(set(valid_track_ids_for_mix))
        if not valid_track_ids_for_mix:
            logging.warning("No valid tracks found for preview mix.")
            return None, None

    # --- Create mix buffer and combine tracks ---
    mix = np.zeros((target_mix_len_samples, 2), dtype=np.float32)
    logger.info(f"Mixing {len(valid_track_ids_for_mix)} tracks. Target length: {target_mix_len_samples / GLOBAL_SR:.2f}s")
    for track_id in valid_track_ids_for_mix:
        t_data = tracks_dict[track_id]
        track_name = t_data.get("name", track_id)
        audio_segment = processed_segments.get(track_id)
        if audio_segment is None or audio_segment.size == 0:
            logger.warning(f"Processed audio missing for '{track_name}'. Skipping.")
            continue
        final_audio_for_track = audio_segment
        should_loop = t_data.get("loop_to_fit", False)
        if not preview and should_loop:  # Only loop in full mix mode
            current_len = len(audio_segment)
            if target_mix_len_samples > 0 and current_len > 0 and current_len < target_mix_len_samples:
                logger.info(f"Looping track '{track_name}' from {current_len} to {target_mix_len_samples} samples for mix.")
                n_repeats = target_mix_len_samples // current_len
                remainder = target_mix_len_samples % current_len
                looped_list = [audio_segment] * n_repeats
                if remainder > 0:
                    looped_list.append(audio_segment[:remainder])
                final_audio_for_track = np.concatenate(looped_list, axis=0)
                logger.debug(f"Looping complete for '{track_name}'. New length: {len(final_audio_for_track)}")
            else:
                logger.debug(f"Looping not needed for '{track_name}' (curr: {current_len}, max: {target_mix_len_samples})")
        current_len = len(final_audio_for_track)
        if current_len < target_mix_len_samples:
            audio_adjusted = np.pad(final_audio_for_track, ((0, target_mix_len_samples - current_len), (0, 0)), mode="constant")
        elif current_len > target_mix_len_samples:
            audio_adjusted = final_audio_for_track[:target_mix_len_samples, :]
        else:
            audio_adjusted = final_audio_for_track.copy()
        pan = t_data.get("pan", 0.0)
        vol = t_data.get("volume", 1.0)
        logger.debug(f"Track '{track_name}': vol={vol:.2f}, pan={pan:.2f}")
        pan_rad = (pan + 1) * np.pi / 4
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)
        panned_audio = np.zeros_like(audio_adjusted)
        panned_audio[:, 0] = audio_adjusted[:, 0] * left_gain
        panned_audio[:, 1] = audio_adjusted[:, 1] * right_gain
        mix += panned_audio
        logger.debug(f"Added track '{track_name}' to mix.")

    final_mix = np.clip(mix, -1.0, 1.0)

    # --- Apply Master Fades (only on full export, not preview) ---
    if not preview and (fade_in_s > 0 or fade_out_s > 0):
        logger.info(f"Applying master fades: In={fade_in_s}s, Out={fade_out_s}s")
        fade_in_samples = int(fade_in_s * GLOBAL_SR)
        fade_out_samples = int(fade_out_s * GLOBAL_SR)
        total_samples = len(final_mix)
        if fade_in_samples > 0:
            fade_in_samples = min(fade_in_samples, total_samples)
            fade_in_curve = np.linspace(0.0, 1.0, fade_in_samples)
            final_mix[:fade_in_samples, 0] *= fade_in_curve
            final_mix[:fade_in_samples, 1] *= fade_in_curve
            logger.debug(f"Applied fade-in over {fade_in_samples} samples.")
        if fade_out_samples > 0:
            fade_out_samples = min(fade_out_samples, total_samples)
            fade_out_curve = np.linspace(1.0, 0.0, fade_out_samples)
            final_mix[total_samples - fade_out_samples :, 0] *= fade_out_curve
            final_mix[total_samples - fade_out_samples :, 1] *= fade_out_curve
            logger.debug(f"Applied fade-out over {fade_out_samples} samples.")
    # -------------------------------------------------------------

    logger.info("Mixing complete.")
    return final_mix.astype(np.float32), target_mix_len_samples  # Return audio and length


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
# 2. Application State Management - NON-DESTRUCTIVE + PREVIEW PATH + TYPE
# ==========================================
# (AppState class remains the same as v4.0)
class AppState:
    """Manages the application state using non-destructive approach + preview path + track type."""

    STATE_KEY = "tracks_non_destructive_v3"

    def __init__(self):
        if self.STATE_KEY not in st.session_state:
            logger.info(f"Initializing '{self.STATE_KEY}'.")
            st.session_state[self.STATE_KEY] = {}
        default_keys = AppState.get_default_track_params().keys()
        tracks_dict = self.get_all_tracks()
        for track_id, track_data in list(tracks_dict.items()):
            changed = False
            if "processed_audio" in track_data:
                del st.session_state[self.STATE_KEY][track_id]["processed_audio"]
                logger.debug(f"Removed old 'processed_audio' key for {track_id}")
                changed = True
            if "preview_temp_file_path" not in track_data:
                logger.warning(f"Track {track_id} missing 'preview_temp_file_path', adding default.")
                st.session_state[self.STATE_KEY][track_id]["preview_temp_file_path"] = None
                changed = True
            if "reverse_audio" not in track_data:
                logger.warning(f"Track {track_id} missing 'reverse_audio', adding default.")
                st.session_state[self.STATE_KEY][track_id]["reverse_audio"] = False
                changed = True
            if "track_type" not in track_data:
                logger.warning(f"Track {track_id} missing 'track_type', adding default.")
                st.session_state[self.STATE_KEY][track_id]["track_type"] = TRACK_TYPE_OTHER
                changed = True
            for key, default_value in AppState.get_default_track_params().items():
                if key not in track_data:
                    logger.warning(f"Track {track_id} missing key '{key}', adding default.")
                    st.session_state[self.STATE_KEY][track_id][key] = default_value
                    changed = True

    @staticmethod
    def get_default_track_params() -> TrackData:
        """Returns default parameters including track_type."""
        return {
            "original_audio": np.zeros((0, 2), dtype=np.float32),
            "sr": GLOBAL_SR,
            "name": "New Track",
            "track_type": TRACK_TYPE_OTHER,
            "volume": 1.0,
            "mute": False,
            "solo": False,
            "speed_factor": 1.0,
            "pitch_shift": 0,
            "pan": 0.0,
            "filter_type": "off",
            "filter_cutoff": 8000.0,
            "loop_to_fit": False,
            "reverse_audio": False,
            "preview_temp_file_path": None,
            "update_counter": 0,
        }

    def _get_tracks_dict(self) -> Dict[TrackID, TrackData]:
        return st.session_state.get(self.STATE_KEY, {})

    def get_all_tracks(self) -> Dict[TrackID, TrackData]:
        return self._get_tracks_dict()

    def get_track(self, track_id: TrackID) -> Optional[TrackData]:
        return self._get_tracks_dict().get(track_id)

    def add_track(self, track_id: TrackID, track_data: TrackData, track_type: TrackType = TRACK_TYPE_OTHER):
        """Adds a new track, storing original audio, parameters, and type."""
        if not isinstance(track_data, dict):
            logger.error(f"Invalid track data type {track_id}")
            return
        default_params = AppState.get_default_track_params()
        final_track_data = {key: track_data.get(key, default_value) for key, default_value in default_params.items()}
        if "original_audio" not in track_data or track_data["original_audio"] is None:
            logger.error(f"Attempted add track {track_id} without original_audio.")
            return
        final_track_data["original_audio"] = track_data["original_audio"]
        final_track_data["sr"] = track_data.get("sr", GLOBAL_SR)
        final_track_data["name"] = track_data.get("name", default_params["name"])
        final_track_data["track_type"] = track_type  # Set the type
        final_track_data["preview_temp_file_path"] = None
        st.session_state[self.STATE_KEY][track_id] = final_track_data
        logger.info(f"Added track ID: {track_id}, Name: '{final_track_data.get('name', 'N/A')}', Type: {track_type}")

    def delete_track(self, track_id: TrackID):
        """Deletes a track and its associated preview temp file."""
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            track_name = tracks[track_id].get("name", "N/A")
            preview_path = tracks[track_id].get("preview_temp_file_path")
            if preview_path and os.path.exists(preview_path):
                try:
                    os.remove(preview_path)
                    logger.info(f"Deleted preview temp file '{preview_path}' for track {track_id}")
                except OSError as e:
                    logger.warning(f"Failed delete preview temp file '{preview_path}': {e}")
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
            if param_name in AppState.get_default_track_params() and param_name != "original_audio":
                st.session_state[self.STATE_KEY][track_id][param_name] = value
            elif param_name == "original_audio":
                logger.error(f"Attempted update original_audio directly for {track_id}.")
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
# 3. TTS Generation (Class Wrapper) - WITH CHUNKING
# ==========================================
# (TTSGenerator class remains the same as v2.15 - Fixed Spinner Error)
class TTSGenerator:
    """Handles Text-to-Speech generation using pyttsx3 with chunking."""

    def __init__(self, chunk_size: int = TTS_CHUNK_SIZE):
        self.engine = None
        self.rate = 200
        self.volume = 1.0
        self.chunk_size = chunk_size
        logger.debug(f"TTSGenerator initialized chunk={chunk_size}, rate={self.rate}")

    def _init_engine(self):
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
        logger.info(f"Starting TTS generation for text length: {len(text)} chars.")
        if not text:
            logger.warning("Empty TTS text provided.")
            return None, None
        temp_chunk_files = []
        audio_chunks = []
        final_sr = None
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
            with st.spinner(f"Synthesizing {num_chunks} audio chunks..."):
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        logger.debug(f"Skipping empty chunk {i + 1}/{num_chunks}")
                        continue
                    chunk_start_time = time.time()
                    progress_placeholder.text(f"Synthesizing audio chunk {i + 1}/{num_chunks}...")
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
                    except Exception as e_chunk:
                        logger.exception(f"Failed synthesize chunk {i + 1}")
                        st.error(f"Error processing chunk {i + 1}: {e_chunk}")
                        if temp_chunk_path in temp_chunk_files:
                            try:
                                temp_chunk_files.remove(temp_chunk_path)
                                logger.debug(f"Removed problematic chunk path {temp_chunk_path}.")
                            except ValueError:
                                logger.warning(f"Could not remove {temp_chunk_path} from list.")
                        continue
                progress_placeholder.text("Combining audio chunks...")
                logger.info("Synthesized all chunks. Concatenating.")
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
            progress_placeholder.empty()
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
            progress_placeholder.empty()
            logger.exception("TTS Gen Failed.")
            st.error(f"TTS Gen Failed: {e}")
            return None, None
        finally:
            progress_placeholder.empty()
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
# 4. UI Management (Class) - COMMUNITY FEATURES
# ==========================================
class UIManager:
    """Handles rendering Streamlit UI components using button-driven previews."""

    def __init__(self, app_state: AppState, tts_generator: TTSGenerator):
        self.app_state = app_state
        self.tts_generator = tts_generator
        logger.debug("UIManager initialized.")

    # (render_sidebar and its helpers _render_uploader, _render_affirmation_inputs, _render_binaural_generator, _render_solfeggio_generator remain the same as v4.0)
    # ...
    def render_sidebar(self):
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
            self._render_frequency_generators()
            st.divider()  # Combined Freq Gens
            self._render_background_generators()
            st.markdown("---")  # Background Noise
            st.info("Edit track details in the main panel.")

    def _render_uploader(self):
        st.subheader("📁 Upload Audio File(s)")
        st.caption("Upload music, voice recordings, or other sounds.")
        uploaded_files = st.file_uploader(
            "Select audio files",
            type=["wav", "mp3", "ogg", "flac"],
            accept_multiple_files=True,
            key="upload_files_key",
            label_visibility="collapsed",
            help="Select one or more audio files.",
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
                        # Deduce likely type based on name? Simple heuristic for now.
                        track_type = TRACK_TYPE_VOICE if "voice" in file.name.lower() or "record" in file.name.lower() else TRACK_TYPE_BACKGROUND
                        track_params.update({"original_audio": audio, "sr": sr, "name": file.name})
                        self.app_state.add_track(track_id, track_params, track_type=track_type)
                        st.success(f"Loaded '{file.name}'")
                        loaded_track_names.append(file.name)
                    else:
                        logger.warning(f"Skipped empty upload: {file.name}")
                        st.warning(f"Skipped empty: {file.name}")

    def _render_affirmation_inputs(self):
        """Renders options for adding affirmations (TTS, File Upload, Record Placeholder)."""
        st.subheader("🗣️ Add Affirmations")
        st.caption("Uses system default TTS voice. Voice selection depends on OS/browser.")
        tab1, tab2, tab3 = st.tabs(["Type Text", "Upload File", "Record Audio"])
        with tab1:
            st.caption("Type or paste affirmations below.")
            affirmation_text = st.text_area(
                "Affirmations (one per line)", height=150, key="affirmation_text_area", label_visibility="collapsed", help="Enter each affirmation on a new line."
            )
            if st.button(
                "Generate Affirmation Track", key="generate_tts_text_key", use_container_width=True, type="primary", help="Convert the text above to a spoken audio track."
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
                    if text is not None:
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
        """Helper to generate TTS and add track, tagged as Affirmation."""
        audio, sr = self.tts_generator.generate(text_content)  # Spinner inside generate
        if audio is not None and sr is not None:
            track_id = str(uuid.uuid4())
            track_params = AppState.get_default_track_params()
            track_params.update({"original_audio": audio, "sr": sr, "name": track_name})
            self.app_state.add_track(track_id, track_params, track_type=TRACK_TYPE_AFFIRMATION)  # Tag as affirmation
            st.success(f"'{track_name}' track generated!")
            st.toast("Affirmation track added!", icon="✅")

    def _render_frequency_generators(self):
        """Renders options for generating Binaural Beats and Solfeggio Tones."""
        st.subheader("🧠✨ Add Frequencies / Tones")
        gen_type = st.radio("Select Type:", ["Binaural Beats", "Solfeggio Tones", "Presets"], key="freq_gen_type", horizontal=True, label_visibility="collapsed")

        if gen_type == "Binaural Beats":
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
                track_params.update({"original_audio": audio, "sr": GLOBAL_SR, "name": f"Binaural {bb_fleft}/{bb_fright}Hz"})
                self.app_state.add_track(track_id, track_params, track_type=TRACK_TYPE_FREQUENCY)
                st.success("Binaural Beats generated!")

        elif gen_type == "Solfeggio Tones":
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
                track_params.update({"original_audio": audio, "sr": GLOBAL_SR, "name": f"Solfeggio {freq}Hz"})
                self.app_state.add_track(track_id, track_params, track_type=TRACK_TYPE_FREQUENCY)
                st.success(f"Solfeggio {freq}Hz generated!")

        elif gen_type == "Presets":
            st.markdown("<small>Generate common frequency patterns.</small>", unsafe_allow_html=True)
            preset_options = {
                "Focus (Alpha 10Hz)": {"type": "binaural", "f_left": 200, "f_right": 210},
                "Relaxation (Theta 5Hz)": {"type": "binaural", "f_left": 150, "f_right": 155},
                "Deep Sleep (Delta 2Hz)": {"type": "binaural", "f_left": 100, "f_right": 102},
                "Love Freq (Solfeggio 528Hz)": {"type": "solfeggio", "freq": 528},
                "Miracle Tone (Solfeggio 417Hz)": {"type": "solfeggio", "freq": 417},
            }
            preset_name = st.selectbox("Select Preset:", list(preset_options.keys()), key="freq_preset_select")
            preset_data = preset_options[preset_name]
            cols_preset = st.columns(2)
            preset_duration = cols_preset[0].number_input("Duration (s)##Preset", 1, 3600, 60, 1, key="preset_duration")
            preset_vol = cols_preset[1].slider("Volume##Preset", 0.0, 1.0, 0.2, 0.05, key="preset_volume", help="Volume (usually kept low)")

            if st.button(f"Generate '{preset_name}' Track", key="generate_preset_freq"):
                with st.spinner("Generating preset frequency..."):
                    audio = None
                    if preset_data["type"] == "binaural":
                        audio = generate_binaural_beats(preset_duration, preset_data["f_left"], preset_data["f_right"], GLOBAL_SR, preset_vol)
                    elif preset_data["type"] == "solfeggio":
                        audio = generate_solfeggio_frequency(preset_duration, preset_data["freq"], GLOBAL_SR, preset_vol)

                    if audio is not None:
                        track_id = str(uuid.uuid4())
                        track_params = AppState.get_default_track_params()
                        track_params.update({"original_audio": audio, "sr": GLOBAL_SR, "name": preset_name})
                        self.app_state.add_track(track_id, track_params, track_type=TRACK_TYPE_FREQUENCY)
                        st.success(f"'{preset_name}' track generated!")
                    else:
                        st.error("Failed to generate audio for selected preset.")

    # --- NEW: Background Noise Generator ---
    def _render_background_generators(self):
        st.subheader("🎵 Add Background Noise")
        noise_type = st.selectbox("Select Noise Type:", ["White Noise"], key="noise_type_select")  # Add more later (Brown, Pink)
        cols_noise = st.columns(2)
        noise_duration = cols_noise[0].number_input("Duration (s)##Noise", 10, 7200, 300, 10, key="noise_duration", help="Length in seconds (will loop if shorter than project).")
        noise_vol = cols_noise[1].slider("Volume##Noise", 0.0, 1.0, 0.5, 0.05, key="noise_volume", help="Loudness (0.0 to 1.0).")

        if st.button(f"Generate {noise_type} Track", key="generate_noise"):
            with st.spinner(f"Generating {noise_type}..."):
                audio = None
                if noise_type == "White Noise":
                    audio = generate_white_noise(noise_duration, GLOBAL_SR, noise_vol)
                # Add elif for Brown, Pink noise here if implementing

                if audio is not None:
                    track_id = str(uuid.uuid4())
                    track_params = AppState.get_default_track_params()
                    track_params.update({"original_audio": audio, "sr": GLOBAL_SR, "name": noise_type, "loop_to_fit": True})  # Default loop to true for noise
                    self.app_state.add_track(track_id, track_params, track_type=TRACK_TYPE_BACKGROUND)
                    st.success(f"{noise_type} track generated!")
                else:
                    st.error(f"Failed to generate {noise_type}.")
        st.caption("More noise types (Brown, Pink, Rain etc.) coming soon!")

    def render_tracks_editor(self):
        """Renders the main editor area with all tracks using previews."""
        st.header("🎚️ Tracks Editor")
        tracks = self.app_state.get_all_tracks()
        if not tracks:
            if "welcome_message_shown" in st.session_state:
                with st.container(border=True):
                    st.markdown("#### ✨ Your Project is Empty!")
                    st.markdown("Ready to start creating? Use the **sidebar on the left** (👈) to:")
                    st.markdown("- Add your **🗣️ Affirmations** (Type/Upload/Record)")
                    st.markdown("- Add a **🎵 Background** sound (Upload/Generate)")
                    st.markdown("- Optionally add **🧠✨ Tones** (Binaural/Solfeggio/Presets)")
                    st.markdown("---")
                    st.markdown("Once you add a track, the editor controls will appear here.")
            return
        st.markdown("Adjust settings below. Click **'Update Preview'** to refresh the 60s preview with current settings.")
        st.divider()
        track_ids_to_delete = []
        logger.debug(f"Rendering editor for {len(tracks)} tracks.")
        for track_id, track_data in list(tracks.items()):
            if track_id not in self.app_state.get_all_tracks():
                continue
            track_name = track_data.get("name", "Unnamed")
            track_type_icon = track_data.get("track_type", TRACK_TYPE_OTHER).split(" ")[0]  # Get icon
            with st.expander(f"{track_type_icon} Track: **{track_name}** (`{track_id[:6]}`)", expanded=True):
                logger.debug(f"Rendering expander for: '{track_name}' ({track_id}) Type: {track_data.get('track_type')}")
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
        """Renders the waveform preview (button-driven) and settings controls."""
        with column:
            try:
                original_audio = track_data.get("original_audio")
                track_sr = track_data.get("sr", GLOBAL_SR)
                full_len_samples = len(original_audio) if original_audio is not None else 0
                full_len_sec = full_len_samples / track_sr if track_sr > 0 else 0
                st.caption(f"SR: {track_sr} Hz | Full Duration: {full_len_sec:.2f}s")

                # --- Waveform Visualization (Button-Driven) ---
                st.markdown(f"**Preview Waveform (First {PREVIEW_DURATION_S}s with Effects)**")
                display_path = track_data.get("preview_temp_file_path")  # Get stored path
                if display_path and os.path.exists(display_path):
                    ws_options = WaveSurferOptions(
                        height=100, normalize=True, wave_color="#A020F0", progress_color="#800080", cursor_color="#333333", cursor_width=1, bar_width=2, bar_gap=1
                    )
                    update_count = track_data.get("update_counter", 0)
                    audix_key = f"audix_{track_id}_{update_count}"
                    logger.debug(f"Displaying existing preview: '{track_data.get('name', 'N/A')}' key={audix_key} path={display_path}")
                    audix(data=display_path, sample_rate=track_sr, wavesurfer_options=ws_options, key=audix_key)
                elif original_audio is None or original_audio.size == 0:
                    st.info("Track has no audio data.")
                else:
                    st.info("Click 'Update Preview' below to generate the waveform.")
                # -------------------------------------------------------------

                st.markdown("---")
                st.markdown("**Track Settings**")
                st.caption("Adjust settings, then click 'Update Preview' below.")

                # --- Add Subliminalize Button for Affirmation Tracks ---
                if track_data.get("track_type") == TRACK_TYPE_AFFIRMATION:
                    if st.button(
                        "⚡ Subliminalize Preset", key=f"subliminalize_{track_id}", help="Quickly set high speed (4x) and low volume (0.05). Click 'Update Preview' after."
                    ):
                        logger.info(f"Subliminalize preset applied to track {track_id}")
                        self.app_state.update_track_param(track_id, "speed_factor", 4.0)
                        self.app_state.update_track_param(track_id, "volume", 0.05)
                        st.toast("Subliminal preset applied! Click 'Update Preview'.", icon="⚡")
                        st.rerun()  # Rerun to reflect slider changes
                # ------------------------------------------------------

                col_fx1_1, col_fx1_2, col_fx1_3 = st.columns([0.6, 1, 1])  # Loop/Reverse, Speed, Pitch
                with col_fx1_1:
                    st.markdown("<br/>", unsafe_allow_html=True)
                    loop_value = st.checkbox(
                        "🔁 Loop", key=f"loop_{track_id}", value=track_data.get("loop_to_fit", False), help="Loop track to fit project during final mix/export? (Preview not shown)"
                    )
                    if loop_value != track_data.get("loop_to_fit"):
                        self.app_state.update_track_param(track_id, "loop_to_fit", loop_value)
                    st.markdown("<br/>", unsafe_allow_html=True)  # Add space
                    reverse_value = st.checkbox(
                        "🔄 Reverse", key=f"reverse_{track_id}", value=track_data.get("reverse_audio", False), help="Reverse audio playback? (Updates preview)"
                    )
                    if reverse_value != track_data.get("reverse_audio"):
                        self.app_state.update_track_param(track_id, "reverse_audio", reverse_value)
                with col_fx1_2:
                    speed = st.slider("Speed", 0.25, 4.0, track_data.get("speed_factor", 1.0), 0.05, key=f"speed_{track_id}", help="Playback speed (>1 faster, <1 slower).")
                    if speed != track_data.get("speed_factor"):
                        self.app_state.update_track_param(track_id, "speed_factor", speed)
                with col_fx1_3:
                    pitch = st.slider("Pitch (semitones)", -12, 12, track_data.get("pitch_shift", 0), 1, key=f"pitch_{track_id}", help="Adjust pitch without changing speed.")
                    if pitch != track_data.get("pitch_shift"):
                        self.app_state.update_track_param(track_id, "pitch_shift", pitch)

                col_fx2_1, col_fx2_2, col_fx2_3, col_fx2_4 = st.columns([1, 1, 1, 1])  # Filter, Cutoff, Volume, Pan
                with col_fx2_1:
                    f_type = st.selectbox(
                        "Filter",
                        ["off", "lowpass", "highpass"],
                        index=["off", "lowpass", "highpass"].index(track_data.get("filter_type", "off")),
                        key=f"filter_type_{track_id}",
                        help="Apply low/high pass filter.",
                    )
                    if f_type != track_data.get("filter_type"):
                        self.app_state.update_track_param(track_id, "filter_type", f_type)
                with col_fx2_2:
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
                with col_fx2_3:
                    vol = st.slider("Volume", 0.0, 2.0, track_data.get("volume", 1.0), 0.05, key=f"vol_{track_id}", help="Adjust loudness (affects preview after Update).")
                    if vol != track_data.get("volume"):
                        self.app_state.update_track_param(track_id, "volume", vol)
                with col_fx2_4:
                    pan = st.slider("Pan", -1.0, 1.0, track_data.get("pan", 0.0), 0.1, key=f"pan_{track_id}", help="Adjust L/R balance (affects preview after Update).")
                    if pan != track_data.get("pan"):
                        self.app_state.update_track_param(track_id, "pan", pan)

                # --- Update Preview Button ---
                if st.button("⚙️ Update Preview", key=f"update_preview_{track_id}", help="Generate the 60s preview waveform/audio with the current settings."):
                    logger.info(f"Update Preview clicked for: '{track_data.get('name', 'N/A')}' ({track_id})")
                    if original_audio is not None and original_audio.size > 0:
                        with st.spinner("Generating preview..."):
                            preview_audio = get_preview_audio(track_data, preview_duration_s=PREVIEW_DURATION_S)
                            if preview_audio is not None and preview_audio.size > 0:
                                new_preview_path = save_audio_to_temp(preview_audio, track_sr)
                                if new_preview_path:
                                    old_preview_path = track_data.get("preview_temp_file_path")
                                    self.app_state.update_track_param(track_id, "preview_temp_file_path", new_preview_path)
                                    self.app_state.increment_update_counter(track_id)
                                    if old_preview_path and old_preview_path != new_preview_path and os.path.exists(old_preview_path):
                                        try:
                                            os.remove(old_preview_path)
                                            logger.info(f"Deleted old preview file: {old_preview_path}")
                                        except OSError as e:
                                            logger.warning(f"Could not delete old preview file {old_preview_path}: {e}")
                                    st.toast("Preview updated.", icon="✅")
                                else:
                                    st.error("Failed to save new preview file.")
                            else:
                                st.error("Failed to generate preview audio.")
                        st.rerun()
                    else:
                        st.warning("No original audio data to generate preview from.")
            except Exception as e:
                logger.exception(f"Error rendering main col for {track_id}")
                st.error(f"Error waveform/effects: {e}")

    def _render_track_controls_col(self, track_id: TrackID, track_data: TrackData, column: st.delta_generator.DeltaGenerator) -> bool:
        """Renders the controls (name, type, mute, solo, delete)."""
        delete_clicked = False
        with column:
            try:
                st.markdown("**Track Details**")
                name = st.text_input("Name", value=track_data.get("name", "Unnamed"), key=f"name_{track_id}", help="Rename track.")
                if name != track_data.get("name"):
                    self.app_state.update_track_param(track_id, "name", name)

                # --- Track Type Selector ---
                current_type = track_data.get("track_type", TRACK_TYPE_OTHER)
                try:
                    current_index = TRACK_TYPES.index(current_type)
                except ValueError:
                    current_index = TRACK_TYPES.index(TRACK_TYPE_OTHER)  # Default if invalid
                new_type = st.selectbox("Type", TRACK_TYPES, index=current_index, key=f"type_{track_id}", help="Categorize this layer (affects icon).")
                if new_type != current_type:
                    self.app_state.update_track_param(track_id, "track_type", new_type)
                    st.rerun()  # Rerun to update icon immediately
                # --------------------------

                st.caption("Mixing (Live Effect)")
                ms_col1, ms_col2 = st.columns(2)
                mute = ms_col1.checkbox("Mute", value=track_data.get("mute", False), key=f"mute_{track_id}", help="Silence track in final mix.")
                if mute != track_data.get("mute"):
                    self.app_state.update_track_param(track_id, "mute", mute)
                solo = ms_col2.checkbox("Solo", value=track_data.get("solo", False), key=f"solo_{track_id}", help="Isolate track(s) in final mix.")
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
        """Renders the master preview, export, and fade controls."""
        st.divider()
        st.header("🔊 Master Output")
        # --- Fade Controls ---
        st.markdown("**Master Fades (Applied on Export)**")
        fade_cols = st.columns(2)
        fade_in_duration = fade_cols[0].slider(
            "Fade In (s)", 0.0, 10.0, st.session_state.get("master_fade_in", 0.0), 0.5, key="master_fade_in", help="Duration of fade-in at the start of the exported file."
        )
        fade_out_duration = fade_cols[1].slider(
            "Fade Out (s)", 0.0, 10.0, st.session_state.get("master_fade_out", 0.0), 0.5, key="master_fade_out", help="Duration of fade-out at the end of the exported file."
        )
        # REMOVED redundant session state writes
        st.markdown("---")
        # --------------------

        # --- Export Filename ---
        default_filename = "mindmorph_mix"
        export_filename_input = st.text_input(
            "Export Filename (no extension):",
            value=st.session_state.get("export_filename", default_filename),
            key="export_filename",
            help="Enter the desired name for the downloaded file.",
        )
        # Basic sanitization: remove invalid characters for filenames
        sanitized_filename = re.sub(r'[\\/*?:"<>|]', "", export_filename_input).strip()
        if not sanitized_filename:
            sanitized_filename = default_filename  # Use default if empty after sanitizing
        # Store sanitized name in a DIFFERENT state key if needed, or just use it directly in download
        # We don't need to write back to st.session_state.export_filename here.
        # -----------------------

        # --- Calculated Duration Display ---
        if "calculated_mix_duration_s" in st.session_state and st.session_state.calculated_mix_duration_s is not None:
            duration_str = f"{st.session_state.calculated_mix_duration_s:.2f}"
            st.info(f"Estimated Full Mix Duration: **{duration_str} seconds**")
        st.caption("Note: Export time depends on total duration and effects used.")
        # -----------------------------------

        # --- Preview/Export Buttons ---
        master_cols = st.columns(2)
        with master_cols[0]:
            st.button(
                "🎧 Preview Mix (10s)",
                key="preview_mix",
                use_container_width=True,
                help="Generate and play first 10s of the final mix with all effects applied.",
                on_click=self._handle_preview_click,
            )
        with master_cols[1]:
            # --- Export Format Selection ---
            export_format = st.radio(
                "Export Format:",
                ["WAV", "MP3"],
                key="export_format",
                horizontal=True,
                help="Choose WAV (lossless, large) or MP3 (compressed, smaller)." if PYDUB_AVAILABLE else "WAV format only (MP3 requires pydub/ffmpeg).",
            )
            # REMOVED redundant session state write: st.session_state.export_format = export_format
            # -----------------------------
            export_disabled = export_format == "MP3" and not PYDUB_AVAILABLE
            export_button_label = f"💾 Export Full Mix (.{export_format.lower()})"
            export_help = f"Generate the complete final mix as .{export_format.lower()}."
            if export_disabled:
                export_help += " MP3 export disabled (pydub/ffmpeg missing)."

            st.button(export_button_label, key="export_mix", use_container_width=True, help=export_help, on_click=self._handle_export_click, disabled=export_disabled)

            # --- Download Button ---
            if "export_buffer" in st.session_state and st.session_state.export_buffer:
                file_ext = st.session_state.get("export_file_ext", "wav")
                # Use the sanitized filename from session state
                download_filename = f"{st.session_state.get('export_filename_final', 'mindmorph_mix')}.{file_ext}"  # Read sanitized name
                mime_type = f"audio/{file_ext}"
                st.download_button(
                    label=f"⬇️ Download: {download_filename}",
                    data=st.session_state.export_buffer,
                    file_name=download_filename,
                    mime=mime_type,
                    key="download_export_key",
                    use_container_width=True,
                )
                # Clear buffer after showing button
                st.session_state.export_buffer = None
                st.session_state.export_file_ext = None
                st.session_state.calculated_mix_duration_s = None  # Clear duration after download
                st.session_state.export_filename_final = None  # Clear final filename
            # ----------------------

    def _handle_preview_click(self):
        """Callback for Preview Mix button."""
        logger.info("Preview Mix button clicked.")
        tracks = self.app_state.get_all_tracks()
        if "preview_audio" in st.session_state:
            del st.session_state.preview_audio
        # Clear calculated duration when generating preview
        if "calculated_mix_duration_s" in st.session_state:
            del st.session_state.calculated_mix_duration_s
        if not tracks:
            st.warning("No tracks loaded.")
            return
        with st.spinner("Generating preview mix..."):
            mix_preview, _ = mix_tracks(tracks, preview=True, fade_in_s=0.0, fade_out_s=0.0)
            if mix_preview is not None and mix_preview.size > 0:
                st.session_state.preview_audio = save_audio(mix_preview, GLOBAL_SR)
                logger.info("Preview generated.")
            else:
                logger.warning("Preview mix empty.")

    def _handle_export_click(self):
        """Callback for Export Mix button. Calculates duration first."""
        logger.info("Export Full Mix button clicked.")
        tracks = self.app_state.get_all_tracks()
        export_format = st.session_state.get("export_format", "WAV").lower()
        fade_in = st.session_state.get("master_fade_in", 0.0)
        fade_out = st.session_state.get("master_fade_out", 0.0)
        # --- Get sanitized filename from state ---
        export_filename_base = st.session_state.get("export_filename", "mindmorph_mix")
        st.session_state.export_filename_final = export_filename_base  # Store for download button
        # -----------------------------------------

        if "export_buffer" in st.session_state:
            del st.session_state.export_buffer
        if "export_file_ext" in st.session_state:
            del st.session_state.export_file_ext
        if "calculated_mix_duration_s" in st.session_state:
            del st.session_state.calculated_mix_duration_s  # Clear old duration
        if not tracks:
            st.warning("No tracks loaded.")
            return

        # --- Calculate Duration First ---
        calculated_max_len = 0
        processed_lengths = {}
        valid_ids_for_len_calc = []
        solo_active = any(t.get("solo", False) for t in tracks.values())
        with st.spinner("Calculating final mix duration..."):
            for track_id, t_data in tracks.items():
                is_active = t_data.get("solo", False) if solo_active else not t_data.get("mute", False)
                original_audio = t_data.get("original_audio")
                if is_active and original_audio is not None and original_audio.size > 0:
                    full_processed = apply_all_effects(t_data)
                    processed_lengths[track_id] = len(full_processed)
                    valid_ids_for_len_calc.append(track_id)
            if valid_ids_for_len_calc:
                calculated_max_len = max(processed_lengths.values())
            st.session_state.calculated_mix_duration_s = calculated_max_len / GLOBAL_SR if GLOBAL_SR > 0 else 0
            logger.info(f"Calculated mix duration (pre-looping): {st.session_state.calculated_mix_duration_s:.2f}s")
        # -----------------------------

        # Now generate the actual mix
        with st.spinner(f"Generating full mix ({export_format.upper()})... This may take time."):
            full_mix, final_mix_len_samples = mix_tracks(tracks, preview=False, fade_in_s=fade_in, fade_out_s=fade_out)
            if final_mix_len_samples is not None:
                st.session_state.calculated_mix_duration_s = final_mix_len_samples / GLOBAL_SR if GLOBAL_SR > 0 else 0
                logger.info(f"Actual final mix duration (post-looping): {st.session_state.calculated_mix_duration_s:.2f}s")

            if full_mix is not None and full_mix.size > 0:
                if export_format == "wav":
                    st.session_state.export_buffer = save_audio(full_mix, GLOBAL_SR)
                    st.session_state.export_file_ext = "wav"
                    logger.info("Full WAV mix generated.")
                elif export_format == "mp3" and PYDUB_AVAILABLE:
                    try:
                        logger.info("Converting full mix to MP3...")
                        audio_int16 = (full_mix * 32767).astype(np.int16)
                        segment = AudioSegment(audio_int16.tobytes(), frame_rate=GLOBAL_SR, sample_width=audio_int16.dtype.itemsize, channels=2)
                        mp3_buffer = BytesIO()
                        segment.export(mp3_buffer, format="mp3", bitrate="192k")
                        mp3_buffer.seek(0)
                        st.session_state.export_buffer = mp3_buffer
                        st.session_state.export_file_ext = "mp3"
                        logger.info("Full MP3 mix generated.")
                    except Exception as e_mp3:
                        logger.exception("Failed to export as MP3.")
                        st.error(f"MP3 Export Failed: {e_mp3}. Check if ffmpeg is installed and accessible.")
                        st.session_state.export_buffer = None
                        st.session_state.export_file_ext = None
                else:
                    logger.error(f"Unsupported export format requested or pydub missing: {export_format}")
                    st.error(f"Cannot export as {export_format.upper()}.")
                    st.session_state.export_buffer = None
                    st.session_state.export_file_ext = None
            else:
                logger.warning("Full mix empty.")
                st.warning("Generated mix is empty.")
                st.session_state.export_buffer = None
                st.session_state.export_file_ext = None

    def render_preview_audio_player(self):
        """Displays the preview audio player if preview data exists in state."""
        # (Implementation remains the same)
        if "preview_audio" in st.session_state and st.session_state.preview_audio:
            st.markdown("**Mix Preview (10s):**")  # Clarified label
            st.audio(st.session_state.preview_audio, format="audio/wav")
            st.session_state.preview_audio = None

    def render_instructions(self):
        """Renders the instructions expander, expanded if no tracks exist."""
        # --- Updated Instructions for Button-Driven Preview ---
        tracks_exist = bool(self.app_state.get_all_tracks())
        st.divider()
        with st.expander("📖 Show Instructions & Notes", expanded=not tracks_exist):
            st.markdown("""
              **Welcome to MindMorph!** Create custom subliminal audio by layering affirmations, sounds, & frequencies.

              **Workflow:**

              1.  **➕ Add Tracks (Sidebar):**
                  * Use the options on the left to add your audio layers. The *full* audio is loaded/generated.

              2.  **🎚️ Edit Tracks (Main Panel):**
                  * Each track appears below. Set its **Type** (Affirmation, Background, etc.) using the dropdown.
                  * **Track Settings (Reverse, Speed, Pitch, Filter, Loop, Volume, Pan):** Adjust these settings.
                  * **Click `⚙️ Update Preview`** to generate a 60-second preview waveform/audio incorporating **all** current settings for that track. *(Note: Looping effect is only calculated during final mix/export, not shown in preview).*
                  * **Track Controls (Mute, Solo):** These affect the final mix directly without needing 'Update Preview'.
                  * `Track Name`: Rename tracks.
                  * `🗑️ Delete Track`: Remove unwanted tracks.

              3.  **🔊 Mix & Export (Bottom Panel):**
                  * Set optional **Master Fade In/Out** durations (applied only to the final export).
                  * Enter a **Filename** for your download.
                  * `🎧 Preview Mix (10s)`: Hear the start of the combined audio. This processes the *full duration* of all tracks with *all* saved settings (fades not applied here).
                  * `💾 Export Full Mix`: Choose WAV or MP3 (if available). This generates the final file using the *full* audio tracks, all settings, and master fades. Click the download button.

              **Tips:**
              * The editor preview is limited to 60s for performance, but the final export uses the full audio.
              * Use low volume for subliminal affirmation tracks (adjust and click 'Update Preview' to hear).
              * High speed (e.g., 2x-4x) is common for affirmation tracks.
              * Use the `🔁 Loop` option for short sounds; it will be applied during Mix/Export.
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
                st.markdown("Adjust **settings** in the main panel.")
                st.caption("Click 'Update Preview' to refresh!")  # Updated caption
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
        app_state = AppState()  # Uses new state structure
        tts_generator = TTSGenerator()  # Uses chunking version now
        ui_manager = UIManager(app_state, tts_generator)
        # benchmarker = Benchmarker(tts_generator) # Optional
        ui_manager.render_sidebar()
        ui_manager.render_tracks_editor()  # Handles empty state & preview rendering
        ui_manager.render_master_controls()
        ui_manager.render_preview_audio_player()  # Display preview mix if available
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
        # --- Changed to log error instead of showing in UI ---
        logger.error(f"Unhandled exception in main: {e}")
        st.error("An unexpected error occurred. Please check the application logs or try reloading.")
        # st.code(traceback.format_exc()) # Avoid showing full traceback in UI
        # ----------------------------------------------------
    # Attempt to stop listener - might not execute reliably
    # logger.info("Attempting to stop logging listener.")
    # if 'listener' in locals() and listener is not None and listener.is_alive(): listener.stop()
