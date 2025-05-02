# audio_io.py
# ==========================================
# Audio File Input/Output Utilities for MindMorph (with Caching)
# ==========================================

import logging
import os
import tempfile
from io import BytesIO
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import streamlit as st  # Import Streamlit for caching decorators

# Import UploadedFile for type hinting without causing circular imports if needed elsewhere
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Import constants from config
from config import GLOBAL_SR

# Define type hints used within this module
AudioData = np.ndarray
SampleRate = int

# Get a logger for this module
logger = logging.getLogger(__name__)

# ==========================================
# File I/O Functions
# ==========================================


# --- Cache the audio loading function ---
# Caching based on the file source (path or content hash), target SR, and duration.
# Note: Streamlit's caching mechanism handles hashing UploadedFile objects based on content.
@st.cache_data(show_spinner="Loading audio file...")
def load_audio(
    file_source: UploadedFile | BytesIO | str,
    target_sr: Optional[SampleRate] = GLOBAL_SR,
    duration: Optional[float] = None,
) -> tuple[Optional[AudioData], Optional[SampleRate]]:
    """
    Loads audio from various sources, ensures stereo, resamples, and optionally limits duration.
    Results are cached based on file source content/path, target SR, and duration.

    Args:
        file_source: An UploadedFile object, BytesIO buffer, or file path string.
        target_sr: The desired sample rate for the output audio. If None, original SR is kept.
        duration: Optional maximum duration in seconds to load from the beginning.
                  If None, loads the entire file.

    Returns:
        A tuple containing:
            - The loaded audio data as a NumPy array (float32, stereo), or None on failure.
            - The sample rate of the loaded audio (target_sr if resampling occurred,
              original SR otherwise), or None on failure.
    """
    # Determine a log-friendly identifier for the source
    if isinstance(file_source, str):
        source_id = file_source
    elif hasattr(file_source, "name"):  # Check if it's likely an UploadedFile
        source_id = f"UploadedFile(name='{file_source.name}', size={getattr(file_source, 'size', 'N/A')})"
    elif isinstance(file_source, BytesIO):
        source_id = f"BytesIO(size={file_source.getbuffer().nbytes})"
    else:
        source_id = f"UnknownSourceType({type(file_source)})"

    logger.info(
        f"Loading audio from: {source_id}, Target SR: {target_sr}, Duration: {duration}s"
    )

    try:
        # Load audio using librosa
        audio, sr = librosa.load(file_source, sr=None, mono=False, duration=duration)
        logger.debug(f"Loaded audio original SR: {sr}, shape: {audio.shape}")

        # --- Ensure Stereo Format ---
        if audio.ndim == 1:
            logger.info("Mono audio detected. Duplicating channel to create stereo.")
            audio = np.stack([audio, audio], axis=-1)
        elif audio.shape[0] == 2 and audio.shape[1] > 2:
            # If shape is (2, samples), transpose to (samples, 2)
            audio = audio.T
        elif audio.ndim > 1 and audio.shape[0] > 2:
            logger.warning(
                f"Audio has more than 2 channels ({audio.shape[0]}). Using only the first two."
            )
            audio = audio[:2, :].T
        elif audio.ndim > 1 and audio.shape[1] > 2:
            logger.warning(
                f"Audio has more than 2 channels ({audio.shape[1]}). Using only the first two."
            )
            audio = audio[:, :2]
        elif audio.ndim > 1 and audio.shape[1] == 1:
            logger.info("Audio has 1 channel dimension. Duplicating.")
            audio = np.concatenate([audio, audio], axis=1)
        elif audio.ndim == 2 and audio.shape[1] == 2:
            logger.debug("Audio is already in desired stereo format (samples, 2).")
        else:
            logger.warning(
                f"Unexpected audio shape {audio.shape}. Attempting to proceed."
            )

        # --- Resample if Necessary ---
        output_sr = sr
        if target_sr is not None and sr != target_sr:
            logger.info(f"Resampling audio from {sr} Hz to {target_sr} Hz.")
            if audio.size > 0:
                audio_float = audio.astype(np.float32)
                if audio_float.ndim == 1:
                    audio_float = np.stack([audio_float, audio_float], axis=-1)

                if audio_float.shape[1] != 2:
                    logger.error(
                        f"Cannot resample, unexpected audio shape after stereo conversion: {audio_float.shape}"
                    )
                    return None, None

                # Librosa expects (channels, samples) for resample, so transpose
                audio_resampled = librosa.resample(
                    audio_float.T, orig_sr=sr, target_sr=target_sr
                )
                # Transpose back to (samples, channels)
                audio = audio_resampled.T
                output_sr = target_sr
            else:
                logger.warning(
                    f"Audio data is empty, cannot resample. Original SR was {sr}Hz."
                )
                output_sr = target_sr  # Still return target SR if requested

        # Ensure final output is float32
        return audio.astype(np.float32), output_sr

    except Exception as e:
        logger.exception(f"Error loading audio from source: {source_id}")
        # Return None, None on error to match signature
        return None, None


def save_audio_to_bytesio(audio: AudioData, sr: SampleRate) -> BytesIO:
    """
    Saves audio data to an in-memory BytesIO buffer as WAV (PCM16).
    (Not cached - this performs an action, not a computation).

    Args:
        audio: The audio data (NumPy array, float32, stereo).
        sr: The sample rate of the audio.

    Returns:
        A BytesIO object containing the WAV audio data. Returns empty buffer on failure.
    """
    buffer = BytesIO()
    logger.debug(
        f"Attempting to save audio (dtype: {audio.dtype}, shape: {audio.shape}, sr: {sr}Hz) to BytesIO buffer."
    )

    if audio is None or audio.size == 0:
        logger.warning("Attempted to save empty audio data to BytesIO.")
        return buffer

    try:
        # --- Robustness Checks ---
        if not isinstance(audio, np.ndarray):
            logger.error(
                f"Input audio is not a numpy array, but {type(audio)}. Cannot save."
            )
            return buffer
        if not np.issubdtype(audio.dtype, np.floating):
            logger.warning(
                f"Input audio dtype is {audio.dtype}, not float. Attempting conversion to float32."
            )
            try:
                audio = audio.astype(np.float32)
            except Exception as e_conv:
                logger.error(f"Failed to convert audio to float32: {e_conv}")
                return buffer
        if audio.ndim == 1:
            logger.warning(
                "Input audio is mono (1D). Converting to stereo for WAV save."
            )
            audio = np.stack([audio, audio], axis=-1)
        elif audio.ndim != 2 or audio.shape[1] != 2:
            logger.error(
                f"Input audio has unexpected shape {audio.shape}. Expected (samples, 2). Cannot save."
            )
            return buffer
        # --- End Checks ---

        logger.debug("Clipping audio data to [-1.0, 1.0].")
        audio_clipped = np.clip(audio, -1.0, 1.0)

        logger.debug("Converting audio data to int16.")
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        logger.debug(
            f"Audio data prepared for writing (dtype: {audio_int16.dtype}, shape: {audio_int16.shape})."
        )

        logger.debug("Writing audio data to buffer using soundfile...")
        sf.write(buffer, audio_int16, sr, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        logger.info("Audio successfully saved to BytesIO buffer.")

    except Exception as e:
        logger.exception(
            f"Error saving audio to BytesIO buffer during sf.write or conversion: {e}"
        )
        buffer = BytesIO()  # Return empty buffer on error

    return buffer


def save_audio_to_temp_file(audio: AudioData, sr: SampleRate) -> str | None:
    """
    Saves audio data to a temporary WAV file on disk (PCM16).
    (Not cached - this performs an action).

    Args:
        audio: The audio data (NumPy array, float32, stereo).
        sr: The sample rate of the audio.

    Returns:
        The file path string to the temporary WAV file, or None on failure.
        The caller is responsible for deleting the file later.
    """
    temp_file_path = None
    logger.debug(f"Attempting to save audio ({audio.shape}, {sr}Hz) to temporary file.")
    if audio is None or audio.size == 0:
        logger.warning("Attempted to save empty audio data to temporary file.")
        return None

    try:
        # --- Robustness checks ---
        if not isinstance(audio, np.ndarray):
            logger.error(
                f"Input audio is not a numpy array (type: {type(audio)}) for temp file save."
            )
            return None
        if not np.issubdtype(audio.dtype, np.floating):
            logger.warning(
                f"Input audio dtype {audio.dtype} is not float for temp file save. Converting."
            )
            try:
                audio = audio.astype(np.float32)
            except Exception as e_conv:
                logger.error(
                    f"Failed to convert audio to float32 for temp file save: {e_conv}"
                )
                return None
        if audio.ndim == 1:
            logger.warning(
                "Input audio is mono (1D) for temp file save. Converting to stereo."
            )
            audio = np.stack([audio, audio], axis=-1)
        elif audio.ndim != 2 or audio.shape[1] != 2:
            logger.error(
                f"Input audio has unexpected shape {audio.shape} for temp file save. Expected (samples, 2)."
            )
            return None
        # --- End checks ---

        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as tmp:
            sf.write(tmp, audio_int16, sr, format="WAV", subtype="PCM_16")
            temp_file_path = tmp.name
        logger.info(f"Audio saved successfully to temporary file: {temp_file_path}")
        return temp_file_path

    except Exception as e:
        logger.exception("Failed to save audio to temporary file.")
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(
                    f"Cleaned up partially created temp file: {temp_file_path}"
                )
            except OSError as e_os:
                logger.warning(
                    f"Failed to clean up partial temp file {temp_file_path}: {e_os}"
                )
        return None
