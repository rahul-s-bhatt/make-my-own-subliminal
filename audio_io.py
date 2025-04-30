# audio_io.py
# ==========================================
# Audio File Input/Output Utilities for MindMorph
# ==========================================

import logging
import os
import tempfile
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

# import streamlit as st # Avoid direct Streamlit UI calls in this module
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Import constants from config
from config import GLOBAL_SR

# Define type hints used within this module
AudioData = np.ndarray
SampleRate = int
# TrackData = Dict[str, Any] # Not directly used in this file

# Get a logger for this module
logger = logging.getLogger(__name__)

# ==========================================
# File I/O Functions
# ==========================================


# <<< MODIFIED: Added duration parameter and updated return type hint >>>
def load_audio(
    file_source: UploadedFile | BytesIO | str, target_sr: Optional[SampleRate] = GLOBAL_SR, duration: Optional[float] = None
) -> tuple[Optional[AudioData], Optional[SampleRate]]:
    """
    Loads audio from various sources, ensures stereo, resamples, and optionally limits duration.

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
    logger.info(f"Loading audio from source type: {type(file_source)}, Target SR: {target_sr}, Duration: {duration}s")
    try:
        # Load audio using librosa
        # sr=None preserves original sample rate, mono=False loads all channels
        # <<< MODIFIED: Pass duration to librosa.load >>>
        audio, sr = librosa.load(file_source, sr=None, mono=False, duration=duration)
        logger.debug(f"Loaded audio original SR: {sr}, shape: {audio.shape}")

        # --- Ensure Stereo Format ---
        # Librosa loads channels first (channels, samples)
        # We want (samples, channels)
        if audio.ndim == 1:
            # Convert mono to stereo by duplicating the channel
            logger.info("Mono audio detected. Duplicating channel to create stereo.")
            audio = np.stack([audio, audio], axis=-1)  # Now (samples, 2)
        elif audio.shape[0] == 2 and audio.shape[1] > 2:
            # If shape is (2, samples), transpose to (samples, 2)
            audio = audio.T
        elif audio.ndim > 1 and audio.shape[0] > 2:  # Check if first dimension is channels > 2
            logger.warning(f"Audio has more than 2 channels ({audio.shape[0]}). Using only the first two.")
            audio = audio[:2, :].T  # Take first 2 channels and transpose
        elif audio.ndim > 1 and audio.shape[1] > 2:  # Check if second dimension is channels > 2 (already transposed?)
            logger.warning(f"Audio has more than 2 channels ({audio.shape[1]}). Using only the first two.")
            audio = audio[:, :2]  # Take first 2 channels
        elif audio.ndim > 1 and audio.shape[1] == 1:
            # If shape is (samples, 1), duplicate channel
            logger.info("Audio has 1 channel dimension. Duplicating.")
            audio = np.concatenate([audio, audio], axis=1)  # Now (samples, 2)
        # Add check for already correct shape (samples, 2)
        elif audio.ndim == 2 and audio.shape[1] == 2:
            logger.debug("Audio is already in desired stereo format (samples, 2).")
        else:
            logger.warning(f"Unexpected audio shape {audio.shape}. Attempting to proceed, but might cause issues.")

        # --- Resample if Necessary and target_sr is specified ---
        output_sr = sr  # Start with original SR
        if target_sr is not None and sr != target_sr:
            logger.info(f"Resampling audio from {sr} Hz to {target_sr} Hz.")
            if audio.size > 0:
                # Ensure audio is float before resampling
                # Librosa expects (channels, samples) for resample, so transpose
                audio_float = audio.astype(np.float32)
                # Handle potential shape issues before transposing
                if audio_float.ndim == 1:  # Should have been converted to stereo already, but double-check
                    audio_float = np.stack([audio_float, audio_float], axis=-1)

                if audio_float.shape[1] != 2:  # If still not (samples, 2) after checks
                    logger.error(f"Cannot resample, unexpected audio shape after stereo conversion: {audio_float.shape}")
                    return None, None  # Indicate failure

                audio_resampled = librosa.resample(audio_float.T, orig_sr=sr, target_sr=target_sr)
                # Transpose back to (samples, channels)
                audio = audio_resampled.T
                output_sr = target_sr  # Update the output SR
            else:
                logger.warning("Audio data is empty, cannot resample. Original SR was {sr}Hz.")
                output_sr = target_sr  # Return target SR even if empty

        # Ensure final output is float32
        return audio.astype(np.float32), output_sr

    except Exception as e:
        logger.exception(f"Error loading audio from source: {type(file_source)}")
        # Removed direct Streamlit call. Log the error.
        # st.error(f"Error loading audio file: {e}")
        # Return None, None on error to match signature
        return None, None


def save_audio_to_bytesio(audio: AudioData, sr: SampleRate) -> BytesIO:
    """
    Saves audio data to an in-memory BytesIO buffer as WAV (PCM16).

    Args:
        audio: The audio data (NumPy array, float32, stereo).
        sr: The sample rate of the audio.

    Returns:
        A BytesIO object containing the WAV audio data. Returns empty buffer on failure.
    """
    buffer = BytesIO()
    logger.debug(f"Saving audio ({audio.shape}, {sr}Hz) to BytesIO buffer.")
    if audio is None or audio.size == 0:
        logger.warning("Attempted to save empty audio data to BytesIO.")
        return buffer  # Return empty buffer

    try:
        # Ensure audio is within [-1.0, 1.0] before converting to int16
        audio_clipped = np.clip(audio, -1.0, 1.0)
        # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        # Write to the buffer using soundfile
        sf.write(buffer, audio_int16, sr, format="WAV", subtype="PCM_16")
        buffer.seek(0)  # Rewind the buffer to the beginning for reading
        logger.debug("Audio successfully saved to BytesIO buffer.")

    except Exception as e:
        logger.exception("Error saving audio to BytesIO buffer.")
        # Removed direct Streamlit call. Log the error.
        # st.error(f"Error saving audio: {e}")
        # Return an empty buffer in case of error
        buffer = BytesIO()

    return buffer


def save_audio_to_temp_file(audio: AudioData, sr: SampleRate) -> str | None:
    """
    Saves audio data to a temporary WAV file on disk (PCM16).

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
        # Ensure audio is within [-1.0, 1.0] before converting to int16
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        # Create a named temporary file (delete=False means we manage deletion)
        # Use 'wb' mode for writing binary data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as tmp:
            sf.write(tmp, audio_int16, sr, format="WAV", subtype="PCM_16")
            temp_file_path = tmp.name
        logger.info(f"Audio saved successfully to temporary file: {temp_file_path}")
        return temp_file_path

    except Exception as e:
        logger.exception("Failed to save audio to temporary file.")
        # Removed direct Streamlit call. Log the error.
        # st.error(f"Failed to save temporary audio file: {e}")
        # Clean up if file was partially created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Cleaned up partially created temp file: {temp_file_path}")
            except OSError as e_os:
                logger.warning(f"Failed to clean up partial temp file {temp_file_path}: {e_os}")
        return None
