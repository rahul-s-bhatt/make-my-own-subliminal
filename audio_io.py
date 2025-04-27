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
import streamlit as st  # TODO: Remove Streamlit UI calls from this module
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


def load_audio(file_source: UploadedFile | BytesIO | str, target_sr: SampleRate = GLOBAL_SR) -> tuple[AudioData, SampleRate]:
    """
    Loads audio from various sources, ensures stereo, and resamples to target SR.

    Args:
        file_source: An UploadedFile object, BytesIO buffer, or file path string.
        target_sr: The desired sample rate for the output audio.

    Returns:
        A tuple containing:
            - The loaded audio data as a NumPy array (float32, stereo).
            - The sample rate of the loaded audio (will be target_sr).
        Returns empty array and target_sr on failure.
    """
    logger.info(f"Loading audio from source type: {type(file_source)}")
    try:
        # Load audio using librosa
        # sr=None preserves original sample rate, mono=False loads all channels
        audio, sr = librosa.load(file_source, sr=None, mono=False)
        logger.debug(f"Loaded audio original SR: {sr}, shape: {audio.shape}")

        # --- Ensure Stereo Format ---
        # Librosa loads channels first (channels, samples)
        # We want (samples, channels)
        if audio.ndim == 1:
            # Convert mono to stereo by duplicating the channel
            logger.warning("Mono audio detected. Duplicating channel to create stereo.")
            audio = np.stack([audio, audio], axis=-1)  # Now (samples, 2)
        elif audio.shape[0] == 2 and audio.shape[1] > 2:
            # If shape is (2, samples), transpose to (samples, 2)
            audio = audio.T
        elif audio.shape[1] > 2:
            # If more than 2 channels, take only the first two
            logger.warning(f"Audio has more than 2 channels ({audio.shape[1]}). Using only the first two.")
            audio = audio[:, :2]
        elif audio.shape[1] == 1:
            # If shape is (samples, 1), duplicate channel
            logger.warning("Audio has 1 channel dimension. Duplicating.")
            audio = np.concatenate([audio, audio], axis=1)  # Now (samples, 2)

        # --- Resample if Necessary ---
        if sr != target_sr:
            logger.info(f"Resampling audio from {sr} Hz to {target_sr} Hz.")
            if audio.size > 0:
                # Ensure audio is float before resampling
                # Librosa expects (channels, samples) for resample, so transpose
                audio_float = audio.astype(np.float32)
                audio_resampled = librosa.resample(audio_float.T, orig_sr=sr, target_sr=target_sr)
                # Transpose back to (samples, channels)
                audio = audio_resampled.T
            else:
                # If audio is empty, just update the sample rate info
                # sr = target_sr # No need to update sr, it's the original rate
                logger.warning("Audio data is empty, cannot resample. Original SR was {sr}Hz.")
                # Return empty array but keep target_sr as the intended rate
                return np.zeros((0, 2), dtype=np.float32), target_sr

        # Ensure final output is float32
        return audio.astype(np.float32), target_sr

    except Exception as e:
        logger.exception(f"Error loading audio from source: {type(file_source)}")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Error loading audio file: {e}")
        # Return an empty stereo array and the target sample rate on error
        return np.zeros((0, 2), dtype=np.float32), target_sr


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
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Error saving audio: {e}")
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
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Failed to save temporary audio file: {e}")
        # Clean up if file was partially created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Cleaned up partially created temp file: {temp_file_path}")
            except OSError as e_os:
                logger.warning(f"Failed to clean up partial temp file {temp_file_path}: {e_os}")
        return None
