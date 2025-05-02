# audio_effects.py
# ==========================================
# Audio Effect Functions for MindMorph (with Caching)
# ==========================================

import logging

import librosa
import librosa.effects
import numpy as np
import streamlit as st  # Import Streamlit for caching decorators
from scipy import signal

# Import constants and types from config
from config import GLOBAL_SR  # Keep GLOBAL_SR for default fallback if needed elsewhere
from config import (
    ULTRASONIC_FILTER_CUTOFF,
    ULTRASONIC_FILTER_ORDER,
    ULTRASONIC_TARGET_FREQ,
)

# Define type hints used within this module
AudioData = np.ndarray
SampleRate = int

# Get a logger for this module
logger = logging.getLogger(__name__)

# ==========================================
# Individual Audio Effect Functions
# ==========================================


# Caching reverse might be minor, but added for consistency.
@st.cache_data(show_spinner=False)  # No spinner for quick effect
def apply_reverse(audio: AudioData) -> AudioData:
    """
    Reverses the audio data along the time axis.
    Cached based on the input audio data.
    """
    logger.debug("Applying reverse effect.")
    if audio is None or audio.size == 0:
        logger.warning("Attempted to reverse empty audio.")
        return audio  # Return unchanged if empty

    try:
        # Slicing [::-1] reverses the array along the first axis (time)
        return np.ascontiguousarray(audio[::-1]).astype(np.float32)
    except Exception as e:
        logger.exception("Error applying reverse effect.")
        # Return original on error instead of st.error
        return audio


@st.cache_data(show_spinner="Applying speed change...")
def apply_speed_change(
    audio: AudioData, sr: SampleRate, speed_factor: float
) -> AudioData:
    """
    Changes the speed of the audio using time stretching.
    Cached based on audio data, sample rate, and speed factor.
    """
    logger.debug(f"Applying speed change: factor={speed_factor:.2f}")

    if audio is None or audio.size == 0:
        logger.warning("Attempted to change speed of empty audio.")
        return audio

    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        logger.error(
            "NaN or Inf detected in audio data before time stretch. Returning original."
        )
        # Removed st.error
        return audio

    if speed_factor <= 0:
        logger.warning(
            f"Invalid speed factor: {speed_factor}. Must be > 0. Returning original."
        )
        # Removed st.warning
        return audio

    if np.isclose(speed_factor, 1.0):
        logger.debug("Speed factor is close to 1.0, skipping time stretch.")
        return audio

    try:
        logger.info(f"Applying time stretch (factor: {speed_factor:.2f})...")
        audio_float = audio.astype(np.float32)
        audio_contiguous_transposed = np.ascontiguousarray(audio_float.T)

        audio_stretched_transposed = librosa.effects.time_stretch(
            y=audio_contiguous_transposed, rate=speed_factor
        )
        return audio_stretched_transposed.T.astype(np.float32)

    except Exception as e:
        logger.exception(
            f"Error during librosa time_stretch with factor {speed_factor:.2f}."
        )
        # Return original on error instead of st.error
        return audio


@st.cache_data(show_spinner="Applying pitch shift...")
def apply_pitch_shift(audio: AudioData, sr: SampleRate, n_steps: float) -> AudioData:
    """
    Shifts the pitch of the audio without changing speed.
    Cached based on audio data, sample rate, and number of steps.
    """
    logger.debug(f"Applying pitch shift: {n_steps:.2f} semitones")

    if audio is None or audio.size == 0:
        logger.warning("Attempted to pitch shift empty audio.")
        return audio

    if np.isclose(n_steps, 0.0):
        logger.debug("Pitch shift is close to 0, skipping pitch shift effect.")
        return audio

    try:
        logger.info(f"Applying pitch shift ({n_steps:.2f} semitones)...")
        max_possible_steps = 12 * np.log2((sr / 2) / 440.0) if sr > 0 else 12
        if abs(n_steps) > max(12, max_possible_steps):
            logger.warning(
                f"Large pitch shift ({n_steps:.1f} steps) requested for SR={sr}Hz. Result quality may be poor."
            )

        audio_float = audio.astype(np.float32)
        audio_contiguous_transposed = np.ascontiguousarray(audio_float.T)

        audio_shifted_transposed = librosa.effects.pitch_shift(
            y=audio_contiguous_transposed, sr=sr, n_steps=n_steps
        )
        return audio_shifted_transposed.T.astype(np.float32)

    except Exception as e:
        logger.exception(f"Error applying pitch shift with {n_steps:.2f} steps.")
        # Return original on error instead of st.error
        return audio


@st.cache_data(show_spinner="Applying ultrasonic shift...")
def apply_ultrasonic_shift(
    audio: AudioData, sr: SampleRate, target_freq: float = ULTRASONIC_TARGET_FREQ
) -> AudioData:
    """
    Shifts audio towards a high target frequency and applies a steep low-pass filter.
    EXPERIMENTAL: Quality depends heavily on source audio and parameters.
    Cached based on audio data, sample rate, and target frequency.
    """
    logger.info(f"Applying ultrasonic shift towards {target_freq} Hz (SR={sr} Hz)")

    if audio is None or audio.size == 0:
        logger.warning("Attempted to apply ultrasonic shift to empty audio.")
        return audio

    try:
        audio_float = audio.astype(np.float32)

        # --- Step 1: Estimate Fundamental and Calculate Pitch Shift ---
        try:
            # Use first channel for pitch track
            pitches, magnitudes = librosa.piptrack(y=audio_float[:, 0], sr=sr)
            median_mag = np.median(magnitudes)
            if median_mag > 1e-6:
                valid_pitches = pitches[magnitudes > median_mag * 0.5]
            else:
                valid_pitches = pitches[magnitudes > 1e-6]
            valid_pitches = valid_pitches[valid_pitches > 0]
        except Exception as e_piptrack:
            logger.warning(
                f"Piptrack failed during ultrasonic shift: {e_piptrack}. Using default fundamental."
            )
            valid_pitches = []

        if len(valid_pitches) > 0:
            fundamental_freq = np.median(valid_pitches)
            logger.debug(f"Estimated fundamental frequency: {fundamental_freq:.2f} Hz")
        else:
            fundamental_freq = 440.0
            logger.warning(
                f"Could not estimate fundamental frequency, using default {fundamental_freq} Hz."
            )

        if fundamental_freq <= 0 or np.isnan(fundamental_freq):
            logger.error(
                f"Invalid fundamental frequency ({fundamental_freq}) estimated. Cannot calculate shift. Returning original."
            )
            # Removed st.error
            return audio

        n_steps = 12 * np.log2(target_freq / fundamental_freq)
        logger.info(
            f"Calculated steps for ultrasonic shift: {n_steps:.2f} semitones (Target: {target_freq} Hz, Estimated Fundamental: {fundamental_freq:.1f} Hz)"
        )
        if abs(n_steps) > 36:
            logger.warning(
                f"Applying very large pitch shift ({n_steps:.1f} steps) for ultrasonic effect."
            )

        # --- Step 2: Apply Pitch Shift ---
        audio_contiguous_transposed = np.ascontiguousarray(audio_float.T)
        audio_shifted_transposed = librosa.effects.pitch_shift(
            y=audio_contiguous_transposed, sr=sr, n_steps=n_steps
        )
        audio_shifted = audio_shifted_transposed.T.astype(np.float32)
        logger.debug(
            f"Pitch shift complete. Shifted audio shape: {audio_shifted.shape}"
        )

        # --- Step 3: Apply Steep Low-Pass Filter ---
        filter_cutoff = ULTRASONIC_FILTER_CUTOFF
        filter_order = ULTRASONIC_FILTER_ORDER
        nyquist = 0.5 * sr
        normalized_cutoff = filter_cutoff / nyquist

        if normalized_cutoff >= 1.0 or normalized_cutoff <= 0:
            logger.warning(
                f"Ultrasonic filter cutoff ({filter_cutoff}Hz) invalid for Nyquist ({nyquist}Hz). Skipping filter."
            )
            return audio_shifted  # Return shifted audio without filter

        logger.info(
            f"Applying {filter_order}-order low-pass Butterworth filter at {filter_cutoff} Hz."
        )
        b, a = signal.butter(filter_order, normalized_cutoff, btype="lowpass")

        min_len_filtfilt = 3 * max(len(a), len(b))
        if audio_shifted.shape[0] <= min_len_filtfilt:
            logger.warning(
                f"Shifted audio too short ({audio_shifted.shape[0]}) for filter length ({min_len_filtfilt}). Skipping filter."
            )
            return audio_shifted  # Return shifted audio without filter

        audio_filtered = signal.filtfilt(b, a, audio_shifted, axis=0)
        logger.debug(
            f"Filtering complete. Filtered audio shape: {audio_filtered.shape}"
        )

        return audio_filtered.astype(np.float32)

    except Exception as e:
        logger.exception("Error applying ultrasonic shift with filtering.")
        # Return original on error instead of st.error
        return audio


@st.cache_data(show_spinner="Applying filter...")
def apply_standard_filter(
    audio: AudioData, sr: SampleRate, filter_type: str, cutoff: float
) -> AudioData:
    """
    Applies a standard Butterworth lowpass or highpass filter.
    Cached based on audio data, sample rate, filter type, and cutoff frequency.
    """
    logger.debug(f"Applying standard filter: type={filter_type}, cutoff={cutoff} Hz")

    if filter_type == "off" or cutoff <= 0:
        return audio

    if audio is None or audio.size == 0:
        logger.warning("Attempted to apply filter to empty audio.")
        return audio

    try:
        audio_float = audio.astype(np.float32)
        nyquist = 0.5 * sr
        normalized_cutoff = cutoff / nyquist

        if normalized_cutoff >= 1.0 or normalized_cutoff <= 0:
            msg = f"Filter cutoff ({cutoff}Hz) is invalid for Nyquist ({nyquist}Hz). Skipping filter."
            logger.warning(msg)
            # Removed st.warning
            return audio

        filter_order = 4
        logger.info(
            f"Applying {filter_order}-order Butterworth {filter_type} filter at {cutoff} Hz."
        )

        if filter_type == "lowpass":
            b, a = signal.butter(filter_order, normalized_cutoff, btype="low")
        elif filter_type == "highpass":
            b, a = signal.butter(filter_order, normalized_cutoff, btype="high")
        else:
            logger.warning(
                f"Unknown filter type specified: {filter_type}. Skipping filter."
            )
            # Removed st.warning
            return audio

        min_len_filtfilt = 3 * max(len(a), len(b))
        if audio_float.shape[0] <= min_len_filtfilt:
            msg = f"Track too short ({audio_float.shape[0]}) for filter length ({min_len_filtfilt}). Skipping filter."
            logger.warning(msg)
            # Removed st.warning
            return audio

        audio_filtered = signal.filtfilt(b, a, audio_float, axis=0)
        return audio_filtered.astype(np.float32)

    except Exception as e:
        logger.exception(
            f"Error applying standard {filter_type} filter at {cutoff} Hz."
        )
        # Return original on error instead of st.error
        return audio
