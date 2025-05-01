# audio_effects.py
# ==========================================
# Audio Effect Functions for MindMorph
# ==========================================

import logging

import librosa
import librosa.effects
import numpy as np
import streamlit as st  # TODO: Remove Streamlit UI calls from this module
from scipy import signal

# Import constants and types from config
from config import (  # Potentially needed for defaults or checks? Added just in case.
    GLOBAL_SR,
    ULTRASONIC_FILTER_CUTOFF,
    ULTRASONIC_FILTER_ORDER,
    ULTRASONIC_TARGET_FREQ,
)

# Define type hints used within this module
AudioData = np.ndarray
SampleRate = int
# TrackData = Dict[str, Any] # Not directly used in this file for individual effects

# Get a logger for this module
logger = logging.getLogger(__name__)

# ==========================================
# Individual Audio Effect Functions
# ==========================================


def apply_reverse(audio: AudioData) -> AudioData:
    """Reverses the audio data along the time axis."""
    logger.debug("Applying reverse effect.")
    if audio is None or audio.size == 0:
        logger.warning("Attempted to reverse empty audio.")
        return audio  # Return unchanged if empty

    try:
        # Slicing [::-1] reverses the array along the first axis (time)
        # Ensure the reversed array is contiguous for potential downstream processing
        return np.ascontiguousarray(audio[::-1]).astype(np.float32)
    except Exception as e:
        logger.exception("Error applying reverse effect.")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Reverse audio effect failed: {e}")
        return audio  # Return original on error


def apply_speed_change(
    audio: AudioData, sr: SampleRate, speed_factor: float
) -> AudioData:
    """Changes the speed of the audio using time stretching."""
    logger.debug(f"Applying speed change: factor={speed_factor:.2f}")

    if audio is None or audio.size == 0:
        logger.warning("Attempted to change speed of empty audio.")
        return audio

    # Check for invalid input data or parameters
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        logger.error("NaN or Inf detected in audio data before time stretch.")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error("Invalid audio data (NaN/Inf) detected before speed change.")
        return audio

    if speed_factor <= 0:
        logger.warning(f"Invalid speed factor: {speed_factor}. Must be > 0.")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.warning("Speed factor must be greater than 0.")
        return audio

    # If speed factor is effectively 1.0, no change needed
    if np.isclose(speed_factor, 1.0):
        logger.debug("Speed factor is close to 1.0, skipping time stretch.")
        return audio

    try:
        logger.info(f"Applying time stretch (factor: {speed_factor:.2f})...")
        # librosa expects (channels, samples) and contiguous array
        # Ensure input is float32 before processing
        audio_float = audio.astype(np.float32)
        audio_contiguous_transposed = np.ascontiguousarray(audio_float.T)

        # Apply time stretching
        audio_stretched_transposed = librosa.effects.time_stretch(
            y=audio_contiguous_transposed, rate=speed_factor
        )
        # Transpose back to (samples, channels)
        return audio_stretched_transposed.T.astype(np.float32)

    except Exception as e:
        logger.exception(
            f"Error during librosa time_stretch with factor {speed_factor:.2f}."
        )
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Speed change effect failed: {e}")
        return audio  # Return original on error


def apply_pitch_shift(audio: AudioData, sr: SampleRate, n_steps: float) -> AudioData:
    """Shifts the pitch of the audio without changing speed."""
    logger.debug(f"Applying pitch shift: {n_steps:.2f} semitones")

    if audio is None or audio.size == 0:
        logger.warning("Attempted to pitch shift empty audio.")
        return audio

    # If pitch shift is effectively 0, no change needed
    if np.isclose(n_steps, 0.0):
        logger.debug("Pitch shift is close to 0, skipping pitch shift effect.")
        return audio

    try:
        logger.info(f"Applying pitch shift ({n_steps:.2f} semitones)...")
        # Check for potentially problematic large shifts relative to SR
        max_possible_steps = 12 * np.log2((sr / 2) / 440.0)  # Rough estimate
        if abs(n_steps) > max(12, max_possible_steps):
            logger.warning(
                f"Large pitch shift ({n_steps:.1f} steps) requested for SR={sr}Hz. Result quality may be poor or contain artifacts."
            )

        # librosa expects (channels, samples) and contiguous array
        # Ensure input is float32
        audio_float = audio.astype(np.float32)
        audio_contiguous_transposed = np.ascontiguousarray(audio_float.T)

        # Apply pitch shifting
        audio_shifted_transposed = librosa.effects.pitch_shift(
            y=audio_contiguous_transposed, sr=sr, n_steps=n_steps
        )
        # Transpose back to (samples, channels)
        return audio_shifted_transposed.T.astype(np.float32)

    except Exception as e:
        logger.exception(f"Error applying pitch shift with {n_steps:.2f} steps.")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Pitch shift effect failed: {e}")
        return audio  # Return original on error


def apply_ultrasonic_shift(
    audio: AudioData, sr: SampleRate, target_freq: float = ULTRASONIC_TARGET_FREQ
) -> AudioData:
    """
    Shifts audio towards a high target frequency and applies a steep low-pass filter.
    EXPERIMENTAL: Quality depends heavily on source audio and parameters.
    """
    logger.info(f"Applying ultrasonic shift towards {target_freq} Hz (SR={sr} Hz)")

    if audio is None or audio.size == 0:
        logger.warning("Attempted to apply ultrasonic shift to empty audio.")
        return audio

    try:
        # Ensure input is float32
        audio_float = audio.astype(np.float32)

        # --- Step 1: Estimate Fundamental and Calculate Pitch Shift ---
        # Use piptrack on the first channel (or mixdown if needed)
        # Consider using a more robust pitch detection if piptrack is unreliable
        try:
            pitches, magnitudes = librosa.piptrack(y=audio_float[:, 0], sr=sr)
            # Get pitches where magnitude is significant (relative to median)
            median_mag = np.median(magnitudes)
            if median_mag > 1e-6:  # Avoid issues with silence
                valid_pitches = pitches[magnitudes > median_mag * 0.5]
            else:
                valid_pitches = pitches[
                    magnitudes > 1e-6
                ]  # Absolute threshold for silence
            valid_pitches = valid_pitches[
                valid_pitches > 0
            ]  # Only positive frequencies
        except Exception as e_piptrack:
            logger.warning(
                f"Piptrack failed during ultrasonic shift: {e_piptrack}. Using default fundamental."
            )
            valid_pitches = []  # Force fallback

        if len(valid_pitches) > 0:
            fundamental_freq = np.median(valid_pitches)
            logger.debug(f"Estimated fundamental frequency: {fundamental_freq:.2f} Hz")
        else:
            fundamental_freq = 440.0  # Default to A4
            logger.warning(
                f"Could not estimate fundamental frequency, using default {fundamental_freq} Hz."
            )

        if fundamental_freq <= 0 or np.isnan(fundamental_freq):
            logger.error(
                f"Invalid fundamental frequency ({fundamental_freq}) estimated. Cannot calculate shift."
            )
            st.error(
                "Could not estimate audio fundamental frequency for ultrasonic shift."
            )
            return audio

        # Calculate required pitch shift in semitones
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
            return audio_shifted

        logger.info(
            f"Applying {filter_order}-order low-pass Butterworth filter at {filter_cutoff} Hz."
        )
        b, a = signal.butter(filter_order, normalized_cutoff, btype="lowpass")

        min_len_filtfilt = 3 * max(len(a), len(b))
        if audio_shifted.shape[0] <= min_len_filtfilt:
            logger.warning(
                f"Shifted audio too short ({audio_shifted.shape[0]}) for filter length ({min_len_filtfilt}). Skipping filter."
            )
            return audio_shifted

        # Apply the filter using filtfilt for zero phase distortion
        audio_filtered = signal.filtfilt(b, a, audio_shifted, axis=0)
        logger.debug(
            f"Filtering complete. Filtered audio shape: {audio_filtered.shape}"
        )

        return audio_filtered.astype(np.float32)

    except Exception as e:
        logger.exception("Error applying ultrasonic shift with filtering.")
        st.error(f"Ultrasonic shift effect failed: {e}")
        return audio  # Return original on error


def apply_standard_filter(
    audio: AudioData, sr: SampleRate, filter_type: str, cutoff: float
) -> AudioData:
    """Applies a standard Butterworth lowpass or highpass filter."""
    logger.debug(f"Applying standard filter: type={filter_type}, cutoff={cutoff} Hz")

    if filter_type == "off" or cutoff <= 0:
        return audio

    if audio is None or audio.size == 0:
        logger.warning("Attempted to apply filter to empty audio.")
        return audio

    try:
        # Ensure input is float32
        audio_float = audio.astype(np.float32)

        nyquist = 0.5 * sr
        normalized_cutoff = cutoff / nyquist

        if normalized_cutoff >= 1.0 or normalized_cutoff <= 0:
            msg = f"Filter cutoff ({cutoff}Hz) is invalid for Nyquist ({nyquist}Hz). Skipping filter."
            logger.warning(msg)
            st.warning(msg)
            return audio

        filter_order = 4  # Standard order
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
            st.warning(f"Unknown filter type: {filter_type}")
            return audio

        min_len_filtfilt = 3 * max(len(a), len(b))
        if audio_float.shape[0] <= min_len_filtfilt:
            msg = f"Track too short ({audio_float.shape[0]}) for filter length ({min_len_filtfilt}). Skipping filter."
            logger.warning(msg)
            st.warning(msg)
            return audio

        # Apply the filter using filtfilt
        audio_filtered = signal.filtfilt(b, a, audio_float, axis=0)
        return audio_filtered.astype(np.float32)

    except Exception as e:
        logger.exception(
            f"Error applying standard {filter_type} filter at {cutoff} Hz."
        )
        st.error(f"Filter effect failed: {e}")
        return audio  # Return original on error
