# audio_utils.py
# ==========================================
# Audio Processing Utilities for MindMorph
# ==========================================

import logging
import os
import tempfile
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import librosa
import librosa.effects
import numpy as np
import soundfile as sf
from scipy import signal
from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st # TODO: Remove Streamlit UI calls from this module

# Import constants and types from config
from config import (
    GLOBAL_SR,
    ULTRASONIC_FILTER_CUTOFF,
    ULTRASONIC_FILTER_ORDER,
    ULTRASONIC_TARGET_FREQ,
)

# Define type hints used within this module
AudioData = np.ndarray
SampleRate = int
TrackData = Dict[str, Any] # Using Dict temporarily, might be replaced by a dataclass later

# Get a logger for this module
logger = logging.getLogger(__name__)

# ==========================================
# 1. File I/O
# ==========================================

def load_audio(
    file_source: UploadedFile | BytesIO | str, target_sr: SampleRate = GLOBAL_SR
) -> tuple[AudioData, SampleRate]:
    """
    Loads audio from various sources, ensures stereo, and resamples to target SR.

    Args:
        file_source: An UploadedFile object, BytesIO buffer, or file path string.
        target_sr: The desired sample rate for the output audio.

    Returns:
        A tuple containing:
            - The loaded audio data as a NumPy array (float32, stereo).
            - The sample rate of the loaded audio.
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
            audio = np.stack([audio, audio], axis=-1) # Now (samples, 2)
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
            audio = np.concatenate([audio, audio], axis=1) # Now (samples, 2)

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
                sr = target_sr
                logger.warning("Audio data is empty, cannot resample. Setting SR to target.")

        # Ensure final output is float32
        return audio.astype(np.float32), target_sr

    except Exception as e:
        logger.exception("Error loading audio.")
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
        return buffer # Return empty buffer

    try:
        # Ensure audio is within [-1.0, 1.0] before converting to int16
        audio_clipped = np.clip(audio, -1.0, 1.0)
        # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        # Write to the buffer using soundfile
        sf.write(buffer, audio_int16, sr, format="WAV", subtype="PCM_16")
        buffer.seek(0) # Rewind the buffer to the beginning for reading
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

        # Create a named temporary file (deleted=False means we manage deletion)
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

# ==========================================
# 2. Audio Generation
# ==========================================

def generate_binaural_beats(
    duration: float, freq_left: float, freq_right: float, sr: SampleRate, volume: float
) -> AudioData:
    """Generates stereo binaural beats."""
    logger.info(
        f"Generating binaural beats: duration={duration}s, L={freq_left}Hz, "
        f"R={freq_right}Hz, SR={sr}Hz, volume={volume:.2f}"
    )
    num_samples = int(sr * duration)
    if num_samples <= 0:
        logger.warning("Duration results in zero samples for binaural beats.")
        return np.zeros((0, 2), dtype=np.float32)

    t = np.linspace(0, duration, num_samples, endpoint=False)
    left_channel = np.sin(2 * np.pi * freq_left * t)
    right_channel = np.sin(2 * np.pi * freq_right * t)

    # Stack channels and apply volume
    audio = np.stack([left_channel, right_channel], axis=1) * volume

    # Clip to prevent values outside [-1.0, 1.0] and set type
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def generate_solfeggio_frequency(
    duration: float, freq: float, sr: SampleRate, volume: float
) -> AudioData:
    """Generates a stereo pure tone (Solfeggio)."""
    logger.info(
        f"Generating Solfeggio tone: duration={duration}s, F={freq}Hz, "
        f"SR={sr}Hz, volume={volume:.2f}"
    )
    num_samples = int(sr * duration)
    if num_samples <= 0:
        logger.warning("Duration results in zero samples for Solfeggio tone.")
        return np.zeros((0, 2), dtype=np.float32)

    t = np.linspace(0, duration, num_samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t)

    # Create stereo by duplicating the sine wave and apply volume
    audio = np.stack([sine_wave, sine_wave], axis=1) * volume

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def generate_isochronic_tones(
    duration: float, carrier_freq: float, pulse_freq: float, sr: SampleRate, volume: float
) -> AudioData:
    """Generates stereo isochronic tones using amplitude modulation."""
    logger.info(
        f"Generating Isochronic tones: duration={duration}s, Carrier={carrier_freq}Hz, "
        f"Pulse={pulse_freq}Hz, SR={sr}Hz, volume={volume:.2f}"
    )
    num_samples = int(sr * duration)
    if num_samples <= 0:
        logger.warning("Duration results in zero samples for Isochronic tones.")
        return np.zeros((0, 2), dtype=np.float32)

    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Carrier wave (the tone being pulsed)
    carrier_wave = np.sin(2 * np.pi * carrier_freq * t)

    # Modulation wave (creates the pulsing effect - square wave)
    # 0.5 * (sign(...) + 1) creates a square wave oscillating between 0 and 1
    modulation_wave = 0.5 * (np.sign(np.sin(2 * np.pi * pulse_freq * t + np.pi / 2)) + 1)

    # Modulate the carrier by the modulation wave
    modulated_signal = carrier_wave * modulation_wave

    # Create stereo output and apply volume
    audio = np.stack([modulated_signal, modulated_signal], axis=1) * volume

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def generate_white_noise(num_samples: int) -> AudioData:
    """Generates stereo white noise (uniform distribution)."""
    logger.debug(f"Generating {num_samples} samples of white noise.")
    if num_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    # Generate random samples between -1.0 and 1.0 for stereo
    return np.random.uniform(-1.0, 1.0, size=(num_samples, 2)).astype(np.float32)


def generate_pink_noise(num_samples: int) -> AudioData:
    """
    Generates stereo pink noise (approximated by filtering white noise).
    Uses Voss-McCartney algorithm for better approximation.
    """
    logger.debug(f"Generating {num_samples} samples of pink noise.")
    if num_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Generate for stereo (2 channels)
    pink_noise = np.zeros((num_samples, 2), dtype=np.float32)
    for channel in range(2):
        # Voss-McCartney algorithm state for this channel
        state = np.zeros(16) # Typically use 16 sources
        key = 0
        white_samples = np.random.uniform(-1.0, 1.0, num_samples) * 0.5 # Start with white noise scaled down

        for i in range(num_samples):
            # Update state based on key changes
            last_key = key
            key += 1
            diff = last_key ^ key
            total = 0
            for j in range(16):
                if (diff >> j) & 1:
                    state[j] = np.random.uniform(-1.0, 1.0) * 0.5
                total += state[j]

            # Add new white sample
            total += white_samples[i]
            pink_noise[i, channel] = total / 17.0 # Normalize by number of sources + white

    # Normalize the final output to be within [-1.0, 1.0]
    max_abs_val = np.max(np.abs(pink_noise))
    if max_abs_val > 1e-6: # Avoid division by zero
        pink_noise /= max_abs_val

    return pink_noise.astype(np.float32)


def generate_brown_noise(num_samples: int) -> AudioData:
    """Generates stereo brown noise (by integrating white noise)."""
    logger.debug(f"Generating {num_samples} samples of brown noise.")
    if num_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Start with white noise
    white = generate_white_noise(num_samples)

    # Integrate the white noise (cumulative sum)
    brown_approx = np.cumsum(white, axis=0)

    # Normalize the result to be within [-1.0, 1.0]
    max_abs_val = np.max(np.abs(brown_approx))
    if max_abs_val > 1e-6: # Avoid division by zero
        brown_approx /= max_abs_val

    return brown_approx.astype(np.float32)


def generate_noise(noise_type: str, duration: float, sr: SampleRate, volume: float) -> Optional[AudioData]:
    """Generates a specified type of noise."""
    logger.info(
        f"Generating {noise_type}: duration={duration}s, SR={sr}Hz, volume={volume:.2f}"
    )
    num_samples = int(sr * duration)
    if num_samples <= 0:
        logger.warning(f"Duration results in zero samples for {noise_type}.")
        return None

    noise_audio: Optional[AudioData] = None
    if noise_type == "White Noise":
        noise_audio = generate_white_noise(num_samples)
    elif noise_type == "Pink Noise":
        noise_audio = generate_pink_noise(num_samples)
    elif noise_type == "Brown Noise":
        noise_audio = generate_brown_noise(num_samples)
    else:
        logger.error(f"Unknown noise type requested: {noise_type}")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Unknown noise type: {noise_type}")
        return None

    if noise_audio is not None:
        # Apply volume and clip
        audio_scaled = np.clip(noise_audio * volume, -1.0, 1.0)
        return audio_scaled.astype(np.float32)

    return None # Should not happen if logic is correct, but for safety

# ==========================================
# 3. Audio Effects
# ==========================================

def apply_reverse(audio: AudioData) -> AudioData:
    """Reverses the audio data along the time axis."""
    logger.debug("Applying reverse effect.")
    if audio is None or audio.size == 0:
        logger.warning("Attempted to reverse empty audio.")
        return audio # Return unchanged if empty

    try:
        # Slicing [::-1] reverses the array along the first axis (time)
        return np.ascontiguousarray(audio[::-1]).astype(np.float32)
    except Exception as e:
        logger.exception("Error applying reverse effect.")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Reverse audio effect failed: {e}")
        return audio # Return original on error


def apply_speed_change(audio: AudioData, sr: SampleRate, speed_factor: float) -> AudioData:
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
        return audio

    try:
        logger.info(f"Applying time stretch (factor: {speed_factor:.2f})...")
        # librosa expects (channels, samples) and contiguous array
        audio_contiguous_transposed = np.ascontiguousarray(audio.T)
        # Apply time stretching
        audio_stretched_transposed = librosa.effects.time_stretch(
            y=audio_contiguous_transposed, rate=speed_factor
        )
        # Transpose back to (samples, channels)
        return audio_stretched_transposed.T.astype(np.float32)

    except Exception as e:
        logger.exception(f"Error during librosa time_stretch with factor {speed_factor:.2f}.")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Speed change effect failed: {e}")
        return audio # Return original on error


def apply_pitch_shift(audio: AudioData, sr: SampleRate, n_steps: float) -> AudioData:
    """Shifts the pitch of the audio without changing speed."""
    logger.debug(f"Applying pitch shift: {n_steps:.2f} semitones")

    if audio is None or audio.size == 0:
        logger.warning("Attempted to pitch shift empty audio.")
        return audio

    # If pitch shift is effectively 0, no change needed
    if np.isclose(n_steps, 0.0):
        return audio

    try:
        logger.info(f"Applying pitch shift ({n_steps:.2f} semitones)...")
        # Check for potentially problematic large shifts relative to SR
        # Note: This is a rough estimate, actual quality depends on content
        max_possible_steps = 12 * np.log2((sr / 2) / 440.0) # Max shift preserving up to A4=440Hz fundamental's octave
        if abs(n_steps) > max(12, max_possible_steps): # Warn for shifts > octave or potentially exceeding Nyquist
             logger.warning(
                 f"Large pitch shift ({n_steps:.1f} steps) requested for SR={sr}Hz. "
                 f"Result quality may be poor or contain artifacts."
             )

        # librosa expects (channels, samples) and contiguous array
        audio_contiguous_transposed = np.ascontiguousarray(audio.T)
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
        return audio # Return original on error


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
        # --- Step 1: Estimate Fundamental and Calculate Pitch Shift ---
        # Use piptrack on the first channel (or mixdown if needed) to estimate pitch
        # Note: piptrack can be computationally intensive and may not be accurate for all sounds
        pitches, magnitudes = librosa.piptrack(y=audio[:, 0], sr=sr)
        # Get pitches where magnitude is significant
        valid_pitches = pitches[magnitudes > np.median(magnitudes) * 0.5] # Use a threshold
        valid_pitches = valid_pitches[valid_pitches > 0] # Only positive frequencies

        if len(valid_pitches) > 0:
            # Use median of valid pitches as a robust estimate of fundamental
            fundamental_freq = np.median(valid_pitches)
            logger.debug(f"Estimated fundamental frequency: {fundamental_freq:.2f} Hz")
        else:
            # Fallback if no valid pitches found
            fundamental_freq = 440.0 # Default to A4
            logger.warning("Could not estimate fundamental frequency, using default {fundamental_freq} Hz.")

        # Avoid division by zero or invalid fundamental
        if fundamental_freq <= 0 or np.isnan(fundamental_freq):
            logger.error(f"Invalid fundamental frequency ({fundamental_freq}) estimated. Cannot calculate shift.")
            # TODO: Remove direct Streamlit call. Raise exception or return error status.
            st.error("Could not estimate audio fundamental frequency for ultrasonic shift.")
            return audio

        # Calculate required pitch shift in semitones
        # n_steps = 12 * log2(target_freq / fundamental_freq)
        n_steps = 12 * np.log2(target_freq / fundamental_freq)
        logger.info(
            f"Calculated steps for ultrasonic shift: {n_steps:.2f} semitones "
            f"(Target: {target_freq} Hz, Estimated Fundamental: {fundamental_freq:.1f} Hz)"
        )

        # Warn if the shift is very large
        if abs(n_steps) > 36: # Warn for shifts greater than 3 octaves
            logger.warning(
                f"Applying very large pitch shift ({n_steps:.1f} steps) for ultrasonic effect. "
                f"Audio quality might be significantly degraded."
            )

        # --- Step 2: Apply Pitch Shift ---
        # librosa expects (channels, samples) and contiguous array
        audio_contiguous_transposed = np.ascontiguousarray(audio.T)
        audio_shifted_transposed = librosa.effects.pitch_shift(
            y=audio_contiguous_transposed, sr=sr, n_steps=n_steps
        )
        audio_shifted = audio_shifted_transposed.T.astype(np.float32)
        logger.debug(f"Pitch shift complete. Shifted audio shape: {audio_shifted.shape}")

        # --- Step 3: Apply Steep Low-Pass Filter ---
        # Use parameters from config
        filter_cutoff = ULTRASONIC_FILTER_CUTOFF
        filter_order = ULTRASONIC_FILTER_ORDER
        nyquist = 0.5 * sr
        normalized_cutoff = filter_cutoff / nyquist

        # Validate cutoff frequency
        if normalized_cutoff >= 1.0:
            logger.warning(
                f"Ultrasonic filter cutoff ({filter_cutoff}Hz) is >= Nyquist frequency ({nyquist}Hz). "
                f"Skipping filter."
            )
            return audio_shifted # Return shifted audio without filtering
        if normalized_cutoff <= 0:
            logger.warning(
                f"Ultrasonic filter cutoff ({filter_cutoff}Hz) is invalid (<= 0). Skipping filter."
            )
            return audio_shifted # Return shifted audio without filtering

        logger.info(
            f"Applying {filter_order}-order low-pass Butterworth filter at {filter_cutoff} Hz "
            f"to remove audible artifacts after ultrasonic shift."
        )
        # Design the Butterworth filter
        b, a = signal.butter(filter_order, normalized_cutoff, btype="lowpass")

        # Check if audio is long enough for filtfilt (prevents instability)
        # Default minimum length for filtfilt is 3 * max(len(a), len(b))
        min_len_filtfilt = 3 * max(len(a), len(b))
        if audio_shifted.shape[0] <= min_len_filtfilt:
            logger.warning(
                f"Shifted audio is too short ({audio_shifted.shape[0]} samples) for the "
                f"filter length ({min_len_filtfilt}). Skipping filter to avoid errors."
            )
            return audio_shifted # Return shifted audio without filtering

        # Apply the filter using filtfilt for zero phase distortion
        # axis=0 applies filter along the time dimension for each channel
        audio_filtered = signal.filtfilt(b, a, audio_shifted, axis=0)
        logger.debug(f"Filtering complete. Filtered audio shape: {audio_filtered.shape}")

        return audio_filtered.astype(np.float32)

    except Exception as e:
        logger.exception("Error applying ultrasonic shift with filtering.")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Ultrasonic shift effect failed: {e}")
        return audio # Return original on error


def apply_standard_filter(
    audio: AudioData, sr: SampleRate, filter_type: str, cutoff: float
) -> AudioData:
    """Applies a standard Butterworth lowpass or highpass filter."""
    logger.debug(f"Applying standard filter: type={filter_type}, cutoff={cutoff} Hz")

    # No filter needed if type is 'off' or cutoff is invalid
    if filter_type == "off" or cutoff <= 0:
        return audio

    if audio is None or audio.size == 0:
        logger.warning("Attempted to apply filter to empty audio.")
        return audio

    try:
        nyquist = 0.5 * sr
        normalized_cutoff = cutoff / nyquist

        # Validate cutoff frequency against Nyquist
        if normalized_cutoff >= 1.0:
            msg = f"Filter cutoff ({cutoff}Hz) >= Nyquist frequency ({nyquist}Hz). Skipping filter."
            logger.warning(msg)
            # TODO: Remove direct Streamlit call. Raise exception or return error status.
            st.warning(msg)
            return audio
        if normalized_cutoff <= 0: # Should be caught by initial check, but good practice
            msg = f"Filter cutoff ({cutoff}Hz) must be positive. Skipping filter."
            logger.warning(msg)
            # TODO: Remove direct Streamlit call. Raise exception or return error status.
            st.warning(msg)
            return audio

        filter_order = 4 # A standard, reasonable order for general filtering
        logger.info(
            f"Applying {filter_order}-order Butterworth {filter_type} filter at {cutoff} Hz."
        )

        # Design the filter based on type
        if filter_type == "lowpass":
            b, a = signal.butter(filter_order, normalized_cutoff, btype="low")
        elif filter_type == "highpass":
            b, a = signal.butter(filter_order, normalized_cutoff, btype="high")
        else:
            logger.warning(f"Unknown filter type specified: {filter_type}. Skipping filter.")
            # TODO: Remove direct Streamlit call. Raise exception or return error status.
            st.warning(f"Unknown filter type: {filter_type}")
            return audio

        # Check if audio is long enough for filtfilt
        min_len_filtfilt = 3 * max(len(a), len(b))
        if audio.shape[0] <= min_len_filtfilt:
            msg = (
                f"Track is too short ({audio.shape[0]} samples) for the filter length "
                f"({min_len_filtfilt}). Skipping filter."
            )
            logger.warning(msg)
            # TODO: Remove direct Streamlit call. Raise exception or return error status.
            st.warning(msg)
            return audio

        # Apply the filter using filtfilt
        audio_filtered = signal.filtfilt(b, a, audio, axis=0)
        return audio_filtered.astype(np.float32)

    except Exception as e:
        logger.exception(f"Error applying standard {filter_type} filter at {cutoff} Hz.")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Filter effect failed: {e}")
        return audio # Return original on error

# ==========================================
# 4. Combined Effects & Preview
# ==========================================

def apply_all_effects(
    track_data: TrackData, audio_segment: Optional[AudioData] = None
) -> AudioData:
    """
    Applies effects sequentially based on track_data settings.
    Order: Reverse -> Speed -> Pitch/Ultrasonic -> Filter.

    Args:
        track_data: Dictionary containing track settings and original audio.
        audio_segment: Optional audio segment to process (e.g., for previews).
                       If None, uses track_data['original_audio'].

    Returns:
        The processed audio data as a NumPy array. Returns empty array on error.
    """
    track_name = track_data.get("name", "Unnamed Track")
    should_reverse = track_data.get("reverse_audio", False)
    use_ultrasonic = track_data.get("ultrasonic_shift", False)
    pitch_shift_steps = track_data.get("pitch_shift", 0.0)
    speed_factor = track_data.get("speed_factor", 1.0)
    filter_type = track_data.get("filter_type", "off")
    filter_cutoff = track_data.get("filter_cutoff", 8000.0)
    sr = track_data.get("sr", GLOBAL_SR)

    if audio_segment is not None:
        # Process the provided segment (e.g., for preview)
        audio = audio_segment.copy()
        log_prefix = f"Applying effects to segment for '{track_name}'"
    else:
        # Process the full original audio from track_data
        original_audio = track_data.get("original_audio")
        if original_audio is None:
            logger.error(f"Track '{track_name}' is missing 'original_audio' data for effects processing.")
            return np.zeros((0, 2), dtype=np.float32)
        audio = original_audio.copy()
        log_prefix = f"Applying effects to full audio for '{track_name}'"

    logger.debug(f"{log_prefix}: Reverse={should_reverse}, Speed={speed_factor:.2f}, "
                 f"Ultrasonic={use_ultrasonic}, Pitch={pitch_shift_steps:.1f}, "
                 f"Filter={filter_type}@{filter_cutoff}Hz")

    if audio.size == 0:
        logger.warning(f"Input audio for effects is empty for '{track_name}'. Returning empty.")
        return audio # Return empty array if input is empty

    # --- Apply Effects Sequentially ---
    # 1. Reverse (Optional)
    if should_reverse:
        audio = apply_reverse(audio)
        logger.debug(f"'{track_name}': Applied reverse.")

    # 2. Speed Change
    # Apply speed change only if factor is not 1.0
    if not np.isclose(speed_factor, 1.0):
        audio = apply_speed_change(audio, sr, speed_factor)
        logger.debug(f"'{track_name}': Applied speed change (factor {speed_factor:.2f}).")


    # 3. Pitch Shift (Ultrasonic OR Regular, mutually exclusive)
    if use_ultrasonic:
        logger.debug(f"'{track_name}': Applying Ultrasonic shift.")
        audio = apply_ultrasonic_shift(audio, sr, target_freq=ULTRASONIC_TARGET_FREQ)
    elif not np.isclose(pitch_shift_steps, 0.0):
        logger.debug(f"'{track_name}': Applying Regular pitch shift ({pitch_shift_steps:.1f} steps).")
        audio = apply_pitch_shift(audio, sr, pitch_shift_steps)
    else:
        logger.debug(f"'{track_name}': No pitch shift applied (Ultrasonic off, Pitch shift is 0).")

    # 4. Filter (Standard Low/High Pass - only if Ultrasonic wasn't applied)
    if not use_ultrasonic and filter_type != "off":
        logger.debug(f"'{track_name}': Applying standard filter ({filter_type} @ {filter_cutoff}Hz).")
        audio = apply_standard_filter(audio, sr, filter_type, filter_cutoff)
    elif use_ultrasonic:
        logger.debug(f"'{track_name}': Skipping standard filter because ultrasonic shift was applied.")
    else:
         logger.debug(f"'{track_name}': No standard filter applied (Ultrasonic off, Filter type is 'off').")


    logger.debug(f"Finished applying effects for '{track_name}'. Output shape: {audio.shape}")
    return audio.astype(np.float32) # Ensure output is float32


def get_preview_audio(
    track_data: TrackData, preview_duration_s: int = PREVIEW_DURATION_S
) -> Optional[AudioData]:
    """
    Generates a preview (first N seconds) of the track with effects, volume, and pan applied.

    Args:
        track_data: Dictionary containing track settings and original audio.
        preview_duration_s: The maximum duration of the preview in seconds.

    Returns:
        The processed preview audio data, or None if generation fails or input is invalid.
    """
    track_name = track_data.get("name", "N/A")
    logger.info(f"Generating preview audio for track '{track_name}' (max {preview_duration_s}s)")

    original_audio = track_data.get("original_audio")
    sr = track_data.get("sr", GLOBAL_SR)

    if original_audio is None or original_audio.size == 0:
        logger.warning(f"No original audio data found for track '{track_name}'. Cannot generate preview.")
        return None
    if sr <= 0:
         logger.warning(f"Invalid sample rate ({sr}) for track '{track_name}'. Cannot generate preview.")
         return None

    try:
        # Calculate number of samples for the preview duration
        preview_samples = min(len(original_audio), int(sr * preview_duration_s))
        if preview_samples <= 0:
            logger.warning(f"Calculated preview samples <= 0 for '{track_name}'. Cannot generate preview.")
            return None

        # Extract the segment for preview processing
        preview_segment = original_audio[:preview_samples].copy()
        logger.debug(f"Extracted preview segment ({preview_samples} samples) for '{track_name}'")

        # Apply all effects (Reverse, Speed, Pitch/Ultrasonic, Filter) to the segment
        logger.debug(f"Applying effects to preview segment for '{track_name}'")
        processed_preview = apply_all_effects(track_data, audio_segment=preview_segment)

        if processed_preview is None or processed_preview.size == 0:
             logger.warning(f"Applying effects resulted in empty audio for '{track_name}' preview.")
             return None

        # Apply Volume and Pan to the processed preview
        vol = track_data.get("volume", 1.0)
        pan = track_data.get("pan", 0.0)
        logger.debug(f"Applying Volume ({vol:.2f}) / Pan ({pan:.2f}) to preview for '{track_name}'")

        # Calculate stereo gains based on pan (-1 L to +1 R)
        # Pan value is mapped to an angle in radians (0 to pi/2)
        # Left gain = cos(angle), Right gain = sin(angle)
        pan_rad = (pan + 1.0) * np.pi / 4.0 # Maps [-1, 1] to [0, pi/2]
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)

        # Apply gains to stereo channels
        if processed_preview.ndim == 2 and processed_preview.shape[1] == 2:
            processed_preview[:, 0] *= left_gain
            processed_preview[:, 1] *= right_gain
        elif processed_preview.ndim == 1: # Handle mono case (apply volume, pan has less effect)
             logger.warning(f"Preview for '{track_name}' is mono after effects. Applying volume, pan ignored.")
             processed_preview *= vol
             # Convert mono back to stereo for consistency if needed, though pan is lost
             processed_preview = np.stack([processed_preview, processed_preview], axis=1)
        else:
            logger.warning(
                f"Processed preview for '{track_name}' has unexpected shape {processed_preview.shape}. "
                f"Cannot apply volume/pan correctly."
            )
            # Attempt to apply volume to first channel if possible as fallback
            try:
                 processed_preview[:, 0] *= vol
            except IndexError:
                 pass # Ignore if not possible

        # Clip final preview audio and ensure correct type
        processed_preview = np.clip(processed_preview, -1.0, 1.0)
        logger.debug(f"Preview generation complete for '{track_name}'. Final shape: {processed_preview.shape}")
        return processed_preview.astype(np.float32)

    except Exception as e:
        logger.exception(f"Error generating preview for track '{track_name}'")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Error generating preview for '{track_name}': {e}")
        return None

# ==========================================
# 5. Mixing Logic (Moved from UIManager/main)
# ==========================================

def mix_tracks(
    tracks_dict: Dict[str, TrackData], # Use str for TrackID key
    target_sr: int = GLOBAL_SR,
    preview: bool = False,
    preview_duration_s: int = MIX_PREVIEW_DURATION_S,
    preview_buffer_s: int = MIX_PREVIEW_PROCESSING_BUFFER_S
) -> Tuple[Optional[AudioData], Optional[int]]:
    """
    Mixes multiple tracks together, handling effects, looping, volume, pan, mute, and solo.

    Args:
        tracks_dict: Dictionary where keys are track IDs and values are TrackData dictionaries.
        target_sr: The sample rate for the final mix.
        preview: If True, generate a shorter preview mix.
        preview_duration_s: Duration of the preview mix in seconds.
        preview_buffer_s: Extra buffer duration for processing preview segments.

    Returns:
        A tuple containing:
            - The final mixed audio data as a NumPy array (float32, stereo), or None on failure.
            - The length of the final mix in samples, or None on failure.
    """
    logger.info(f"Starting track mixing. Preview mode: {preview}. Target SR: {target_sr}Hz.")

    if not tracks_dict:
        logger.warning("Mix called with no tracks provided.")
        return None, None

    # --- Determine Active Tracks and Estimate Lengths ---
    valid_track_ids_for_mix = []
    estimated_processed_lengths = {}
    # Check if any track is soloed
    solo_active = any(t_data.get("solo", False) for t_data in tracks_dict.values())
    logger.debug(f"Solo active: {solo_active}")

    logger.info("Step 1: Determining active tracks and estimating lengths after speed changes.")
    for track_id, t_data in tracks_dict.items():
        # Determine if track should be included in the mix
        is_active = False
        original_audio = t_data.get("original_audio")
        has_audio = original_audio is not None and original_audio.size > 0
        is_muted = t_data.get("mute", False)

        if has_audio:
            if solo_active:
                # If solo is active, only include soloed tracks
                is_active = t_data.get("solo", False)
            else:
                # If solo is not active, include unmuted tracks
                is_active = not is_muted

        if is_active:
            valid_track_ids_for_mix.append(track_id)
            original_len = len(original_audio)
            speed_factor = t_data.get("speed_factor", 1.0)
            # Estimate length after speed change (avoid division by zero)
            estimated_len = int(original_len / speed_factor) if speed_factor > 0 else original_len
            estimated_processed_lengths[track_id] = estimated_len
            logger.debug(
                f"Track '{t_data.get('name', track_id)}': Active. "
                f"Original len={original_len}, Speed={speed_factor:.2f}, Estimated len={estimated_len}"
            )
        else:
            reason = "no audio data" if not has_audio else ("muted" if is_muted else "not soloed")
            logger.debug(f"Skipping track '{t_data.get('name', track_id)}' from mix ({reason}).")

    if not valid_track_ids_for_mix:
        logger.warning("No active tracks with audio found for mixing.")
        return None, None

    # --- Determine Target Mix Length ---
    # Find the maximum estimated length among active tracks (before looping)
    target_mix_len_samples = max(estimated_processed_lengths.values()) if estimated_processed_lengths else 0
    logger.info(f"Target mix length based on estimations (pre-looping): {target_mix_len_samples} samples ({target_mix_len_samples / target_sr:.2f}s)")

    # Adjust length and processing buffer for preview mode
    process_samples = 0 # How much audio to process for each track in preview mode
    if preview:
        preview_target_len = int(target_sr * preview_duration_s)
        if target_mix_len_samples > preview_target_len:
            logger.info(f"Preview mode: Limiting mix length from {target_mix_len_samples} to {preview_target_len} samples.")
            target_mix_len_samples = preview_target_len
        # Calculate how much raw audio needs processing to cover the preview + buffer
        process_duration_s = preview_duration_s + preview_buffer_s
        process_samples = int(target_sr * process_duration_s)
        logger.debug(f"Preview processing buffer: {process_samples} samples ({process_duration_s:.1f}s)")


    if target_mix_len_samples <= 0:
        logger.warning("Target mix length is zero or negative. Cannot create mix.")
        return None, None

    # Initialize the master mix buffer
    mix_buffer = np.zeros((target_mix_len_samples, 2), dtype=np.float32)
    logger.info(f"Mixing {len(valid_track_ids_for_mix)} tracks sequentially. Initial mix buffer length: {target_mix_len_samples / target_sr:.2f}s")

    # --- Pre-process Preview Segments (if in preview mode) ---
    # This avoids re-applying effects repeatedly inside the main loop for previews
    processed_preview_segments = {}
    if preview:
        logger.info("Step 2: Pre-processing segments for preview mode.")
        for track_id in valid_track_ids_for_mix:
            t_data = tracks_dict[track_id]
            original_audio = t_data.get("original_audio")
            if original_audio is not None and original_audio.size > 0:
                # Process only the required segment length
                segment_samples = min(len(original_audio), process_samples)
                segment = original_audio[:segment_samples].copy()
                logger.debug(f"Processing PREVIEW segment ({segment_samples} samples) for track '{t_data.get('name', track_id)}'.")
                processed_preview_segments[track_id] = apply_all_effects(t_data, audio_segment=segment)
            else:
                 # Should not happen based on earlier check, but handle defensively
                processed_preview_segments[track_id] = None
                logger.warning(f"Track '{t_data.get('name', track_id)}' unexpectedly missing audio during preview pre-processing.")

    # --- Process and Add Each Track to Mix ---
    logger.info("Step 3: Processing and adding tracks to the mix buffer.")
    actual_max_len_samples = target_mix_len_samples # Track the true max length after looping

    for track_id in valid_track_ids_for_mix:
        t_data = tracks_dict[track_id]
        track_name = t_data.get("name", track_id)
        logger.debug(f"Processing and mixing track: '{track_name}'")

        # Get the processed audio (either pre-processed preview or process full track)
        processed_audio: Optional[AudioData] = None
        if preview:
            processed_audio = processed_preview_segments.get(track_id)
        else:
            # Process the full original audio with effects
            processed_audio = apply_all_effects(t_data) # No segment passed

        if processed_audio is None or processed_audio.size == 0:
            logger.warning(f"Processing resulted in empty audio for '{track_name}'. Skipping track.")
            continue

        actual_processed_len = len(processed_audio)
        logger.debug(f"Track '{track_name}': Actual processed length after effects = {actual_processed_len} samples.")

        # --- Handle Looping (only for full mix, not preview) ---
        final_audio_for_track = processed_audio # Start with the processed audio
        should_loop = t_data.get("loop_to_fit", False)

        if not preview and should_loop:
            # Only loop if the track is shorter than the current target mix length
            if actual_processed_len > 0 and actual_processed_len < target_mix_len_samples:
                logger.info(f"Looping track '{track_name}' from {actual_processed_len} samples up to {target_mix_len_samples} samples.")
                num_repeats = target_mix_len_samples // actual_processed_len
                remainder = target_mix_len_samples % actual_processed_len

                # Create a list of audio segments to concatenate
                looped_list = [processed_audio] * num_repeats
                if remainder > 0:
                    looped_list.append(processed_audio[:remainder]) # Add the remaining part

                try:
                    # Concatenate the segments
                    final_audio_for_track = np.concatenate(looped_list, axis=0)
                    # Update the overall maximum length if this looped track is now the longest
                    actual_max_len_samples = max(actual_max_len_samples, len(final_audio_for_track))
                    logger.debug(f"Looping complete for '{track_name}'. New length: {len(final_audio_for_track)} samples.")
                except ValueError as e_concat:
                    # Handle potential errors during concatenation (e.g., shape mismatch)
                    logger.error(f"Error concatenating looped audio for '{track_name}': {e_concat}. Using non-looped audio instead.")
                    final_audio_for_track = processed_audio # Fallback to non-looped
            else:
                logger.debug(f"Looping not needed or applicable for '{track_name}' (Length: {actual_processed_len}, Target: {target_mix_len_samples})")
                # Update max length even if not looped
                actual_max_len_samples = max(actual_max_len_samples, actual_processed_len)


        # --- Resize Mix Buffer if Looping Extended Max Length ---
        # This ensures the buffer is large enough to hold the longest track
        if len(mix_buffer) < actual_max_len_samples:
            logger.warning(
                f"Resizing mix buffer from {len(mix_buffer)} to {actual_max_len_samples} samples "
                f"due to looping/track length of '{track_name}'."
            )
            # Pad the existing buffer with zeros at the end
            mix_buffer = np.pad(
                mix_buffer,
                ((0, actual_max_len_samples - len(mix_buffer)), (0, 0)),
                mode="constant"
            )
            # Update the target length to the new maximum
            target_mix_len_samples = actual_max_len_samples


        # --- Adjust Track Length to Match Mix Buffer ---
        current_track_len = len(final_audio_for_track)
        if current_track_len < target_mix_len_samples:
            # Pad shorter tracks with silence at the end
            audio_adjusted = np.pad(
                final_audio_for_track,
                ((0, target_mix_len_samples - current_track_len), (0, 0)),
                mode="constant"
            )
        elif current_track_len > target_mix_len_samples:
            # Truncate longer tracks (should only happen in preview mode or if looping logic changes)
             logger.warning(f"Track '{track_name}' ({current_track_len}) longer than mix buffer ({target_mix_len_samples}). Truncating.")
             audio_adjusted = final_audio_for_track[:target_mix_len_samples, :]
        else:
            # Length matches exactly
            audio_adjusted = final_audio_for_track

        # --- Apply Volume and Pan ---
        pan = t_data.get("pan", 0.0)
        vol = t_data.get("volume", 1.0)
        logger.debug(f"Track '{track_name}': Applying final vol={vol:.2f}, pan={pan:.2f}")

        # Calculate stereo gains
        pan_rad = (pan + 1.0) * np.pi / 4.0
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)

        # Add the adjusted track audio to the mix buffer
        if audio_adjusted.ndim == 2 and audio_adjusted.shape[1] == 2:
            mix_buffer[:, 0] += audio_adjusted[:, 0] * left_gain
            mix_buffer[:, 1] += audio_adjusted[:, 1] * right_gain
        elif audio_adjusted.ndim == 1: # Handle mono case
            logger.warning(f"Track '{track_name}' is mono during final mixing stage. Applying volume and splitting.")
            # Apply volume and distribute equally to L/R channels (approx -3dB each)
            mono_scaled = audio_adjusted * vol * 0.7071 # 1/sqrt(2)
            mix_buffer[:, 0] += mono_scaled
            mix_buffer[:, 1] += mono_scaled
        else:
             logger.error(f"Track '{track_name}' has unexpected shape {audio_adjusted.shape} during mixing. Cannot add to buffer.")


        logger.debug(f"Added track '{track_name}' to mix buffer.")
        # Clean up intermediate arrays to free memory
        del processed_audio, final_audio_for_track, audio_adjusted

    # --- Finalize Mix ---
    # Clip the final mix to prevent values outside [-1.0, 1.0]
    final_mix = np.clip(mix_buffer, -1.0, 1.0)
    final_mix_len = len(final_mix)
    logger.info(f"Mixing complete. Final mix length: {final_mix_len} samples ({final_mix_len / target_sr:.2f}s).")

    return final_mix.astype(np.float32), final_mix_len

