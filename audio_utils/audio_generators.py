# audio_generators.py
# ==========================================
# Audio Signal Generation Utilities for MindMorph
# ==========================================

import logging
from typing import Optional

import numpy as np
import streamlit as st  # TODO: Remove Streamlit UI calls from this module
from scipy import signal  # Used for pink noise generation approximation

# Import constants from config if needed (e.g., defaults)
# from config import GLOBAL_SR # Not strictly needed if SR is always passed in

# Define type hints used within this module
AudioData = np.ndarray
SampleRate = int

# Get a logger for this module
logger = logging.getLogger(__name__)

# ==========================================
# Tone Generation Functions
# ==========================================


def generate_binaural_beats(
    duration: float, freq_left: float, freq_right: float, sr: SampleRate, volume: float
) -> AudioData:
    """Generates stereo binaural beats."""
    logger.info(
        f"Generating binaural beats: duration={duration}s, L={freq_left}Hz, R={freq_right}Hz, SR={sr}Hz, volume={volume:.2f}"
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
        f"Generating Solfeggio tone: duration={duration}s, F={freq}Hz, SR={sr}Hz, volume={volume:.2f}"
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
    duration: float,
    carrier_freq: float,
    pulse_freq: float,
    sr: SampleRate,
    volume: float,
) -> AudioData:
    """Generates stereo isochronic tones using amplitude modulation."""
    logger.info(
        f"Generating Isochronic tones: duration={duration}s, Carrier={carrier_freq}Hz, Pulse={pulse_freq}Hz, SR={sr}Hz, volume={volume:.2f}"
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
    modulation_wave = 0.5 * (
        np.sign(np.sin(2 * np.pi * pulse_freq * t + np.pi / 2)) + 1
    )

    # Modulate the carrier by the modulation wave
    modulated_signal = carrier_wave * modulation_wave

    # Create stereo output and apply volume
    audio = np.stack([modulated_signal, modulated_signal], axis=1) * volume

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ==========================================
# Noise Generation Functions
# ==========================================


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
    pink_noise_out = np.zeros((num_samples, 2), dtype=np.float32)
    for channel in range(2):
        # Voss-McCartney algorithm state for this channel
        state = np.zeros(16)  # Typically use 16 sources/octaves
        key = 0
        # Generate white noise source, scaled down slightly
        white_samples = np.random.uniform(-1.0, 1.0, num_samples) * 0.5

        for i in range(num_samples):
            # Update state based on key changes (detects powers of 2)
            last_key = key
            key += 1
            diff = last_key ^ key  # Find changed bits
            total = 0
            for j in range(16):  # Iterate through octaves/sources
                if (diff >> j) & 1:  # If j-th bit changed, update this source
                    state[j] = np.random.uniform(-1.0, 1.0) * 0.5
                total += state[j]  # Sum contributions from all sources

            # Add new white sample for highest frequency component
            total += white_samples[i]
            # Store the result, normalizing by number of sources + white sample
            pink_noise_out[i, channel] = total / 17.0

    # Normalize the final output to be within [-1.0, 1.0]
    max_abs_val = np.max(np.abs(pink_noise_out))
    if max_abs_val > 1e-6:  # Avoid division by zero
        pink_noise_out /= max_abs_val

    return pink_noise_out.astype(np.float32)


def generate_brown_noise(num_samples: int) -> AudioData:
    """Generates stereo brown noise (by integrating white noise)."""
    logger.debug(f"Generating {num_samples} samples of brown noise.")
    if num_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Start with white noise
    white = generate_white_noise(num_samples)

    # Integrate the white noise (cumulative sum) along the time axis (axis=0)
    brown_approx = np.cumsum(white, axis=0)

    # Normalize the result to be within [-1.0, 1.0]
    # Find the maximum absolute value across the entire signal
    max_abs_val = np.max(np.abs(brown_approx))
    if max_abs_val > 1e-6:  # Avoid division by zero for silent input
        brown_approx /= max_abs_val

    return brown_approx.astype(np.float32)


def generate_noise(
    noise_type: str, duration: float, sr: SampleRate, volume: float
) -> Optional[AudioData]:
    """
    Generates a specified type of noise (White, Pink, Brown).

    Args:
        noise_type: The type of noise ("White Noise", "Pink Noise", "Brown Noise").
        duration: Duration of the noise in seconds.
        sr: The sample rate.
        volume: The volume multiplier (0.0 to 1.0).

    Returns:
        The generated noise audio data, or None if type is unknown or duration is invalid.
    """
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

    # This part should ideally not be reached if logic is correct
    logger.error(f"Noise generation failed unexpectedly for type: {noise_type}")
    return None
