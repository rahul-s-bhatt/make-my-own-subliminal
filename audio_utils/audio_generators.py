# audio_generators.py
# ==========================================
# Audio Signal Generation Utilities for MindMorph (with Caching)
# ==========================================

import logging
from typing import Optional

import numpy as np
import streamlit as st  # Import Streamlit for caching decorators
from scipy import signal  # Used for pink noise generation approximation

# Define type hints used within this module
AudioData = np.ndarray
SampleRate = int

# Get a logger for this module
logger = logging.getLogger(__name__)

# ==========================================
# Tone Generation Functions
# ==========================================


@st.cache_data(show_spinner="Generating binaural beats...")
def generate_binaural_beats(
    duration: float, freq_left: float, freq_right: float, sr: SampleRate, volume: float
) -> AudioData:
    """
    Generates stereo binaural beats.
    Cached based on duration, frequencies, sample rate, and volume.
    """
    logger.info(
        f"Generating binaural beats: duration={duration}s, L={freq_left}Hz, R={freq_right}Hz, SR={sr}Hz, volume={volume:.2f}"
    )
    num_samples = int(sr * duration)
    if num_samples <= 0:
        logger.warning("Duration results in zero samples for binaural beats.")
        return np.zeros((0, 2), dtype=np.float32)  # Return empty array

    t = np.linspace(0, duration, num_samples, endpoint=False)
    left_channel = np.sin(2 * np.pi * freq_left * t)
    right_channel = np.sin(2 * np.pi * freq_right * t)

    audio = np.stack([left_channel, right_channel], axis=1) * volume
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


@st.cache_data(show_spinner="Generating Solfeggio tone...")
def generate_solfeggio_frequency(
    duration: float, freq: float, sr: SampleRate, volume: float
) -> AudioData:
    """
    Generates a stereo pure tone (Solfeggio).
    Cached based on duration, frequency, sample rate, and volume.
    """
    logger.info(
        f"Generating Solfeggio tone: duration={duration}s, F={freq}Hz, SR={sr}Hz, volume={volume:.2f}"
    )
    num_samples = int(sr * duration)
    if num_samples <= 0:
        logger.warning("Duration results in zero samples for Solfeggio tone.")
        return np.zeros((0, 2), dtype=np.float32)  # Return empty array

    t = np.linspace(0, duration, num_samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t)

    audio = np.stack([sine_wave, sine_wave], axis=1) * volume
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


@st.cache_data(show_spinner="Generating isochronic tones...")
def generate_isochronic_tones(
    duration: float,
    carrier_freq: float,
    pulse_freq: float,
    sr: SampleRate,
    volume: float,
) -> AudioData:
    """
    Generates stereo isochronic tones using amplitude modulation.
    Cached based on duration, frequencies, sample rate, and volume.
    """
    logger.info(
        f"Generating Isochronic tones: duration={duration}s, Carrier={carrier_freq}Hz, Pulse={pulse_freq}Hz, SR={sr}Hz, volume={volume:.2f}"
    )
    num_samples = int(sr * duration)
    if num_samples <= 0:
        logger.warning("Duration results in zero samples for Isochronic tones.")
        return np.zeros((0, 2), dtype=np.float32)  # Return empty array

    t = np.linspace(0, duration, num_samples, endpoint=False)
    carrier_wave = np.sin(2 * np.pi * carrier_freq * t)
    modulation_wave = 0.5 * (
        np.sign(np.sin(2 * np.pi * pulse_freq * t + np.pi / 2)) + 1
    )
    modulated_signal = carrier_wave * modulation_wave

    audio = np.stack([modulated_signal, modulated_signal], axis=1) * volume
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ==========================================
# Noise Generation Functions (Internal Helpers - Not Cached Directly)
# ==========================================
# Caching the individual helpers might add overhead if generate_noise is always called.
# We cache generate_noise instead.


def _generate_white_noise(num_samples: int) -> AudioData:
    """Generates stereo white noise (uniform distribution)."""
    logger.debug(f"Generating {num_samples} samples of white noise.")
    if num_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.random.uniform(-1.0, 1.0, size=(num_samples, 2)).astype(np.float32)


def _generate_pink_noise(num_samples: int) -> AudioData:
    """Generates stereo pink noise (Voss-McCartney approximation)."""
    logger.debug(f"Generating {num_samples} samples of pink noise.")
    if num_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    pink_noise_out = np.zeros((num_samples, 2), dtype=np.float32)
    for channel in range(2):
        state = np.zeros(16)
        key = 0
        white_samples = np.random.uniform(-1.0, 1.0, num_samples) * 0.5

        for i in range(num_samples):
            last_key = key
            key += 1
            diff = last_key ^ key
            total = 0
            for j in range(16):
                if (diff >> j) & 1:
                    state[j] = np.random.uniform(-1.0, 1.0) * 0.5
                total += state[j]

            total += white_samples[i]
            pink_noise_out[i, channel] = total / 17.0

    max_abs_val = np.max(np.abs(pink_noise_out))
    if max_abs_val > 1e-6:
        pink_noise_out /= max_abs_val

    return pink_noise_out.astype(np.float32)


def _generate_brown_noise(num_samples: int) -> AudioData:
    """Generates stereo brown noise (by integrating white noise)."""
    logger.debug(f"Generating {num_samples} samples of brown noise.")
    if num_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    white = _generate_white_noise(num_samples)  # Use internal helper
    brown_approx = np.cumsum(white, axis=0)

    max_abs_val = np.max(np.abs(brown_approx))
    if max_abs_val > 1e-6:
        brown_approx /= max_abs_val

    return brown_approx.astype(np.float32)


# ==========================================
# Main Noise Generation Function (Cached)
# ==========================================


@st.cache_data(show_spinner="Generating noise...")
def generate_noise(
    noise_type: str, duration: float, sr: SampleRate, volume: float
) -> Optional[AudioData]:
    """
    Generates a specified type of noise (White, Pink, Brown).
    Cached based on noise type, duration, sample rate, and volume.

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
        return None  # Return None for invalid duration

    noise_audio: Optional[AudioData] = None
    if noise_type == "White Noise":
        noise_audio = _generate_white_noise(num_samples)
    elif noise_type == "Pink Noise":
        noise_audio = _generate_pink_noise(num_samples)
    elif noise_type == "Brown Noise":
        noise_audio = _generate_brown_noise(num_samples)
    else:
        logger.error(f"Unknown noise type requested: {noise_type}")
        # Removed st.error call
        return None  # Return None for unknown type

    if noise_audio is not None:
        # Apply volume and clip
        audio_scaled = np.clip(noise_audio * volume, -1.0, 1.0)
        return audio_scaled.astype(np.float32)

    logger.error(f"Noise generation failed unexpectedly for type: {noise_type}")
    return None  # Should not be reached ideally
