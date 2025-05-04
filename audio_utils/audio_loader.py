# audio_loader.py
# ==========================================
# Loads or regenerates full audio data based on source information.
# (No caching added here; relies on caching in called functions)
# ==========================================

import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as np

# Removed: import streamlit as st # No longer needed for st.error
# Import functions that ARE cached or will be
from audio_utils.audio_generators import (
    generate_binaural_beats,
    generate_isochronic_tones,
    generate_noise,
    generate_solfeggio_frequency,
)
from audio_utils.audio_io import load_audio  # This is cached

# Import constants and types
from config import GLOBAL_SR

# Import Base TTS interface (the generate method within should be cached)
from tts.base_tts import BaseTTSGenerator

if TYPE_CHECKING:
    from audio_utils.audio_state_definitions import (
        AudioData,
        SampleRate,
        SourceInfo,
        SourceInfoFrequency,
        SourceInfoNoise,
        SourceInfoTTS,
        SourceInfoUpload,
    )

# Get a logger for this module
logger = logging.getLogger(__name__)


def load_or_regenerate_audio(
    source_info: Optional["SourceInfo"],
    tts_generator: BaseTTSGenerator,
    sr_hint: int = GLOBAL_SR,
    required_duration_samples: Optional[int] = None,
) -> Optional[Tuple["AudioData", "SampleRate"]]:
    """
    Loads or regenerates the FULL audio data based on source_info.
    Relies on caching within the called functions (load_audio, generate_noise, etc.).

    Args:
        source_info: The dictionary containing source type and parameters.
        tts_generator: An initialized instance conforming to BaseTTSGenerator.
        sr_hint: The expected sample rate (from track data).
        required_duration_samples: Target duration in samples for generated tracks.

    Returns:
        A tuple containing (AudioData, sample_rate), or None if loading/generation fails.
    """
    if not source_info:
        logger.error("load_or_regenerate_audio: Missing source_info.")
        return None
    if (
        not tts_generator and source_info.get("type") == "tts"
    ):  # Check generator only if needed
        logger.error(
            "load_or_regenerate_audio: Missing tts_generator instance for TTS track."
        )
        return None

    source_type = source_info.get("type")
    full_audio: Optional["AudioData"] = None
    final_sr: int = sr_hint

    logger.info(
        f"load_or_regenerate_audio: Loading/Regenerating full audio (Type: {source_type})"
    )

    try:
        if source_type == "upload":
            upload_info = cast("SourceInfoUpload", source_info)
            temp_file_path = upload_info.get("temp_file_path")
            upload_info.get("original_filename", "Unknown File")
            if temp_file_path and os.path.exists(temp_file_path):
                # Call the cached load_audio function
                full_audio_tuple = load_audio(
                    temp_file_path, target_sr=sr_hint, duration=None
                )
                if full_audio_tuple:
                    full_audio, loaded_sr = full_audio_tuple
                    if loaded_sr:
                        final_sr = loaded_sr
                    else:  # Should not happen if load_audio returns tuple, but safety check
                        full_audio = None
                else:
                    full_audio = None  # load_audio failed
            else:
                logger.error(
                    f"Temporary upload file not found or inaccessible: {temp_file_path}"
                )
                # Removed: st.error(...)
                return None

        elif source_type == "tts":
            tts_info = cast("SourceInfoTTS", source_info)
            text = tts_info.get("text")
            if text:
                logger.info("Regenerating TTS audio using provided generator...")
                # Call the cached generate method of the TTS generator
                full_audio_tuple = tts_generator.generate(text)
                if full_audio_tuple:
                    full_audio, loaded_sr = full_audio_tuple
                    if loaded_sr and loaded_sr != sr_hint:
                        logger.warning(
                            f"TTS generated audio at {loaded_sr}Hz, but expected {sr_hint}Hz. Check TTS implementation."
                        )
                        final_sr = loaded_sr  # Trust the generator's SR
                    elif loaded_sr:
                        final_sr = loaded_sr
                    else:  # Should not happen if generate returns tuple
                        full_audio = None
                else:  # tts_generator.generate failed
                    full_audio = None

                if full_audio is None:
                    logger.error("TTS regeneration failed.")
                    # Removed: st.error(...)
                    return None
            else:
                logger.error("Missing text in source_info for TTS track.")
                return None

        elif source_type == "noise":
            noise_info = cast("SourceInfoNoise", source_info)
            noise_type = noise_info.get("noise_type")
            target_duration_s = noise_info.get("target_duration_s")
            duration_s = (
                (required_duration_samples / sr_hint)
                if required_duration_samples is not None and sr_hint > 0
                else target_duration_s
            )
            # Provide a default duration if none is available
            if duration_s is None or duration_s <= 0:
                logger.warning(
                    f"Invalid or missing duration for noise generation. Defaulting to 300s."
                )
                duration_s = 300
            if noise_type:
                # Call the cached generate_noise function
                full_audio = generate_noise(
                    noise_type, duration_s, sr_hint, volume=1.0
                )  # Volume applied later in pipeline/mix
                final_sr = sr_hint
            else:
                logger.error("Missing noise_type for noise track.")
                return None

        elif source_type == "frequency":
            freq_info = cast("SourceInfoFrequency", source_info)
            freq_type = freq_info.get("freq_type")
            target_duration_s = freq_info.get("target_duration_s")
            duration_s = (
                (required_duration_samples / sr_hint)
                if required_duration_samples is not None and sr_hint > 0
                else target_duration_s
            )
            # Provide a default duration if none is available
            if duration_s is None or duration_s <= 0:
                logger.warning(
                    f"Invalid or missing duration for frequency generation. Defaulting to 300s."
                )
                duration_s = 300
            if freq_type:
                gen_volume = 1.0  # Volume applied later in pipeline/mix
                # Call the cached frequency generation functions
                if freq_type == "binaural":
                    f_left = freq_info.get("f_left")
                    f_right = freq_info.get("f_right")
                    if f_left is not None and f_right is not None:
                        full_audio = generate_binaural_beats(
                            duration_s, f_left, f_right, sr_hint, gen_volume
                        )
                    else:
                        logger.error("Missing binaural params")
                        return None
                elif freq_type == "isochronic":
                    carrier = freq_info.get("carrier")
                    pulse = freq_info.get("pulse")
                    if carrier is not None and pulse is not None:
                        full_audio = generate_isochronic_tones(
                            duration_s, carrier, pulse, sr_hint, gen_volume
                        )
                    else:
                        logger.error("Missing isochronic params")
                        return None
                elif freq_type == "solfeggio":
                    freq = freq_info.get("freq")
                    if freq is not None:
                        full_audio = generate_solfeggio_frequency(
                            duration_s, freq, sr_hint, gen_volume
                        )
                    else:
                        logger.error("Missing solfeggio params")
                        return None
                else:
                    logger.error(f"Unknown frequency type '{freq_type}'.")
                    return None
                final_sr = sr_hint  # Generators use the provided sr_hint
            else:
                logger.error("Missing freq_type for frequency track.")
                return None

        else:
            logger.error(f"Unknown source_type '{source_type}'.")
            return None

    except Exception as e:
        logger.exception(f"Error processing source_type '{source_type}'")
        # Removed: st.error(...)
        return None

    # Final checks
    if full_audio is None:
        logger.error(f"Failed to get full audio data (Type: {source_type}).")
        return None
    if full_audio.size == 0:
        logger.warning(f"Loaded/generated audio is empty (Type: {source_type}).")
        # Return empty audio and SR, don't treat as failure
        return full_audio.astype(np.float32), final_sr

    logger.info(
        f"Successfully retrieved full audio (Type: {source_type}, {len(full_audio) / final_sr:.2f}s)"
    )
    return full_audio.astype(np.float32), final_sr
