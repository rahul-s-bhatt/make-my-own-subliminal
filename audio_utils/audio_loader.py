# audio_loader.py
# ==========================================
# Loads or regenerates full audio data based on source information.
# ==========================================

import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as np
import streamlit as st  # For st.error display

from audio_utils.audio_generators import (
    generate_binaural_beats,
    generate_isochronic_tones,
    generate_noise,
    generate_solfeggio_frequency,
)

# Import necessary components
from audio_utils.audio_io import load_audio
from config import GLOBAL_SR
from tts_generator import TTSGenerator  # Assuming simple instantiation is okay

# Import type definitions
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
    sr_hint: int = GLOBAL_SR,  # Use SR from track data as hint
    required_duration_samples: Optional[int] = None,
) -> Optional[Tuple["AudioData", "SampleRate"]]:
    """
    Loads or regenerates the FULL audio data based on source_info.

    Args:
        source_info: The dictionary containing source type and parameters.
        sr_hint: The expected sample rate (from track data). Used for resampling uploads
                 and as the target for generation.
        required_duration_samples: The target duration in samples needed for looping
                                   generated tracks (noise, frequency). If None, uses
                                   a default or inherent duration from source_info.

    Returns:
        A tuple containing (AudioData, sample_rate), or None if loading/generation fails.
    """
    if not source_info:
        logger.error("load_or_regenerate_audio: Missing source_info.")
        return None

    source_type = source_info.get("type")
    full_audio: Optional["AudioData"] = None
    final_sr: int = sr_hint  # Assume target SR unless loaded audio differs

    logger.info(
        f"load_or_regenerate_audio: Loading/Regenerating full audio (Type: {source_type})"
    )

    try:
        if source_type == "upload":
            # Cast for type checker happiness after checking 'type'
            upload_info = cast("SourceInfoUpload", source_info)
            temp_file_path = upload_info.get("temp_file_path")
            original_filename = upload_info.get("original_filename", "Unknown File")
            if temp_file_path and os.path.exists(temp_file_path):
                # Load full audio, resample to sr_hint if necessary
                full_audio, loaded_sr = load_audio(
                    temp_file_path, target_sr=sr_hint, duration=None
                )
                if loaded_sr:
                    final_sr = loaded_sr  # Use the actual loaded SR
                else:
                    full_audio = None  # Indicate failure if load_audio failed
            else:
                logger.error(
                    f"load_or_regenerate_audio: Temporary upload file not found for '{original_filename}' at path: {temp_file_path}"
                )
                st.error(
                    f"Source file '{original_filename}' missing. Please remove and re-add the track.",
                    icon="⚠️",
                )
                return None  # Critical error, cannot proceed

        elif source_type == "tts":
            tts_info = cast("SourceInfoTTS", source_info)
            text = tts_info.get("text")
            if text:
                tts_gen = TTSGenerator()  # Assuming simple instantiation works
                # TTS generator should ideally return audio at GLOBAL_SR
                full_audio, loaded_sr = tts_gen.generate(text)
                if loaded_sr and loaded_sr != sr_hint:
                    logger.warning(
                        f"TTS generated audio at {loaded_sr}Hz, expected {sr_hint}Hz. Resampling may occur later if needed."
                    )
                    # Store the actual SR from TTS
                    final_sr = loaded_sr
                elif loaded_sr:
                    final_sr = loaded_sr  # Store the confirmed SR
                # Handle potential failure from tts_gen.generate
                if full_audio is None:
                    logger.error("TTS generation failed.")
                    st.error("Text-to-Speech generation failed.")
                    return None

            else:
                logger.error(
                    "load_or_regenerate_audio: Missing text in source_info for TTS track."
                )
                return None

        elif source_type == "noise":
            noise_info = cast("SourceInfoNoise", source_info)
            noise_type = noise_info.get("noise_type")
            target_duration_s = noise_info.get("target_duration_s")
            # Use required_duration if provided (for looping), else use hint or default
            duration_s = (
                (required_duration_samples / sr_hint)
                if required_duration_samples is not None and sr_hint > 0
                else target_duration_s
            )
            if duration_s is None:
                duration_s = 300  # Default if no hint and no requirement

            if noise_type:
                # Generate noise at the target sample rate
                full_audio = generate_noise(noise_type, duration_s, sr_hint, volume=1.0)
                final_sr = sr_hint
            else:
                logger.error(
                    "load_or_regenerate_audio: Missing noise_type for noise track."
                )
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
            if duration_s is None:
                duration_s = 300  # Default

            if freq_type:
                gen_volume = 1.0  # Generate at full amplitude
                # Generate frequencies at the target sample rate
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
                    logger.error(
                        f"load_or_regenerate_audio: Unknown frequency type '{freq_type}'."
                    )
                    return None
                final_sr = sr_hint  # Generation uses the target SR
            else:
                logger.error(
                    "load_or_regenerate_audio: Missing freq_type for frequency track."
                )
                return None

        else:
            logger.error(
                f"load_or_regenerate_audio: Unknown source_type '{source_type}'."
            )
            return None

    except Exception as e:
        logger.exception(
            f"load_or_regenerate_audio: Error processing source_type '{source_type}'"
        )
        st.error(f"Error preparing audio: {e}", icon="🔥")
        return None

    # Final check if audio data was successfully obtained
    if full_audio is None:
        logger.error(
            f"load_or_regenerate_audio: Failed to get full audio data (Type: {source_type})."
        )
        return None
    if full_audio.size == 0:
        logger.warning(
            f"load_or_regenerate_audio: Loaded/generated audio is empty (Type: {source_type})."
        )
        # Return empty array but with correct SR
        return full_audio.astype(np.float32), final_sr

    logger.info(
        f"load_or_regenerate_audio: Successfully retrieved full audio (Type: {source_type}, {len(full_audio) / final_sr:.2f}s)"
    )
    # Ensure float32 output
    return full_audio.astype(np.float32), final_sr
