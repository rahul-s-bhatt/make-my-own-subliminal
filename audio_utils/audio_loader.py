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

# --- MODIFIED IMPORT ---
# Import the Base TTS interface
from tts.base_tts import BaseTTSGenerator

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


# --- MODIFIED FUNCTION SIGNATURE ---
def load_or_regenerate_audio(
    source_info: Optional["SourceInfo"],
    tts_generator: BaseTTSGenerator,  # Added TTS generator instance argument
    sr_hint: int = GLOBAL_SR,
    required_duration_samples: Optional[int] = None,
) -> Optional[Tuple["AudioData", "SampleRate"]]:
    # --- END MODIFIED FUNCTION SIGNATURE ---
    """
    Loads or regenerates the FULL audio data based on source_info.

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
    # --- ADDED CHECK for tts_generator ---
    if not tts_generator:
        logger.error("load_or_regenerate_audio: Missing tts_generator instance.")
        # This function might be called for non-TTS tracks, but if it's a TTS track,
        # the generator is required. We'll check specifically in the TTS block.
        # return None # Or handle based on whether it's needed for the source_type
    # --- END ADDED CHECK ---

    source_type = source_info.get("type")
    full_audio: Optional["AudioData"] = None
    final_sr: int = sr_hint

    logger.info(f"load_or_regenerate_audio: Loading/Regenerating full audio (Type: {source_type})")

    try:
        if source_type == "upload":
            # (Upload logic remains the same)
            upload_info = cast("SourceInfoUpload", source_info)
            temp_file_path = upload_info.get("temp_file_path")
            original_filename = upload_info.get("original_filename", "Unknown File")
            if temp_file_path and os.path.exists(temp_file_path):
                full_audio, loaded_sr = load_audio(temp_file_path, target_sr=sr_hint, duration=None)
                if loaded_sr:
                    final_sr = loaded_sr
                else:
                    full_audio = None
            else:
                logger.error(f"Temporary upload file not found: {temp_file_path}")
                st.error(f"Source file '{original_filename}' missing. Please re-add track.", icon="⚠️")
                return None

        elif source_type == "tts":
            # --- MODIFIED TTS HANDLING ---
            # Check if TTS generator was provided (essential for this type)
            if not tts_generator:
                logger.error("load_or_regenerate_audio: TTS generator instance is required for TTS tracks but was not provided.")
                st.error("TTS engine not available to regenerate audio.", icon="🔥")
                return None

            tts_info = cast("SourceInfoTTS", source_info)
            text = tts_info.get("text")
            if text:
                logger.info("Regenerating TTS audio using provided generator...")
                # Use the passed-in tts_generator instance
                full_audio, loaded_sr = tts_generator.generate(text)
                # Remove: tts_gen = TTSGenerator()
                # Remove: full_audio, loaded_sr = tts_gen.generate(text)

                # The new generator should handle resampling internally to GLOBAL_SR
                if loaded_sr and loaded_sr != sr_hint:  # sr_hint should be GLOBAL_SR here
                    logger.warning(f"TTS generated audio at {loaded_sr}Hz, but expected {sr_hint}Hz. Check TTS implementation.")
                    # Trust the SR returned by the generator
                    final_sr = loaded_sr
                elif loaded_sr:
                    final_sr = loaded_sr
                else:  # Handle potential failure from tts_generator.generate
                    full_audio = None  # Indicate failure

                if full_audio is None:
                    logger.error("TTS regeneration failed.")
                    st.error("Text-to-Speech regeneration failed.")
                    return None
            else:
                logger.error("Missing text in source_info for TTS track.")
                return None
            # --- END MODIFIED TTS HANDLING ---

        elif source_type == "noise":
            # (Noise logic remains the same)
            noise_info = cast("SourceInfoNoise", source_info)
            noise_type = noise_info.get("noise_type")
            target_duration_s = noise_info.get("target_duration_s")
            duration_s = (required_duration_samples / sr_hint) if required_duration_samples is not None and sr_hint > 0 else target_duration_s
            if duration_s is None:
                duration_s = 300
            if noise_type:
                full_audio = generate_noise(noise_type, duration_s, sr_hint, volume=1.0)
                final_sr = sr_hint
            else:
                logger.error("Missing noise_type for noise track.")
                return None

        elif source_type == "frequency":
            # (Frequency logic remains the same)
            freq_info = cast("SourceInfoFrequency", source_info)
            freq_type = freq_info.get("freq_type")
            target_duration_s = freq_info.get("target_duration_s")
            duration_s = (required_duration_samples / sr_hint) if required_duration_samples is not None and sr_hint > 0 else target_duration_s
            if duration_s is None:
                duration_s = 300
            if freq_type:
                gen_volume = 1.0
                # ... (generation logic for different freq types) ...
                if freq_type == "binaural":
                    f_left = freq_info.get("f_left")
                    f_right = freq_info.get("f_right")
                    if f_left is not None and f_right is not None:
                        full_audio = generate_binaural_beats(duration_s, f_left, f_right, sr_hint, gen_volume)
                    else:
                        logger.error("Missing binaural params")
                        return None
                elif freq_type == "isochronic":
                    carrier = freq_info.get("carrier")
                    pulse = freq_info.get("pulse")
                    if carrier is not None and pulse is not None:
                        full_audio = generate_isochronic_tones(duration_s, carrier, pulse, sr_hint, gen_volume)
                    else:
                        logger.error("Missing isochronic params")
                        return None
                elif freq_type == "solfeggio":
                    freq = freq_info.get("freq")
                    if freq is not None:
                        full_audio = generate_solfeggio_frequency(duration_s, freq, sr_hint, gen_volume)
                    else:
                        logger.error("Missing solfeggio params")
                        return None
                else:
                    logger.error(f"Unknown frequency type '{freq_type}'.")
                    return None
                final_sr = sr_hint
            else:
                logger.error("Missing freq_type for frequency track.")
                return None

        else:
            logger.error(f"Unknown source_type '{source_type}'.")
            return None

    except Exception as e:
        logger.exception(f"Error processing source_type '{source_type}'")
        st.error(f"Error preparing audio: {e}", icon="🔥")
        return None

    # Final checks (remain the same)
    if full_audio is None:
        logger.error(f"Failed to get full audio data (Type: {source_type}).")
        return None
    if full_audio.size == 0:
        logger.warning(f"Loaded/generated audio is empty (Type: {source_type}).")
        return full_audio.astype(np.float32), final_sr

    logger.info(f"Successfully retrieved full audio (Type: {source_type}, {len(full_audio) / final_sr:.2f}s)")
    return full_audio.astype(np.float32), final_sr
