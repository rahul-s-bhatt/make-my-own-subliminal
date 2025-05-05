# audio_loader.py
# ==========================================
# Loads or regenerates audio data based on source information.
# STEP 2 OPTIMIZED: Added get_or_generate_preview_snippet function.
# ==========================================

import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as np
import streamlit as st  # Import streamlit for caching

# Import functions that ARE cached or will be
from audio_utils.audio_generators import (
    generate_binaural_beats,
    generate_isochronic_tones,
    generate_noise,
    generate_solfeggio_frequency,
)
from audio_utils.audio_io import load_audio  # This is cached
from config import (
    GLOBAL_SR,  # Use config for default duration
    TRACK_SNIPPET_DURATION_S,
)
from tts.base_tts import BaseTTSGenerator

# Import necessary types and classes
if TYPE_CHECKING:
    from app_state import AppState  # Import AppState for type hinting
    from audio_utils.audio_state_definitions import TrackID  # Add TrackID
    from audio_utils.audio_state_definitions import (
        AudioData,
        SampleRate,
        SourceInfo,
        SourceInfoFrequency,
        SourceInfoNoise,
        SourceInfoTTS,
        SourceInfoUpload,
    )

logger = logging.getLogger(__name__)


# --- NEW FUNCTION: Get or Generate Preview Snippet ---
# @st.cache_data # Caching might be complex due to AppState/TTSGenerator args. Rely on AppState storage for now.
def get_or_generate_preview_snippet(
    track_id: "TrackID",
    app_state: "AppState",
    tts_generator: Optional[BaseTTSGenerator],  # Make optional as not all types need it
    preview_duration_s: float = TRACK_SNIPPET_DURATION_S,
) -> Optional[Tuple["AudioData", "SampleRate"]]:
    """
    Retrieves the cached raw audio snippet for a track, or loads/generates
    it on demand if not already cached in the AppState.

    Args:
        track_id: The ID of the track.
        app_state: The application state manager instance.
        tts_generator: An initialized TTS generator instance (required for TTS tracks).
        preview_duration_s: The desired duration of the snippet in seconds.

    Returns:
        A tuple containing (AudioData, sample_rate) for the raw snippet,
        or None if the snippet cannot be retrieved or generated.
    """
    log_prefix = f"Snippet for Track {track_id[-6:]}"
    logger.debug(
        f"{log_prefix}: Requesting preview snippet (max {preview_duration_s}s)."
    )

    # 1. Check AppState cache first
    cached_snippet = app_state.get_track_snippet(track_id)
    track_data = app_state.get_track(track_id)  # Get full track data once

    if track_data is None:
        logger.error(f"{log_prefix}: Track data not found in state.")
        return None

    cached_sr = track_data.get("sr")

    if cached_snippet is not None and cached_sr is not None:
        # Basic duration check - might need regeneration if duration setting changed
        # For simplicity, we assume if snippet exists, it's valid for now.
        # More robust check could compare expected length vs cached length.
        logger.debug(f"{log_prefix}: Found cached snippet in AppState.")
        return cached_snippet, cached_sr

    # 2. If not cached, load/generate based on source_info
    logger.info(f"{log_prefix}: Snippet not cached. Loading/generating...")
    source_info = track_data.get("source_info")
    if not source_info:
        logger.error(f"{log_prefix}: Missing source_info. Cannot generate snippet.")
        return None

    source_type = source_info.get("type")
    generated_snippet: Optional["AudioData"] = None
    generated_sr: Optional["SampleRate"] = None

    try:
        if source_type == "upload":
            upload_info = cast("SourceInfoUpload", source_info)
            temp_file_path = upload_info.get("temp_file_path")
            if temp_file_path and os.path.exists(temp_file_path):
                logger.debug(
                    f"{log_prefix}: Loading snippet from upload: {temp_file_path}"
                )
                # Use cached load_audio, but limit duration
                snippet_tuple = load_audio(
                    temp_file_path,
                    target_sr=GLOBAL_SR,  # Load towards global SR
                    duration=preview_duration_s,
                )
                if snippet_tuple:
                    generated_snippet, generated_sr = snippet_tuple
            else:
                logger.error(
                    f"{log_prefix}: Upload source file missing: {temp_file_path}"
                )
                # Optionally: Update track state to indicate missing file?

        elif source_type == "tts":
            if not tts_generator:
                logger.error(
                    f"{log_prefix}: TTS generator instance is required but missing."
                )
                return None
            tts_info = cast("SourceInfoTTS", source_info)
            text = tts_info.get("text")
            if text:
                logger.debug(f"{log_prefix}: Generating TTS snippet...")
                # Use cached generate method with max duration
                snippet_tuple = tts_generator.generate(
                    text, max_duration_s=preview_duration_s
                )
                if snippet_tuple:
                    generated_snippet, generated_sr = snippet_tuple
            else:
                logger.error(
                    f"{log_prefix}: Missing text in source_info for TTS snippet."
                )

        elif source_type == "noise":
            noise_info = cast("SourceInfoNoise", source_info)
            noise_type = noise_info.get("noise_type")
            if noise_type:
                logger.debug(
                    f"{log_prefix}: Generating Noise snippet ({noise_type})..."
                )
                # Use cached generate_noise
                generated_snippet = generate_noise(
                    noise_type, preview_duration_s, GLOBAL_SR, volume=1.0
                )
                generated_sr = GLOBAL_SR
            else:
                logger.error(f"{log_prefix}: Missing noise_type for noise snippet.")

        elif source_type == "frequency":
            freq_info = cast("SourceInfoFrequency", source_info)
            freq_type = freq_info.get("freq_type")
            gen_volume = 1.0  # Volume applied later
            if freq_type:
                logger.debug(
                    f"{log_prefix}: Generating Frequency snippet ({freq_type})..."
                )
                # Use cached frequency generators
                if freq_type == "binaural":
                    f_left = freq_info.get("f_left")
                    f_right = freq_info.get("f_right")
                    if f_left is not None and f_right is not None:
                        generated_snippet = generate_binaural_beats(
                            preview_duration_s, f_left, f_right, GLOBAL_SR, gen_volume
                        )
                    else:
                        logger.error(f"{log_prefix}: Missing binaural params")
                elif freq_type == "isochronic":
                    carrier = freq_info.get("carrier")
                    pulse = freq_info.get("pulse")
                    if carrier is not None and pulse is not None:
                        generated_snippet = generate_isochronic_tones(
                            preview_duration_s, carrier, pulse, GLOBAL_SR, gen_volume
                        )
                    else:
                        logger.error(f"{log_prefix}: Missing isochronic params")
                elif freq_type == "solfeggio":
                    freq = freq_info.get("freq")
                    if freq is not None:
                        generated_snippet = generate_solfeggio_frequency(
                            preview_duration_s, freq, GLOBAL_SR, gen_volume
                        )
                    else:
                        logger.error(f"{log_prefix}: Missing solfeggio params")
                else:
                    logger.error(f"{log_prefix}: Unknown frequency type '{freq_type}'.")
                generated_sr = GLOBAL_SR  # Generators use GLOBAL_SR
            else:
                logger.error(f"{log_prefix}: Missing freq_type for frequency snippet.")

        else:
            logger.error(
                f"{log_prefix}: Unknown source_type '{source_type}' for snippet generation."
            )

    except Exception as e:
        logger.exception(
            f"{log_prefix}: Error generating snippet for source_type '{source_type}'"
        )
        return None  # Fail gracefully

    # 3. Check result and update AppState cache
    if (
        generated_snippet is not None
        and generated_sr is not None
        and generated_snippet.size > 0
    ):
        logger.info(
            f"{log_prefix}: Snippet generated successfully ({generated_snippet.shape}, {generated_sr}Hz). Caching in AppState."
        )
        # Ensure float32 before caching
        generated_snippet = generated_snippet.astype(np.float32)
        app_state.update_track_snippet(track_id, generated_snippet, generated_sr)
        return generated_snippet, generated_sr
    elif generated_snippet is not None:  # Empty snippet generated
        logger.warning(
            f"{log_prefix}: Generated snippet is empty (Type: {source_type})."
        )
        # Cache the empty snippet? Or return None? Returning None might trigger regeneration next time.
        # Let's cache the empty result to avoid repeated failed attempts.
        app_state.update_track_snippet(
            track_id, generated_snippet, generated_sr
        )  # generated_sr might be None here
        return generated_snippet, (
            generated_sr if generated_sr else GLOBAL_SR
        )  # Return empty array and best guess SR
    else:  # Generation failed
        logger.error(f"{log_prefix}: Failed to generate snippet (Type: {source_type}).")
        return None


# --- Load or Regenerate FULL Audio (Remains mostly the same) ---
def load_or_regenerate_audio(
    source_info: Optional["SourceInfo"],
    tts_generator: Optional[BaseTTSGenerator],  # Make optional
    sr_hint: int = GLOBAL_SR,  # Keep sr_hint, might be from track state if snippet existed
    required_duration_samples: Optional[int] = None,
) -> Optional[Tuple["AudioData", "SampleRate"]]:
    """
    Loads or regenerates the FULL audio data based on source_info.
    Relies on caching within the called functions (load_audio, generate_noise, etc.).

    Args:
        source_info: The dictionary containing source type and parameters.
        tts_generator: An initialized instance conforming to BaseTTSGenerator (required for TTS).
        sr_hint: The expected sample rate (e.g., from track data or GLOBAL_SR).
        required_duration_samples: Target duration in samples for generated tracks.

    Returns:
        A tuple containing (AudioData, sample_rate), or None if loading/generation fails.
    """
    if not source_info:
        logger.error("load_or_regenerate_audio: Missing source_info.")
        return None

    source_type = source_info.get("type")
    # Check TTS generator only if needed
    if source_type == "tts" and not tts_generator:
        logger.error(
            "load_or_regenerate_audio: Missing tts_generator instance for TTS track."
        )
        return None

    full_audio: Optional["AudioData"] = None
    final_sr: int = sr_hint if sr_hint > 0 else GLOBAL_SR  # Use hint or default

    logger.info(
        f"load_or_regenerate_audio: Loading/Regenerating full audio (Type: {source_type})"
    )

    try:
        if source_type == "upload":
            upload_info = cast("SourceInfoUpload", source_info)
            temp_file_path = upload_info.get("temp_file_path")
            if temp_file_path and os.path.exists(temp_file_path):
                # Call cached load_audio for FULL duration
                full_audio_tuple = load_audio(
                    temp_file_path,
                    target_sr=final_sr,  # Load towards final SR
                    duration=None,  # Load full file
                )
                if full_audio_tuple:
                    full_audio, loaded_sr = full_audio_tuple
                    final_sr = loaded_sr  # Update final_sr based on actual loaded rate
                else:
                    full_audio = None
            else:
                logger.error(f"Temporary upload file not found: {temp_file_path}")
                return None

        elif source_type == "tts":
            tts_info = cast("SourceInfoTTS", source_info)
            text = tts_info.get("text")
            if text and tts_generator:  # Check tts_generator again
                logger.info("Regenerating FULL TTS audio...")
                # Call cached generate method (NO max duration)
                full_audio_tuple = tts_generator.generate(text, max_duration_s=None)
                if full_audio_tuple:
                    full_audio, loaded_sr = full_audio_tuple
                    final_sr = loaded_sr  # Trust the generator's SR
                else:
                    full_audio = None

                if full_audio is None:
                    logger.error("FULL TTS regeneration failed.")
                    return None
            elif not text:
                logger.error("Missing text in source_info for TTS track.")
                return None
            # else: # tts_generator is None, already checked

        elif source_type == "noise":
            noise_info = cast("SourceInfoNoise", source_info)
            noise_type = noise_info.get("noise_type")
            target_duration_s = noise_info.get("target_duration_s")
            # Determine duration for full generation
            duration_s = (
                (required_duration_samples / final_sr)
                if required_duration_samples is not None and final_sr > 0
                else target_duration_s
            )
            if duration_s is None or duration_s <= 0:
                duration_s = 300  # Default
            if noise_type:
                # Call cached generate_noise
                full_audio = generate_noise(
                    noise_type, duration_s, final_sr, volume=1.0
                )
                # final_sr remains unchanged
            else:
                logger.error("Missing noise_type for noise track.")
                return None

        elif source_type == "frequency":
            freq_info = cast("SourceInfoFrequency", source_info)
            freq_type = freq_info.get("freq_type")
            target_duration_s = freq_info.get("target_duration_s")
            # Determine duration for full generation
            duration_s = (
                (required_duration_samples / final_sr)
                if required_duration_samples is not None and final_sr > 0
                else target_duration_s
            )
            if duration_s is None or duration_s <= 0:
                duration_s = 300  # Default
            if freq_type:
                gen_volume = 1.0
                # Call cached frequency generators
                if freq_type == "binaural":
                    f_left = freq_info.get("f_left")
                    f_right = freq_info.get("f_right")
                    if f_left is not None and f_right is not None:
                        full_audio = generate_binaural_beats(
                            duration_s, f_left, f_right, final_sr, gen_volume
                        )
                    else:
                        logger.error("Missing binaural params")
                        return None
                elif freq_type == "isochronic":
                    carrier = freq_info.get("carrier")
                    pulse = freq_info.get("pulse")
                    if carrier is not None and pulse is not None:
                        full_audio = generate_isochronic_tones(
                            duration_s, carrier, pulse, final_sr, gen_volume
                        )
                    else:
                        logger.error("Missing isochronic params")
                        return None
                elif freq_type == "solfeggio":
                    freq = freq_info.get("freq")
                    if freq is not None:
                        full_audio = generate_solfeggio_frequency(
                            duration_s, freq, final_sr, gen_volume
                        )
                    else:
                        logger.error("Missing solfeggio params")
                        return None
                else:
                    logger.error(f"Unknown frequency type '{freq_type}'.")
                    return None
                # final_sr remains unchanged
            else:
                logger.error("Missing freq_type for frequency track.")
                return None

        else:
            logger.error(f"Unknown source_type '{source_type}'.")
            return None

    except Exception as e:
        logger.exception(f"Error processing source_type '{source_type}' for full audio")
        return None

    # Final checks
    if full_audio is None:
        logger.error(f"Failed to get full audio data (Type: {source_type}).")
        return None
    if full_audio.size == 0:
        logger.warning(f"Loaded/generated full audio is empty (Type: {source_type}).")
        # Return empty audio and SR
        return full_audio.astype(np.float32), final_sr

    logger.info(
        f"Successfully retrieved full audio (Type: {source_type}, {len(full_audio) / final_sr:.2f}s)"
    )
    return full_audio.astype(np.float32), final_sr
