# audio_preview.py
# ==========================================
# Generates audio previews for the Advanced Editor tracks.
# (No caching added here; relies on caching in called functions)
# ==========================================

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import streamlit as st  # Import for consistency, though not used for caching here

# Import the effects pipeline (relies on cached effects)
from audio_utils.audio_effects_pipeline import apply_all_effects

# Import constants and types
from config import GLOBAL_SR, PREVIEW_DURATION_S

# Define type hints used within this module
AudioData = np.ndarray
if TYPE_CHECKING:
    try:
        # Assuming TrackDataDict might be defined elsewhere, e.g., app_state
        from app_state import TrackDataDict
    except ImportError:
        # Fallback if TrackDataDict isn't easily importable or defined centrally
        from typing import Any, Dict

        TrackDataDict = Dict[str, Any]


# Get a logger for this module
logger = logging.getLogger(__name__)


# No caching added - relies on cached apply_all_effects
def get_preview_audio(
    track_data: "TrackDataDict", preview_duration_s: int = PREVIEW_DURATION_S
) -> Optional[AudioData]:
    """
    Generates a preview of the track using its snippet with effects, volume, and pan applied.
    Relies on caching within apply_all_effects (via cached individual effects).

    Args:
        track_data: Dictionary containing track settings and the audio snippet.
        preview_duration_s: The maximum duration of the preview in seconds.

    Returns:
        The processed preview audio data, or None if generation fails or input is invalid.
    """
    track_name = track_data.get("name", "N/A")
    track_id = track_data.get("track_id", "N/A")  # Use track_id if available
    log_prefix = f"Preview for '{track_name}' ({track_id})"
    logger.info(f"{log_prefix}: Generating preview (max {preview_duration_s}s)")

    audio_snippet = track_data.get("audio_snippet")
    sr = track_data.get("sr", GLOBAL_SR)

    if audio_snippet is None or audio_snippet.size == 0:
        logger.warning(
            f"{log_prefix}: No audio snippet found. Cannot generate preview."
        )
        return None
    if sr <= 0:
        logger.warning(
            f"{log_prefix}: Invalid sample rate ({sr}). Cannot generate preview."
        )
        return None

    try:
        # Apply all effects (Reverse, Speed, Pitch/Ultrasonic, Filter) to the snippet
        # This function uses cached individual effect functions internally
        logger.debug(f"{log_prefix}: Applying effects to snippet.")
        processed_preview = apply_all_effects(track_data, audio_segment=audio_snippet)

        if processed_preview is None or processed_preview.size == 0:
            logger.warning(
                f"{log_prefix}: Effects resulted in empty audio. Cannot generate preview."
            )
            return None

        # Determine Preview Length after Effects
        target_preview_samples = int(sr * preview_duration_s)
        current_processed_len = len(processed_preview)

        # Loop snippet if needed and possible
        if (
            track_data.get("loop_to_fit", False)
            and current_processed_len > 0
            and current_processed_len < target_preview_samples
        ):
            logger.debug(
                f"{log_prefix}: Looping snippet ({current_processed_len} -> {target_preview_samples})"
            )
            num_repeats = target_preview_samples // current_processed_len
            remainder = target_preview_samples % current_processed_len
            looped_list = [processed_preview] * num_repeats
            if remainder > 0:
                looped_list.append(processed_preview[:remainder])
            try:
                processed_preview = np.concatenate(looped_list, axis=0)
                logger.debug(
                    f"{log_prefix}: Looping complete. New length: {len(processed_preview)}."
                )
            except ValueError as e_concat:
                logger.error(
                    f"{log_prefix}: Error concatenating looped preview: {e_concat}. Using non-looped."
                )
                # Fallback to padding/truncating original processed snippet
                if len(processed_preview) > target_preview_samples:
                    processed_preview = processed_preview[:target_preview_samples]
                elif len(processed_preview) < target_preview_samples:
                    processed_preview = np.pad(
                        processed_preview,
                        ((0, target_preview_samples - len(processed_preview)), (0, 0)),
                        mode="constant",
                    )

        # Truncate if longer than target (handles non-looped or failed loop cases)
        if len(processed_preview) > target_preview_samples:
            logger.debug(
                f"{log_prefix}: Truncating preview to {target_preview_samples} samples."
            )
            processed_preview = processed_preview[:target_preview_samples]
        # Pad if shorter than target (handles non-looped short snippets or failed loops)
        elif len(processed_preview) < target_preview_samples:
            logger.debug(
                f"{log_prefix}: Padding preview to {target_preview_samples} samples."
            )
            processed_preview = np.pad(
                processed_preview,
                ((0, target_preview_samples - len(processed_preview)), (0, 0)),
                mode="constant",
            )

        # Apply Volume and Pan
        vol = track_data.get("volume", 1.0)
        pan = track_data.get("pan", 0.0)
        logger.debug(f"{log_prefix}: Applying Volume ({vol:.2f}) / Pan ({pan:.2f}).")

        pan_rad = (pan + 1.0) * np.pi / 4.0  # Map [-1, 1] to [0, pi/2]
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)

        # Ensure stereo before applying gains
        if processed_preview.ndim == 1:
            logger.warning(
                f"{log_prefix}: Preview is mono after effects. Applying vol, pan ignored."
            )
            processed_preview *= vol  # Apply overall volume
            # Convert to stereo for consistent output shape
            processed_preview = np.stack([processed_preview, processed_preview], axis=1)
        elif processed_preview.shape[1] == 2:
            processed_preview[:, 0] *= left_gain
            processed_preview[:, 1] *= right_gain
        else:
            logger.warning(
                f"{log_prefix}: Unexpected shape {processed_preview.shape}. Cannot apply vol/pan."
            )
            # Attempt to apply volume to first channel as fallback? Or just skip?
            # Skipping might be safer.
            pass  # Skip applying vol/pan if shape is wrong

        # Final clip and type check
        processed_preview = np.clip(processed_preview, -1.0, 1.0)
        logger.debug(
            f"{log_prefix}: Preview generation complete. Final shape: {processed_preview.shape}"
        )
        return processed_preview.astype(np.float32)

    except Exception as e:
        logger.exception(f"{log_prefix}: Error generating preview.")
        return None
