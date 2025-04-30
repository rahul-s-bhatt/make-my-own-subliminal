# audio_preview.py
# ==========================================
# Generates audio previews for the Advanced Editor tracks.
# ==========================================

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

# Import the effects pipeline
from audio_effects_pipeline import apply_all_effects

# Import constants and types
from config import GLOBAL_SR, PREVIEW_DURATION_S

# Define type hints used within this module
AudioData = np.ndarray
if TYPE_CHECKING:
    try:
        from app_state import TrackDataDict
    except ImportError:
        from typing import Any, Dict

        TrackDataDict = Dict[str, Any]


# Get a logger for this module
logger = logging.getLogger(__name__)


def get_preview_audio(track_data: "TrackDataDict", preview_duration_s: int = PREVIEW_DURATION_S) -> Optional[AudioData]:
    """
    Generates a preview of the track using its snippet with effects, volume, and pan applied.

    Args:
        track_data: Dictionary containing track settings and the audio snippet.
        preview_duration_s: The maximum duration of the preview in seconds.

    Returns:
        The processed preview audio data, or None if generation fails or input is invalid.
    """
    track_name = track_data.get("name", "N/A")
    # Attempt to get track_id if it exists in the dict, otherwise use N/A
    track_id = track_data.get("track_id", "N/A")
    logger.info(f"Generating preview audio for track '{track_name}' ({track_id}) using snippet (max {preview_duration_s}s)")

    audio_snippet = track_data.get("audio_snippet")
    sr = track_data.get("sr", GLOBAL_SR)

    if audio_snippet is None or audio_snippet.size == 0:
        logger.warning(f"No audio snippet data found for track '{track_name}'. Cannot generate preview.")
        return None
    if sr <= 0:
        logger.warning(f"Invalid sample rate ({sr}) for track '{track_name}'. Cannot generate preview.")
        return None

    try:
        # Apply all effects (Reverse, Speed, Pitch/Ultrasonic, Filter) to the snippet
        logger.debug(f"Applying effects to preview snippet for '{track_name}'")
        processed_preview = apply_all_effects(track_data, audio_segment=audio_snippet)

        if processed_preview is None or processed_preview.size == 0:
            logger.warning(f"Applying effects resulted in empty audio for '{track_name}' preview.")
            return None

        # Determine Preview Length after Effects
        target_preview_samples = int(sr * preview_duration_s)
        current_processed_len = len(processed_preview)

        # Loop snippet if needed
        if track_data.get("loop_to_fit", False) and current_processed_len > 0 and current_processed_len < target_preview_samples:
            logger.debug(f"Looping processed snippet for '{track_name}' preview ({current_processed_len} -> {target_preview_samples})")
            num_repeats = target_preview_samples // current_processed_len
            remainder = target_preview_samples % current_processed_len
            looped_list = [processed_preview] * num_repeats
            if remainder > 0:
                looped_list.append(processed_preview[:remainder])
            try:
                processed_preview = np.concatenate(looped_list, axis=0)
                logger.debug(f"Looping complete for preview. New length: {len(processed_preview)}.")
            except ValueError as e_concat:
                logger.error(f"Error concatenating looped preview for '{track_name}': {e_concat}. Using non-looped.")
                if len(processed_preview) > target_preview_samples:
                    processed_preview = processed_preview[:target_preview_samples]
                elif len(processed_preview) < target_preview_samples:
                    processed_preview = np.pad(processed_preview, ((0, target_preview_samples - len(processed_preview)), (0, 0)), mode="constant")

        # Truncate if longer than target
        elif len(processed_preview) > target_preview_samples:
            logger.debug(f"Truncating processed preview for '{track_name}' to {target_preview_samples} samples.")
            processed_preview = processed_preview[:target_preview_samples]

        # Apply Volume and Pan
        vol = track_data.get("volume", 1.0)
        pan = track_data.get("pan", 0.0)
        logger.debug(f"Applying Volume ({vol:.2f}) / Pan ({pan:.2f}) to preview for '{track_name}'")

        pan_rad = (pan + 1.0) * np.pi / 4.0
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)

        if processed_preview.ndim == 2 and processed_preview.shape[1] == 2:
            processed_preview[:, 0] *= left_gain
            processed_preview[:, 1] *= right_gain
        elif processed_preview.ndim == 1:
            logger.warning(f"Preview for '{track_name}' is mono after effects. Applying volume, pan ignored.")
            processed_preview *= vol
            processed_preview = np.stack([processed_preview, processed_preview], axis=1)
        else:
            logger.warning(f"Processed preview for '{track_name}' has unexpected shape {processed_preview.shape}. Cannot apply volume/pan.")
            try:
                processed_preview[:, 0] *= vol
            except IndexError:
                pass

        processed_preview = np.clip(processed_preview, -1.0, 1.0)
        logger.debug(f"Preview generation complete for '{track_name}'. Final shape: {processed_preview.shape}")
        return processed_preview.astype(np.float32)

    except Exception as e:
        logger.exception(f"Error generating preview for track '{track_name}'")
        return None
