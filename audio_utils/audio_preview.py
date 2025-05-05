# audio_preview.py
# ==========================================
# Generates audio previews for the Advanced Editor tracks.
# STEP 3 OPTIMIZED: Modified get_preview_audio to accept raw snippet and SR.
# ==========================================

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from audio_utils.audio_effects_pipeline import apply_all_effects

# --- MODIFIED: Added SampleRate ---
from audio_utils.audio_state_definitions import AudioData, SampleRate
from config import GLOBAL_SR, PREVIEW_DURATION_S  # Use config default

if TYPE_CHECKING:
    from app_state import TrackDataDict  # Keep this

logger = logging.getLogger(__name__)


# --- MODIFIED: Function Signature and Logic ---
def get_preview_audio(
    raw_snippet: Optional[AudioData],
    sr: Optional[SampleRate],
    track_data: "TrackDataDict",  # Still need track_data for effects settings
    preview_duration_s: int = PREVIEW_DURATION_S,
) -> Optional[AudioData]:
    """
    Generates a *processed* preview (with effects, vol, pan) from a raw audio snippet.
    Relies on caching within apply_all_effects.

    Args:
        raw_snippet: The raw audio data snippet (NumPy array).
        sr: The sample rate of the raw snippet.
        track_data: Dictionary containing track settings (effects, vol, pan, etc.).
        preview_duration_s: The maximum duration of the preview in seconds.

    Returns:
        The processed preview audio data, or None if generation fails or input is invalid.
    """
    track_name = track_data.get("name", "N/A")
    track_id = track_data.get("id", "N/A")  # Use track_id from track_data
    log_prefix = f"Processed Preview for '{track_name}' ({track_id[-6:]})"
    logger.info(
        f"{log_prefix}: Generating processed preview (max {preview_duration_s}s)"
    )

    # Use the passed-in raw_snippet and sr
    if raw_snippet is None or raw_snippet.size == 0:
        logger.warning(
            f"{log_prefix}: No raw audio snippet provided. Cannot generate preview."
        )
        return None
    if sr is None or sr <= 0:
        logger.warning(
            f"{log_prefix}: Invalid sample rate ({sr}). Cannot generate preview."
        )
        return None

    try:
        # Apply all effects (Reverse, Speed, Pitch/Ultrasonic, Filter) to the raw snippet
        # This function uses cached individual effect functions internally
        logger.debug(f"{log_prefix}: Applying effects to raw snippet.")
        # Pass the raw snippet directly to the effects pipeline
        processed_preview = apply_all_effects(track_data, audio_segment=raw_snippet)

        if processed_preview is None or processed_preview.size == 0:
            logger.warning(
                f"{log_prefix}: Effects resulted in empty audio. Cannot generate preview."
            )
            return None

        # --- Duration handling (Loop/Truncate/Pad) remains the same ---
        target_preview_samples = int(sr * preview_duration_s)
        current_processed_len = len(processed_preview)

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
                    f"{log_prefix}: Error concatenating looped preview: {e_concat}. Using non-looped/padded."
                )
                # Fallback padding/truncating
                if len(processed_preview) > target_preview_samples:
                    processed_preview = processed_preview[:target_preview_samples]
                elif len(processed_preview) < target_preview_samples:
                    processed_preview = np.pad(
                        processed_preview,
                        ((0, target_preview_samples - len(processed_preview)), (0, 0)),
                        mode="constant",
                    )

        # Truncate/Pad if needed (handles non-looped or failed loops)
        if len(processed_preview) > target_preview_samples:
            logger.debug(
                f"{log_prefix}: Truncating preview to {target_preview_samples} samples."
            )
            processed_preview = processed_preview[:target_preview_samples]
        elif len(processed_preview) < target_preview_samples:
            logger.debug(
                f"{log_prefix}: Padding preview to {target_preview_samples} samples."
            )
            processed_preview = np.pad(
                processed_preview,
                ((0, target_preview_samples - len(processed_preview)), (0, 0)),
                mode="constant",
            )

        # --- Apply Volume and Pan (Remains the same) ---
        vol = track_data.get("volume", 1.0)
        pan = track_data.get("pan", 0.0)
        logger.debug(f"{log_prefix}: Applying Volume ({vol:.2f}) / Pan ({pan:.2f}).")
        pan_rad = (pan + 1.0) * np.pi / 4.0
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)

        if processed_preview.ndim == 1:
            logger.warning(
                f"{log_prefix}: Preview is mono after effects. Applying vol, pan ignored."
            )
            processed_preview *= vol
            processed_preview = np.stack([processed_preview, processed_preview], axis=1)
        elif processed_preview.shape[1] == 2:
            processed_preview[:, 0] *= left_gain
            processed_preview[:, 1] *= right_gain
        else:
            logger.warning(
                f"{log_prefix}: Unexpected shape {processed_preview.shape}. Cannot apply vol/pan."
            )

        # Final clip and type check
        processed_preview = np.clip(processed_preview, -1.0, 1.0)
        logger.debug(
            f"{log_prefix}: Processed preview generation complete. Final shape: {processed_preview.shape}"
        )
        return processed_preview.astype(np.float32)

    except Exception as e:
        logger.exception(f"{log_prefix}: Error generating processed preview.")
        return None
