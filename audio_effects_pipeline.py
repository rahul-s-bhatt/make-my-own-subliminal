# audio_effects_pipeline.py
# ==========================================
# Applies a sequence of audio effects based on track settings.
# ==========================================

import logging
from typing import TYPE_CHECKING

import numpy as np

# Import individual effect functions
from audio_effects import apply_pitch_shift, apply_reverse, apply_speed_change, apply_standard_filter, apply_ultrasonic_shift

# Import constants and types
from config import GLOBAL_SR

# Define type hints used within this module
AudioData = np.ndarray
if TYPE_CHECKING:
    # Use TrackDataDict for type hinting if available, otherwise fallback
    try:
        from app_state import TrackDataDict
    except ImportError:
        from typing import Any, Dict

        TrackDataDict = Dict[str, Any]


# Get a logger for this module
logger = logging.getLogger(__name__)


def apply_all_effects(track_data: "TrackDataDict", audio_segment: AudioData) -> AudioData:
    """
    Applies effects sequentially based on track_data settings to the provided audio segment.
    Order: Reverse -> Speed -> Pitch/Ultrasonic -> Filter.

    Args:
        track_data: Dictionary containing track settings.
        audio_segment: The audio segment (snippet or full) to process.

    Returns:
        The processed audio data as a NumPy array. Returns empty array on error or if input is empty.

    Raises:
        ValueError: If audio_segment is None.
    """
    if audio_segment is None:
        raise ValueError("apply_all_effects requires an audio_segment.")

    track_name = track_data.get("name", "Unnamed Track")
    # Attempt to get track_id if it exists in the dict, otherwise use N/A
    track_id = track_data.get("track_id", "N/A")

    if audio_segment.size == 0:
        logger.warning(f"Input audio segment for effects is empty for '{track_name}'. Returning empty.")
        return audio_segment  # Return the empty array

    audio = audio_segment.copy()  # Work on a copy
    log_prefix = f"Applying effects to segment for '{track_name}' ({track_id})"

    # Get settings from track_data
    should_reverse = track_data.get("reverse_audio", False)
    use_ultrasonic = track_data.get("ultrasonic_shift", False)
    pitch_shift_steps = track_data.get("pitch_shift", 0.0)
    speed_factor = track_data.get("speed_factor", 1.0)
    filter_type = track_data.get("filter_type", "off")
    filter_cutoff = track_data.get("filter_cutoff", 8000.0)
    sr = track_data.get("sr", GLOBAL_SR)

    logger.debug(
        f"{log_prefix}: Reverse={should_reverse}, Speed={speed_factor:.2f}, Ultrasonic={use_ultrasonic}, Pitch={pitch_shift_steps:.1f}, Filter={filter_type}@{filter_cutoff}Hz, SR={sr}"
    )

    # --- Apply Effects Sequentially Using Imported Functions ---
    try:
        # 1. Reverse (Optional)
        if should_reverse:
            audio = apply_reverse(audio)
            logger.debug(f"'{track_name}': Applied reverse.")

        # 2. Speed Change
        if not np.isclose(speed_factor, 1.0):
            audio = apply_speed_change(audio, sr, speed_factor)
            logger.debug(f"'{track_name}': Applied speed change (factor {speed_factor:.2f}).")

        # 3. Pitch Shift (Ultrasonic OR Regular, mutually exclusive)
        if use_ultrasonic:
            logger.debug(f"'{track_name}': Applying Ultrasonic shift.")
            audio = apply_ultrasonic_shift(audio, sr)
        elif not np.isclose(pitch_shift_steps, 0.0):
            logger.debug(f"'{track_name}': Applying Regular pitch shift ({pitch_shift_steps:.1f} steps).")
            audio = apply_pitch_shift(audio, sr, pitch_shift_steps)

        # 4. Filter (Standard Low/High Pass - only if Ultrasonic wasn't applied)
        if not use_ultrasonic and filter_type != "off":
            logger.debug(f"'{track_name}': Applying standard filter ({filter_type} @ {filter_cutoff}Hz).")
            audio = apply_standard_filter(audio, sr, filter_type, filter_cutoff)

    except Exception as e:
        logger.exception(f"Error during effect application pipeline for track '{track_name}'.")
        # Return empty array on pipeline failure
        return np.zeros((0, 2), dtype=np.float32)

    logger.debug(f"Finished applying effects for '{track_name}'. Output shape: {audio.shape}")
    return audio.astype(np.float32)  # Ensure output is float32
