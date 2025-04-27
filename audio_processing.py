# audio_processing.py
# ==========================================
# High-Level Audio Processing Pipeline for MindMorph
# (Applying effects, creating previews, mixing tracks)
# ==========================================

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st  # TODO: Remove Streamlit UI calls from this module

# Import individual effect functions
from audio_effects import apply_pitch_shift, apply_reverse, apply_speed_change, apply_standard_filter, apply_ultrasonic_shift

# Import constants and types from config
from config import GLOBAL_SR, MIX_PREVIEW_DURATION_S, MIX_PREVIEW_PROCESSING_BUFFER_S, PREVIEW_DURATION_S

# Define type hints used within this module
AudioData = np.ndarray
SampleRate = int
TrackData = Dict[str, Any]  # Using Dict temporarily, might be replaced by a dataclass later

# Get a logger for this module
logger = logging.getLogger(__name__)

# ==========================================
# Combined Effects & Preview Generation
# ==========================================


def apply_all_effects(track_data: TrackData, audio_segment: Optional[AudioData] = None) -> AudioData:
    """
    Applies effects sequentially based on track_data settings.
    Order: Reverse -> Speed -> Pitch/Ultrasonic -> Filter.

    Args:
        track_data: Dictionary containing track settings and original audio.
        audio_segment: Optional audio segment to process (e.g., for previews).
                       If None, uses track_data['original_audio'].

    Returns:
        The processed audio data as a NumPy array. Returns empty array on error or if input is empty.
    """
    track_name = track_data.get("name", "Unnamed Track")
    should_reverse = track_data.get("reverse_audio", False)
    use_ultrasonic = track_data.get("ultrasonic_shift", False)
    pitch_shift_steps = track_data.get("pitch_shift", 0.0)
    speed_factor = track_data.get("speed_factor", 1.0)
    filter_type = track_data.get("filter_type", "off")
    filter_cutoff = track_data.get("filter_cutoff", 8000.0)
    sr = track_data.get("sr", GLOBAL_SR)

    if audio_segment is not None:
        # Process the provided segment (e.g., for preview)
        if audio_segment.size == 0:
            logger.warning(f"Input audio segment for effects is empty for '{track_name}'. Returning empty.")
            return audio_segment
        audio = audio_segment.copy()
        log_prefix = f"Applying effects to segment for '{track_name}'"
    else:
        # Process the full original audio from track_data
        original_audio = track_data.get("original_audio")
        if original_audio is None:
            logger.error(f"Track '{track_name}' is missing 'original_audio' data for effects processing.")
            return np.zeros((0, 2), dtype=np.float32)  # Return empty stereo array
        if original_audio.size == 0:
            logger.warning(f"Input 'original_audio' for effects is empty for '{track_name}'. Returning empty.")
            return original_audio  # Return empty array
        audio = original_audio.copy()
        log_prefix = f"Applying effects to full audio for '{track_name}'"

    logger.debug(
        f"{log_prefix}: Reverse={should_reverse}, Speed={speed_factor:.2f}, Ultrasonic={use_ultrasonic}, Pitch={pitch_shift_steps:.1f}, Filter={filter_type}@{filter_cutoff}Hz"
    )

    # --- Apply Effects Sequentially Using Imported Functions ---
    try:
        # 1. Reverse (Optional)
        if should_reverse:
            audio = apply_reverse(audio)
            logger.debug(f"'{track_name}': Applied reverse.")

        # 2. Speed Change
        # Apply speed change only if factor is not 1.0
        if not np.isclose(speed_factor, 1.0):
            audio = apply_speed_change(audio, sr, speed_factor)
            logger.debug(f"'{track_name}': Applied speed change (factor {speed_factor:.2f}).")

        # 3. Pitch Shift (Ultrasonic OR Regular, mutually exclusive)
        if use_ultrasonic:
            logger.debug(f"'{track_name}': Applying Ultrasonic shift.")
            audio = apply_ultrasonic_shift(audio, sr)  # target_freq uses default from config via audio_effects
        elif not np.isclose(pitch_shift_steps, 0.0):
            logger.debug(f"'{track_name}': Applying Regular pitch shift ({pitch_shift_steps:.1f} steps).")
            audio = apply_pitch_shift(audio, sr, pitch_shift_steps)
        else:
            logger.debug(f"'{track_name}': No pitch shift applied (Ultrasonic off, Pitch shift is 0).")

        # 4. Filter (Standard Low/High Pass - only if Ultrasonic wasn't applied)
        if not use_ultrasonic and filter_type != "off":
            logger.debug(f"'{track_name}': Applying standard filter ({filter_type} @ {filter_cutoff}Hz).")
            audio = apply_standard_filter(audio, sr, filter_type, filter_cutoff)
        elif use_ultrasonic:
            logger.debug(f"'{track_name}': Skipping standard filter because ultrasonic shift was applied.")
        else:
            logger.debug(f"'{track_name}': No standard filter applied (Ultrasonic off, Filter type is 'off').")

    except Exception as e:
        logger.exception(f"Error during effect application pipeline for track '{track_name}'.")
        st.error(f"Failed to apply effects to track '{track_name}': {e}")
        # Return empty array on pipeline failure? Or original audio? Returning empty for now.
        return np.zeros((0, 2), dtype=np.float32)

    logger.debug(f"Finished applying effects for '{track_name}'. Output shape: {audio.shape}")
    return audio.astype(np.float32)  # Ensure output is float32


def get_preview_audio(track_data: TrackData, preview_duration_s: int = PREVIEW_DURATION_S) -> Optional[AudioData]:
    """
    Generates a preview (first N seconds) of the track with effects, volume, and pan applied.

    Args:
        track_data: Dictionary containing track settings and original audio.
        preview_duration_s: The maximum duration of the preview in seconds.

    Returns:
        The processed preview audio data, or None if generation fails or input is invalid.
    """
    track_name = track_data.get("name", "N/A")
    logger.info(f"Generating preview audio for track '{track_name}' (max {preview_duration_s}s)")

    original_audio = track_data.get("original_audio")
    sr = track_data.get("sr", GLOBAL_SR)

    if original_audio is None or original_audio.size == 0:
        logger.warning(f"No original audio data found for track '{track_name}'. Cannot generate preview.")
        return None
    if sr <= 0:
        logger.warning(f"Invalid sample rate ({sr}) for track '{track_name}'. Cannot generate preview.")
        return None

    try:
        # Calculate number of samples for the preview duration
        preview_samples = min(len(original_audio), int(sr * preview_duration_s))
        if preview_samples <= 0:
            logger.warning(f"Calculated preview samples <= 0 for '{track_name}'. Cannot generate preview.")
            return None

        # Extract the segment for preview processing
        preview_segment = original_audio[:preview_samples].copy()
        logger.debug(f"Extracted preview segment ({preview_samples} samples) for '{track_name}'")

        # Apply all effects (Reverse, Speed, Pitch/Ultrasonic, Filter) to the segment
        logger.debug(f"Applying effects to preview segment for '{track_name}'")
        processed_preview = apply_all_effects(track_data, audio_segment=preview_segment)  # Uses the function above

        if processed_preview is None or processed_preview.size == 0:
            logger.warning(f"Applying effects resulted in empty audio for '{track_name}' preview.")
            return None

        # Apply Volume and Pan to the processed preview
        vol = track_data.get("volume", 1.0)
        pan = track_data.get("pan", 0.0)
        logger.debug(f"Applying Volume ({vol:.2f}) / Pan ({pan:.2f}) to preview for '{track_name}'")

        # Calculate stereo gains based on pan (-1 L to +1 R)
        pan_rad = (pan + 1.0) * np.pi / 4.0  # Maps [-1, 1] to [0, pi/2]
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)

        # Apply gains to stereo channels
        if processed_preview.ndim == 2 and processed_preview.shape[1] == 2:
            processed_preview[:, 0] *= left_gain
            processed_preview[:, 1] *= right_gain
        elif processed_preview.ndim == 1:  # Handle mono case
            logger.warning(f"Preview for '{track_name}' is mono after effects. Applying volume, pan ignored.")
            processed_preview *= vol
            processed_preview = np.stack([processed_preview, processed_preview], axis=1)  # Convert back to stereo
        else:
            logger.warning(f"Processed preview for '{track_name}' has unexpected shape {processed_preview.shape}. Cannot apply volume/pan.")
            # Attempt to apply volume to first channel if possible as fallback
            try:
                processed_preview[:, 0] *= vol
            except IndexError:
                pass

        # Clip final preview audio and ensure correct type
        processed_preview = np.clip(processed_preview, -1.0, 1.0)
        logger.debug(f"Preview generation complete for '{track_name}'. Final shape: {processed_preview.shape}")
        return processed_preview.astype(np.float32)

    except Exception as e:
        logger.exception(f"Error generating preview for track '{track_name}'")
        # TODO: Remove direct Streamlit call. Raise exception or return error status.
        st.error(f"Error generating preview for '{track_name}': {e}")
        return None


# ==========================================
# Track Mixing Logic
# ==========================================


def mix_tracks(
    tracks_dict: Dict[str, TrackData],  # Use str for TrackID key
    target_sr: int = GLOBAL_SR,
    preview: bool = False,
    preview_duration_s: int = MIX_PREVIEW_DURATION_S,
    preview_buffer_s: int = MIX_PREVIEW_PROCESSING_BUFFER_S,
) -> Tuple[Optional[AudioData], Optional[int]]:
    """
    Mixes multiple tracks together, handling effects, looping, volume, pan, mute, and solo.

    Args:
        tracks_dict: Dictionary where keys are track IDs and values are TrackData dictionaries.
        target_sr: The sample rate for the final mix.
        preview: If True, generate a shorter preview mix.
        preview_duration_s: Duration of the preview mix in seconds.
        preview_buffer_s: Extra buffer duration for processing preview segments.

    Returns:
        A tuple containing:
            - The final mixed audio data as a NumPy array (float32, stereo), or None on failure.
            - The length of the final mix in samples, or None on failure.
    """
    logger.info(f"Starting track mixing. Preview mode: {preview}. Target SR: {target_sr}Hz.")

    if not tracks_dict:
        logger.warning("Mix called with no tracks provided.")
        return None, None

    # --- Determine Active Tracks and Estimate Lengths ---
    valid_track_ids_for_mix = []
    estimated_processed_lengths = {}
    solo_active = any(t_data.get("solo", False) for t_data in tracks_dict.values())
    logger.debug(f"Solo active: {solo_active}")

    logger.info("Step 1: Determining active tracks and estimating lengths after speed changes.")
    for track_id, t_data in tracks_dict.items():
        is_active = False
        original_audio = t_data.get("original_audio")
        has_audio = original_audio is not None and original_audio.size > 0
        is_muted = t_data.get("mute", False)

        if has_audio:
            if solo_active:
                is_active = t_data.get("solo", False)
            else:
                is_active = not is_muted

        if is_active:
            valid_track_ids_for_mix.append(track_id)
            original_len = len(original_audio)
            speed_factor = t_data.get("speed_factor", 1.0)
            estimated_len = int(original_len / speed_factor) if speed_factor > 0 else original_len
            estimated_processed_lengths[track_id] = estimated_len
            logger.debug(f"Track '{t_data.get('name', track_id)}': Active. Est len={estimated_len}")
        else:
            reason = "no audio" if not has_audio else ("muted" if is_muted else "not soloed")
            logger.debug(f"Skipping track '{t_data.get('name', track_id)}' from mix ({reason}).")

    if not valid_track_ids_for_mix:
        logger.warning("No active tracks with audio found for mixing.")
        return None, None

    # --- Determine Target Mix Length ---
    target_mix_len_samples = max(estimated_processed_lengths.values()) if estimated_processed_lengths else 0
    logger.info(f"Target mix length (pre-looping): {target_mix_len_samples} samples ({target_mix_len_samples / target_sr:.2f}s)")

    process_samples = 0
    if preview:
        preview_target_len = int(target_sr * preview_duration_s)
        if target_mix_len_samples > preview_target_len:
            logger.info(f"Preview mode: Limiting mix length to {preview_target_len} samples.")
            target_mix_len_samples = preview_target_len
        process_duration_s = preview_duration_s + preview_buffer_s
        process_samples = int(target_sr * process_duration_s)
        logger.debug(f"Preview processing buffer: {process_samples} samples ({process_duration_s:.1f}s)")

    if target_mix_len_samples <= 0:
        logger.warning("Target mix length is zero or negative. Cannot create mix.")
        return None, None

    # Initialize the master mix buffer
    mix_buffer = np.zeros((target_mix_len_samples, 2), dtype=np.float32)
    logger.info(f"Mixing {len(valid_track_ids_for_mix)} tracks. Initial buffer: {target_mix_len_samples / target_sr:.2f}s")

    # --- Pre-process Preview Segments ---
    processed_preview_segments = {}
    if preview:
        logger.info("Step 2: Pre-processing segments for preview mode.")
        for track_id in valid_track_ids_for_mix:
            t_data = tracks_dict[track_id]
            original_audio = t_data.get("original_audio")
            if original_audio is not None and original_audio.size > 0:
                segment_samples = min(len(original_audio), process_samples)
                segment = original_audio[:segment_samples].copy()
                logger.debug(f"Processing PREVIEW segment ({segment_samples}) for '{t_data.get('name', track_id)}'.")
                processed_preview_segments[track_id] = apply_all_effects(t_data, audio_segment=segment)  # Use function from this module
            else:
                processed_preview_segments[track_id] = None
                logger.warning(f"Track '{t_data.get('name', track_id)}' missing audio during preview pre-processing.")

    # --- Process and Add Each Track to Mix ---
    logger.info("Step 3: Processing and adding tracks to the mix buffer.")
    actual_max_len_samples = target_mix_len_samples

    for track_id in valid_track_ids_for_mix:
        t_data = tracks_dict[track_id]
        track_name = t_data.get("name", track_id)
        logger.debug(f"Processing and mixing track: '{track_name}'")

        processed_audio: Optional[AudioData] = None
        if preview:
            processed_audio = processed_preview_segments.get(track_id)
        else:
            processed_audio = apply_all_effects(t_data)  # Use function from this module

        if processed_audio is None or processed_audio.size == 0:
            logger.warning(f"Processing resulted in empty audio for '{track_name}'. Skipping.")
            continue

        actual_processed_len = len(processed_audio)
        logger.debug(f"Track '{track_name}': Actual processed length = {actual_processed_len} samples.")

        # --- Handle Looping ---
        final_audio_for_track = processed_audio
        should_loop = t_data.get("loop_to_fit", False)

        if not preview and should_loop:
            if actual_processed_len > 0 and actual_processed_len < target_mix_len_samples:
                logger.info(f"Looping track '{track_name}' from {actual_processed_len} to {target_mix_len_samples}.")
                num_repeats = target_mix_len_samples // actual_processed_len
                remainder = target_mix_len_samples % actual_processed_len
                looped_list = [processed_audio] * num_repeats
                if remainder > 0:
                    looped_list.append(processed_audio[:remainder])
                try:
                    final_audio_for_track = np.concatenate(looped_list, axis=0)
                    actual_max_len_samples = max(actual_max_len_samples, len(final_audio_for_track))
                    logger.debug(f"Looping complete for '{track_name}'. New length: {len(final_audio_for_track)}.")
                except ValueError as e_concat:
                    logger.error(f"Error concatenating looped audio for '{track_name}': {e_concat}. Using non-looped.")
                    final_audio_for_track = processed_audio  # Fallback
            else:
                logger.debug(f"Looping not needed/applicable for '{track_name}'.")
                actual_max_len_samples = max(actual_max_len_samples, actual_processed_len)

        # --- Resize Mix Buffer if Needed ---
        if len(mix_buffer) < actual_max_len_samples:
            logger.warning(f"Resizing mix buffer from {len(mix_buffer)} to {actual_max_len_samples} for '{track_name}'.")
            mix_buffer = np.pad(mix_buffer, ((0, actual_max_len_samples - len(mix_buffer)), (0, 0)), mode="constant")
            target_mix_len_samples = actual_max_len_samples  # Update target length

        # --- Adjust Track Length ---
        current_track_len = len(final_audio_for_track)
        if current_track_len < target_mix_len_samples:
            audio_adjusted = np.pad(final_audio_for_track, ((0, target_mix_len_samples - current_track_len), (0, 0)), mode="constant")
        elif current_track_len > target_mix_len_samples:
            logger.warning(f"Track '{track_name}' ({current_track_len}) longer than mix buffer ({target_mix_len_samples}). Truncating.")
            audio_adjusted = final_audio_for_track[:target_mix_len_samples, :]
        else:
            audio_adjusted = final_audio_for_track

        # --- Apply Volume and Pan ---
        pan = t_data.get("pan", 0.0)
        vol = t_data.get("volume", 1.0)
        logger.debug(f"Track '{track_name}': Applying final vol={vol:.2f}, pan={pan:.2f}")
        pan_rad = (pan + 1.0) * np.pi / 4.0
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)

        if audio_adjusted.ndim == 2 and audio_adjusted.shape[1] == 2:
            mix_buffer[:, 0] += audio_adjusted[:, 0] * left_gain
            mix_buffer[:, 1] += audio_adjusted[:, 1] * right_gain
        elif audio_adjusted.ndim == 1:
            logger.warning(f"Track '{track_name}' is mono during final mixing. Applying volume and splitting.")
            mono_scaled = audio_adjusted * vol * 0.7071
            mix_buffer[:, 0] += mono_scaled
            mix_buffer[:, 1] += mono_scaled
        else:
            logger.error(f"Track '{track_name}' has unexpected shape {audio_adjusted.shape}. Cannot add to buffer.")

        logger.debug(f"Added track '{track_name}' to mix buffer.")
        del processed_audio, final_audio_for_track, audio_adjusted  # Memory cleanup

    # --- Finalize Mix ---
    final_mix = np.clip(mix_buffer, -1.0, 1.0)
    final_mix_len = len(final_mix)
    logger.info(f"Mixing complete. Final mix length: {final_mix_len} samples ({final_mix_len / target_sr:.2f}s).")

    return final_mix.astype(np.float32), final_mix_len
