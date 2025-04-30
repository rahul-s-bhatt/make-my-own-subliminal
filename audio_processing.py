# audio_processing.py
# ==========================================
# High-Level Audio Processing Pipeline for MindMorph
# (Applying effects, creating previews, mixing tracks)
# ==========================================

import logging
from typing import Any, Dict, Optional, Tuple

import librosa  # <<< Added librosa import for resampling
import numpy as np

# <<< Updated AppState import and types >>>
# AppState needed only for the advanced mix_tracks function
from app_state import AppState, TrackDataDict, TrackID

# import streamlit as st # Avoid direct Streamlit UI calls
# Import individual effect functions
from audio_effects import apply_pitch_shift, apply_reverse, apply_speed_change, apply_standard_filter, apply_ultrasonic_shift

# Import constants and types from config
from config import (
    GLOBAL_SR,
    MIX_PREVIEW_DURATION_S,
    # MIX_PREVIEW_PROCESSING_BUFFER_S, # No longer needed
    PREVIEW_DURATION_S,
)

# Define type hints used within this module
AudioData = np.ndarray
SampleRate = int
# TrackData = TrackDataDict # Use the specific type hint

# Get a logger for this module
logger = logging.getLogger(__name__)

# ==========================================
# Combined Effects & Preview Generation (For Advanced Editor)
# ==========================================


# <<< MODIFIED: Requires audio_segment, uses TrackDataDict >>>
def apply_all_effects(track_data: TrackDataDict, audio_segment: AudioData) -> AudioData:
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
    track_id = track_data.get("track_id", "N/A")  # Assuming track_id might be added

    if audio_segment.size == 0:
        logger.warning(f"Input audio segment for effects is empty for '{track_name}'. Returning empty.")
        return audio_segment

    audio = audio_segment.copy()
    log_prefix = f"Applying effects to segment for '{track_name}' ({track_id})"

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

    try:
        if should_reverse:
            audio = apply_reverse(audio)
            logger.debug(f"'{track_name}': Applied reverse.")
        if not np.isclose(speed_factor, 1.0):
            audio = apply_speed_change(audio, sr, speed_factor)
            logger.debug(f"'{track_name}': Applied speed change (factor {speed_factor:.2f}).")
        if use_ultrasonic:
            logger.debug(f"'{track_name}': Applying Ultrasonic shift.")
            audio = apply_ultrasonic_shift(audio, sr)
        elif not np.isclose(pitch_shift_steps, 0.0):
            logger.debug(f"'{track_name}': Applying Regular pitch shift ({pitch_shift_steps:.1f} steps).")
            audio = apply_pitch_shift(audio, sr, pitch_shift_steps)
        if not use_ultrasonic and filter_type != "off":
            logger.debug(f"'{track_name}': Applying standard filter ({filter_type} @ {filter_cutoff}Hz).")
            audio = apply_standard_filter(audio, sr, filter_type, filter_cutoff)

    except Exception as e:
        logger.exception(f"Error during effect application pipeline for track '{track_name}'.")
        return np.zeros((0, 2), dtype=np.float32)

    logger.debug(f"Finished applying effects for '{track_name}'. Output shape: {audio.shape}")
    return audio.astype(np.float32)


# <<< MODIFIED: Uses audio_snippet from track_data >>>
def get_preview_audio(track_data: TrackDataDict, preview_duration_s: int = PREVIEW_DURATION_S) -> Optional[AudioData]:
    """
    Generates a preview of the track using its snippet with effects, volume, and pan applied.

    Args:
        track_data: Dictionary containing track settings and the audio snippet.
        preview_duration_s: The maximum duration of the preview in seconds.

    Returns:
        The processed preview audio data, or None if generation fails or input is invalid.
    """
    track_name = track_data.get("name", "N/A")
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
        logger.debug(f"Applying effects to preview snippet for '{track_name}'")
        processed_preview = apply_all_effects(track_data, audio_segment=audio_snippet)

        if processed_preview is None or processed_preview.size == 0:
            logger.warning(f"Applying effects resulted in empty audio for '{track_name}' preview.")
            return None

        target_preview_samples = int(sr * preview_duration_s)
        current_processed_len = len(processed_preview)

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

        elif len(processed_preview) > target_preview_samples:
            logger.debug(f"Truncating processed preview for '{track_name}' to {target_preview_samples} samples.")
            processed_preview = processed_preview[:target_preview_samples]

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


# ==========================================
# Track Mixing Logic (For Advanced Editor)
# ==========================================


# <<< MODIFIED: Added app_state parameter, uses TrackDataDict >>>
def mix_tracks(
    app_state: AppState,  # <<< Added AppState instance
    tracks_dict: Dict[TrackID, TrackDataDict],
    target_sr: int = GLOBAL_SR,
    preview: bool = False,
    preview_duration_s: int = MIX_PREVIEW_DURATION_S,
    # preview_buffer_s: int = MIX_PREVIEW_PROCESSING_BUFFER_S, # Buffer less critical with snippets
) -> Tuple[Optional[AudioData], Optional[int]]:
    """
    Mixes multiple tracks together for the Advanced Editor.
    Uses snippets for preview mode and reloads/regenerates full audio for export mode.
    Args:
        app_state: The AppState instance (needed for get_full_audio).
        tracks_dict: Dictionary where keys are track IDs and values are TrackDataDict dictionaries.
        target_sr: The sample rate for the final mix.
        preview: If True, generate a shorter preview mix using snippets.
        preview_duration_s: Duration of the preview mix in seconds.
    Returns:
        A tuple containing: (mixed audio data, length in samples), or (None, None) on failure.
    """
    logger.info(f"Starting ADVANCED track mixing. Preview mode: {preview}. Target SR: {target_sr}Hz.")
    # --- (Implementation for Advanced Editor - Snippet Handling) ---
    # ... (Code from previous version - id: audio_processing_py) ...
    # ... (Ensure this complex logic remains unchanged for the advanced editor) ...
    if not tracks_dict:
        logger.warning("Mix called with no tracks provided.")
        return None, None

    active_track_ids = []
    solo_active = any(t_data.get("solo", False) for t_data in tracks_dict.values())
    logger.debug(f"Solo active: {solo_active}")

    for track_id, t_data in tracks_dict.items():
        has_audio_source = t_data.get("audio_snippet") is not None or t_data.get("source_info") is not None
        is_muted = t_data.get("mute", False)
        is_active = False
        if has_audio_source:
            if solo_active:
                is_active = t_data.get("solo", False)
            else:
                is_active = not is_muted
        if is_active:
            active_track_ids.append(track_id)
        else:
            logger.debug(f"Skipping track '{t_data.get('name', track_id)}' from mix.")

    if not active_track_ids:
        logger.warning("No active tracks with audio source found for mixing.")
        return None, None

    processed_tracks: Dict[TrackID, AudioData] = {}
    track_lengths_samples: Dict[TrackID, int] = {}

    if preview:
        logger.info(f"Step 1 (Preview): Processing snippets for {len(active_track_ids)} active tracks.")
        target_mix_len_samples = int(target_sr * preview_duration_s)
        for track_id in active_track_ids:
            t_data = tracks_dict[track_id]
            track_name = t_data.get("name", track_id)
            audio_snippet = t_data.get("audio_snippet")
            if audio_snippet is None or audio_snippet.size == 0:
                logger.warning(f"Preview: Skipping track '{track_name}' due to missing/empty snippet.")
                continue
            processed_snippet = apply_all_effects(t_data, audio_segment=audio_snippet)
            if processed_snippet is None or processed_snippet.size == 0:
                logger.warning(f"Preview: Skipping track '{track_name}' after effects resulted in empty audio.")
                continue
            processed_len = len(processed_snippet)
            final_preview_segment = processed_snippet
            if t_data.get("loop_to_fit", False) and processed_len > 0 and processed_len < target_mix_len_samples:
                logger.debug(f"Preview: Looping processed snippet for '{track_name}' ({processed_len} -> {target_mix_len_samples})")
                num_repeats = target_mix_len_samples // processed_len
                remainder = target_mix_len_samples % processed_len
                looped_list = [processed_snippet] * num_repeats
                if remainder > 0:
                    looped_list.append(processed_snippet[:remainder])
                try:
                    final_preview_segment = np.concatenate(looped_list, axis=0)
                except ValueError as e_concat:
                    logger.error(f"Preview: Error concatenating looped preview for '{track_name}': {e_concat}. Using non-looped.")
                    if processed_len > target_mix_len_samples:
                        final_preview_segment = processed_snippet[:target_mix_len_samples]
                    elif processed_len < target_mix_len_samples:
                        final_preview_segment = np.pad(processed_snippet, ((0, target_mix_len_samples - processed_len), (0, 0)), mode="constant")
            if len(final_preview_segment) > target_mix_len_samples:
                final_preview_segment = final_preview_segment[:target_mix_len_samples]
            processed_tracks[track_id] = final_preview_segment
            track_lengths_samples[track_id] = len(final_preview_segment)
        if not processed_tracks:
            logger.warning("Preview: No tracks successfully processed.")
            return None, None
        final_mix_len = target_mix_len_samples
    else:  # Export Mode
        logger.info(f"Step 1 (Export): Processing non-looping tracks to find max length.")
        max_len_non_looping = 0
        non_looping_processed: Dict[TrackID, AudioData] = {}
        for track_id in active_track_ids:
            t_data = tracks_dict[track_id]
            track_name = t_data.get("name", track_id)
            if not t_data.get("loop_to_fit", False):
                logger.debug(f"Export: Processing non-looping track '{track_name}'")
                full_audio_tuple = app_state.get_full_audio(track_id)
                if full_audio_tuple is None:
                    continue
                full_audio, _ = full_audio_tuple
                processed_full = apply_all_effects(t_data, audio_segment=full_audio)
                if processed_full is None or processed_full.size == 0:
                    continue
                non_looping_processed[track_id] = processed_full
                track_len = len(processed_full)
                track_lengths_samples[track_id] = track_len
                max_len_non_looping = max(max_len_non_looping, track_len)
                logger.debug(f"Export: Non-looping '{track_name}' processed length: {track_len}")
        target_mix_len_samples = max_len_non_looping
        logger.info(f"Step 2 (Export): Target mix length determined by non-looping tracks: {target_mix_len_samples} samples ({target_mix_len_samples / target_sr:.2f}s)")
        if target_mix_len_samples <= 0:
            logger.warning("Export: Target mix length is zero. Check non-looping tracks.")
            if not target_mix_len_samples and active_track_ids:
                logger.info("Export: Estimating from looping tracks' source info duration hints.")
                max_loop_target_len = 0
                for track_id in active_track_ids:
                    t_data = tracks_dict[track_id]
                    if t_data.get("loop_to_fit", False):
                        source_info = t_data.get("source_info")
                        sr = t_data.get("sr", GLOBAL_SR)
                        if source_info and "target_duration_s" in source_info and source_info["target_duration_s"] is not None and sr > 0:
                            est_len = int(source_info["target_duration_s"] * sr)
                            max_loop_target_len = max(max_loop_target_len, est_len)
                if max_loop_target_len > 0:
                    target_mix_len_samples = max_loop_target_len
                else:
                    logger.error("Export: Cannot determine target mix length.")
                    return None, None
        logger.info(f"Step 3 (Export): Processing looping tracks to match target length {target_mix_len_samples}.")
        for track_id in active_track_ids:
            if track_id in non_looping_processed:
                processed_tracks[track_id] = non_looping_processed[track_id]
                continue
            t_data = tracks_dict[track_id]
            track_name = t_data.get("name", track_id)
            logger.debug(f"Export: Processing looping track '{track_name}'")
            full_audio_tuple = app_state.get_full_audio(track_id, required_duration_samples=target_mix_len_samples)
            if full_audio_tuple is None:
                continue
            full_audio, _ = full_audio_tuple
            processed_full = apply_all_effects(t_data, audio_segment=full_audio)
            if processed_full is None or processed_full.size == 0:
                continue
            processed_len = len(processed_full)
            track_lengths_samples[track_id] = processed_len
            final_audio_for_track = processed_full
            if processed_len > 0 and processed_len < target_mix_len_samples:
                logger.debug(f"Export: Looping track '{track_name}' from {processed_len} to {target_mix_len_samples}.")
                num_repeats = target_mix_len_samples // processed_len
                remainder = target_mix_len_samples % processed_len
                looped_list = [processed_full] * num_repeats
                if remainder > 0:
                    looped_list.append(processed_full[:remainder])
                try:
                    final_audio_for_track = np.concatenate(looped_list, axis=0)
                except ValueError as e_concat:
                    logger.error(f"Export: Error concatenating looped audio for '{track_name}': {e_concat}. Using non-looped.")
                    final_audio_for_track = np.pad(processed_full, ((0, target_mix_len_samples - processed_len), (0, 0)), mode="constant")
            if len(final_audio_for_track) > target_mix_len_samples:
                final_audio_for_track = final_audio_for_track[:target_mix_len_samples]
            elif len(final_audio_for_track) < target_mix_len_samples:
                final_audio_for_track = np.pad(final_audio_for_track, ((0, target_mix_len_samples - len(final_audio_for_track)), (0, 0)), mode="constant")
            processed_tracks[track_id] = final_audio_for_track
        if not processed_tracks:
            logger.warning("Export: No tracks successfully processed.")
            return None, None
        final_mix_len = target_mix_len_samples

    logger.info(f"Step 4: Adding {len(processed_tracks)} processed tracks to mix buffer (Length: {final_mix_len} samples).")
    mix_buffer = np.zeros((final_mix_len, 2), dtype=np.float32)
    for track_id, processed_audio in processed_tracks.items():
        t_data = tracks_dict[track_id]
        track_name = t_data.get("name", track_id)
        current_len = len(processed_audio)
        if current_len > final_mix_len:
            audio_adjusted = processed_audio[:final_mix_len]
        elif current_len < final_mix_len:
            logger.warning(f"Track '{track_name}' length ({current_len}) is less than final mix length ({final_mix_len}) before final add. Padding.")
            audio_adjusted = np.pad(processed_audio, ((0, final_mix_len - current_len), (0, 0)), mode="constant")
        else:
            audio_adjusted = processed_audio
        pan = t_data.get("pan", 0.0)
        vol = t_data.get("volume", 1.0)
        logger.debug(f"Track '{track_name}': Applying final vol={vol:.2f}, pan={pan:.2f}")
        pan_rad = (pan + 1.0) * np.pi / 4.0
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)
        try:
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
        except ValueError as e_add:
            logger.error(f"Error adding track '{track_name}' to mix buffer: {e_add}. Buffer: {mix_buffer.shape}, Track: {audio_adjusted.shape}")
        logger.debug(f"Added track '{track_name}' to mix buffer.")

    final_mix = np.clip(mix_buffer, -1.0, 1.0)
    logger.info(f"Mixing complete. Final mix length: {final_mix_len} samples ({final_mix_len / target_sr:.2f}s).")
    return final_mix.astype(np.float32), final_mix_len


# ==========================================
# Track Mixing Logic (For Quick Wizard)
# ==========================================


# <<< NEW FUNCTION for Quick Wizard >>>
def mix_wizard_tracks(
    affirmation_audio: Optional[Tuple[AudioData, SampleRate]],
    background_audio: Optional[Tuple[AudioData, SampleRate]],
    frequency_audio: Optional[Tuple[AudioData, SampleRate]],
    affirmation_speed: float = 10.0,  # Wizard default speed
    affirmation_volume: float = 0.05,  # Wizard default volume
    background_volume: float = 1.0,  # Assume full volume unless specified otherwise
    frequency_volume: float = 1.0,  # Assume full volume unless specified otherwise
    target_sr: int = GLOBAL_SR,
) -> Optional[AudioData]:
    """
    Mixes tracks specifically for the Quick Create Wizard workflow.

    Applies fixed processing (speed/volume) to affirmations and loops
    background/frequency tracks to match the affirmation duration.

    Args:
        affirmation_audio: Tuple (audio_data, sr) for the affirmation track.
        background_audio: Optional tuple (audio_data, sr) for the background track.
        frequency_audio: Optional tuple (audio_data, sr) for the frequency track.
        affirmation_speed: Speed factor for the affirmation track.
        affirmation_volume: Volume factor for the affirmation track.
        background_volume: Volume factor for the background track.
        frequency_volume: Volume factor for the frequency track.
        target_sr: The target sample rate for the mix.

    Returns:
        The final mixed audio data (stereo, float32), or None on failure.
    """
    logger.info("Starting Quick Wizard track mixing.")
    processed_tracks = []  # List to hold processed numpy arrays

    if affirmation_audio is None or affirmation_audio[0].size == 0:
        logger.error("Wizard Mix Error: Affirmation audio is missing or empty.")
        return None
    aff_audio, aff_sr = affirmation_audio

    # --- 1. Process Affirmation Track ---
    try:
        logger.debug(f"Wizard: Processing affirmation (Speed: {affirmation_speed}, Vol: {affirmation_volume})")
        # a) Resample if needed
        if aff_sr != target_sr:
            logger.debug(f"Wizard: Resampling affirmation from {aff_sr} to {target_sr}")
            # Ensure input is float for librosa resample
            aff_audio_float = aff_audio.astype(np.float32)
            # Librosa expects (channels, samples) or (samples,)
            if aff_audio_float.ndim == 2:
                aff_audio = librosa.resample(aff_audio_float.T, orig_sr=aff_sr, target_sr=target_sr).T
            else:  # Mono
                aff_audio = librosa.resample(aff_audio_float, orig_sr=aff_sr, target_sr=target_sr)
            aff_sr = target_sr  # Update sample rate after resampling

        # b) Apply speed change (ensure apply_speed_change handles mono/stereo)
        processed_aff = apply_speed_change(aff_audio, aff_sr, affirmation_speed)

        # c) Ensure stereo after speed change (which might return mono)
        if processed_aff.ndim == 1:
            processed_aff = np.stack([processed_aff, processed_aff], axis=-1)

        # d) Apply volume
        processed_aff *= affirmation_volume
        processed_tracks.append(processed_aff)
        logger.debug(f"Wizard: Affirmation processed. Length: {len(processed_aff)} samples.")
    except Exception as e:
        logger.exception("Wizard Mix Error: Failed to process affirmation audio.")
        return None

    target_mix_len_samples = len(processed_aff)
    if target_mix_len_samples <= 0:
        logger.error("Wizard Mix Error: Processed affirmation audio has zero length.")
        return None

    # --- 2. Process Background Track (Looping) ---
    if background_audio and background_audio[0].size > 0:
        bg_audio, bg_sr = background_audio
        try:
            logger.debug(f"Wizard: Processing background (Vol: {background_volume})")
            # a) Resample if needed
            if bg_sr != target_sr:
                logger.debug(f"Wizard: Resampling background from {bg_sr} to {target_sr}")
                bg_audio_float = bg_audio.astype(np.float32)
                if bg_audio_float.ndim == 2:
                    bg_audio = librosa.resample(bg_audio_float.T, orig_sr=bg_sr, target_sr=target_sr).T
                else:
                    bg_audio = librosa.resample(bg_audio_float, orig_sr=bg_sr, target_sr=target_sr)
                bg_sr = target_sr

            # b) Ensure stereo
            if bg_audio.ndim == 1:
                bg_audio = np.stack([bg_audio, bg_audio], axis=-1)

            # c) Loop/Truncate to fit affirmation length
            processed_bg = bg_audio
            bg_len = len(processed_bg)
            if bg_len > 0 and bg_len < target_mix_len_samples:
                num_repeats = target_mix_len_samples // bg_len
                remainder = target_mix_len_samples % bg_len
                looped_list = [processed_bg] * num_repeats
                if remainder > 0:
                    looped_list.append(processed_bg[:remainder])
                processed_bg = np.concatenate(looped_list, axis=0)
            elif bg_len > target_mix_len_samples:
                processed_bg = processed_bg[:target_mix_len_samples]

            # d) Apply volume
            processed_bg *= background_volume
            processed_tracks.append(processed_bg)
            logger.debug(f"Wizard: Background processed. Length: {len(processed_bg)} samples.")
        except Exception as e:
            logger.exception("Wizard Mix Error: Failed to process background audio. Skipping background.")
            # Continue without background

    # --- 3. Process Frequency Track (Looping) ---
    if frequency_audio and frequency_audio[0].size > 0:
        freq_audio, freq_sr = frequency_audio
        try:
            logger.debug(f"Wizard: Processing frequency (Vol: {frequency_volume})")
            # a) Resample if needed
            if freq_sr != target_sr:
                logger.debug(f"Wizard: Resampling frequency from {freq_sr} to {target_sr}")
                freq_audio_float = freq_audio.astype(np.float32)
                if freq_audio_float.ndim == 2:
                    freq_audio = librosa.resample(freq_audio_float.T, orig_sr=freq_sr, target_sr=target_sr).T
                else:
                    freq_audio = librosa.resample(freq_audio_float, orig_sr=freq_sr, target_sr=target_sr)
                freq_sr = target_sr

            # b) Ensure stereo
            if freq_audio.ndim == 1:
                freq_audio = np.stack([freq_audio, freq_audio], axis=-1)

            # c) Loop/Truncate to fit affirmation length
            processed_freq = freq_audio
            freq_len = len(processed_freq)
            if freq_len > 0 and freq_len < target_mix_len_samples:
                num_repeats = target_mix_len_samples // freq_len
                remainder = target_mix_len_samples % freq_len
                looped_list = [processed_freq] * num_repeats
                if remainder > 0:
                    looped_list.append(processed_freq[:remainder])
                processed_freq = np.concatenate(looped_list, axis=0)
            elif freq_len > target_mix_len_samples:
                processed_freq = processed_freq[:target_mix_len_samples]

            # d) Apply volume
            processed_freq *= frequency_volume
            processed_tracks.append(processed_freq)
            logger.debug(f"Wizard: Frequency processed. Length: {len(processed_freq)} samples.")
        except Exception as e:
            logger.exception("Wizard Mix Error: Failed to process frequency audio. Skipping frequency.")
            # Continue without frequency

    # --- 4. Mix All Processed Tracks ---
    if not processed_tracks:
        logger.error("Wizard Mix Error: No tracks were successfully processed for mixing.")
        return None

    # Initialize mix buffer to the target length
    mix_buffer = np.zeros((target_mix_len_samples, 2), dtype=np.float32)

    logger.info(f"Wizard: Summing {len(processed_tracks)} processed tracks.")
    for track in processed_tracks:
        # Ensure track length matches buffer before adding
        track_len = len(track)
        if track_len == target_mix_len_samples:
            # Ensure track is stereo before adding
            if track.ndim == 1:  # Should not happen if steps above work, but check
                track = np.stack([track, track], axis=-1)
            if track.shape[1] == 2:
                mix_buffer += track
            else:
                logger.warning(f"Wizard Mix Error: Track has unexpected shape {track.shape}. Skipping add.")
        else:
            # This shouldn't happen if looping/truncating worked correctly
            logger.error(f"Wizard Mix Error: Processed track length ({track_len}) mismatch with target ({target_mix_len_samples}). Cannot add.")
            # Optionally pad/truncate here as a fallback, but indicates prior error
            # return None # Safer to fail if lengths mismatch unexpectedly

    # --- 5. Finalize Mix ---
    final_mix = np.clip(mix_buffer, -1.0, 1.0)
    logger.info(f"Wizard mixing complete. Final mix length: {len(final_mix)} samples.")
    return final_mix.astype(np.float32)
