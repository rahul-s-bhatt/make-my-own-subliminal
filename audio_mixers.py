# audio_mixers.py
# ==========================================
# Functions for mixing audio tracks for both Advanced Editor and Quick Wizard.
# ==========================================

import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import librosa  # Needed for wizard mixer resampling
import numpy as np

# Import individual effects needed by wizard mixer
from audio_effects import apply_speed_change

# Import the effects pipeline
from audio_effects_pipeline import apply_all_effects

# <<< MODIFIED: Import the new audio loader function >>>
from audio_loader import load_or_regenerate_audio

# <<< MODIFIED: Import definitions from the new definitions file >>>
from audio_state_definitions import AudioData, SampleRate, SourceInfo, TrackDataDict, TrackID

# Import constants and types
from config import GLOBAL_SR, MIX_PREVIEW_DURATION_S

# Get a logger for this module
logger = logging.getLogger(__name__)


# ==========================================
# Track Mixing Logic (For Advanced Editor)
# ==========================================


def mix_tracks(
    # <<< MODIFIED: No longer needs AppState directly >>>
    # app_state: "AppState",
    tracks_dict: Dict[TrackID, TrackDataDict],
    target_sr: int = GLOBAL_SR,
    preview: bool = False,
    preview_duration_s: int = MIX_PREVIEW_DURATION_S,
) -> Tuple[Optional[AudioData], Optional[int]]:
    """
    Mixes multiple tracks together for the Advanced Editor.
    Uses snippets for preview mode and reloads/regenerates full audio for export mode.
    Args:
        tracks_dict: Dictionary where keys are track IDs and values are TrackDataDict dictionaries.
        target_sr: The sample rate for the final mix.
        preview: If True, generate a shorter preview mix using snippets.
        preview_duration_s: Duration of the preview mix in seconds.
    Returns:
        A tuple containing: (mixed audio data, length in samples), or (None, None) on failure.
    """
    logger.info(f"Starting ADVANCED track mixing. Preview mode: {preview}. Target SR: {target_sr}Hz.")
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

    # --- Preview Mode ---
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

    # --- Export Mode ---
    else:
        logger.info(f"Step 1 (Export): Processing non-looping tracks to find max length.")
        max_len_non_looping = 0
        non_looping_processed: Dict[TrackID, AudioData] = {}
        for track_id in active_track_ids:
            t_data = tracks_dict[track_id]
            track_name = t_data.get("name", track_id)
            if not t_data.get("loop_to_fit", False):
                logger.debug(f"Export: Processing non-looping track '{track_name}'")
                # <<< MODIFIED: Call audio_loader function >>>
                full_audio_tuple = load_or_regenerate_audio(t_data.get("source_info"), sr_hint=t_data.get("sr", GLOBAL_SR))
                if full_audio_tuple is None:
                    logger.warning(f"Export: Failed get full audio for non-looping '{track_name}'. Skip.")
                    continue
                full_audio, loaded_sr = full_audio_tuple
                # Update track_data SR if it was different (e.g., from TTS) - might affect effects
                if loaded_sr != t_data.get("sr", GLOBAL_SR):
                    logger.info(f"Updating SR for track '{track_name}' from {t_data.get('sr')} to {loaded_sr} based on loaded audio.")
                    t_data["sr"] = loaded_sr  # Modify dict in place for effects calculation

                processed_full = apply_all_effects(t_data, audio_segment=full_audio)
                if processed_full is None or processed_full.size == 0:
                    logger.warning(f"Export: Effects empty for non-looping '{track_name}'. Skip.")
                    continue
                non_looping_processed[track_id] = processed_full
                track_len = len(processed_full)
                track_lengths_samples[track_id] = track_len
                max_len_non_looping = max(max_len_non_looping, track_len)
                logger.debug(f"Export: Non-looping '{track_name}' processed length: {track_len}")

        target_mix_len_samples = max_len_non_looping
        logger.info(f"Step 2 (Export): Target mix length determined: {target_mix_len_samples} samples ({target_mix_len_samples / target_sr:.2f}s)")
        if target_mix_len_samples <= 0:
            logger.warning("Export: Target mix length is zero. Trying estimate from looping tracks.")
            max_loop_target_len = 0
            for track_id in active_track_ids:
                t_data = tracks_dict[track_id]
                if t_data.get("loop_to_fit", False):
                    source_info = t_data.get("source_info")
                    sr = t_data.get("sr", GLOBAL_SR)
                    if isinstance(source_info, dict) and "target_duration_s" in source_info and source_info["target_duration_s"] is not None and sr > 0:
                        est_len = int(source_info["target_duration_s"] * sr)
                        max_loop_target_len = max(max_loop_target_len, est_len)
            if max_loop_target_len > 0:
                target_mix_len_samples = max_loop_target_len
                logger.info(f"Using estimated length: {target_mix_len_samples}")
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
            # <<< MODIFIED: Call audio_loader function >>>
            full_audio_tuple = load_or_regenerate_audio(t_data.get("source_info"), sr_hint=t_data.get("sr", GLOBAL_SR), required_duration_samples=target_mix_len_samples)
            if full_audio_tuple is None:
                logger.warning(f"Export: Failed get full audio for looping '{track_name}'. Skip.")
                continue
            full_audio, loaded_sr = full_audio_tuple
            if loaded_sr != t_data.get("sr", GLOBAL_SR):
                t_data["sr"] = loaded_sr  # Update SR if needed

            processed_full = apply_all_effects(t_data, audio_segment=full_audio)
            if processed_full is None or processed_full.size == 0:
                logger.warning(f"Export: Effects empty for looping '{track_name}'. Skip.")
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
                    logger.error(f"Export: Error concatenating looped audio for '{track_name}': {e_concat}. Using non-looped/padded.")
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

    # --- Final Mixing ---
    logger.info(f"Step 4: Adding {len(processed_tracks)} processed tracks to mix buffer (Length: {final_mix_len} samples).")
    mix_buffer = np.zeros((final_mix_len, 2), dtype=np.float32)
    for track_id, processed_audio in processed_tracks.items():
        t_data = tracks_dict[track_id]
        track_name = t_data.get("name", track_id)
        current_len = len(processed_audio)
        if current_len > final_mix_len:
            audio_adjusted = processed_audio[:final_mix_len]
        elif current_len < final_mix_len:
            logger.warning(f"Padding track '{track_name}' length ({current_len}) to match final mix ({final_mix_len}).")
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
                logger.warning(f"Track '{track_name}' is mono during final mixing. Splitting.")
                mono_scaled = audio_adjusted * vol * 0.7071
                mix_buffer[:, 0] += mono_scaled
                mix_buffer[:, 1] += mono_scaled
            else:
                logger.error(f"Track '{track_name}' has unexpected shape {audio_adjusted.shape}. Cannot add.")
        except ValueError as e_add:
            logger.error(f"Error adding track '{track_name}' to mix buffer: {e_add}. Buffer: {mix_buffer.shape}, Track: {audio_adjusted.shape}")
        logger.debug(f"Added track '{track_name}' to mix buffer.")

    final_mix = np.clip(mix_buffer, -1.0, 1.0)
    logger.info(f"Mixing complete. Final mix length: {final_mix_len} samples ({final_mix_len / target_sr:.2f}s).")
    return final_mix.astype(np.float32), final_mix_len


# ==========================================
# Track Mixing Logic (For Quick Wizard)
# ==========================================


def mix_wizard_tracks(
    affirmation_audio: Optional[Tuple[AudioData, SampleRate]],
    background_audio: Optional[Tuple[AudioData, SampleRate]],
    frequency_audio: Optional[Tuple[AudioData, SampleRate]],
    affirmation_speed: float = 10.0,
    affirmation_volume: float = 0.05,
    background_volume: float = 1.0,
    frequency_volume: float = 1.0,
    target_sr: int = GLOBAL_SR,
) -> Optional[AudioData]:
    """
    Mixes tracks specifically for the Quick Create Wizard workflow.
    Args: (See previous definition)
    Returns: The final mixed audio data (stereo, float32), or None on failure.
    """
    logger.info("Starting Quick Wizard track mixing.")
    processed_tracks = []
    if affirmation_audio is None or affirmation_audio[0].size == 0:
        logger.error("Wizard Mix: Affirmation missing.")
        return None
    aff_audio, aff_sr = affirmation_audio
    try:
        logger.debug(f"Wizard: Processing affirmation (Speed: {affirmation_speed}, Vol: {affirmation_volume})")
        if aff_sr != target_sr:
            logger.debug(f"Wizard: Resampling affirmation from {aff_sr} to {target_sr}")
            aff_audio_float = aff_audio.astype(np.float32)
            if aff_audio_float.ndim == 2:
                aff_audio = librosa.resample(aff_audio_float.T, orig_sr=aff_sr, target_sr=target_sr).T
            else:
                aff_audio = librosa.resample(aff_audio_float, orig_sr=aff_sr, target_sr=target_sr)
            aff_sr = target_sr
        processed_aff = apply_speed_change(aff_audio, aff_sr, affirmation_speed)
        if processed_aff.ndim == 1:
            processed_aff = np.stack([processed_aff, processed_aff], axis=-1)
        processed_aff *= affirmation_volume
        processed_tracks.append(processed_aff)
        logger.debug(f"Wizard: Affirmation processed. Length: {len(processed_aff)} samples.")
    except Exception as e:
        logger.exception("Wizard Mix: Failed process affirmation.")
        return None
    target_mix_len_samples = len(processed_aff)
    if target_mix_len_samples <= 0:
        logger.error("Wizard Mix: Processed affirmation zero length.")
        return None
    if background_audio and background_audio[0].size > 0:
        bg_audio, bg_sr = background_audio
        try:
            logger.debug(f"Wizard: Processing background (Vol: {background_volume})")
            if bg_sr != target_sr:
                logger.debug(f"Wizard: Resampling background from {bg_sr} to {target_sr}")
                bg_audio_float = bg_audio.astype(np.float32)
                if bg_audio_float.ndim == 2:
                    bg_audio = librosa.resample(bg_audio_float.T, orig_sr=bg_sr, target_sr=target_sr).T
                else:
                    bg_audio = librosa.resample(bg_audio_float, orig_sr=bg_sr, target_sr=target_sr)
                bg_sr = target_sr
            if bg_audio.ndim == 1:
                bg_audio = np.stack([bg_audio, bg_audio], axis=-1)
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
            processed_bg *= background_volume
            processed_tracks.append(processed_bg)
            logger.debug(f"Wizard: Background processed. Length: {len(processed_bg)} samples.")
        except Exception as e:
            logger.exception("Wizard Mix: Failed process background. Skipping.")
    if frequency_audio and frequency_audio[0].size > 0:
        freq_audio, freq_sr = frequency_audio
        try:
            logger.debug(f"Wizard: Processing frequency (Vol: {frequency_volume})")
            if freq_sr != target_sr:
                logger.debug(f"Wizard: Resampling frequency from {freq_sr} to {target_sr}")
                freq_audio_float = freq_audio.astype(np.float32)
                if freq_audio_float.ndim == 2:
                    freq_audio = librosa.resample(freq_audio_float.T, orig_sr=freq_sr, target_sr=target_sr).T
                else:
                    freq_audio = librosa.resample(freq_audio_float, orig_sr=freq_sr, target_sr=target_sr)
                freq_sr = target_sr
            if freq_audio.ndim == 1:
                freq_audio = np.stack([freq_audio, freq_audio], axis=-1)
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
            processed_freq *= frequency_volume
            processed_tracks.append(processed_freq)
            logger.debug(f"Wizard: Frequency processed. Length: {len(processed_freq)} samples.")
        except Exception as e:
            logger.exception("Wizard Mix: Failed process frequency. Skipping.")
    if not processed_tracks:
        logger.error("Wizard Mix: No tracks processed.")
        return None
    mix_buffer = np.zeros((target_mix_len_samples, 2), dtype=np.float32)
    logger.info(f"Wizard: Summing {len(processed_tracks)} processed tracks.")
    for track in processed_tracks:
        track_len = len(track)
        if track_len == target_mix_len_samples:
            if track.ndim == 1:
                track = np.stack([track, track], axis=-1)
            if track.shape[1] == 2:
                mix_buffer += track
            else:
                logger.warning(f"Wizard Mix: Track shape {track.shape} unexpected. Skip add.")
        else:
            logger.error(f"Wizard Mix: Track length {track_len} != target {target_mix_len_samples}. Skip add.")
    final_mix = np.clip(mix_buffer, -1.0, 1.0)
    logger.info(f"Wizard mixing complete. Final length: {len(final_mix)} samples.")
    return final_mix.astype(np.float32)
