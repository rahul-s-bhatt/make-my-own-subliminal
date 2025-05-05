# audio_mixers.py
# ==========================================
# Functions for mixing audio tracks for both Advanced Editor and Quick Wizard.
# Removed redundant speed change from mix_wizard_tracks.
# ==========================================

import logging
import math  # Import math for ceil
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import streamlit as st  # Not used directly, but potentially by imported modules

# --- REMOVED: Import apply_speed_change (no longer needed here) ---
# from audio_utils.audio_effects import apply_speed_change
# -------------------------------------------------------------------
# Import the effects pipeline (relies on cached effects)
from audio_utils.audio_effects_pipeline import apply_all_effects

# Import the audio loader function (relies on cached load/generate functions)
from audio_utils.audio_loader import load_or_regenerate_audio

# Import definitions from the new definitions file
from audio_utils.audio_state_definitions import (
    AudioData,
    SampleRate,
    TrackDataDict,
    TrackID,
)

# Import constants and types
from config import GLOBAL_SR, MIX_PREVIEW_DURATION_S

# Import BaseTTSGenerator for type hinting
from tts.base_tts import BaseTTSGenerator

# Get a logger for this module
logger = logging.getLogger(__name__)


# ==========================================
# Track Mixing Logic (For Advanced Editor)
# ==========================================
# (mix_tracks function remains unchanged from previous context)
def mix_tracks(
    tracks_dict: Dict[TrackID, TrackDataDict],
    tts_generator: Optional[BaseTTSGenerator],  # Make optional as it might be None
    target_sr: int = GLOBAL_SR,
    preview: bool = False,
    preview_duration_s: int = MIX_PREVIEW_DURATION_S,
) -> Tuple[Optional[AudioData], Optional[int]]:
    """
    Mixes multiple tracks together for the Advanced Editor.
    Uses snippets for preview mode and reloads/regenerates full audio for export mode.
    Relies on caching within load_or_regenerate_audio and apply_all_effects.

    Args:
        tracks_dict: Dictionary where keys are track IDs and values are TrackDataDict dictionaries.
        tts_generator: An initialized TTS generator instance (needed for regeneration, can be None).
        target_sr: The sample rate for the final mix.
        preview: If True, generate a shorter preview mix using snippets.
        preview_duration_s: Duration of the preview mix in seconds.

    Returns:
        A tuple containing: (mixed audio data, length in samples), or (None, None) on failure.
    """
    logger.info(
        f"Starting ADVANCED track mixing. Preview mode: {preview}. Target SR: {target_sr}Hz."
    )
    if not tracks_dict:
        logger.warning("Mix called with no tracks provided.")
        return None, None

    # --- Track Selection Logic (Solo/Mute) ---
    active_track_ids = []
    solo_active = any(t_data.get("solo", False) for t_data in tracks_dict.values())
    logger.debug(f"Solo active: {solo_active}")
    for track_id, t_data in tracks_dict.items():
        # Check if track has *potential* audio (source info exists)
        has_audio_source = t_data.get("source_info") is not None
        is_muted = t_data.get("mute", False)
        is_active = False
        if has_audio_source:
            is_active = t_data.get("solo", False) if solo_active else not is_muted
        if is_active:
            active_track_ids.append(track_id)
        else:
            logger.debug(
                f"Skipping track '{t_data.get('name', track_id)}' from mix (Muted/NoSolo/NoSource)."
            )

    if not active_track_ids:
        logger.warning("No active tracks with audio source found for mixing.")
        return None, None

    processed_tracks: Dict[TrackID, AudioData] = {}
    track_lengths_samples: Dict[TrackID, int] = {}

    # --- Preview Mode ---
    if preview:
        logger.info(
            f"Step 1 (Preview): Processing snippets for {len(active_track_ids)} active tracks."
        )
        target_mix_len_samples = int(target_sr * preview_duration_s)
        for track_id in active_track_ids:
            t_data = tracks_dict[track_id]
            track_name = t_data.get("name", track_id)
            # --- Use the raw snippet from track_data (should be populated by TrackEditorManager) ---
            audio_snippet = t_data.get("audio_snippet")
            snippet_sr = t_data.get("sr")  # Get the SR associated with the snippet
            # ---

            if audio_snippet is None or audio_snippet.size == 0 or snippet_sr is None:
                logger.warning(
                    f"Preview: Skipping track '{track_name}' due to missing/empty snippet or SR."
                )
                continue

            # Apply effects to the raw snippet
            processed_snippet = apply_all_effects(t_data, audio_segment=audio_snippet)
            if processed_snippet is None or processed_snippet.size == 0:
                logger.warning(
                    f"Preview: Skipping track '{track_name}' after effects resulted in empty audio."
                )
                continue

            # --- Loop/Pad/Truncate processed snippet ---
            processed_len = len(processed_snippet)
            final_preview_segment = processed_snippet
            if (
                t_data.get("loop_to_fit", False)
                and processed_len > 0
                and processed_len < target_mix_len_samples
            ):
                num_repeats = target_mix_len_samples // processed_len
                remainder = target_mix_len_samples % processed_len
                looped_list = [processed_snippet] * num_repeats
                if remainder > 0:
                    looped_list.append(processed_snippet[:remainder])
                try:
                    final_preview_segment = np.concatenate(looped_list, axis=0)
                except ValueError as e_concat:
                    logger.error(
                        f"Preview: Error concatenating looped preview for '{track_name}': {e_concat}. Using non-looped/padded."
                    )
                    # Fallback padding/truncating
                    if processed_len > target_mix_len_samples:
                        final_preview_segment = processed_snippet[
                            :target_mix_len_samples
                        ]
                    elif processed_len < target_mix_len_samples:
                        final_preview_segment = np.pad(
                            processed_snippet,
                            ((0, target_mix_len_samples - processed_len), (0, 0)),
                            mode="constant",
                        )

            # Final truncate/pad to ensure exact length
            if len(final_preview_segment) > target_mix_len_samples:
                final_preview_segment = final_preview_segment[:target_mix_len_samples]
            elif len(final_preview_segment) < target_mix_len_samples:
                final_preview_segment = np.pad(
                    final_preview_segment,
                    ((0, target_mix_len_samples - len(final_preview_segment)), (0, 0)),
                    mode="constant",
                )
            # --- End Loop/Pad/Truncate ---

            processed_tracks[track_id] = final_preview_segment
            track_lengths_samples[track_id] = len(final_preview_segment)

        if not processed_tracks:
            logger.warning("Preview: No tracks successfully processed.")
            return None, None
        final_mix_len = target_mix_len_samples

    # --- Export Mode ---
    else:
        logger.info(
            f"Step 1 (Export): Processing non-looping tracks to find max length."
        )
        max_len_non_looping = 0
        non_looping_processed: Dict[TrackID, AudioData] = {}
        for track_id in active_track_ids:
            t_data = tracks_dict[track_id]
            track_name = t_data.get("name", track_id)
            if not t_data.get("loop_to_fit", False):
                logger.debug(f"Export: Processing non-looping track '{track_name}'")
                # --- CORRECT: Calls load_or_regenerate_audio for full audio ---
                full_audio_tuple = load_or_regenerate_audio(
                    t_data.get("source_info"),
                    tts_generator,  # Pass generator
                    t_data.get("sr", GLOBAL_SR),  # Use track SR hint if available
                )
                # ---
                if full_audio_tuple is None:
                    logger.warning(
                        f"Export: Failed get full audio for non-looping '{track_name}'. Skip."
                    )
                    continue
                full_audio, loaded_sr = full_audio_tuple
                # Update track's SR in the dictionary if it changed during load/gen
                if loaded_sr != t_data.get("sr", GLOBAL_SR):
                    logger.info(
                        f"Updating SR for track '{track_name}' to {loaded_sr} based on full load."
                    )
                    t_data["sr"] = (
                        loaded_sr  # Modify the dictionary directly (careful if it's a copy)
                    )
                    # If tracks_dict is a deep copy, this change won't persist outside.
                    # If it's the actual session state dict, it will. Assume it's usable for effects.

                # Apply effects to the full audio
                processed_full = apply_all_effects(t_data, audio_segment=full_audio)
                if processed_full is None or processed_full.size == 0:
                    logger.warning(
                        f"Export: Effects empty for non-looping '{track_name}'. Skip."
                    )
                    continue
                non_looping_processed[track_id] = processed_full
                track_len = len(processed_full)
                track_lengths_samples[track_id] = track_len
                max_len_non_looping = max(max_len_non_looping, track_len)
                logger.debug(
                    f"Export: Non-looping '{track_name}' processed length: {track_len}"
                )

        target_mix_len_samples = max_len_non_looping
        logger.info(
            f"Step 2 (Export): Target mix length determined: {target_mix_len_samples} samples ({target_mix_len_samples / target_sr:.2f}s)"
        )

        # Estimate length if no non-looping tracks provided content
        if target_mix_len_samples <= 0:
            logger.warning(
                "Export: Target mix length zero. Trying estimate from looping tracks."
            )
            max_loop_target_len = 0
            for track_id in active_track_ids:
                t_data = tracks_dict[track_id]
                if t_data.get("loop_to_fit", False):
                    source_info = t_data.get("source_info")
                    sr = t_data.get("sr", GLOBAL_SR)
                    if (
                        isinstance(source_info, dict)
                        and "target_duration_s" in source_info
                        and source_info["target_duration_s"] is not None
                        and sr > 0
                    ):
                        try:
                            max_loop_target_len = max(
                                max_loop_target_len,
                                int(float(source_info["target_duration_s"]) * sr),
                            )
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Could not parse target_duration_s for {track_id}"
                            )
            if max_loop_target_len > 0:
                target_mix_len_samples = max_loop_target_len
                logger.info(
                    f"Using estimated length from looping tracks: {target_mix_len_samples}"
                )
            else:
                logger.error(
                    "Export: Cannot determine target mix length from any track."
                )
                return None, None

        logger.info(
            f"Step 3 (Export): Processing looping tracks to match target length {target_mix_len_samples}."
        )
        for track_id in active_track_ids:
            if track_id in non_looping_processed:
                processed_tracks[track_id] = non_looping_processed[
                    track_id
                ]  # Use already processed non-looping
                continue
            # Process looping tracks
            t_data = tracks_dict[track_id]
            if t_data.get("loop_to_fit", False):
                track_name = t_data.get("name", track_id)
                logger.debug(f"Export: Processing looping track '{track_name}'")
                # --- CORRECT: Calls load_or_regenerate_audio for full audio ---
                full_audio_tuple = load_or_regenerate_audio(
                    t_data.get("source_info"),
                    tts_generator,  # Pass generator
                    t_data.get("sr", GLOBAL_SR),  # Use track SR hint if available
                    target_mix_len_samples,  # Pass target length hint
                )
                # ---
                if full_audio_tuple is None:
                    logger.warning(
                        f"Export: Failed get full audio for looping '{track_name}'. Skip."
                    )
                    continue
                full_audio, loaded_sr = full_audio_tuple
                # Update track's SR if needed
                if loaded_sr != t_data.get("sr", GLOBAL_SR):
                    logger.info(
                        f"Updating SR for track '{track_name}' to {loaded_sr} based on full load."
                    )
                    t_data["sr"] = loaded_sr

                # Apply effects to full audio
                processed_full = apply_all_effects(t_data, audio_segment=full_audio)
                if processed_full is None or processed_full.size == 0:
                    logger.warning(
                        f"Export: Effects empty for looping '{track_name}'. Skip."
                    )
                    continue

                # --- Loop/Pad/Truncate processed looping track ---
                processed_len = len(processed_full)
                track_lengths_samples[track_id] = (
                    processed_len  # Store original processed length
                )
                final_audio_for_track = processed_full
                if processed_len > 0 and processed_len < target_mix_len_samples:
                    logger.debug(
                        f"Export: Looping track '{track_name}' from {processed_len} to {target_mix_len_samples}."
                    )
                    num_repeats = target_mix_len_samples // processed_len
                    remainder = target_mix_len_samples % processed_len
                    looped_list = [processed_full] * num_repeats
                    if remainder > 0:
                        looped_list.append(processed_full[:remainder])
                    try:
                        final_audio_for_track = np.concatenate(looped_list, axis=0)
                    except ValueError as e_concat:
                        logger.error(
                            f"Export: Error concat loop '{track_name}': {e_concat}. Using pad."
                        )
                        final_audio_for_track = np.pad(
                            processed_full,
                            ((0, target_mix_len_samples - processed_len), (0, 0)),
                            mode="constant",
                        )
                # Final truncate/pad
                if len(final_audio_for_track) > target_mix_len_samples:
                    final_audio_for_track = final_audio_for_track[
                        :target_mix_len_samples
                    ]
                elif len(final_audio_for_track) < target_mix_len_samples:
                    logger.warning(
                        f"Padding looping track '{track_name}' to meet final length."
                    )
                    final_audio_for_track = np.pad(
                        final_audio_for_track,
                        (
                            (0, target_mix_len_samples - len(final_audio_for_track)),
                            (0, 0),
                        ),
                        mode="constant",
                    )
                # --- End Loop/Pad/Truncate ---
                processed_tracks[track_id] = final_audio_for_track

        logger.info(
            f"Step 3b (Export): Padding non-looping tracks to final length {target_mix_len_samples}."
        )
        for track_id, audio_data in processed_tracks.items():
            # Only pad/truncate tracks that were originally non-looping
            if not tracks_dict[track_id].get("loop_to_fit", False):
                current_len = len(audio_data)
                if current_len < target_mix_len_samples:
                    logger.debug(
                        f"Padding non-looping {track_id} from {current_len} to {target_mix_len_samples}"
                    )
                    processed_tracks[track_id] = np.pad(
                        audio_data,
                        ((0, target_mix_len_samples - current_len), (0, 0)),
                        mode="constant",
                    )
                elif current_len > target_mix_len_samples:
                    # This shouldn't happen if max_len_non_looping was calculated correctly, but safety check
                    logger.warning(
                        f"Truncating non-looping {track_id} from {current_len} to {target_mix_len_samples}"
                    )
                    processed_tracks[track_id] = audio_data[:target_mix_len_samples]

        if not processed_tracks:
            logger.warning("Export: No tracks processed after looping/padding.")
            return None, None
        final_mix_len = target_mix_len_samples

    # --- Final Mixing ---
    logger.info(
        f"Step 4: Adding {len(processed_tracks)} processed tracks to mix buffer (Length: {final_mix_len} samples)."
    )
    # Ensure mix buffer matches target SR (should be consistent by now)
    mix_buffer = np.zeros((final_mix_len, 2), dtype=np.float32)
    for track_id, processed_audio in processed_tracks.items():
        t_data = tracks_dict[track_id]
        track_name = t_data.get("name", track_id)
        if len(processed_audio) != final_mix_len:
            logger.error(
                f"Track '{track_name}' length mismatch ({len(processed_audio)} vs {final_mix_len}). Skip add."
            )
            continue
        # Apply final volume and pan
        pan = t_data.get("pan", 0.0)
        vol = t_data.get("volume", 1.0)
        logger.debug(
            f"Track '{track_name}': Applying final vol={vol:.2f}, pan={pan:.2f}"
        )
        pan_rad = (pan + 1.0) * np.pi / 4.0
        left_gain = vol * np.cos(pan_rad)
        right_gain = vol * np.sin(pan_rad)
        try:
            # Ensure stereo before adding
            if processed_audio.ndim == 1:
                logger.warning(
                    f"Track '{track_name}' mono in final mix buffer stage. Applying gains to both channels."
                )
                mix_buffer[:, 0] += processed_audio * left_gain
                mix_buffer[:, 1] += processed_audio * right_gain
            elif processed_audio.shape[1] == 2:
                mix_buffer[:, 0] += processed_audio[:, 0] * left_gain
                mix_buffer[:, 1] += processed_audio[:, 1] * right_gain
            else:
                logger.error(
                    f"Track '{track_name}' unexpected shape {processed_audio.shape}. Skip add."
                )
                continue
        except ValueError as e_add:
            logger.error(
                f"Error adding track '{track_name}' to mix: {e_add}. Buffer: {mix_buffer.shape}, Track: {processed_audio.shape}"
            )
            continue
        logger.debug(f"Added track '{track_name}' to mix buffer.")

    # Final clipping
    final_mix = np.clip(mix_buffer, -1.0, 1.0)
    logger.info(
        f"Mixing complete. Final mix length: {final_mix_len} samples ({final_mix_len / target_sr:.2f}s)."
    )
    return final_mix.astype(np.float32), final_mix_len


# ==========================================
# Track Mixing Logic (For Quick Wizard)
# ==========================================
# No caching added here - relies on caching in called functions
def mix_wizard_tracks(
    affirmation_audio: Optional[Tuple[AudioData, SampleRate]],
    background_audio: Optional[Tuple[AudioData, SampleRate]],
    frequency_audio: Optional[Tuple[AudioData, SampleRate]],
    # --- REMOVED affirmation_speed parameter ---
    affirmation_volume: float = 0.05,  # Keep volume parameters
    background_volume: float = 1.0,
    frequency_volume: float = 1.0,
    target_sr: int = GLOBAL_SR,
) -> Optional[AudioData]:
    """
    Mixes tracks specifically for the Quick Create Wizard workflow.
    Assumes affirmation audio is already speed-adjusted if necessary.
    Relies on audio being passed in already loaded/generated.
    """
    logger.info("Starting Quick Wizard track mixing.")
    processed_tracks = []
    target_mix_len_samples = 0  # Determine from affirmation track

    # --- Process Affirmation ---
    if affirmation_audio is None or affirmation_audio[0].size == 0:
        logger.error("Wizard Mix: Affirmation audio missing or empty.")
        return None
    aff_audio, aff_sr = affirmation_audio
    try:
        logger.debug(f"Wizard: Processing affirmation (Vol: {affirmation_volume})")
        # Resample if needed (should ideally already be target_sr)
        if aff_sr != target_sr:
            logger.warning(
                f"Wizard: Resampling affirmation from {aff_sr} to {target_sr} in mixer."
            )
            aff_audio_float = aff_audio.astype(np.float32)
            # Handle mono/stereo resampling
            if aff_audio_float.ndim == 1:
                aff_audio = librosa.resample(
                    aff_audio_float, orig_sr=aff_sr, target_sr=target_sr
                )
            elif aff_audio_float.shape[1] == 1:  # Treat as mono
                aff_audio = librosa.resample(
                    aff_audio_float[:, 0], orig_sr=aff_sr, target_sr=target_sr
                )
            elif aff_audio_float.shape[1] == 2:  # Stereo
                aff_audio = librosa.resample(
                    aff_audio_float.T, orig_sr=aff_sr, target_sr=target_sr
                ).T
            else:  # Unexpected shape
                logger.error(
                    f"Wizard: Affirmation has unexpected shape {aff_audio_float.shape} for resampling."
                )
                return None
            aff_sr = target_sr  # Update SR after resampling

        # --- REMOVED call to apply_speed_change ---
        processed_aff = aff_audio  # Use the audio directly
        # ------------------------------------------

        # Ensure stereo and apply volume
        if processed_aff.ndim == 1:
            processed_aff = np.stack([processed_aff, processed_aff], axis=-1)
        elif processed_aff.shape[1] != 2:
            logger.error(
                f"Affirmation audio has unexpected shape {processed_aff.shape} after processing."
            )
            return None

        processed_aff = processed_aff.astype(np.float32) * affirmation_volume
        processed_tracks.append(processed_aff)
        target_mix_len_samples = len(processed_aff)
        logger.debug(
            f"Wizard: Affirmation processed. Length: {target_mix_len_samples} samples."
        )
    except Exception as e:
        logger.exception("Wizard Mix: Failed to process affirmation.")
        return None

    if target_mix_len_samples <= 0:
        logger.error("Wizard Mix: Processed affirmation has zero length.")
        return None

    # --- Process Background (if exists) ---
    if background_audio and background_audio[0].size > 0:
        bg_audio, bg_sr = background_audio
        try:
            logger.debug(f"Wizard: Processing background (Vol: {background_volume})")
            # Resample if needed
            if bg_sr != target_sr:
                logger.warning(
                    f"Wizard: Resampling background from {bg_sr} to {target_sr} in mixer."
                )
                bg_audio_float = bg_audio.astype(np.float32)
                if bg_audio_float.ndim == 1:
                    bg_audio = librosa.resample(
                        bg_audio_float, orig_sr=bg_sr, target_sr=target_sr
                    )
                elif bg_audio_float.shape[1] == 1:
                    bg_audio = librosa.resample(
                        bg_audio_float[:, 0], orig_sr=bg_sr, target_sr=target_sr
                    )
                elif bg_audio_float.shape[1] == 2:
                    bg_audio = librosa.resample(
                        bg_audio_float.T, orig_sr=bg_sr, target_sr=target_sr
                    ).T
                else:
                    raise ValueError(
                        f"Background audio not mono/stereo: {bg_audio_float.shape}"
                    )
                bg_sr = target_sr

            # Ensure stereo
            if bg_audio.ndim == 1:
                bg_audio = np.stack([bg_audio, bg_audio], axis=-1)
            elif bg_audio.shape[1] != 2:
                raise ValueError(
                    f"Background audio not stereo after potential resample: {bg_audio.shape}"
                )

            # Loop/Truncate/Pad
            processed_bg = bg_audio
            bg_len = len(processed_bg)
            if bg_len > 0 and bg_len < target_mix_len_samples:
                num_repeats = math.ceil(target_mix_len_samples / bg_len)
                looped_audio = np.tile(
                    processed_bg, (num_repeats, 1)
                )  # Tile expects 2D for axis 1 repeat
                processed_bg = looped_audio[:target_mix_len_samples]
            elif bg_len > target_mix_len_samples:
                processed_bg = processed_bg[:target_mix_len_samples]
            elif bg_len < target_mix_len_samples:  # Pad if shorter
                processed_bg = np.pad(
                    processed_bg,
                    ((0, target_mix_len_samples - bg_len), (0, 0)),
                    mode="constant",
                )

            processed_bg = processed_bg.astype(np.float32) * background_volume
            processed_tracks.append(processed_bg)
            logger.debug(
                f"Wizard: Background processed. Length: {len(processed_bg)} samples."
            )
        except Exception as e:
            logger.exception("Wizard Mix: Failed process background. Skipping.")

    # --- Process Frequency (if exists) ---
    if frequency_audio and frequency_audio[0].size > 0:
        freq_audio, freq_sr = frequency_audio
        try:
            logger.debug(f"Wizard: Processing frequency (Vol: {frequency_volume})")
            # Resample if needed
            if freq_sr != target_sr:
                logger.warning(
                    f"Wizard: Resampling frequency from {freq_sr} to {target_sr} in mixer."
                )
                freq_audio_float = freq_audio.astype(np.float32)
                if freq_audio_float.ndim == 1:
                    freq_audio = librosa.resample(
                        freq_audio_float, orig_sr=freq_sr, target_sr=target_sr
                    )
                elif freq_audio_float.shape[1] == 1:
                    freq_audio = librosa.resample(
                        freq_audio_float[:, 0], orig_sr=freq_sr, target_sr=target_sr
                    )
                elif freq_audio_float.shape[1] == 2:
                    freq_audio = librosa.resample(
                        freq_audio_float.T, orig_sr=freq_sr, target_sr=target_sr
                    ).T
                else:
                    raise ValueError(
                        f"Frequency audio not mono/stereo: {freq_audio_float.shape}"
                    )
                freq_sr = target_sr

            # Ensure stereo
            if freq_audio.ndim == 1:
                freq_audio = np.stack([freq_audio, freq_audio], axis=-1)
            elif freq_audio.shape[1] != 2:
                raise ValueError(
                    f"Frequency audio not stereo after potential resample: {freq_audio.shape}"
                )

            # Loop/Truncate/Pad
            processed_freq = freq_audio
            freq_len = len(processed_freq)
            if freq_len > 0 and freq_len < target_mix_len_samples:
                num_repeats = math.ceil(target_mix_len_samples / freq_len)
                looped_audio = np.tile(processed_freq, (num_repeats, 1))
                processed_freq = looped_audio[:target_mix_len_samples]
            elif freq_len > target_mix_len_samples:
                processed_freq = processed_freq[:target_mix_len_samples]
            elif freq_len < target_mix_len_samples:
                processed_freq = np.pad(
                    processed_freq,
                    ((0, target_mix_len_samples - freq_len), (0, 0)),
                    mode="constant",
                )

            processed_freq = processed_freq.astype(np.float32) * frequency_volume
            processed_tracks.append(processed_freq)
            logger.debug(
                f"Wizard: Frequency processed. Length: {len(processed_freq)} samples."
            )
        except Exception as e:
            logger.exception("Wizard Mix: Failed process frequency. Skipping.")

    # --- Final mix (Wizard) ---
    if not processed_tracks:
        logger.error("Wizard Mix: No tracks available for mixing.")
        return None

    mix_buffer = np.zeros((target_mix_len_samples, 2), dtype=np.float32)
    logger.info(f"Wizard: Summing {len(processed_tracks)} processed tracks.")
    for i, track in enumerate(processed_tracks):
        track_len = len(track)
        if track_len == target_mix_len_samples:
            # Ensure track is stereo before adding
            if track.ndim == 1:
                track = np.stack(
                    [track, track], axis=-1
                )  # Should be stereo already from processing steps
            if track.shape[1] == 2:
                mix_buffer += track
            else:
                logger.warning(
                    f"Wizard Mix: Track {i} shape {track.shape} unexpected. Skip add."
                )
        else:
            logger.error(
                f"Wizard Mix: Track {i} len {track_len} != target {target_mix_len_samples}. Skip add."
            )

    final_mix = np.clip(mix_buffer, -1.0, 1.0)
    logger.info(f"Wizard mixing complete. Final length: {len(final_mix)} samples.")
    return final_mix.astype(np.float32)
