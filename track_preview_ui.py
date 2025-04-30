# track_preview_ui.py
# ==========================================
# Track Preview Column UI for MindMorph Editor
# ==========================================

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

# Import necessary components from other modules
# <<< Updated AppState import >>>
from app_state import AppState, TrackDataDict, TrackID
from audio_io import save_audio_to_temp_file
from audio_processing import AudioData, get_preview_audio
from config import (
    GLOBAL_SR,
    PREVIEW_DURATION_S,
    TRACK_TYPE_AFFIRMATION,  # Needed for subliminalize button
    ULTRASONIC_TARGET_FREQ,
)

# Import UI-specific libraries conditionally
try:
    from streamlit_advanced_audio import WaveSurferOptions, audix

    STREAMLIT_ADVANCED_AUDIO_AVAILABLE = True
except ImportError:
    STREAMLIT_ADVANCED_AUDIO_AVAILABLE = False

    # Define dummy functions if library is missing
    def WaveSurferOptions(**kwargs):
        return None

    def audix(*args, **kwargs):
        pass  # Dummy function


logger = logging.getLogger(__name__)


class TrackPreviewUI:
    """Handles rendering the main preview and controls column for a track."""

    def __init__(self, app_state: AppState):
        """
        Initializes the TrackPreviewUI.

        Args:
            app_state: An instance of the AppState class.
        """
        self.app_state = app_state
        logger.debug("TrackPreviewUI initialized.")

    # <<< MODIFIED: Hash based on snippet size/presence >>>
    def _calculate_preview_hash(self, track_data: TrackDataDict) -> int:
        """Calculates a hash based on settings relevant to the audio preview."""
        audio_snippet = track_data.get("audio_snippet")
        # Hash relevant settings + snippet info (size as a proxy for content change)
        params_to_hash = (
            track_data.get("volume", 1.0),
            track_data.get("speed_factor", 1.0),
            track_data.get("pitch_shift", 0.0),
            track_data.get("pan", 0.0),
            track_data.get("filter_type", "off"),
            track_data.get("filter_cutoff", 8000.0),
            track_data.get("reverse_audio", False),
            track_data.get("ultrasonic_shift", False),
            # Use snippet size instead of original_audio size
            audio_snippet.size if audio_snippet is not None else 0,
            track_data.get("sr", GLOBAL_SR),
            # Include loop setting as it affects preview generation
            track_data.get("loop_to_fit", False),
        )
        settings_hash = hash(params_to_hash)
        return settings_hash

    # <<< MODIFIED: Use audio_snippet >>>
    def render_preview_column(self, track_id: TrackID, track_data: TrackDataDict, column: st.delta_generator.DeltaGenerator):
        """Renders the main controls for a single track (waveform, settings, preview button)."""
        with column:
            try:
                # <<< Get audio_snippet instead of original_audio >>>
                audio_snippet = track_data.get("audio_snippet")
                track_sr = track_data.get("sr", GLOBAL_SR)
                app_mode = st.session_state.get("app_mode", "Easy")
                source_info = track_data.get("source_info")  # Get source info for context

                # --- Waveform Preview Display ---
                st.markdown("**Preview (First 60s with Effects Applied)**")
                display_path = None
                preview_cache_hit = False

                # <<< Check if audio_snippet exists >>>
                if audio_snippet is not None and audio_snippet.size > 0:
                    current_settings_hash = self._calculate_preview_hash(track_data)
                    stored_hash = track_data.get("preview_settings_hash")
                    stored_path = track_data.get("preview_temp_file_path")

                    # Check if cached preview is valid
                    if stored_hash is not None and current_settings_hash == stored_hash and stored_path and isinstance(stored_path, str) and os.path.exists(stored_path):
                        display_path = stored_path
                        preview_cache_hit = True
                    # Clean up if path exists in state but file doesn't
                    elif stored_path and isinstance(stored_path, str) and not os.path.exists(stored_path):
                        logger.warning(f"Cached preview file missing for track {track_id}: {stored_path}. Clearing path.")
                        self.app_state.update_track_param(track_id, "preview_temp_file_path", None)
                        self.app_state.update_track_param(track_id, "preview_settings_hash", None)

                    # --- Display Logic ---
                    if display_path:
                        if STREAMLIT_ADVANCED_AUDIO_AVAILABLE:
                            ws_options = WaveSurferOptions(
                                height=100, normalize=True, wave_color="#A020F0", progress_color="#800080", cursor_color="#333333", cursor_width=1, bar_width=2, bar_gap=1
                            )
                            update_count = track_data.get("update_counter", 0)
                            audix_key = f"audix_preview_{track_id}_{update_count}"
                            try:
                                audix(data=display_path, sample_rate=track_sr, wavesurfer_options=ws_options, key=audix_key)
                            except Exception as audix_err:
                                logger.error(f"Error rendering audix waveform for {track_id}: {audix_err}")
                                st.warning("Waveform preview failed to render. Showing standard audio player instead.")
                                try:
                                    st.audio(display_path, format="audio/wav")
                                except Exception as audio_err:
                                    logger.error(f"Fallback st.audio player also failed for {track_id}: {audio_err}")
                                    st.error("Could not display audio preview.")
                        else:
                            st.audio(display_path, format="audio/wav")
                    elif not preview_cache_hit:
                        st.info("Settings changed or preview not generated. Click 'Update Preview' below.")

                else:  # No audio snippet data
                    if source_info and source_info.get("type") == "upload":
                        # Check if the temporary file path exists in source_info
                        temp_file_path = source_info.get("temp_file_path")
                        original_filename = source_info.get("original_filename", "N/A")
                        if temp_file_path and os.path.exists(temp_file_path):
                            st.warning(f"Audio snippet missing for '{original_filename}'. Try 'Update Preview'.", icon="⚠️")
                        else:
                            st.error(f"Source file missing for '{original_filename}'. Please remove and re-add the track.", icon="❌")
                    else:
                        st.warning("Track currently has no audio snippet data. Try 'Update Preview'.")

                # Display track info (Show snippet duration)
                snippet_len_samples = len(audio_snippet) if audio_snippet is not None else 0
                snippet_len_sec = snippet_len_samples / track_sr if track_sr > 0 else 0
                st.caption(f"Snippet Duration: {snippet_len_sec:.2f}s | SR: {track_sr} Hz")
                st.markdown("---")

                # --- Basic Settings (Volume, Pan) ---
                st.markdown("**Track Settings**")
                basic_cols = st.columns(2)
                with basic_cols[0]:
                    vol = st.slider("Volume", 0.0, 2.0, float(track_data.get("volume", 1.0)), 0.05, key=f"vol_{track_id}", help="Adjust track loudness.")
                    if not np.isclose(vol, track_data.get("volume", 1.0)):
                        self.app_state.update_track_param(track_id, "volume", vol)
                with basic_cols[1]:
                    pan = st.slider("Pan", -1.0, 1.0, float(track_data.get("pan", 0.0)), 0.1, key=f"pan_{track_id}", help="Adjust stereo balance.")
                    if not np.isclose(pan, track_data.get("pan", 0.0)):
                        self.app_state.update_track_param(track_id, "pan", pan)

                # --- Subliminalize Preset Button ---
                if track_data.get("track_type") == TRACK_TYPE_AFFIRMATION:
                    if st.button("⚡ Quick Subliminal Settings", key=f"subliminalize_preset_{track_id}", help="Applies Speed=10x, Volume=0.05."):
                        logger.info(f"Subliminalize preset applied to track {track_id}")
                        self.app_state.update_track_param(track_id, "speed_factor", 10.0)
                        self.app_state.update_track_param(track_id, "volume", 0.05)
                        st.toast("Speed set to 10x, Volume to 0.05. Click 'Update Preview'.", icon="⚡")
                        st.rerun()

                # --- Advanced Settings (Conditional on App Mode) ---
                if app_mode == "Advanced":
                    st.markdown("**Advanced Effects**")
                    st.caption("Adjust speed, pitch, filter, etc. Click 'Update Preview' after changes.")

                    ultrasonic_enabled = track_data.get("ultrasonic_shift", False)
                    ultrasonic_checkbox = st.checkbox(
                        "🔊 Ultrasonic Shift ('Silent')",
                        value=ultrasonic_enabled,
                        key=f"ultrasonic_{track_id}",
                        help=f"EXPERIMENTAL: Shifts audio towards {ULTRASONIC_TARGET_FREQ}Hz. Disables Pitch Shift.",
                    )
                    if ultrasonic_checkbox != ultrasonic_enabled:
                        self.app_state.update_track_param(track_id, "ultrasonic_shift", ultrasonic_checkbox)
                        st.rerun()

                    col_fx1_1, col_fx1_2, col_fx1_3 = st.columns([0.6, 1, 1])
                    with col_fx1_1:
                        st.markdown("<br/>", unsafe_allow_html=True)
                        loop_value = st.checkbox("🔁 Loop", value=track_data.get("loop_to_fit", False), key=f"loop_{track_id}", help="Loop track during final mix?")
                        if loop_value != track_data.get("loop_to_fit"):
                            self.app_state.update_track_param(track_id, "loop_to_fit", loop_value)
                        st.markdown("<br/>", unsafe_allow_html=True)
                        reverse_value = st.checkbox("🔄 Reverse", value=track_data.get("reverse_audio", False), key=f"reverse_{track_id}", help="Reverse audio playback?")
                        if reverse_value != track_data.get("reverse_audio"):
                            self.app_state.update_track_param(track_id, "reverse_audio", reverse_value)

                    with col_fx1_2:
                        speed = st.slider("Speed", 0.1, 16.0, float(track_data.get("speed_factor", 1.0)), 0.05, key=f"speed_{track_id}", help="Playback speed factor.")
                        if not np.isclose(speed, track_data.get("speed_factor", 1.0)):
                            self.app_state.update_track_param(track_id, "speed_factor", speed)

                    with col_fx1_3:
                        pitch_disabled = track_data.get("ultrasonic_shift", False)
                        pitch = st.slider(
                            "Pitch (semitones)",
                            -24,
                            24,
                            int(track_data.get("pitch_shift", 0)),
                            1,
                            key=f"pitch_{track_id}",
                            help="Adjust pitch. Disabled if Ultrasonic is active.",
                            disabled=pitch_disabled,
                        )
                        if not pitch_disabled and pitch != track_data.get("pitch_shift", 0):
                            self.app_state.update_track_param(track_id, "pitch_shift", float(pitch))

                    col_fx2_1, col_fx2_2 = st.columns(2)
                    with col_fx2_1:
                        filter_disabled = track_data.get("ultrasonic_shift", False)
                        filter_options = ["off", "lowpass", "highpass"]
                        current_filter_type = track_data.get("filter_type", "off")
                        try:
                            filter_index = filter_options.index(current_filter_type)
                        except ValueError:
                            filter_index = 0
                        f_type = st.selectbox(
                            "Filter",
                            filter_options,
                            index=filter_index,
                            key=f"filter_type_{track_id}",
                            help="Apply standard filter. Disabled if Ultrasonic is active.",
                            disabled=filter_disabled,
                        )
                        if not filter_disabled and f_type != current_filter_type:
                            self.app_state.update_track_param(track_id, "filter_type", f_type)
                            st.rerun()

                    with col_fx2_2:
                        filter_type_active = track_data.get("filter_type", "off") != "off"
                        cutoff_disabled = filter_disabled or not filter_type_active
                        max_cutoff = (track_sr / 2.0 - 1.0) if track_sr else 20000.0
                        min_cutoff = 20.0
                        f_cutoff = st.number_input(
                            f"Cutoff Freq (Hz)",
                            min_cutoff,
                            max(min_cutoff, max_cutoff),
                            float(track_data.get("filter_cutoff", 8000.0)),
                            100.0,
                            key=f"filter_cutoff_{track_id}",
                            disabled=cutoff_disabled,
                            help=f"Filter cutoff ({min_cutoff:.0f}Hz - {max_cutoff:.0f}Hz).",
                            format="%.1f",
                        )
                        f_cutoff_clamped = max(min_cutoff, min(f_cutoff, max_cutoff))
                        if not cutoff_disabled and not np.isclose(f_cutoff_clamped, track_data.get("filter_cutoff", 8000.0)):
                            self.app_state.update_track_param(track_id, "filter_cutoff", f_cutoff_clamped)

                # --- Update Preview Button ---
                st.markdown("---")
                # <<< Disable button if audio_snippet is missing >>>
                update_disabled = audio_snippet is None or audio_snippet.size == 0
                if st.button(
                    "⚙️ Update Preview",
                    key=f"update_track_preview_{track_id}",
                    help="Generate the 60s preview using the snippet and current settings.",
                    disabled=update_disabled,
                    use_container_width=True,
                ):
                    self._handle_update_preview(track_id, track_data)  # Call helper method

            except Exception as e:
                logger.exception(f"Error rendering preview column for track {track_id}")
                st.error(f"An error occurred displaying controls for this track: {e}")

    # <<< Helper method for handling preview generation >>>
    def _handle_update_preview(self, track_id: TrackID, track_data: TrackDataDict):
        """Handles the logic when 'Update Preview' is clicked."""
        logger.info(f"Update Preview clicked for: '{track_data.get('name', 'N/A')}' ({track_id})")
        audio_snippet = track_data.get("audio_snippet")
        track_sr = track_data.get("sr", GLOBAL_SR)

        if audio_snippet is None or audio_snippet.size == 0:
            st.error("Cannot generate preview, audio snippet data is missing.")
            logger.error(f"Preview generation failed for {track_id}: audio_snippet is missing.")
            return

        with st.spinner("Generating preview audio..."):
            try:
                # <<< Pass audio_snippet to get_preview_audio >>>
                # get_preview_audio needs to be adapted to accept snippet instead of full track_data if necessary
                # Assuming get_preview_audio primarily needs the snippet and other params from track_data
                preview_audio = get_preview_audio(track_data, preview_duration_s=PREVIEW_DURATION_S)

                if preview_audio is not None and preview_audio.size > 0:
                    new_preview_path = save_audio_to_temp_file(preview_audio, track_sr)
                    if new_preview_path:
                        old_preview_path = track_data.get("preview_temp_file_path")
                        new_settings_hash = self._calculate_preview_hash(track_data)
                        # Update state with new path and hash
                        self.app_state.update_track_param(track_id, "preview_temp_file_path", new_preview_path)
                        self.app_state.update_track_param(track_id, "preview_settings_hash", new_settings_hash)
                        self.app_state.increment_update_counter(track_id)  # Force audix refresh
                        # Clean up old preview file
                        if old_preview_path and isinstance(old_preview_path, str) and old_preview_path != new_preview_path and os.path.exists(old_preview_path):
                            try:
                                os.remove(old_preview_path)
                                logger.info(f"Deleted old preview file: {old_preview_path}")
                            except OSError as e:
                                logger.warning(f"Could not delete old preview file {old_preview_path}: {e}")
                        st.toast("Preview updated.", icon="✅")
                        st.rerun()
                    else:
                        logger.error(f"Failed to save new preview temp file for track {track_id}.")
                        st.error("Failed to save updated preview file.")
                else:
                    logger.error(f"Failed to generate preview audio for track {track_id} (get_preview_audio returned None or empty).")
                    st.error("Failed to generate preview audio. Check settings or logs.")
            except Exception as e:
                logger.exception(f"Error during preview generation for track {track_id}")
                st.error(f"Error generating preview: {e}")
