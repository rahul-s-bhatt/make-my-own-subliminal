# track_editor_manager.py
# ==========================================
# Track Editor UI Management for MindMorph
# ==========================================

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st
from PIL import Image

# Import necessary components from other modules
from app_state import AppState, TrackData, TrackID, TrackType

# --- Updated Audio Imports ---
from audio_io import save_audio_to_temp_file
from audio_processing import AudioData, get_preview_audio

# --- End Updated Audio Imports ---
from config import (
    FAVICON_PATH,  # For empty state message
    GLOBAL_SR,
    PREVIEW_DURATION_S,
    TRACK_TYPE_AFFIRMATION,
    TRACK_TYPE_OTHER,
    TRACK_TYPES,  # Import TRACK_TYPES here as it's used in _render_track_controls_col
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
        # This dummy function won't be called if the library is missing,
        # as the code checks STREAMLIT_ADVANCED_AUDIO_AVAILABLE first.
        # If it were called, raising an error might be more explicit.
        # However, the current logic relies on the boolean flag.
        pass


# Get a logger for this module
logger = logging.getLogger(__name__)


class TrackEditorManager:
    """Handles rendering the main track editor panel with track controls."""

    def __init__(self, app_state: AppState):
        """
        Initializes the TrackEditorManager.

        Args:
            app_state: An instance of the AppState class.
        """
        self.app_state = app_state
        logger.debug("TrackEditorManager initialized.")

    # --- Helper Method ---
    def _calculate_preview_hash(self, track_data: TrackData) -> int:
        """Calculates a hash based on settings relevant to the audio preview."""
        # Define the parameters that affect the audio output for preview
        params_to_hash = (
            track_data.get("volume", 1.0),
            track_data.get("speed_factor", 1.0),
            track_data.get("pitch_shift", 0.0),
            track_data.get("pan", 0.0),
            track_data.get("filter_type", "off"),
            track_data.get("filter_cutoff", 8000.0),
            track_data.get("reverse_audio", False),
            track_data.get("ultrasonic_shift", False),
            track_data.get("original_audio").size if track_data.get("original_audio") is not None else 0,
            track_data.get("sr", GLOBAL_SR),
        )
        settings_hash = hash(params_to_hash)
        # Reduced logging frequency for hash calculation
        # logger.debug(f"Calculated preview hash for track '{track_data.get('name', 'N/A')}': {settings_hash}")
        return settings_hash

    # --- Main Rendering Method ---

    def render_tracks_editor(self):
        """Renders the main editor area with track controls."""
        st.header("🎚️ Tracks Editor")
        tracks = self.app_state.get_all_tracks()
        app_mode = st.session_state.get("app_mode", "Easy")

        # --- Handle Empty State ---
        if not tracks:
            if "welcome_message_shown" in st.session_state:
                col_icon, col_text = st.columns([1, 5])
                with col_icon:
                    st.markdown("<br/>", unsafe_allow_html=True)
                    if os.path.exists(FAVICON_PATH):
                        try:
                            st.image(Image.open(FAVICON_PATH), width=80)
                        except Exception:
                            st.markdown("🧠", unsafe_allow_html=True)
                    else:
                        st.markdown("🧠", unsafe_allow_html=True)
                with col_text:
                    st.subheader("✨ Let's Create Your Subliminal!")
                    st.markdown("Your project is empty. Use the **sidebar on the left** (👈) to add your first audio layer.")
                    st.markdown("- **Upload** your own audio files (music, voice).")
                    st.markdown("- Generate **Affirmations** from text or a file.")
                    st.markdown("- Add background **Noise** (White, Pink, Brown).")
                    if app_mode == "Easy":
                        st.markdown("- Add **Frequency Presets** for focus, relaxation, etc.")
                    else:
                        st.markdown("- Add specific **Frequencies/Tones** or use Presets.")
                st.markdown("---")
                st.info("Once you add a track, its editor controls will appear here.")
            return  # Stop rendering if no tracks

        # --- Render Tracks ---
        st.caption(f"Current Mode: **{app_mode}** | Tracks: {len(tracks)}")
        st.markdown("Adjust settings for each track below. Click **'Update Preview'** inside a track's panel to refresh its 60s preview with the latest settings applied.")
        st.divider()

        track_ids_to_delete = []
        logger.debug(f"Rendering editor for {len(tracks)} tracks.")

        track_ids = list(tracks.keys())
        for track_id in track_ids:
            track_data = self.app_state.get_track(track_id)
            if track_data is None:
                logger.warning(f"Track {track_id} not found during editor rendering, likely deleted.")
                continue

            track_name = track_data.get("name", "Unnamed Track")
            track_type_str = track_data.get("track_type", TRACK_TYPE_OTHER)
            track_type_icon = track_type_str.split(" ")[0] if " " in track_type_str else "⚪"

            expander_label = f"{track_type_icon} Track: **{track_name}**"
            if track_data.get("source_type") == "upload" and track_data.get("original_audio") is None:
                expander_label += "  ⚠️ Missing Source File"
            expander_label += f"  (`...{track_id[-6:]}`)"

            with st.expander(expander_label, expanded=True):
                logger.debug(f"Rendering expander for: '{track_name}' ({track_id}), Type: {track_type_str}")
                col_main, col_controls = st.columns([3, 1])

                self._render_track_main_col(track_id, track_data, col_main)
                deleted = self._render_track_controls_col(track_id, track_data, col_controls)

                if deleted:
                    track_ids_to_delete.append(track_id)

        # --- Process Deletions ---
        if track_ids_to_delete:
            deleted_count = 0
            for tid in track_ids_to_delete:
                if self.app_state.delete_track(tid):  # delete_track handles preview file cleanup
                    deleted_count += 1
            if deleted_count > 0:
                st.toast(f"Deleted {deleted_count} track(s).")
                logger.info(f"Processed deletion of {deleted_count} tracks.")
                st.rerun()  # Rerun to update the UI immediately after deletion

    # --- Private Rendering Methods for Track Editor Sections ---

    def _render_track_main_col(self, track_id: TrackID, track_data: TrackData, column: st.delta_generator.DeltaGenerator):
        """Renders the main controls for a single track (waveform, settings, preview button)."""
        with column:
            try:
                original_audio = track_data.get("original_audio")
                track_sr = track_data.get("sr", GLOBAL_SR)
                app_mode = st.session_state.get("app_mode", "Easy")

                # --- Waveform Preview Display ---
                st.markdown("**Preview (First 60s with Effects Applied)**")
                display_path = None
                preview_cache_hit = False

                if original_audio is not None and original_audio.size > 0:
                    current_settings_hash = self._calculate_preview_hash(track_data)
                    stored_hash = track_data.get("preview_settings_hash")
                    stored_path = track_data.get("preview_temp_file_path")

                    if stored_hash is not None and current_settings_hash == stored_hash and stored_path and isinstance(stored_path, str) and os.path.exists(stored_path):
                        display_path = stored_path
                        preview_cache_hit = True
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
                                # Attempt to render the waveform preview
                                audix(data=display_path, sample_rate=track_sr, wavesurfer_options=ws_options, key=audix_key)
                            except Exception as audix_err:
                                # --- FALLBACK: Render st.audio if audix fails ---
                                logger.error(f"Error rendering audix waveform for {track_id}: {audix_err}")
                                st.warning("Waveform preview failed to render. Showing standard audio player instead.")
                                try:
                                    st.audio(display_path, format="audio/wav")
                                except Exception as audio_err:
                                    logger.error(f"Fallback st.audio player also failed for {track_id}: {audio_err}")
                                    st.error("Could not display audio preview.")
                                # --- End Fallback ---
                        else:
                            # If library not available, just show standard player
                            st.audio(display_path, format="audio/wav")
                    elif not preview_cache_hit:
                        # If no display path and cache miss, prompt user to update
                        st.info("Settings changed or preview not generated. Click 'Update Preview' below.")

                else:  # No original audio data
                    if track_data.get("source_type") == "upload":
                        st.warning(f"Source file '{track_data.get('original_filename', 'N/A')}' needs to be re-uploaded.", icon="⚠️")
                    else:
                        st.warning("Track currently has no audio data.")

                full_len_samples = len(original_audio) if original_audio is not None else 0
                full_len_sec = full_len_samples / track_sr if track_sr > 0 else 0
                st.caption(f"Full Duration: {full_len_sec:.2f}s | SR: {track_sr} Hz")
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
                update_disabled = original_audio is None or original_audio.size == 0
                if st.button(
                    "⚙️ Update Preview",
                    key=f"update_track_preview_{track_id}",
                    help="Generate the 60s preview with current settings.",
                    disabled=update_disabled,
                    use_container_width=True,
                ):
                    logger.info(f"Update Preview clicked for: '{track_data.get('name', 'N/A')}' ({track_id})")
                    with st.spinner("Generating preview audio..."):
                        # Use function from audio_processing
                        preview_audio = get_preview_audio(track_data, preview_duration_s=PREVIEW_DURATION_S)
                        if preview_audio is not None and preview_audio.size > 0:
                            # Use function from audio_io
                            new_preview_path = save_audio_to_temp_file(preview_audio, track_sr)
                            if new_preview_path:
                                old_preview_path = track_data.get("preview_temp_file_path")
                                new_settings_hash = self._calculate_preview_hash(track_data)
                                self.app_state.update_track_param(track_id, "preview_temp_file_path", new_preview_path)
                                self.app_state.update_track_param(track_id, "preview_settings_hash", new_settings_hash)
                                self.app_state.increment_update_counter(track_id)  # Force audix refresh
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
                            logger.error(f"Failed to generate preview audio for track {track_id}.")
                            st.error("Failed to generate preview audio. Check settings.")

            except Exception as e:
                logger.exception(f"Error rendering main column for track {track_id}")
                st.error(f"An error occurred displaying controls for this track: {e}")

    def _render_track_controls_col(self, track_id: TrackID, track_data: TrackData, column: st.delta_generator.DeltaGenerator) -> bool:
        """Renders the side controls for a track (Name, Type, Mute, Solo, Delete)."""
        delete_clicked = False
        with column:
            try:
                st.markdown("**Track Details**")
                current_name = track_data.get("name", "Unnamed Track")
                new_name = st.text_input("Track Name", value=current_name, key=f"name_{track_id}", help="Rename this track.")
                if new_name != current_name and new_name.strip():
                    self.app_state.update_track_param(track_id, "name", new_name.strip())
                    st.rerun()

                current_type = track_data.get("track_type", TRACK_TYPE_OTHER)
                # TRACK_TYPES imported from config at top of file
                try:
                    current_index = TRACK_TYPES.index(current_type)
                except ValueError:
                    current_index = TRACK_TYPES.index(TRACK_TYPE_OTHER)
                new_type = st.selectbox("Track Type", TRACK_TYPES, index=current_index, key=f"type_{track_id}", help="Categorize this layer.")
                if new_type != current_type:
                    self.app_state.update_track_param(track_id, "track_type", new_type)
                    st.rerun()

                st.caption("Mixing Controls")
                ms_col1, ms_col2 = st.columns(2)
                with ms_col1:
                    mute_value = track_data.get("mute", False)
                    mute = st.checkbox("Mute", value=mute_value, key=f"mute_{track_id}", help="Silence track in mix.")
                    if mute != mute_value:
                        self.app_state.update_track_param(track_id, "mute", mute)
                with ms_col2:
                    solo_value = track_data.get("solo", False)
                    solo = st.checkbox("Solo", value=solo_value, key=f"solo_{track_id}", help="Isolate track(s) in mix.")
                    if solo != solo_value:
                        self.app_state.update_track_param(track_id, "solo", solo)

                st.markdown("---")
                if st.button("🗑️ Delete Track", key=f"delete_{track_id}", help="Permanently remove track.", type="secondary", use_container_width=True):
                    delete_clicked = True
                    logger.info(f"Delete button clicked for track {track_id} ('{track_data.get('name', 'N/A')}')")

            except Exception as e:
                logger.exception(f"Error rendering controls column for track {track_id}")
                st.error(f"Error displaying track controls: {e}")

        return delete_clicked
