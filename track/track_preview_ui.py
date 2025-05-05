# track_preview_ui.py
# ==========================================
# Track Preview Column UI for MindMorph Editor
# FIX: Corrected arguments passed to get_preview_audio in _handle_update_preview.
# ==========================================

import logging
import os

import numpy as np
import streamlit as st

# Import necessary components from other modules
from app_state import AppState, TrackDataDict, TrackID
from audio_utils.audio_io import save_audio_to_temp_file

# --- Import the UPDATED get_preview_audio function ---
from audio_utils.audio_preview import get_preview_audio
from audio_utils.audio_state_definitions import (  # Import SampleRate
    AudioData,
    SampleRate,
)
from config import (
    GLOBAL_SR,
    PREVIEW_DURATION_S,
    QUICK_SUBLIMINAL_PRESET_SPEED,
    QUICK_SUBLIMINAL_PRESET_VOLUME,
    TRACK_TYPE_AFFIRMATION,
    ULTRASONIC_TARGET_FREQ,
)

# Import UI-specific libraries conditionally
try:
    from streamlit_advanced_audio import WaveSurferOptions, audix

    STREAMLIT_ADVANCED_AUDIO_AVAILABLE = True
except ImportError:
    STREAMLIT_ADVANCED_AUDIO_AVAILABLE = False

    def WaveSurferOptions(**kwargs):
        return None

    def audix(*args, **kwargs):
        pass


logger = logging.getLogger(__name__)


class TrackPreviewUI:
    """Handles rendering the main preview and controls column for a track."""

    def __init__(self, app_state: AppState):
        """Initializes the TrackPreviewUI."""
        self.app_state = app_state
        logger.debug("TrackPreviewUI initialized.")

    def _calculate_preview_hash(self, track_data: TrackDataDict) -> int:
        """Calculates a hash based on settings relevant to the audio preview."""
        # (This function remains the same)
        audio_snippet = track_data.get("audio_snippet")
        params_to_hash = (
            track_data.get("volume", 1.0),
            track_data.get("speed_factor", 1.0),
            track_data.get("pitch_shift", 0.0),
            track_data.get("pan", 0.0),
            track_data.get("filter_type", "off"),
            track_data.get("filter_cutoff", 8000.0),
            track_data.get("reverse_audio", False),
            track_data.get("ultrasonic_shift", False),
            audio_snippet.size if audio_snippet is not None else 0,
            track_data.get("sr", GLOBAL_SR),
            track_data.get("loop_to_fit", False),
        )
        return hash(params_to_hash)

    def render_preview_column(
        self,
        track_id: TrackID,
        track_data: TrackDataDict,
        column: st.delta_generator.DeltaGenerator,
    ):
        """Renders the main controls for a single track (waveform, settings, preview button)."""
        # (This function remains largely the same, UI rendering logic is unchanged)
        with column:
            try:
                audio_snippet = track_data.get("audio_snippet")
                track_sr = track_data.get("sr")  # Get SR which should be populated now
                app_mode = st.session_state.get("app_mode", "Easy")
                source_info = track_data.get("source_info")

                st.markdown(
                    f"**Preview (First {PREVIEW_DURATION_S}s with Effects Applied)**"
                )
                display_path = None
                preview_cache_hit = False

                # --- Logic to check for cached preview file (remains same) ---
                if (
                    audio_snippet is not None
                    and audio_snippet.size > 0
                    and track_sr is not None
                ):
                    current_settings_hash = self._calculate_preview_hash(track_data)
                    stored_hash = track_data.get("preview_settings_hash")
                    stored_path = track_data.get("preview_temp_file_path")
                    if (
                        current_settings_hash == stored_hash
                        and stored_path
                        and isinstance(stored_path, str)
                        and os.path.exists(stored_path)
                    ):
                        display_path = stored_path
                        preview_cache_hit = True
                    elif (
                        stored_path
                        and isinstance(stored_path, str)
                        and not os.path.exists(stored_path)
                    ):
                        logger.warning(
                            f"Cached preview file missing for track {track_id}: {stored_path}. Clearing path."
                        )
                        # Use update_track_preview_file for consistency if available, else direct update
                        if hasattr(self.app_state, "update_track_preview_file"):
                            self.app_state.update_track_preview_file(
                                track_id, None, None
                            )
                        else:  # Fallback
                            if "preview_temp_file_path" in track_data:
                                track_data["preview_temp_file_path"] = None
                            if "preview_settings_hash" in track_data:
                                track_data["preview_settings_hash"] = None
                            # Note: Direct modification might not persist if track_data is a copy

                    # --- Waveform/Audio Player Rendering (remains same) ---
                    if display_path:
                        if STREAMLIT_ADVANCED_AUDIO_AVAILABLE:
                            ws_options = WaveSurferOptions(
                                height=100,
                                normalize=True,
                                wave_color="#A020F0",
                                progress_color="#800080",
                                cursor_color="#333333",
                                cursor_width=1,
                                bar_width=2,
                                bar_gap=1,
                            )
                            update_count = track_data.get("update_counter", 0)
                            audix_key = f"audix_preview_{track_id}_{update_count}"
                            try:
                                audix(
                                    data=display_path,
                                    sample_rate=track_sr,
                                    wavesurfer_options=ws_options,
                                    key=audix_key,
                                )
                            except Exception as audix_err:
                                logger.error(
                                    f"Error rendering audix waveform for {track_id}: {audix_err}"
                                )
                                st.warning(
                                    "Waveform preview failed. Showing standard player."
                                )
                                try:
                                    st.audio(display_path, format="audio/wav")
                                except Exception as audio_err:
                                    logger.error(
                                        f"Fallback st.audio player failed for {track_id}: {audio_err}"
                                    )
                                    st.error("Could not display audio preview.")
                        else:
                            st.audio(display_path, format="audio/wav")
                    elif not preview_cache_hit:
                        st.info(
                            "Settings changed or preview not generated. Click 'Update Preview'."
                        )
                # --- Handling missing snippet/source (remains same) ---
                else:
                    if source_info and source_info.get("type") == "upload":
                        temp_file_path = source_info.get("temp_file_path")
                        original_filename = source_info.get("original_filename", "N/A")
                        if temp_file_path and os.path.exists(str(temp_file_path)):
                            st.warning(
                                f"Audio snippet missing for '{original_filename}'. Click 'Update Preview'.",
                                icon="⏳",
                            )
                        else:
                            st.error(
                                f"Source file missing for '{original_filename}'. Please remove and re-add the track.",
                                icon="❌",
                            )
                    elif track_data.get("audio_snippet") is None:
                        st.warning(
                            "Track preview needs to be generated. Click 'Update Preview'.",
                            icon="⏳",
                        )
                    else:  # Should have snippet but maybe SR is missing?
                        st.warning(
                            "Track preview data incomplete. Click 'Update Preview'.",
                            icon="⚠️",
                        )

                # --- Display Snippet Info (remains same) ---
                snippet_len_samples = (
                    len(audio_snippet) if audio_snippet is not None else 0
                )
                snippet_sr_display = track_sr if track_sr is not None else "N/A"
                snippet_len_sec = (
                    snippet_len_samples / track_sr
                    if track_sr is not None and track_sr > 0
                    else 0
                )
                st.caption(
                    f"Snippet Duration: {snippet_len_sec:.2f}s | SR: {snippet_sr_display} Hz"
                )
                st.markdown("---")

                # --- Track Settings UI (Volume, Pan, Presets, Advanced Effects - remains same) ---
                st.markdown("**Track Settings**")
                basic_cols = st.columns(2)
                with basic_cols[0]:
                    vol = st.slider(
                        "Volume",
                        0.0,
                        2.0,
                        float(track_data.get("volume", 1.0)),
                        0.05,
                        key=f"vol_{track_id}",
                        help="Adjust track loudness.",
                    )
                    if not np.isclose(vol, track_data.get("volume", 1.0)):
                        self.app_state.update_track_param(track_id, "volume", vol)
                with basic_cols[1]:
                    pan = st.slider(
                        "Pan",
                        -1.0,
                        1.0,
                        float(track_data.get("pan", 0.0)),
                        0.1,
                        key=f"pan_{track_id}",
                        help="Adjust stereo balance.",
                    )
                    if not np.isclose(pan, track_data.get("pan", 0.0)):
                        self.app_state.update_track_param(track_id, "pan", pan)

                if track_data.get("track_type") == TRACK_TYPE_AFFIRMATION:
                    preset_help_text = f"Applies Speed={QUICK_SUBLIMINAL_PRESET_SPEED}x, Volume={QUICK_SUBLIMINAL_PRESET_VOLUME}."
                    if st.button(
                        "⚡ Quick Subliminal Settings",
                        key=f"subliminalize_preset_{track_id}",
                        help=preset_help_text,
                    ):
                        logger.info(f"Subliminalize preset applied to track {track_id}")
                        self.app_state.update_track_param(
                            track_id, "speed_factor", QUICK_SUBLIMINAL_PRESET_SPEED
                        )
                        self.app_state.update_track_param(
                            track_id, "volume", QUICK_SUBLIMINAL_PRESET_VOLUME
                        )
                        st.toast(
                            f"Speed set to {QUICK_SUBLIMINAL_PRESET_SPEED}x, Volume to {QUICK_SUBLIMINAL_PRESET_VOLUME}. Click 'Update Preview'.",
                            icon="⚡",
                        )
                        st.rerun()

                if app_mode == "Advanced":
                    st.markdown("**Advanced Effects**")
                    st.caption(
                        "Adjust speed, pitch, filter, etc. Click 'Update Preview' after changes."
                    )
                    ultrasonic_enabled = track_data.get("ultrasonic_shift", False)
                    ultrasonic_checkbox = st.checkbox(
                        "🔊 Ultrasonic Shift ('Silent')",
                        value=ultrasonic_enabled,
                        key=f"ultrasonic_{track_id}",
                        help=f"EXPERIMENTAL: Shifts audio towards {ULTRASONIC_TARGET_FREQ}Hz. Disables Pitch Shift.",
                    )
                    if ultrasonic_checkbox != ultrasonic_enabled:
                        self.app_state.update_track_param(
                            track_id, "ultrasonic_shift", ultrasonic_checkbox
                        )
                        st.rerun()

                    col_fx1_1, col_fx1_2, col_fx1_3 = st.columns([0.6, 1, 1])
                    with col_fx1_1:
                        st.markdown("<br/>", unsafe_allow_html=True)
                        loop_value = st.checkbox(
                            "🔁 Loop",
                            value=track_data.get("loop_to_fit", False),
                            key=f"loop_{track_id}",
                            help="Loop track during final mix?",
                        )
                        if loop_value != track_data.get("loop_to_fit"):
                            self.app_state.update_track_param(
                                track_id, "loop_to_fit", loop_value
                            )
                        st.markdown("<br/>", unsafe_allow_html=True)
                        reverse_value = st.checkbox(
                            "🔄 Reverse",
                            value=track_data.get("reverse_audio", False),
                            key=f"reverse_{track_id}",
                            help="Reverse audio playback?",
                        )
                        if reverse_value != track_data.get("reverse_audio"):
                            self.app_state.update_track_param(
                                track_id, "reverse_audio", reverse_value
                            )
                    with col_fx1_2:
                        speed = st.slider(
                            "Speed",
                            0.1,
                            16.0,
                            float(track_data.get("speed_factor", 1.0)),
                            0.05,
                            key=f"speed_{track_id}",
                            help="Playback speed factor.",
                        )
                        if not np.isclose(speed, track_data.get("speed_factor", 1.0)):
                            self.app_state.update_track_param(
                                track_id, "speed_factor", speed
                            )
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
                        if not pitch_disabled and pitch != track_data.get(
                            "pitch_shift", 0
                        ):
                            self.app_state.update_track_param(
                                track_id, "pitch_shift", float(pitch)
                            )

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
                            self.app_state.update_track_param(
                                track_id, "filter_type", f_type
                            )
                            st.rerun()
                    with col_fx2_2:
                        filter_type_active = (
                            track_data.get("filter_type", "off") != "off"
                        )
                        cutoff_disabled = filter_disabled or not filter_type_active
                        # Use actual track_sr if available for max cutoff, else default
                        effective_sr = (
                            track_sr
                            if track_sr is not None and track_sr > 0
                            else GLOBAL_SR
                        )
                        max_cutoff = effective_sr / 2.0 - 1.0
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
                        if not cutoff_disabled and not np.isclose(
                            f_cutoff_clamped, track_data.get("filter_cutoff", 8000.0)
                        ):
                            self.app_state.update_track_param(
                                track_id, "filter_cutoff", f_cutoff_clamped
                            )

                # --- Update Preview Button ---
                st.markdown("---")
                # Disable button if raw snippet or SR is missing
                update_disabled = (
                    audio_snippet is None or audio_snippet.size == 0 or track_sr is None
                )
                update_preview_help = f"Generate the {PREVIEW_DURATION_S}s preview using the raw snippet and current settings."
                if st.button(
                    "⚙️ Update Preview",
                    key=f"update_track_preview_{track_id}",
                    help=update_preview_help,
                    disabled=update_disabled,
                    use_container_width=True,
                ):
                    # Call the handler which now uses the correct arguments for get_preview_audio
                    self._handle_update_preview(track_id, track_data)

            except Exception as e:
                logger.exception(f"Error rendering preview column for track {track_id}")
                st.error(f"An error occurred displaying controls for this track: {e}")

    # --- MODIFIED: _handle_update_preview to call get_preview_audio correctly ---
    def _handle_update_preview(self, track_id: TrackID, track_data: TrackDataDict):
        """Handles the logic when 'Update Preview' is clicked."""
        logger.info(
            f"Update Preview clicked for: '{track_data.get('name', 'N/A')}' ({track_id})"
        )
        # Get the raw snippet and its SR from track_data (should be populated by TrackEditorManager)
        raw_audio_snippet = track_data.get("audio_snippet")
        track_sr: Optional[SampleRate] = track_data.get("sr")  # Explicitly type hint

        if raw_audio_snippet is None or raw_audio_snippet.size == 0:
            st.error("Cannot generate preview, raw audio snippet data is missing.")
            logger.error(
                f"Preview generation failed for {track_id}: raw audio_snippet is missing."
            )
            return
        if track_sr is None or track_sr <= 0:
            st.error(
                "Cannot generate preview, track sample rate is missing or invalid."
            )
            logger.error(
                f"Preview generation failed for {track_id}: track SR is missing or invalid ({track_sr})."
            )
            return

        with st.spinner("Generating preview audio..."):
            try:
                # --- CORRECTED CALL to get_preview_audio ---
                # Pass the raw snippet, its SR, and the full track_data dictionary
                preview_audio = get_preview_audio(
                    raw_snippet=raw_audio_snippet,
                    sr=track_sr,
                    track_data=track_data,  # Pass the full dict for effect settings
                    preview_duration_s=PREVIEW_DURATION_S,
                )
                # --- END CORRECTION ---

                # --- Rest of the logic remains the same ---
                if preview_audio is not None and preview_audio.size > 0:
                    new_preview_path = save_audio_to_temp_file(preview_audio, track_sr)
                    if new_preview_path:
                        old_preview_path = track_data.get("preview_temp_file_path")
                        new_settings_hash = self._calculate_preview_hash(track_data)
                        # Use update_track_preview_file for consistency if available
                        if hasattr(self.app_state, "update_track_preview_file"):
                            self.app_state.update_track_preview_file(
                                track_id, new_preview_path, new_settings_hash
                            )
                        else:  # Fallback
                            self.app_state.update_track_param(
                                track_id, "preview_temp_file_path", new_preview_path
                            )
                            self.app_state.update_track_param(
                                track_id, "preview_settings_hash", new_settings_hash
                            )

                        self.app_state.increment_update_counter(
                            track_id
                        )  # Trigger UI update if needed
                        # Clean up old file
                        if (
                            old_preview_path
                            and isinstance(old_preview_path, str)
                            and old_preview_path != new_preview_path
                            and os.path.exists(old_preview_path)
                        ):
                            try:
                                os.remove(old_preview_path)
                                logger.info(
                                    f"Deleted old preview file: {old_preview_path}"
                                )
                            except OSError as e:
                                logger.warning(
                                    f"Could not delete old preview file {old_preview_path}: {e}"
                                )
                        st.toast("Preview updated.", icon="✅")
                        st.rerun()
                    else:
                        logger.error(
                            f"Failed to save new preview temp file for track {track_id}."
                        )
                        st.error("Failed to save updated preview file.")
                else:
                    logger.error(
                        f"Failed to generate preview audio for track {track_id} (get_preview_audio returned None or empty)."
                    )
                    st.error(
                        "Failed to generate preview audio. Check settings or logs."
                    )
            except Exception as e:
                logger.exception(
                    f"Error during preview generation for track {track_id}"
                )
                st.error(f"Error generating preview: {e}")
