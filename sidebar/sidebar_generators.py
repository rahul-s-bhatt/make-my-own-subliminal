# sidebar_generators.py
# ==========================================
# Frequency and Noise Generation UI for MindMorph Sidebar
# STEP 1 OPTIMIZED: Removed immediate snippet generation on track add.
# ==========================================

import logging
from typing import Optional

import streamlit as st

from app_state import AppState

# --- REMOVED: audio_generators imports (no longer needed here) ---
# from audio_utils.audio_generators import (...)
from audio_utils.audio_state_definitions import (
    AudioData,
)  # Keep for type hints if needed elsewhere
from audio_utils.audio_state_definitions import SourceInfoFrequency, SourceInfoNoise
from config import (
    GENERATOR_SNIPPET_DURATION_S,
)  # Keep for potential future use or reference
from config import GLOBAL_SR  # Keep for potential future use or reference
from config import MAX_TRACK_LIMIT, TRACK_TYPE_BACKGROUND, TRACK_TYPE_FREQUENCY

logger = logging.getLogger(__name__)


class SidebarGenerators:
    """Handles rendering frequency and noise generator UI in the sidebar."""

    def __init__(self, app_state: AppState):
        """Initializes the SidebarGenerators."""
        self.app_state = app_state
        logger.debug("SidebarGenerators initialized.")

    def _check_track_limit(self, adding_count: int = 1) -> bool:
        """Checks if adding tracks would exceed the limit."""
        current_count = len(self.app_state.get_all_tracks())
        if current_count + adding_count > MAX_TRACK_LIMIT:
            logger.warning(f"Track limit ({MAX_TRACK_LIMIT}) reached.")
            st.error(f"Cannot add more tracks. Limit: {MAX_TRACK_LIMIT}.", icon="🚫")
            return False
        return True

    def render_frequency_generators(self):
        """Renders frequency/tone generation options based on app mode."""
        # (UI Rendering remains the same)
        st.subheader("🧠✨ Add Frequencies / Tones")
        app_mode = st.session_state.get("app_mode", "Easy")
        limit_reached = len(self.app_state.get_all_tracks()) >= MAX_TRACK_LIMIT
        if limit_reached:
            st.warning(f"Track limit ({MAX_TRACK_LIMIT}) reached.", icon="⚠️")

        available_gen_types = (
            ["Presets"]
            if app_mode == "Easy"
            else ["Binaural Beats", "Solfeggio Tones", "Isochronic Tones", "Presets"]
        )
        st.caption(
            "Use presets for common patterns."
            if app_mode == "Easy"
            else "Generate specific tones or use presets."
        )

        gen_type_key = "sidebar_frequency_gen_type"
        default_gen_type = "Presets"
        if (
            gen_type_key in st.session_state
            and st.session_state[gen_type_key] in available_gen_types
        ):
            default_gen_type = st.session_state[gen_type_key]
        try:
            default_index = available_gen_types.index(default_gen_type)
        except ValueError:
            default_index = 0
        gen_type = st.radio(
            "Select Type:",
            available_gen_types,
            index=default_index,
            key=gen_type_key,
            horizontal=True,
            label_visibility="collapsed",
            disabled=limit_reached,
        )

        if gen_type == "Presets":
            self._render_frequency_presets(disabled=limit_reached)
        elif gen_type == "Binaural Beats" and app_mode == "Advanced":
            self._render_binaural_generator(disabled=limit_reached)
        elif gen_type == "Solfeggio Tones" and app_mode == "Advanced":
            self._render_solfeggio_generator(disabled=limit_reached)
        elif gen_type == "Isochronic Tones" and app_mode == "Advanced":
            self._render_isochronic_generator(disabled=limit_reached)

    def render_background_generators(self):
        """Renders options for generating background noise."""
        # (UI Rendering remains the same)
        st.subheader("🎵 Add Background Noise")
        limit_reached = len(self.app_state.get_all_tracks()) >= MAX_TRACK_LIMIT
        if limit_reached:
            st.warning(f"Track limit ({MAX_TRACK_LIMIT}) reached.", icon="⚠️")
        noise_options = ["White Noise", "Pink Noise", "Brown Noise"]
        noise_type = st.selectbox(
            "Select Noise Type:",
            noise_options,
            key="sidebar_noise_type",
            help="Choose background noise.",
            disabled=limit_reached,
        )
        noise_duration_req = st.number_input(
            "Target Duration (s)",
            10,
            7200,
            300,
            10,
            key="sidebar_noise_duration",
            help="Target length for final mix. Noise will loop.",
            disabled=limit_reached,
        )

        # --- MODIFIED: Generate Button Logic ---
        if st.button(
            f"➕ Add {noise_type} Track",
            key="sidebar_generate_noise",
            disabled=limit_reached,
        ):
            if not self._check_track_limit(adding_count=1):
                return

            # --- REMOVED: Snippet generation ---
            # with st.spinner(f"Generating {noise_type} snippet..."):
            #     audio_snippet = generate_noise(...)

            # --- Create SourceInfo and add track state ---
            logger.info(f"Adding {noise_type} track state.")
            source_info: SourceInfoNoise = {
                "type": "noise",
                "noise_type": noise_type,
                "target_duration_s": noise_duration_req,
            }
            initial_params = {
                "name": noise_type,
                "track_type": TRACK_TYPE_BACKGROUND,
                "loop_to_fit": True,
                "volume": 0.5,  # Default volume for background
            }
            # Call app_state.add_track (New signature)
            new_track_id = self.app_state.add_track(
                source_info=source_info,
                initial_params=initial_params,
            )
            if new_track_id:
                st.success(
                    f"'{noise_type}' track added (ID: {new_track_id[:6]}). Preview pending."
                )
                st.rerun()
            else:
                logger.error(f"Failed to add {noise_type} track state.")
                st.error(
                    f"Failed to add track for {noise_type}. Limit might be reached."
                )
            # --- End of modification ---

    def _render_frequency_presets(self, disabled: bool = False):
        # (UI Rendering remains the same)
        st.markdown(
            "<small>Generate common frequency patterns.</small>", unsafe_allow_html=True
        )
        preset_options = {
            "Focus (Alpha 10Hz Binaural)": {
                "type": "binaural",
                "f_left": 200.0,
                "f_right": 210.0,
            },
            "Relaxation (Theta 5Hz Binaural)": {
                "type": "binaural",
                "f_left": 150.0,
                "f_right": 155.0,
            },
            "Deep Sleep (Delta 2Hz Binaural)": {
                "type": "binaural",
                "f_left": 100.0,
                "f_right": 102.0,
            },
            "Love Frequency (Solfeggio 528Hz)": {"type": "solfeggio", "freq": 528.0},
            "Miracle Tone (Solfeggio 417Hz)": {"type": "solfeggio", "freq": 417.0},
        }
        preset_name = st.selectbox(
            "Select Preset:",
            list(preset_options.keys()),
            key="sidebar_freq_preset_select",
            disabled=disabled,
        )
        preset_data = preset_options[preset_name]
        cols_preset = st.columns(2)
        with cols_preset[0]:
            preset_duration_req = st.number_input(
                "Target Duration (s)",
                10,
                7200,
                300,
                10,
                key="sidebar_preset_duration",
                help="Target length. Tone will loop.",
                disabled=disabled,
            )
        with cols_preset[1]:
            preset_initial_vol = st.slider(
                "Initial Volume",
                0.0,
                1.0,
                0.2,
                0.05,
                key="sidebar_preset_volume",
                help="Initial volume.",
                disabled=disabled,
            )

        # --- MODIFIED: Generate Button Logic ---
        if st.button(
            f"➕ Add '{preset_name}' Track",
            key="sidebar_generate_preset",
            disabled=disabled,
        ):
            if not self._check_track_limit(adding_count=1):
                return

            # --- REMOVED: Snippet generation ---
            # with st.spinner(f"Generating {preset_name} snippet..."):
            #     audio_snippet = None; source_info = None; gen_volume = 1.0
            #     if preset_data["type"] == "binaural": audio_snippet = generate_binaural_beats(...)
            #     elif preset_data["type"] == "solfeggio": audio_snippet = generate_solfeggio_frequency(...)

            # --- Create SourceInfo and add track state ---
            source_info: Optional[SourceInfoFrequency] = None
            if preset_data["type"] == "binaural":
                source_info = {
                    "type": "frequency",
                    "freq_type": "binaural",
                    "f_left": preset_data["f_left"],
                    "f_right": preset_data["f_right"],
                    "target_duration_s": preset_duration_req,
                    "freq": None,
                    "carrier": None,
                    "pulse": None,
                }
            elif preset_data["type"] == "solfeggio":
                source_info = {
                    "type": "frequency",
                    "freq_type": "solfeggio",
                    "freq": preset_data["freq"],
                    "target_duration_s": preset_duration_req,
                    "f_left": None,
                    "f_right": None,
                    "carrier": None,
                    "pulse": None,
                }

            if source_info:
                logger.info(f"Adding {preset_name} track state.")
                initial_params = {
                    "name": preset_name,
                    "track_type": TRACK_TYPE_FREQUENCY,
                    "loop_to_fit": True,
                    "volume": preset_initial_vol,
                }
                new_track_id = self.app_state.add_track(
                    source_info=source_info, initial_params=initial_params
                )
                if new_track_id:
                    st.success(
                        f"'{preset_name}' track added (ID: {new_track_id[:6]}). Preview pending."
                    )
                    st.rerun()
                else:
                    logger.error(f"Failed to add {preset_name} track state.")
                    st.error(
                        f"Failed to add track for {preset_name}. Limit might be reached."
                    )
            else:
                logger.error(
                    f"Could not determine source_info for preset: {preset_name}"
                )
                st.error(f"Internal error creating track for preset '{preset_name}'.")
            # --- End of modification ---

    def _render_binaural_generator(self, disabled: bool = False):
        # (UI Rendering remains the same)
        st.markdown(
            "<small>Generates stereo tones (requires headphones).</small>",
            unsafe_allow_html=True,
        )
        bb_cols = st.columns(2)
        with bb_cols[0]:
            bb_duration_req = st.number_input(
                "Target Duration (s)",
                10,
                7200,
                300,
                10,
                key="sidebar_bb_duration",
                help="Length. Tone will loop.",
                disabled=disabled,
            )
            bb_fleft = st.number_input(
                "Left Freq (Hz)",
                20.0,
                1000.0,
                200.0,
                0.1,
                format="%.1f",
                key="sidebar_bb_freq_left",
                help="Left ear frequency.",
                disabled=disabled,
            )
        with bb_cols[1]:
            bb_initial_vol = st.slider(
                "Initial Volume##BB",
                0.0,
                1.0,
                0.3,
                0.05,
                key="sidebar_bb_volume",
                help="Initial loudness.",
                disabled=disabled,
            )
            bb_fright = st.number_input(
                "Right Freq (Hz)",
                20.0,
                1000.0,
                210.0,
                0.1,
                format="%.1f",
                key="sidebar_bb_freq_right",
                help="Right ear frequency.",
                disabled=disabled,
            )
        beat_freq = abs(bb_fleft - bb_fright)
        st.caption(f"Resulting Beat Frequency: {beat_freq:.1f} Hz")

        # --- MODIFIED: Generate Button Logic ---
        if st.button(
            "➕ Add Binaural Track",
            key="sidebar_generate_bb",
            help="Add track.",
            disabled=disabled,
        ):
            if not self._check_track_limit(adding_count=1):
                return

            # --- REMOVED: Snippet generation ---
            # with st.spinner("Generating Binaural Beats snippet..."):
            #     audio_snippet = generate_binaural_beats(...)

            # --- Create SourceInfo and add track state ---
            logger.info(
                f"Adding Binaural track state ({bb_fleft:.1f}Hz L / {bb_fright:.1f}Hz R)."
            )
            source_info: SourceInfoFrequency = {
                "type": "frequency",
                "freq_type": "binaural",
                "f_left": bb_fleft,
                "f_right": bb_fright,
                "target_duration_s": bb_duration_req,
                "freq": None,
                "carrier": None,
                "pulse": None,
            }
            default_name = f"Binaural {bb_fleft:.1f}Hz L / {bb_fright:.1f}Hz R ({beat_freq:.1f}Hz Beat)"
            initial_params = {
                "name": default_name,
                "track_type": TRACK_TYPE_FREQUENCY,
                "loop_to_fit": True,
                "volume": bb_initial_vol,
            }
            new_track_id = self.app_state.add_track(
                source_info=source_info, initial_params=initial_params
            )
            if new_track_id:
                st.success(
                    f"'{default_name}' added (ID: {new_track_id[:6]}). Preview pending."
                )
                st.rerun()
            else:
                logger.error(f"Failed to add Binaural track state.")
                st.error(
                    f"Failed to add track for '{default_name}'. Limit might be reached."
                )
            # --- End of modification ---

    def _render_solfeggio_generator(self, disabled: bool = False):
        # (UI Rendering remains the same)
        st.markdown(
            "<small>Generates pure tones based on Solfeggio frequencies.</small>",
            unsafe_allow_html=True,
        )
        freqs = [174.0, 285.0, 396.0, 417.0, 528.0, 639.0, 741.0, 852.0, 963.0]
        freq_labels = {f: f"{f} Hz" for f in freqs}
        cols = st.columns(2)
        with cols[0]:
            freq = st.selectbox(
                "Frequency (Hz)",
                freqs,
                index=4,
                key="sidebar_solf_freq",
                format_func=lambda x: freq_labels.get(x, f"{x} Hz"),
                help="Select Solfeggio frequency.",
                disabled=disabled,
            )
        with cols[1]:
            duration_req = st.number_input(
                "Target Duration (s)##Solf",
                10,
                7200,
                300,
                10,
                key="sidebar_solf_duration",
                help="Length. Tone will loop.",
                disabled=disabled,
            )
        initial_vol = st.slider(
            "Initial Volume##Solf",
            0.0,
            1.0,
            0.3,
            0.05,
            key="sidebar_solf_volume",
            help="Initial loudness.",
            disabled=disabled,
        )

        # --- MODIFIED: Generate Button Logic ---
        if st.button(
            "➕ Add Solfeggio Track",
            key="sidebar_generate_solf",
            help="Add track.",
            disabled=disabled,
        ):
            if not self._check_track_limit(adding_count=1):
                return

            # --- REMOVED: Snippet generation ---
            # with st.spinner("Generating Solfeggio Tone snippet..."):
            #     audio_snippet = generate_solfeggio_frequency(...)

            # --- Create SourceInfo and add track state ---
            logger.info(f"Adding Solfeggio {freq}Hz track state.")
            source_info: SourceInfoFrequency = {
                "type": "frequency",
                "freq_type": "solfeggio",
                "freq": freq,
                "target_duration_s": duration_req,
                "f_left": None,
                "f_right": None,
                "carrier": None,
                "pulse": None,
            }
            default_name = f"Solfeggio {freq}Hz"
            initial_params = {
                "name": default_name,
                "track_type": TRACK_TYPE_FREQUENCY,
                "loop_to_fit": True,
                "volume": initial_vol,
            }
            new_track_id = self.app_state.add_track(
                source_info=source_info, initial_params=initial_params
            )
            if new_track_id:
                st.success(
                    f"'{default_name}' added (ID: {new_track_id[:6]}). Preview pending."
                )
                st.rerun()
            else:
                logger.error(f"Failed to add Solfeggio track state.")
                st.error(
                    f"Failed to add track for '{default_name}'. Limit might be reached."
                )
            # --- End of modification ---

    def _render_isochronic_generator(self, disabled: bool = False):
        # (UI Rendering remains the same)
        st.markdown(
            "<small>Generates rhythmic pulses of a single tone.</small>",
            unsafe_allow_html=True,
        )
        iso_cols = st.columns(2)
        with iso_cols[0]:
            iso_duration_req = st.number_input(
                "Target Duration (s)##Iso",
                10,
                7200,
                300,
                10,
                key="sidebar_iso_duration",
                help="Length. Tone will loop.",
                disabled=disabled,
            )
            iso_carrier = st.number_input(
                "Carrier Freq (Hz)",
                20.0,
                1000.0,
                150.0,
                0.1,
                format="%.1f",
                key="sidebar_iso_carrier",
                help="Base tone frequency.",
                disabled=disabled,
            )
        with iso_cols[1]:
            iso_initial_vol = st.slider(
                "Initial Volume##Iso",
                0.0,
                1.0,
                0.4,
                0.05,
                key="sidebar_iso_volume",
                help="Initial loudness.",
                disabled=disabled,
            )
            iso_pulse = st.number_input(
                "Pulse Freq (Hz)",
                0.1,
                40.0,
                10.0,
                0.1,
                format="%.1f",
                key="sidebar_iso_pulse",
                help="Pulses per second.",
                disabled=disabled,
            )

        # --- MODIFIED: Generate Button Logic ---
        if st.button(
            "➕ Add Isochronic Track",
            key="sidebar_generate_iso",
            help="Add track.",
            disabled=disabled,
        ):
            if not self._check_track_limit(adding_count=1):
                return

            # --- REMOVED: Snippet generation ---
            # with st.spinner("Generating Isochronic Tones snippet..."):
            #     audio_snippet = generate_isochronic_tones(...)

            # --- Create SourceInfo and add track state ---
            logger.info(
                f"Adding Isochronic track state ({iso_carrier:.1f}Hz / {iso_pulse:.1f}Hz Pulse)."
            )
            source_info: SourceInfoFrequency = {
                "type": "frequency",
                "freq_type": "isochronic",
                "carrier": iso_carrier,
                "pulse": iso_pulse,
                "target_duration_s": iso_duration_req,
                "f_left": None,
                "f_right": None,
                "freq": None,
            }
            default_name = f"Isochronic {iso_carrier:.1f}Hz / {iso_pulse:.1f}Hz Pulse"
            initial_params = {
                "name": default_name,
                "track_type": TRACK_TYPE_FREQUENCY,
                "loop_to_fit": True,
                "volume": iso_initial_vol,
            }
            new_track_id = self.app_state.add_track(
                source_info=source_info, initial_params=initial_params
            )
            if new_track_id:
                st.success(
                    f"'{default_name}' added (ID: {new_track_id[:6]}). Preview pending."
                )
                st.rerun()
            else:
                logger.error(f"Failed to add Isochronic track state.")
                st.error(
                    f"Failed to add track for '{default_name}'. Limit might be reached."
                )
            # --- End of modification ---
