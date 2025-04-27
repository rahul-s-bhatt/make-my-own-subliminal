# sidebar_generators.py
# ==========================================
# Frequency and Noise Generation UI for MindMorph Sidebar
# ==========================================

import logging
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

# Import necessary components from other modules
from app_state import AppState, TrackData, TrackType
from audio_generators import generate_binaural_beats, generate_isochronic_tones, generate_noise, generate_solfeggio_frequency  # Use the specific generators module

# Type hint (ensure consistency across files)
try:
    from audio_processing import AudioData
except ImportError:
    AudioData = np.ndarray
from config import GLOBAL_SR, TRACK_TYPE_BACKGROUND, TRACK_TYPE_FREQUENCY, get_default_track_params

# Get a logger for this module
logger = logging.getLogger(__name__)


class SidebarGenerators:
    """Handles rendering frequency and noise generator UI in the sidebar."""

    def __init__(self, app_state: AppState):
        """
        Initializes the SidebarGenerators.

        Args:
            app_state: An instance of the AppState class.
        """
        self.app_state = app_state
        logger.debug("SidebarGenerators initialized.")

    # --- Public Rendering Methods ---

    def render_frequency_generators(self):
        """Renders frequency/tone generation options based on app mode."""
        st.subheader("🧠✨ Add Frequencies / Tones")
        app_mode = st.session_state.get("app_mode", "Easy")

        if app_mode == "Easy":
            available_gen_types = ["Presets"]
            st.caption("Use presets for common frequency patterns.")
        else:  # Advanced mode
            available_gen_types = ["Binaural Beats", "Solfeggio Tones", "Isochronic Tones", "Presets"]
            st.caption("Generate specific tones or use presets.")

        gen_type_key = "sidebar_frequency_gen_type"  # Keep same key for consistency
        default_gen_type = "Presets"
        if gen_type_key in st.session_state and st.session_state[gen_type_key] in available_gen_types:
            default_gen_type = st.session_state[gen_type_key]
        try:
            default_index = available_gen_types.index(default_gen_type)
        except ValueError:
            default_index = 0

        gen_type = st.radio("Select Frequency Type:", available_gen_types, index=default_index, key=gen_type_key, horizontal=True, label_visibility="collapsed")

        # Call private methods to render specific generator UI
        if gen_type == "Presets":
            self._render_frequency_presets()
        elif gen_type == "Binaural Beats" and app_mode == "Advanced":
            self._render_binaural_generator()
        elif gen_type == "Solfeggio Tones" and app_mode == "Advanced":
            self._render_solfeggio_generator()
        elif gen_type == "Isochronic Tones" and app_mode == "Advanced":
            self._render_isochronic_generator()

    def render_background_generators(self):
        """Renders options for generating background noise."""
        st.subheader("🎵 Add Background Noise")
        noise_options = ["White Noise", "Pink Noise", "Brown Noise"]
        noise_type = st.selectbox("Select Noise Type:", noise_options, key="sidebar_noise_type", help="Choose a type of background noise.")

        cols_noise = st.columns(2)
        with cols_noise[0]:
            noise_duration = st.number_input("Duration (s)##Noise", 10, 7200, 300, 10, key="sidebar_noise_duration", help="Length in seconds. Will loop if needed.")
        with cols_noise[1]:
            noise_vol = st.slider("Volume##Noise", 0.0, 1.0, 0.5, 0.05, key="sidebar_noise_volume", help="Loudness of the noise.")

        if st.button(f"Generate {noise_type} Track", key="sidebar_generate_noise"):
            with st.spinner(f"Generating {noise_type}..."):
                audio = generate_noise(noise_type, noise_duration, GLOBAL_SR, noise_vol)  # From audio_generators
                if audio is not None and audio.size > 0:
                    track_params = get_default_track_params()
                    default_name = f"{noise_type}"
                    track_params.update(
                        {
                            "original_audio": audio,
                            "sr": GLOBAL_SR,
                            "name": default_name,
                            "loop_to_fit": True,
                            "source_type": "noise",
                            "gen_noise_type": noise_type,
                            "gen_duration": noise_duration,
                            "gen_volume": noise_vol,
                        }
                    )
                    new_track_id = self.app_state.add_track(track_params, track_type=TRACK_TYPE_BACKGROUND)
                    st.success(f"'{default_name}' track generated (ID: {new_track_id[:6]})!")
                    st.rerun()  # Rerun to show new track
                elif audio is not None:
                    st.warning(f"Generated {noise_type} was empty.")
                else:
                    logger.error(f"Failed to generate {noise_type}.")  # generate_noise shows error

    # --- Private Rendering Methods for Specific Generators ---

    def _render_frequency_presets(self):
        """Renders the UI for selecting and generating frequency presets."""
        st.markdown("<small>Generate common frequency patterns like Alpha for focus or Delta for sleep.</small>", unsafe_allow_html=True)
        preset_options = {
            "Focus (Alpha 10Hz Binaural)": {"type": "binaural", "f_left": 200.0, "f_right": 210.0},
            "Relaxation (Theta 5Hz Binaural)": {"type": "binaural", "f_left": 150.0, "f_right": 155.0},
            "Deep Sleep (Delta 2Hz Binaural)": {"type": "binaural", "f_left": 100.0, "f_right": 102.0},
            "Love Frequency (Solfeggio 528Hz)": {"type": "solfeggio", "freq": 528.0},
            "Miracle Tone (Solfeggio 417Hz)": {"type": "solfeggio", "freq": 417.0},
        }
        preset_name = st.selectbox("Select Preset:", list(preset_options.keys()), key="sidebar_freq_preset_select")
        preset_data = preset_options[preset_name]

        cols_preset = st.columns(2)
        with cols_preset[0]:
            preset_duration = st.number_input("Duration (s)", 1, 7200, 60, 1, key="sidebar_preset_duration", help="Duration in seconds.")
        with cols_preset[1]:
            preset_vol = st.slider("Volume", 0.0, 1.0, 0.2, 0.05, key="sidebar_preset_volume", help="Volume (usually kept low).")

        if st.button(f"Generate '{preset_name}' Track", key="sidebar_generate_preset"):
            with st.spinner("Generating preset frequency..."):
                audio: Optional[AudioData] = None
                gen_params: Dict[str, Any] = {}
                default_track_name = preset_name

                if preset_data["type"] == "binaural":
                    audio = generate_binaural_beats(preset_duration, preset_data["f_left"], preset_data["f_right"], GLOBAL_SR, preset_vol)
                    gen_params = {
                        "source_type": "binaural_preset",
                        "gen_duration": preset_duration,
                        "gen_freq_left": preset_data["f_left"],
                        "gen_freq_right": preset_data["f_right"],
                        "gen_volume": preset_vol,
                    }
                elif preset_data["type"] == "solfeggio":
                    audio = generate_solfeggio_frequency(preset_duration, preset_data["freq"], GLOBAL_SR, preset_vol)
                    gen_params = {"source_type": "solfeggio_preset", "gen_duration": preset_duration, "gen_freq": preset_data["freq"], "gen_volume": preset_vol}

                if audio is not None and audio.size > 0:
                    track_params = get_default_track_params()
                    track_params.update({"original_audio": audio, "sr": GLOBAL_SR, "name": default_track_name})
                    track_params.update(gen_params)
                    new_track_id = self.app_state.add_track(track_params, track_type=TRACK_TYPE_FREQUENCY)
                    st.success(f"'{default_track_name}' track generated (ID: {new_track_id[:6]})!")
                    st.rerun()  # Rerun to show new track
                elif audio is not None:
                    st.warning("Generated preset audio was empty.")
                else:
                    st.error("Failed to generate audio for the selected preset.")

    def _render_binaural_generator(self):
        """Renders UI for generating custom Binaural Beats (Advanced Mode)."""
        st.markdown("<small>Generates stereo tones potentially inducing brainwave states (requires headphones).</small>", unsafe_allow_html=True)
        bb_cols = st.columns(2)
        with bb_cols[0]:
            bb_duration = st.number_input("Duration (s)", 1, 7200, 60, 1, key="sidebar_bb_duration", help="Length in seconds.")
            bb_fleft = st.number_input("Left Freq (Hz)", 20.0, 1000.0, 200.0, 0.1, format="%.1f", key="sidebar_bb_freq_left", help="Left ear frequency.")
        with bb_cols[1]:
            bb_vol = st.slider("Volume##BB", 0.0, 1.0, 0.3, 0.05, key="sidebar_bb_volume", help="Loudness (0.0 to 1.0). Usually kept low.")
            bb_fright = st.number_input("Right Freq (Hz)", 20.0, 1000.0, 210.0, 0.1, format="%.1f", key="sidebar_bb_freq_right", help="Right ear frequency.")

        beat_freq = abs(bb_fleft - bb_fright)
        st.caption(f"Resulting Beat Frequency: {beat_freq:.1f} Hz")

        if st.button("Generate Binaural Track", key="sidebar_generate_bb", help="Create track with these settings."):
            with st.spinner("Generating Binaural Beats..."):
                audio = generate_binaural_beats(bb_duration, bb_fleft, bb_fright, GLOBAL_SR, bb_vol)
            if audio is not None and audio.size > 0:
                track_params = get_default_track_params()
                default_name = f"Binaural {bb_fleft:.1f}Hz L / {bb_fright:.1f}Hz R ({beat_freq:.1f}Hz Beat)"
                track_params.update(
                    {
                        "original_audio": audio,
                        "sr": GLOBAL_SR,
                        "name": default_name,
                        "source_type": "binaural",
                        "gen_duration": bb_duration,
                        "gen_freq_left": bb_fleft,
                        "gen_freq_right": bb_fright,
                        "gen_volume": bb_vol,
                    }
                )
                new_track_id = self.app_state.add_track(track_params, track_type=TRACK_TYPE_FREQUENCY)
                st.success(f"'{default_name}' generated (ID: {new_track_id[:6]})!")
                st.rerun()  # Rerun to show new track
            elif audio is not None:
                st.warning("Generated binaural audio was empty.")
            else:
                st.error("Failed to generate binaural beats.")

    def _render_solfeggio_generator(self):
        """Renders UI for generating Solfeggio Tones (Advanced Mode)."""
        st.markdown("<small>Generates pure tones based on historical Solfeggio frequencies.</small>", unsafe_allow_html=True)
        freqs = [174.0, 285.0, 396.0, 417.0, 528.0, 639.0, 741.0, 852.0, 963.0]
        freq_labels = {f: f"{f} Hz" for f in freqs}

        cols = st.columns(2)
        with cols[0]:
            freq = st.selectbox("Frequency (Hz)", freqs, index=4, key="sidebar_solf_freq", format_func=lambda x: freq_labels.get(x, f"{x} Hz"), help="Select Solfeggio frequency.")
        with cols[1]:
            duration = st.number_input("Duration (s)##Solf", 1, 7200, 60, 1, key="sidebar_solf_duration", help="Length in seconds.")

        vol = st.slider("Volume##Solf", 0.0, 1.0, 0.3, 0.05, key="sidebar_solf_volume", help="Loudness (0.0 to 1.0). Usually kept low.")

        if st.button("Generate Solfeggio Track", key="sidebar_generate_solf", help="Create track with this tone."):
            with st.spinner("Generating Solfeggio Tone..."):
                audio = generate_solfeggio_frequency(duration, freq, GLOBAL_SR, vol)
            if audio is not None and audio.size > 0:
                track_params = get_default_track_params()
                default_name = f"Solfeggio {freq}Hz"
                track_params.update(
                    {"original_audio": audio, "sr": GLOBAL_SR, "name": default_name, "source_type": "solfeggio", "gen_duration": duration, "gen_freq": freq, "gen_volume": vol}
                )
                new_track_id = self.app_state.add_track(track_params, track_type=TRACK_TYPE_FREQUENCY)
                st.success(f"'{default_name}' generated (ID: {new_track_id[:6]})!")
                st.rerun()  # Rerun to show new track
            elif audio is not None:
                st.warning("Generated Solfeggio audio was empty.")
            else:
                st.error("Failed to generate Solfeggio tone.")

    def _render_isochronic_generator(self):
        """Renders UI for generating Isochronic Tones (Advanced Mode)."""
        st.markdown("<small>Generates rhythmic pulses of a single tone (headphones not required).</small>", unsafe_allow_html=True)
        iso_cols = st.columns(2)
        with iso_cols[0]:
            iso_duration = st.number_input("Duration (s)##Iso", 1, 7200, 60, 1, key="sidebar_iso_duration", help="Length in seconds.")
            iso_carrier = st.number_input("Carrier Freq (Hz)", 20.0, 1000.0, 150.0, 0.1, format="%.1f", key="sidebar_iso_carrier", help="The base tone frequency being pulsed.")
        with iso_cols[1]:
            iso_vol = st.slider("Volume##Iso", 0.0, 1.0, 0.4, 0.05, key="sidebar_iso_volume", help="Loudness (0.0 to 1.0).")
            iso_pulse = st.number_input(
                "Pulse Freq (Hz)", 0.1, 40.0, 10.0, 0.1, format="%.1f", key="sidebar_iso_pulse", help="How many times per second the tone pulses (e.g., 10Hz for Alpha)."
            )

        if st.button("Generate Isochronic Track", key="sidebar_generate_iso", help="Create track with these settings."):
            with st.spinner("Generating Isochronic Tones..."):
                audio = generate_isochronic_tones(iso_duration, iso_carrier, iso_pulse, GLOBAL_SR, iso_vol)
            if audio is not None and audio.size > 0:
                track_params = get_default_track_params()
                default_name = f"Isochronic {iso_carrier:.1f}Hz / {iso_pulse:.1f}Hz Pulse"
                track_params.update(
                    {
                        "original_audio": audio,
                        "sr": GLOBAL_SR,
                        "name": default_name,
                        "source_type": "isochronic",
                        "gen_duration": iso_duration,
                        "gen_carrier_freq": iso_carrier,
                        "gen_pulse_freq": iso_pulse,
                        "gen_volume": iso_vol,
                    }
                )
                new_track_id = self.app_state.add_track(track_params, track_type=TRACK_TYPE_FREQUENCY)
                st.success(f"'{default_name}' generated (ID: {new_track_id[:6]})!")
                st.rerun()  # Rerun to show new track
            elif audio is not None:
                st.warning("Generated Isochronic audio was empty.")
            else:
                st.error("Failed to generate Isochronic tones.")
