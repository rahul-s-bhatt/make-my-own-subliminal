# wizard_steps/step_3_frequency.py
# ==========================================
# UI Rendering for Wizard Step 3: Frequency/Tone
# ==========================================

import logging

# Import TYPE_CHECKING if AudioData type hint is needed without circular import
from typing import (
    TYPE_CHECKING,
    Optional,  # For type hinting
)

import streamlit as st

# Import necessary components
from audio_generators import (
    generate_binaural_beats,
    generate_isochronic_tones,  # If needed later
    generate_solfeggio_frequency,
)
from audio_io import save_audio_to_bytesio
from config import GLOBAL_SR

if TYPE_CHECKING:
    from audio_processing import AudioData

# Define presets locally or import from a shared config/wizard_config
WIZARD_FREQ_PRESETS = {
    "None": None,  # Option to skip
    "Focus (Alpha 10Hz Binaural)": {"type": "binaural", "f_left": 200.0, "f_right": 210.0, "vol": 0.2},
    "Relaxation (Theta 5Hz Binaural)": {"type": "binaural", "f_left": 150.0, "f_right": 155.0, "vol": 0.2},
    "Deep Sleep (Delta 2Hz Binaural)": {"type": "binaural", "f_left": 100.0, "f_right": 102.0, "vol": 0.2},
    "Love Frequency (Solfeggio 528Hz)": {"type": "solfeggio", "freq": 528.0, "vol": 0.2},
    "Miracle Tone (Solfeggio 417Hz)": {"type": "solfeggio", "freq": 417.0, "vol": 0.2},
}
WIZARD_FREQ_PRESET_NAMES = list(WIZARD_FREQ_PRESETS.keys())

logger = logging.getLogger(__name__)


def render_step_3(wizard):
    """
    Renders the UI for Step 3: Frequency/Tone.

    Args:
        wizard: The instance of the main QuickWizard class.
    """
    st.subheader("Step 3: Add Frequency/Tone (Optional)")
    st.write("Add a background frequency like Binaural Beats or Solfeggio Tones.")

    # Get current preset index
    try:
        current_preset_index = WIZARD_FREQ_PRESET_NAMES.index(st.session_state.wizard_frequency_choice)
    except ValueError:
        current_preset_index = 0  # Default to "None"

    preset_name = st.selectbox("Select Frequency Preset:", WIZARD_FREQ_PRESET_NAMES, key="wizard_freq_preset_select", index=current_preset_index)

    # Update state and clear audio if choice changed
    if preset_name != st.session_state.wizard_frequency_choice:
        st.session_state.wizard_frequency_choice = preset_name
        st.session_state.wizard_frequency_audio = None
        st.session_state.wizard_frequency_sr = None
        # Reset volume to preset default when changing *to* a non-"None" preset
        if preset_name != "None":
            preset_data = WIZARD_FREQ_PRESETS[preset_name]
            st.session_state.wizard_frequency_volume = preset_data["vol"]
        st.rerun()  # Rerun to update UI and generate audio if needed

    # --- Logic for Selected Preset ---
    if preset_name != "None":
        preset_data = WIZARD_FREQ_PRESETS[preset_name]

        # Volume Slider - use the value from session state
        current_volume = st.session_state.wizard_frequency_volume
        new_volume = st.slider("Frequency Volume", 0.0, 1.0, current_volume, 0.05, key="wizard_freq_vol")

        # Update state and regenerate audio if volume changed
        if new_volume != current_volume:
            st.session_state.wizard_frequency_volume = new_volume
            st.session_state.wizard_frequency_audio = None  # Force regeneration
            st.session_state.wizard_frequency_sr = None
            st.rerun()

        # Generate audio if not already generated for this preset/volume
        if st.session_state.get("wizard_frequency_audio") is None:
            with st.spinner(f"Generating {preset_name}..."):
                audio: Optional["AudioData"] = None
                duration = 60  # Generate 60s sample, will be looped/truncated later
                volume = st.session_state.wizard_frequency_volume  # Use current volume from state

                try:
                    if preset_data["type"] == "binaural":
                        audio = generate_binaural_beats(duration, preset_data["f_left"], preset_data["f_right"], GLOBAL_SR, volume)
                    elif preset_data["type"] == "solfeggio":
                        audio = generate_solfeggio_frequency(duration, preset_data["freq"], GLOBAL_SR, volume)
                    # Add isochronic if presets exist for it

                    if audio is not None:
                        st.session_state.wizard_frequency_audio = audio
                        st.session_state.wizard_frequency_sr = GLOBAL_SR
                        # Optional preview:
                        # st.audio(save_audio_to_bytesio(audio, GLOBAL_SR), format="audio/wav")
                        st.rerun()  # Rerun now that audio exists
                    else:
                        st.error(f"Failed to generate {preset_name}.")
                except Exception as e:
                    logger.error(f"Error generating frequency {preset_name}: {e}")
                    st.error(f"Failed to generate {preset_name}: {e}")

    else:
        # Ensure audio is cleared if "None" is selected
        if st.session_state.get("wizard_frequency_audio") is not None:
            st.session_state.wizard_frequency_audio = None
            st.session_state.wizard_frequency_sr = None
        st.caption("No frequency tone will be added.")

    # --- Navigation Buttons ---
    st.divider()
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        st.button("Back: Background Sound", on_click=wizard._go_to_step, args=(2,), key="wizard_step3_back")
    with col_nav2:
        # Enable Next button regardless of choice (as frequency is optional)
        st.button("Next: Export", on_click=wizard._go_to_step, args=(4,), type="primary", key="wizard_step3_next")
