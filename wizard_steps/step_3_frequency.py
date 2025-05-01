# wizard_steps/step_3_frequency.py
# ==========================================
# UI Rendering for Wizard Step 3: Frequency/Tone
# ==========================================

import logging

# Import TYPE_CHECKING if AudioData type hint is needed without circular import
from typing import Optional  # For type hinting
from typing import TYPE_CHECKING

import numpy as np  # Import NumPy
import streamlit as st

# Import necessary components
from audio_utils.audio_generators import generate_isochronic_tones  # If needed later
from audio_utils.audio_generators import (
    generate_binaural_beats,
    generate_solfeggio_frequency,
)
from config import GLOBAL_SR

# Import AudioData definition for type hinting
if TYPE_CHECKING:
    from audio_utils.audio_state_definitions import AudioData

# Define presets locally or import from a shared config/wizard_config
WIZARD_FREQ_PRESETS = {
    "None": None,  # Option to skip
    "Focus (Alpha 10Hz Binaural)": {
        "type": "binaural",
        "f_left": 200.0,
        "f_right": 210.0,
        "vol": 0.2,
    },
    "Relaxation (Theta 5Hz Binaural)": {
        "type": "binaural",
        "f_left": 150.0,
        "f_right": 155.0,
        "vol": 0.2,
    },
    "Deep Sleep (Delta 2Hz Binaural)": {
        "type": "binaural",
        "f_left": 100.0,
        "f_right": 102.0,
        "vol": 0.2,
    },
    "Love Frequency (Solfeggio 528Hz)": {
        "type": "solfeggio",
        "freq": 528.0,
        "vol": 0.2,
    },
    "Miracle Tone (Solfeggio 417Hz)": {"type": "solfeggio", "freq": 417.0, "vol": 0.2},
}
WIZARD_FREQ_PRESET_NAMES = list(WIZARD_FREQ_PRESETS.keys())
WIZARD_FREQ_PREVIEW_DURATION_S = 10  # Generate only 10 seconds for preview/step

logger = logging.getLogger(__name__)


def _process_frequency_choice(wizard):
    """
    Helper function to generate frequency audio based on choice.
    This ensures audio is ready before moving to the next step if needed,
    or handles the 'None' case.
    """
    preset_name = st.session_state.get("wizard_frequency_choice", "None")
    audio_loaded = st.session_state.get("wizard_frequency_audio") is not None

    if preset_name != "None" and not audio_loaded:
        preset_data = WIZARD_FREQ_PRESETS.get(preset_name)
        if preset_data:
            volume = st.session_state.wizard_frequency_volume  # Use current volume
            logger.info(f"Generating frequency '{preset_name}' before proceeding.")
            with st.spinner(f"Generating {preset_name}..."):
                audio: Optional["AudioData"] = None
                try:
                    if preset_data["type"] == "binaural":
                        audio = generate_binaural_beats(
                            WIZARD_FREQ_PREVIEW_DURATION_S,
                            preset_data["f_left"],
                            preset_data["f_right"],
                            GLOBAL_SR,
                            volume,
                        )
                    elif preset_data["type"] == "solfeggio":
                        audio = generate_solfeggio_frequency(
                            WIZARD_FREQ_PREVIEW_DURATION_S,
                            preset_data["freq"],
                            GLOBAL_SR,
                            volume,
                        )
                    # Add other types if needed

                    if audio is not None:
                        st.session_state.wizard_frequency_audio = audio
                        st.session_state.wizard_frequency_sr = GLOBAL_SR
                    else:
                        st.error(f"Failed to generate {preset_name}.")
                        logger.error(
                            f"Failed to generate frequency {preset_name} before Step 4."
                        )
                except Exception as e:
                    logger.error(f"Error generating frequency {preset_name}: {e}")
                    st.error(f"Failed to generate {preset_name}: {e}")
        else:
            logger.warning(
                f"Preset data not found for '{preset_name}' during processing."
            )
    elif preset_name == "None":
        # Ensure audio state is cleared if 'None' is chosen
        if audio_loaded:
            st.session_state.wizard_frequency_audio = None
            st.session_state.wizard_frequency_sr = None
            logger.info("Cleared frequency audio state as 'None' was selected.")


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
        current_preset_index = WIZARD_FREQ_PRESET_NAMES.index(
            st.session_state.wizard_frequency_choice
        )
    except ValueError:
        current_preset_index = 0  # Default to "None"

    preset_name = st.selectbox(
        "Select Frequency Preset:",
        WIZARD_FREQ_PRESET_NAMES,
        key="wizard_freq_preset_select",
        index=current_preset_index,
    )

    # Update state and clear audio if choice changed
    if preset_name != st.session_state.wizard_frequency_choice:
        st.session_state.wizard_frequency_choice = preset_name
        st.session_state.wizard_frequency_audio = None
        st.session_state.wizard_frequency_sr = None
        # Reset volume to preset default when changing *to* a non-"None" preset
        if preset_name != "None":
            preset_data = WIZARD_FREQ_PRESETS.get(preset_name)  # Use get safely
            if preset_data:
                st.session_state.wizard_frequency_volume = preset_data.get(
                    "vol", 0.2
                )  # Use get with default
            else:
                st.session_state.wizard_frequency_volume = 0.2  # Fallback default
        else:
            # Optionally reset volume if switching back to None, or keep last setting
            # st.session_state.wizard_frequency_volume = 0.2 # Reset to default
            pass
        st.rerun()  # Rerun to update UI and generate audio if needed

    # --- Logic for Selected Preset ---
    if preset_name != "None":
        preset_data = WIZARD_FREQ_PRESETS.get(preset_name)  # Use .get for safety
        if not preset_data:
            st.error(f"Configuration error: Preset data for '{preset_name}' not found.")
            return  # Stop rendering if preset data is missing

        # Volume Slider - use the value from session state
        current_volume = st.session_state.wizard_frequency_volume
        new_volume = st.slider(
            "Frequency Volume", 0.0, 1.0, current_volume, 0.05, key="wizard_freq_vol"
        )

        # Update state and regenerate audio if volume changed
        if not np.isclose(
            new_volume, current_volume
        ):  # Use isclose for float comparison
            st.session_state.wizard_frequency_volume = new_volume
            st.session_state.wizard_frequency_audio = None  # Force regeneration
            st.session_state.wizard_frequency_sr = None
            st.rerun()

        # Generate audio if not already generated for this preset/volume
        if st.session_state.get("wizard_frequency_audio") is None:
            st.info(
                f"Generating a {WIZARD_FREQ_PREVIEW_DURATION_S}-second sample of {preset_name}. This will be looped during final export."
            )
            with st.spinner(f"Generating {preset_name}..."):
                audio: Optional["AudioData"] = None
                duration = WIZARD_FREQ_PREVIEW_DURATION_S
                volume = (
                    st.session_state.wizard_frequency_volume
                )  # Use current volume from state

                try:
                    if preset_data["type"] == "binaural":
                        audio = generate_binaural_beats(
                            duration,
                            preset_data["f_left"],
                            preset_data["f_right"],
                            GLOBAL_SR,
                            volume,
                        )
                    elif preset_data["type"] == "solfeggio":
                        audio = generate_solfeggio_frequency(
                            duration, preset_data["freq"], GLOBAL_SR, volume
                        )
                    # Add isochronic if presets exist for it

                    if audio is not None:
                        st.session_state.wizard_frequency_audio = audio
                        st.session_state.wizard_frequency_sr = GLOBAL_SR
                        # Optional preview: Show a small snippet
                        # preview_snippet = audio[:int(5 * GLOBAL_SR)] # 5 second preview
                        # preview_buffer = save_audio_to_bytesio(preview_snippet, GLOBAL_SR)
                        # if preview_buffer: st.audio(preview_buffer, format="audio/wav")
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
    # Use 3 columns for Home, Back, Next
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])  # Adjust ratios

    with col_nav_1:
        # Add Back to Home button
        if st.button(
            "🏠 Back to Home",
            key="wizard_step3_home",
            use_container_width=True,
            help="Exit Wizard and return to main menu.",
        ):
            wizard._reset_wizard_state()
            # st.rerun() is handled by reset_wizard_state indirectly

    with col_nav_2:
        if st.button(
            "⬅️ Back: Background", key="wizard_step3_back", use_container_width=True
        ):
            wizard._go_to_step(2)

    with col_nav_3:
        if st.button(
            "Next: Export Mix ➡️",
            key="wizard_step3_next",
            type="primary",
            use_container_width=True,
        ):
            # Process frequency choice before moving on
            _process_frequency_choice(wizard)  # Call helper to generate if needed
            wizard._go_to_step(4)
