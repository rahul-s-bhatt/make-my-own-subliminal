# wizard_steps/step_3_frequency.py
# ==========================================
# UI Rendering for Wizard Step 3: Frequency Selection
# Uses constants from quick_wizard_config.py
# ==========================================

import gc
import logging
from typing import Optional

import streamlit as st

# Import constants from the central config file
from .quick_wizard_config import FREQUENCY_TYPES  # Import list of types
from .quick_wizard_config import LEGACY_FREQ_AUDIO_KEY  # For cleanup
from .quick_wizard_config import LEGACY_FREQ_SR_KEY  # For cleanup
from .quick_wizard_config import (
    DEFAULT_BASE_FREQ,
    DEFAULT_BEAT_FREQ,
    DEFAULT_FREQ_CHOICE,
    DEFAULT_FREQ_PARAMS,
    DEFAULT_PULSE_FREQ,
    FREQ_CHOICE_KEY,
    FREQ_PARAMS_KEY,
)

logger = logging.getLogger(__name__)

# --- Key definitions moved to config ---
# --- Frequency options moved to config ---
# --- Default frequencies moved to config ---


# --- Helper Functions ---
def _validate_frequency(
    freq_str: str, min_val: float = 1.0, max_val: float = 1000.0
) -> Optional[float]:
    """Validates if a string can be converted to a float within a range."""
    try:
        freq = float(freq_str)
        if min_val <= freq <= max_val:
            return freq
        else:
            st.error(f"Frequency must be between {min_val} Hz and {max_val} Hz.")
            return None
    except ValueError:
        st.error("Please enter a valid number for the frequency.")
        return None


# --- Main Rendering Function ---
def render_step_3(wizard):
    """
    Renders the UI for Step 3: Frequency Selection.
    Stores the choice and parameters, does NOT generate audio.

    Args:
        wizard: An instance of the QuickWizard class.
    """
    st.subheader("Step 3: Add Frequency (Optional)")
    st.write("Choose a frequency type like Binaural Beats or Isochronic Tones...")

    # --- Initialization handled by initialize_wizard_state ---

    # --- Frequency Type Selection ---
    try:
        current_choice = st.session_state.get(FREQ_CHOICE_KEY, DEFAULT_FREQ_CHOICE)
        current_choice_index = FREQUENCY_TYPES.index(current_choice)
    except ValueError:
        current_choice_index = 0  # Default to 'None'
        st.session_state[FREQ_CHOICE_KEY] = DEFAULT_FREQ_CHOICE

    selected_choice = st.radio(
        "Frequency Type:",
        options=FREQUENCY_TYPES,  # Use list from config
        index=current_choice_index,
        key="wizard_freq_choice_radio",
        horizontal=True,
        help="Select the type of frequency to add, or 'None'.",
    )

    # --- Update State and Clear Old Audio/Settings if Choice Changes ---
    if selected_choice != st.session_state.get(
        FREQ_CHOICE_KEY
    ):  # Check against state directly
        logger.info(f"Frequency choice changed to: {selected_choice}")
        st.session_state[FREQ_CHOICE_KEY] = selected_choice
        # Clear legacy audio data
        cleared_audio = st.session_state.pop(LEGACY_FREQ_AUDIO_KEY, None) is not None
        cleared_audio |= st.session_state.pop(LEGACY_FREQ_SR_KEY, None) is not None
        if cleared_audio:
            logger.debug("Cleared legacy frequency audio/sr state.")
            gc.collect()
        # Reset parameters to default when type changes?
        st.session_state[FREQ_PARAMS_KEY] = {}  # Clear params or set defaults
        st.rerun()

    # --- Parameter Inputs based on Selection ---
    current_params = st.session_state.get(FREQ_PARAMS_KEY, DEFAULT_FREQ_PARAMS)
    params_changed = False
    new_params = current_params.copy()

    if selected_choice == "Binaural Beats":
        st.markdown("**Binaural Beat Parameters:**")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            base_freq_str = st.text_input(
                "Base Frequency (Hz):",
                value=str(current_params.get("base_freq", DEFAULT_BASE_FREQ)),
                key="wizard_binaural_base_freq",
            )
            valid_base = _validate_frequency(base_freq_str, 20.0, 500.0)
            if valid_base is not None and valid_base != current_params.get("base_freq"):
                new_params["base_freq"] = valid_base
                params_changed = True
        with col_b2:
            beat_freq_str = st.text_input(
                "Beat Frequency (Hz):",
                value=str(current_params.get("beat_freq", DEFAULT_BEAT_FREQ)),
                key="wizard_binaural_beat_freq",
            )
            valid_beat = _validate_frequency(beat_freq_str, 0.1, 30.0)
            if valid_beat is not None and valid_beat != current_params.get("beat_freq"):
                new_params["beat_freq"] = valid_beat
                params_changed = True
        display_base = new_params.get("base_freq", "?")
        display_beat = new_params.get("beat_freq", "?")
        if isinstance(display_base, (int, float)) and isinstance(
            display_beat, (int, float)
        ):
            st.caption(
                f"Generates tones: {display_base - display_beat / 2.0:.2f} Hz (L) & {display_base + display_beat / 2.0:.2f} Hz (R)"
            )
        else:
            st.caption("Enter valid frequencies above.")

    elif selected_choice == "Isochronic Tones":
        st.markdown("**Isochronic Tone Parameters:**")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            base_freq_str = st.text_input(
                "Tone Frequency (Hz):",
                value=str(current_params.get("base_freq", DEFAULT_BASE_FREQ)),
                key="wizard_isochronic_base_freq",
            )
            valid_base = _validate_frequency(base_freq_str, 20.0, 500.0)
            if valid_base is not None and valid_base != current_params.get("base_freq"):
                new_params["base_freq"] = valid_base
                params_changed = True
        with col_i2:
            pulse_freq_str = st.text_input(
                "Pulse Rate (Hz):",
                value=str(current_params.get("pulse_freq", DEFAULT_PULSE_FREQ)),
                key="wizard_isochronic_pulse_freq",
            )
            valid_pulse = _validate_frequency(pulse_freq_str, 0.1, 30.0)
            if valid_pulse is not None and valid_pulse != current_params.get(
                "pulse_freq"
            ):
                new_params["pulse_freq"] = valid_pulse
                params_changed = True
        display_base = new_params.get("base_freq", "?")
        display_pulse = new_params.get("pulse_freq", "?")
        st.caption(
            f"Generates a {display_base} Hz tone pulsing {display_pulse} times per second."
        )

    elif selected_choice == "None":
        st.info("No frequency will be added.")

    if params_changed:
        is_valid = True  # Assume valid unless proven otherwise
        if selected_choice == "Binaural Beats" and (
            "base_freq" not in new_params or "beat_freq" not in new_params
        ):
            is_valid = False
        elif selected_choice == "Isochronic Tones" and (
            "base_freq" not in new_params or "pulse_freq" not in new_params
        ):
            is_valid = False
        if is_valid:
            st.session_state[FREQ_PARAMS_KEY] = new_params
            logger.info(f"Frequency parameters updated: {new_params}")
            st.rerun()

    st.divider()
    # --- Navigation ---
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])
    with col_nav_1:
        if st.button(
            "🏠 Back to Home", key="wizard_step3_home", use_container_width=True
        ):
            wizard._reset_wizard_state()
    with col_nav_2:
        if st.button(
            "⬅️ Back: Background", key="wizard_step3_back", use_container_width=True
        ):
            wizard._go_to_step(2)
    with col_nav_3:
        next_disabled = False
        current_params_check = st.session_state.get(
            FREQ_PARAMS_KEY, {}
        )  # Check current state
        if selected_choice == "Binaural Beats" and (
            "base_freq" not in current_params_check
            or "beat_freq" not in current_params_check
        ):
            next_disabled = True
        elif selected_choice == "Isochronic Tones" and (
            "base_freq" not in current_params_check
            or "pulse_freq" not in current_params_check
        ):
            next_disabled = True
        if st.button(
            "Next: Mix & Export ➡️",
            key="wizard_step3_next",
            type="primary",
            disabled=next_disabled,
            use_container_width=True,
        ):
            if not next_disabled:
                wizard._go_to_step(4)
            else:
                st.warning("Please ensure valid frequency parameters are entered.")
