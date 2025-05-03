# wizard_state.py
# ==========================================
# Session State Management for Quick Wizard
# ==========================================

import logging

import streamlit as st

from config import GLOBAL_SR  # Needed for defaults maybe

logger = logging.getLogger(__name__)

# Define default values here if needed, or rely on initialization logic
DEFAULT_BG_VOLUME = 0.7
DEFAULT_FREQ_VOLUME = 0.2
DEFAULT_NOISE_TYPE = "White Noise"
DEFAULT_FILENAME = "my_quick_subliminal"
DEFAULT_EXPORT_FORMAT = "WAV"
DEFAULT_APPLY_QUICK_SETTINGS = True
# --- Default affirmation volume ---
DEFAULT_AFFIRM_VOLUME = 1.0
# --- END ---


def initialize_wizard_state():
    """Initializes necessary session state variables for the wizard if they don't exist."""
    state_defaults = {
        "wizard_step": 1,
        "wizard_affirmation_text": "",
        "wizard_affirmation_audio": None,
        "wizard_affirmation_sr": None,
        "wizard_affirmation_source": None,
        # --- Affirmation volume state ---
        "wizard_affirmation_volume": DEFAULT_AFFIRM_VOLUME,
        # --- END ---
        "wizard_background_choice": "none",
        "wizard_background_choice_label": "None (Skip)",
        "wizard_background_audio": None,
        "wizard_background_sr": None,
        "wizard_background_volume": DEFAULT_BG_VOLUME,
        "wizard_background_noise_type": DEFAULT_NOISE_TYPE,
        "wizard_frequency_choice": "None",
        "wizard_frequency_audio": None,
        "wizard_frequency_sr": None,
        "wizard_frequency_volume": DEFAULT_FREQ_VOLUME,
        "wizard_output_filename": DEFAULT_FILENAME,
        "wizard_export_format": DEFAULT_EXPORT_FORMAT,
        "wizard_apply_quick_settings": DEFAULT_APPLY_QUICK_SETTINGS,
        "wizard_export_buffer": None,
        "wizard_export_error": None,
        "wizard_original_affirmation_text": None,
        "wizard_affirm_text_pending_update": None,
        "wizard_affirm_truncated_pending": False,
        "wizard_processing_active": False,
        # --- ADDED: Preview state ---
        "wizard_preview_buffer": None,
        "wizard_preview_error": None,
        # --- END ADDED ---
    }
    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def reset_wizard_state():
    """Clears all wizard-specific session state variables and resets workflow selection."""
    logger.info("Resetting Quick Wizard state.")
    # Ensure processing flag is reset if wizard is exited prematurely
    st.session_state.wizard_processing_active = False
    keys_to_delete = [k for k in st.session_state if k.startswith("wizard_")]
    for key in keys_to_delete:
        if key != "wizard_processing_active":  # Don't delete the flag itself here
            del st.session_state[key]

    if "selected_workflow" in st.session_state:
        del st.session_state["selected_workflow"]
        logger.debug("Deleted session state key: selected_workflow")

    # Re-initialize defaults after clearing
    initialize_wizard_state()
    logger.info("Quick Wizard state reset complete.")
