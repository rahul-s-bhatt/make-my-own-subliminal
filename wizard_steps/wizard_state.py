# wizard_steps/wizard_state.py
# ==========================================
# Session State Management for Quick Wizard
# Uses constants from quick_wizard_config.py
# ==========================================

import gc
import logging

import streamlit as st

# Import constants from the central config file
from .quick_wizard_config import *  # Import all constants

logger = logging.getLogger(__name__)

# --- Defaults moved to config ---


def initialize_wizard_state():
    """Initializes necessary session state variables for the wizard if they don't exist."""
    # Use constants for keys and defaults
    state_defaults = {
        WIZARD_STEP_KEY: DEFAULT_STEP,
        WIZARD_PROCESSING_ACTIVE_KEY: DEFAULT_PROCESSING_ACTIVE,  # For export
        WIZARD_PREVIEW_ACTIVE_KEY: DEFAULT_PREVIEW_ACTIVE,  # For preview <--- NEW
        AFFIRMATION_TEXT_KEY: DEFAULT_AFFIRMATION_TEXT,
        AFFIRMATION_SOURCE_KEY: DEFAULT_AFFIRMATION_SOURCE,
        AFFIRM_APPLY_SPEED_KEY: DEFAULT_APPLY_SPEED,
        AFFIRMATION_VOLUME_KEY: DEFAULT_AFFIRMATION_VOLUME,
        BG_CHOICE_KEY: DEFAULT_BG_CHOICE,
        BG_CHOICE_LABEL_KEY: DEFAULT_BG_CHOICE_LABEL,
        BG_UPLOADED_FILE_KEY: DEFAULT_BG_UPLOADED_FILE,
        BG_NOISE_TYPE_KEY: DEFAULT_NOISE_TYPE,
        BG_VOLUME_KEY: DEFAULT_BG_VOLUME,
        FREQ_CHOICE_KEY: DEFAULT_FREQ_CHOICE,
        FREQ_PARAMS_KEY: DEFAULT_FREQ_PARAMS.copy(),  # Use copy for mutable default
        FREQ_VOLUME_KEY: DEFAULT_FREQ_VOLUME,
        OUTPUT_FILENAME_KEY: DEFAULT_OUTPUT_FILENAME,
        EXPORT_FORMAT_KEY: DEFAULT_EXPORT_FORMAT,
        EXPORT_BUFFER_KEY: None,
        EXPORT_ERROR_KEY: None,
        PREVIEW_BUFFER_KEY: None,
        PREVIEW_ERROR_KEY: None,
        # Legacy keys initialized to None for potential checks/cleanup
        LEGACY_AFFIRM_AUDIO_KEY: None,
        LEGACY_AFFIRM_SR_KEY: None,
        LEGACY_BG_AUDIO_KEY: None,
        LEGACY_BG_SR_KEY: None,
        LEGACY_FREQ_AUDIO_KEY: None,
        LEGACY_FREQ_SR_KEY: None,
        # Add any other keys previously missed
        "wizard_step1_processing": False,  # Example if this was used
        "wizard_uploaded_audio_file": None,  # Example if this was used
    }
    initialized_count = 0
    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            initialized_count += 1
    if initialized_count > 0:
        logger.debug(f"Initialized {initialized_count} wizard state keys.")


def reset_wizard_state():
    """Clears all wizard-specific session state variables and resets workflow selection."""
    logger.info("Resetting Quick Wizard state.")
    # Define all keys managed by this wizard using constants
    keys_to_clear = [
        WIZARD_STEP_KEY,
        WIZARD_PROCESSING_ACTIVE_KEY,  # Export processing flag
        WIZARD_PREVIEW_ACTIVE_KEY,  # Preview processing flag <--- NEW
        AFFIRMATION_TEXT_KEY,
        AFFIRMATION_SOURCE_KEY,
        AFFIRM_APPLY_SPEED_KEY,
        AFFIRMATION_VOLUME_KEY,
        BG_CHOICE_KEY,
        BG_CHOICE_LABEL_KEY,
        BG_UPLOADED_FILE_KEY,
        BG_NOISE_TYPE_KEY,
        BG_VOLUME_KEY,
        FREQ_CHOICE_KEY,
        FREQ_PARAMS_KEY,
        FREQ_VOLUME_KEY,
        OUTPUT_FILENAME_KEY,
        EXPORT_FORMAT_KEY,
        EXPORT_BUFFER_KEY,
        EXPORT_ERROR_KEY,
        PREVIEW_BUFFER_KEY,
        PREVIEW_ERROR_KEY,
        # Include legacy keys if they might exist from previous versions
        LEGACY_AFFIRM_AUDIO_KEY,
        LEGACY_AFFIRM_SR_KEY,
        LEGACY_BG_AUDIO_KEY,
        LEGACY_BG_SR_KEY,
        LEGACY_FREQ_AUDIO_KEY,
        LEGACY_FREQ_SR_KEY,
        # Include any other specific keys
        "wizard_step1_processing",
        "wizard_uploaded_audio_file",
    ]

    popped_count = 0
    for key in keys_to_clear:
        if key in st.session_state:
            # Use pop with default None to avoid KeyError if key somehow missing
            st.session_state.pop(key, None)
            popped_count += 1

    # Clear any other keys starting with 'wizard_' just in case (optional)
    # other_wizard_keys = [
    #     k for k in st.session_state if k.startswith("wizard_") and k not in keys_to_clear
    # ]
    # for key in other_wizard_keys:
    #     st.session_state.pop(key, None)
    #     popped_count += 1

    # Clear workflow selection if it exists
    if "selected_workflow" in st.session_state:
        st.session_state.pop("selected_workflow", None)
        popped_count += 1

    if popped_count > 0:
        logger.debug(f"Cleared {popped_count} session state keys during reset.")
        gc.collect()

    # Re-initialize to defaults after clearing
    initialize_wizard_state()
    logger.info("Quick Wizard state reset complete.")
