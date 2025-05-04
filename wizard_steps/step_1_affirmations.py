# wizard_steps/step_1_affirmations.py
# ==========================================
# UI Rendering for Wizard Step 1: Affirmations Input
# Uses constants from quick_wizard_config.py
# ==========================================

import logging

import streamlit as st

# Import constants from the central config file
from .quick_wizard_config import (
    AFFIRM_APPLY_SPEED_KEY,
    AFFIRMATION_SOURCE_KEY,
    AFFIRMATION_TEXT_KEY,
    DEFAULT_AFFIRMATION_SOURCE,
    DEFAULT_AFFIRMATION_TEXT,
    DEFAULT_APPLY_SPEED,
)

# Import config values potentially needed from main config
try:
    from config import QUICK_SUBLIMINAL_PRESET_SPEED
except ImportError:
    QUICK_SUBLIMINAL_PRESET_SPEED = 2.0  # Fallback
    logging.warning("Could not import QUICK_SUBLIMINAL_PRESET_SPEED from config.")

logger = logging.getLogger(__name__)

# --- Key definition moved to config ---
# AFFIRM_APPLY_SPEED_KEY = "wizard_apply_speed_change"


def render_step_1(wizard):
    """
    Renders the UI for Step 1: Affirmations Input (Text Area Only).
    Stores text and speed setting, does NOT generate audio.

    Args:
        wizard: An instance of the QuickWizard class.
    """
    st.subheader("Step 1: Enter Your Affirmations")
    st.write(
        "Type or paste the affirmations you want to include. The audio will be generated later in the process."
    )

    # --- Set Affirmation Source ---
    # Use constant key
    st.session_state[AFFIRMATION_SOURCE_KEY] = (
        DEFAULT_AFFIRMATION_SOURCE  # Usually 'text'
    )

    # --- Affirmation Text Area ---
    st.markdown("**Affirmation Text:**")
    st.text_area(
        "Enter affirmations here (one per line recommended):",
        key="wizard_affirm_text_area",  # Keep widget key distinct if needed, but use constant for state access
        value=st.session_state.get(AFFIRMATION_TEXT_KEY, DEFAULT_AFFIRMATION_TEXT),
        height=200,
        help="Enter the positive statements you want to use.",
        on_change=wizard.sync_affirmation_text,  # Syncs text state using AFFIRMATION_TEXT_KEY
    )

    affirmation_text = st.session_state.get(AFFIRMATION_TEXT_KEY, "")
    text_is_present = bool(affirmation_text and affirmation_text.strip())

    # --- Optional Speed Setting ---
    st.divider()
    st.markdown("**Optional Settings:**")

    # Checkbox linked via key. Initialization handled by initialize_wizard_state
    st.checkbox(
        f"Apply Speed Change ({QUICK_SUBLIMINAL_PRESET_SPEED}x Speed)",
        key=AFFIRM_APPLY_SPEED_KEY,  # Use constant key
        help=f"Check this to speed up the affirmation audio...",
    )

    # Logging uses the constant key
    logger.info(
        f"DEBUG STEP 1 - Value of st.session_state[{AFFIRM_APPLY_SPEED_KEY}] AFTER checkbox render: {st.session_state.get(AFFIRM_APPLY_SPEED_KEY, DEFAULT_APPLY_SPEED)}"
    )

    st.divider()

    # --- Navigation ---
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])

    with col_nav_1:  # Home
        if st.button(
            "🏠 Back to Home", key="wizard_step1_home", use_container_width=True
        ):
            wizard._reset_wizard_state()

    with col_nav_2:  # Back (Disabled)
        st.button(
            "⬅️ Back", key="wizard_step1_back", disabled=True, use_container_width=True
        )

    with col_nav_3:  # Next
        next_disabled = not text_is_present
        next_tooltip = (
            "Proceed to Step 2 (Background)."
            if not next_disabled
            else "Please enter affirmation text first."
        )
        if st.button(
            "Next: Background ➡️",
            key="wizard_step1_next",
            type="primary",
            disabled=next_disabled,
            help=next_tooltip,
            use_container_width=True,
        ):
            if not next_disabled:
                wizard._go_to_step(2)
            else:
                st.warning("Please enter some affirmation text before proceeding.")
                st.rerun()
