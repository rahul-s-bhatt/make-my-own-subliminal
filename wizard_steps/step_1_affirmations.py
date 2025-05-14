# wizard_steps/step_1_affirmations.py
# ==========================================
# UI Rendering for Wizard Step 1: Affirmations Input
# Uses constants from quick_wizard_config.py
# Includes Affirmation Expansion Functionality
# ==========================================

import logging

import streamlit as st

from affirmation_expander import expand_affirmations

# Import constants from the central config file
# Make sure to add the new keys here or import them if defined in quick_wizard_config
from .quick_wizard_config import (
    AFFIRM_APPLY_SPEED_KEY,
    AFFIRMATION_SOURCE_KEY,
    AFFIRMATION_TEXT_KEY,  # Key for the actual affirmation text value
    DEFAULT_AFFIRMATION_SOURCE,
    DEFAULT_AFFIRMATION_TEXT,
    DEFAULT_APPLY_SPEED,
    WIZARD_AFFIRM_ORIGINAL_TEXT_KEY,
    # Define or import new keys for expansion
    WIZARD_AFFIRM_TEXT_AREA_WIDGET_KEY,  # Key for the st.text_area widget itself
    WIZARD_EXPANSION_RESULT_TEXT_KEY,
    WIZARD_EXPANSION_WAS_TRUNCATED_KEY,
    WIZARD_PENDING_AFFIRMATION_TEXT_UPDATE_KEY,
    WIZARD_PENDING_UPDATE_WAS_TRUNCATED_KEY,
)

# Import config values potentially needed from main config
try:
    from config import MAX_AFFIRMATION_CHARS, QUICK_SUBLIMINAL_PRESET_SPEED
except ImportError:
    QUICK_SUBLIMINAL_PRESET_SPEED = 2.0  # Fallback
    MAX_AFFIRMATION_CHARS = 1000  # Fallback
    logging.warning("Could not import QUICK_SUBLIMINAL_PRESET_SPEED or MAX_AFFIRMATION_CHARS from config.")

logger = logging.getLogger(__name__)


def initialize_wizard_expansion_state():
    """Initializes session state keys required for the expansion feature if they don't exist."""
    if WIZARD_AFFIRM_ORIGINAL_TEXT_KEY not in st.session_state:
        st.session_state[WIZARD_AFFIRM_ORIGINAL_TEXT_KEY] = None
    if WIZARD_EXPANSION_RESULT_TEXT_KEY not in st.session_state:
        st.session_state[WIZARD_EXPANSION_RESULT_TEXT_KEY] = None
    if WIZARD_EXPANSION_WAS_TRUNCATED_KEY not in st.session_state:
        st.session_state[WIZARD_EXPANSION_WAS_TRUNCATED_KEY] = False
    if WIZARD_PENDING_AFFIRMATION_TEXT_UPDATE_KEY not in st.session_state:
        st.session_state[WIZARD_PENDING_AFFIRMATION_TEXT_UPDATE_KEY] = None
    if WIZARD_PENDING_UPDATE_WAS_TRUNCATED_KEY not in st.session_state:
        st.session_state[WIZARD_PENDING_UPDATE_WAS_TRUNCATED_KEY] = False
    if AFFIRMATION_TEXT_KEY not in st.session_state:
        st.session_state[AFFIRMATION_TEXT_KEY] = DEFAULT_AFFIRMATION_TEXT


def render_step_1(wizard):
    """
    Renders the UI for Step 1: Affirmations Input with Expansion.
    Stores text and speed setting, does NOT generate audio.

    Args:
        wizard: An instance of the QuickWizard class. (Currently not used for sync)
    """
    st.subheader("Step 1: Enter Your Affirmations")
    st.write("Type or paste the affirmations you want to include. You can also use the 'Expand Affirmations' feature to enhance your text.")

    initialize_wizard_expansion_state()

    if st.session_state.get(WIZARD_PENDING_AFFIRMATION_TEXT_UPDATE_KEY) is not None:
        logger.debug("Applying pending affirmation text update for wizard.")
        st.session_state[AFFIRMATION_TEXT_KEY] = st.session_state[WIZARD_PENDING_AFFIRMATION_TEXT_UPDATE_KEY]
        st.session_state[WIZARD_AFFIRM_TEXT_AREA_WIDGET_KEY] = st.session_state[AFFIRMATION_TEXT_KEY]

        if st.session_state.get(WIZARD_PENDING_UPDATE_WAS_TRUNCATED_KEY):
            st.warning(
                f"⚠️ Expanded text was shortened to fit {MAX_AFFIRMATION_CHARS} characters.",
                icon="✂️",
            )
        st.session_state[WIZARD_PENDING_AFFIRMATION_TEXT_UPDATE_KEY] = None
        st.session_state[WIZARD_PENDING_UPDATE_WAS_TRUNCATED_KEY] = False

    st.session_state[AFFIRMATION_SOURCE_KEY] = DEFAULT_AFFIRMATION_SOURCE

    st.markdown("**Affirmation Text:**")

    def handle_text_area_change():
        # Always sync the canonical state (AFFIRMATION_TEXT_KEY) from the widget's current value
        # (st.session_state[WIZARD_AFFIRM_TEXT_AREA_WIDGET_KEY]).
        if WIZARD_AFFIRM_TEXT_AREA_WIDGET_KEY in st.session_state:
            typed_text = st.session_state[WIZARD_AFFIRM_TEXT_AREA_WIDGET_KEY]
            st.session_state[AFFIRMATION_TEXT_KEY] = typed_text
            logger.debug(f"In handle_text_area_change: Synced AFFIRMATION_TEXT_KEY from widget. New value: '{typed_text[:50]}...'")
        else:
            logger.warning(f"Widget key {WIZARD_AFFIRM_TEXT_AREA_WIDGET_KEY} not found in session_state during on_change. Setting AFFIRMATION_TEXT_KEY to empty string.")
            st.session_state[AFFIRMATION_TEXT_KEY] = ""

        # Clear expansion state if text is manually edited after an expansion attempt.
        # This logic uses AFFIRMATION_TEXT_KEY, which has just been updated.
        current_text_for_expansion_check = st.session_state.get(AFFIRMATION_TEXT_KEY, "")
        original_for_expansion = st.session_state.get(WIZARD_AFFIRM_ORIGINAL_TEXT_KEY)
        expansion_result = st.session_state.get(WIZARD_EXPANSION_RESULT_TEXT_KEY)

        if original_for_expansion is not None and current_text_for_expansion_check != original_for_expansion and current_text_for_expansion_check != expansion_result:
            st.session_state[WIZARD_AFFIRM_ORIGINAL_TEXT_KEY] = None
            st.session_state[WIZARD_EXPANSION_RESULT_TEXT_KEY] = None
            st.session_state[WIZARD_EXPANSION_WAS_TRUNCATED_KEY] = False
            logger.debug("Cleared wizard expansion backup and result due to manual edit of affirmation text.")

    affirmation_text_value = st.session_state.get(AFFIRMATION_TEXT_KEY, DEFAULT_AFFIRMATION_TEXT)

    st.text_area(
        "Enter affirmations here (one per line recommended):",
        key=WIZARD_AFFIRM_TEXT_AREA_WIDGET_KEY,
        value=affirmation_text_value,
        height=200,
        max_chars=MAX_AFFIRMATION_CHARS,
        help="Enter the positive statements you want to use. You can expand them below.",
        on_change=handle_text_area_change,
    )
    # The caption will update on the rerun after handle_text_area_change updates AFFIRMATION_TEXT_KEY
    st.caption(f"{len(st.session_state.get(AFFIRMATION_TEXT_KEY, ''))} / {MAX_AFFIRMATION_CHARS} characters")

    expand_col, undo_col = st.columns(2)
    # Use the most up-to-date text from AFFIRMATION_TEXT_KEY for button logic
    current_affirm_text_for_button = st.session_state.get(AFFIRMATION_TEXT_KEY, "")

    with expand_col:
        expand_disabled = not current_affirm_text_for_button.strip() or not callable(globals().get("expand_affirmations"))
        if st.button(
            "✨ Expand Affirmations",
            key="wizard_expand_affirmations",
            disabled=expand_disabled,
            use_container_width=True,
            help="Uses AI to expand and elaborate on your affirmations.",
        ):
            if callable(globals().get("expand_affirmations")):
                with st.spinner("Expanding affirmations..."):
                    try:
                        st.session_state[WIZARD_AFFIRM_ORIGINAL_TEXT_KEY] = current_affirm_text_for_button
                        expanded_text, truncated = expand_affirmations(
                            base_text=current_affirm_text_for_button,
                            max_chars=MAX_AFFIRMATION_CHARS,
                        )
                        st.session_state[WIZARD_EXPANSION_RESULT_TEXT_KEY] = expanded_text
                        st.session_state[WIZARD_EXPANSION_WAS_TRUNCATED_KEY] = truncated
                        st.session_state[WIZARD_PENDING_AFFIRMATION_TEXT_UPDATE_KEY] = None
                        st.session_state[WIZARD_PENDING_UPDATE_WAS_TRUNCATED_KEY] = False
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Error during wizard affirmation expansion: {e}", exc_info=True)
                        st.error(f"Failed to expand affirmations: {e}")
                        st.session_state[WIZARD_AFFIRM_ORIGINAL_TEXT_KEY] = None
                        st.session_state[WIZARD_EXPANSION_RESULT_TEXT_KEY] = None
                        st.session_state[WIZARD_EXPANSION_WAS_TRUNCATED_KEY] = False
            else:
                st.error("Expansion function is not available.")

    with undo_col:
        undo_disabled = st.session_state.get(WIZARD_AFFIRM_ORIGINAL_TEXT_KEY) is None
        if st.button("↩️ Undo Expansion", key="wizard_undo_expansion", disabled=undo_disabled, use_container_width=True, help="Revert to the text before the last expansion."):
            original_text = st.session_state.get(WIZARD_AFFIRM_ORIGINAL_TEXT_KEY)
            if original_text is not None:
                st.session_state[WIZARD_PENDING_AFFIRMATION_TEXT_UPDATE_KEY] = original_text
                st.session_state[WIZARD_PENDING_UPDATE_WAS_TRUNCATED_KEY] = False
                st.session_state[WIZARD_AFFIRM_ORIGINAL_TEXT_KEY] = None
                st.session_state[WIZARD_EXPANSION_RESULT_TEXT_KEY] = None
                st.session_state[WIZARD_EXPANSION_WAS_TRUNCATED_KEY] = False
                st.rerun()

    if st.session_state.get(WIZARD_EXPANSION_RESULT_TEXT_KEY) is not None:
        st.markdown("**Suggested Expansions:**")
        if st.session_state[WIZARD_EXPANSION_WAS_TRUNCATED_KEY]:
            st.warning(
                f"⚠️ Expanded text was automatically shortened to fit the {MAX_AFFIRMATION_CHARS} character limit.",
                icon="✂️",
            )
        st.text_area(
            "Expanded Affirmations Result (Review Only):",
            value=st.session_state[WIZARD_EXPANSION_RESULT_TEXT_KEY],
            height=150,
            key="wizard_expansion_result_display_area",
            label_visibility="collapsed",
            help="This is a preview of the expanded affirmations. Click 'Use Expanded Text' to apply.",
            disabled=True,
        )
        if st.button("✅ Use Expanded Text", key="wizard_use_expanded_text", use_container_width=True, type="primary"):
            st.session_state[WIZARD_PENDING_AFFIRMATION_TEXT_UPDATE_KEY] = st.session_state[WIZARD_EXPANSION_RESULT_TEXT_KEY]
            st.session_state[WIZARD_PENDING_UPDATE_WAS_TRUNCATED_KEY] = st.session_state[WIZARD_EXPANSION_WAS_TRUNCATED_KEY]
            st.session_state[WIZARD_AFFIRM_ORIGINAL_TEXT_KEY] = None
            st.session_state[WIZARD_EXPANSION_RESULT_TEXT_KEY] = None
            st.session_state[WIZARD_EXPANSION_WAS_TRUNCATED_KEY] = False
            st.rerun()

    st.divider()
    st.markdown("**Optional Settings:**")
    st.checkbox(
        f"Apply Speed Change ({QUICK_SUBLIMINAL_PRESET_SPEED}x Speed)",
        key=AFFIRM_APPLY_SPEED_KEY,
        help=f"Check this to speed up the affirmation audio to approximately {QUICK_SUBLIMINAL_PRESET_SPEED}x.",
    )
    logger.debug(f"Step 1: Apply speed change checkbox state: {st.session_state.get(AFFIRM_APPLY_SPEED_KEY, DEFAULT_APPLY_SPEED)}")
    st.divider()

    # For navigation, use the up-to-date text from AFFIRMATION_TEXT_KEY
    text_is_present = bool(st.session_state.get(AFFIRMATION_TEXT_KEY, "") and st.session_state.get(AFFIRMATION_TEXT_KEY, "").strip())

    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])
    with col_nav_1:
        if st.button("🏠 Back to Home", key="wizard_step1_home", use_container_width=True):
            if hasattr(wizard, "_reset_wizard_state") and callable(wizard._reset_wizard_state):
                wizard._reset_wizard_state()
            else:
                # Basic reset if wizard method not available
                keys_to_reset = [
                    AFFIRMATION_TEXT_KEY,
                    WIZARD_AFFIRM_ORIGINAL_TEXT_KEY,
                    WIZARD_EXPANSION_RESULT_TEXT_KEY,
                    WIZARD_EXPANSION_WAS_TRUNCATED_KEY,
                    WIZARD_PENDING_AFFIRMATION_TEXT_UPDATE_KEY,
                    WIZARD_PENDING_UPDATE_WAS_TRUNCATED_KEY,
                    AFFIRM_APPLY_SPEED_KEY,
                ]
                for key in keys_to_reset:
                    if DEFAULT_AFFIRMATION_TEXT and key == AFFIRMATION_TEXT_KEY:
                        st.session_state[key] = DEFAULT_AFFIRMATION_TEXT
                    elif DEFAULT_APPLY_SPEED and key == AFFIRM_APPLY_SPEED_KEY:
                        st.session_state[key] = DEFAULT_APPLY_SPEED
                    elif isinstance(st.session_state.get(key), bool):
                        st.session_state[key] = False
                    else:
                        st.session_state[key] = None
                st.session_state[AFFIRMATION_TEXT_KEY] = DEFAULT_AFFIRMATION_TEXT  # ensure default text
            st.rerun()
    with col_nav_2:
        st.button("⬅️ Back", key="wizard_step1_back", disabled=True, use_container_width=True)
    with col_nav_3:
        next_disabled = not text_is_present
        next_tooltip = "Proceed to Step 2 (Background)." if not next_disabled else "Please enter affirmation text first."
        if st.button(
            "Next: Background ➡️",
            key="wizard_step1_next",
            type="primary",
            disabled=next_disabled,
            help=next_tooltip,
            use_container_width=True,
        ):
            if not next_disabled:
                if hasattr(wizard, "_go_to_step") and callable(wizard._go_to_step):
                    wizard._go_to_step(2)
                    st.rerun()
                else:
                    st.error("Navigation function wizard._go_to_step() not found.")
            else:
                st.warning("Please enter some affirmation text before proceeding.")
