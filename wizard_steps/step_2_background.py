# wizard_steps/step_2_background.py
# ==========================================
# UI Rendering for Wizard Step 2: Background Selection
# Uses constants from quick_wizard_config.py
# ==========================================

import logging

import streamlit as st

# Import constants from the central config file
from .quick_wizard_config import NOISE_TYPES  # Import the list of noise types
from .quick_wizard_config import (
    BG_CHOICE_KEY,
    BG_CHOICE_LABEL_KEY,
    BG_NOISE_TYPE_KEY,
    BG_UPLOADED_FILE_KEY,
    DEFAULT_BG_CHOICE,
    DEFAULT_BG_CHOICE_LABEL,
    DEFAULT_NOISE_TYPE,
)

# Import audio loading utilities if needed (may not be needed here anymore)
try:
    pass

    AUDIO_IO_AVAILABLE = True
except ImportError:
    AUDIO_IO_AVAILABLE = False
    # logging.warning("audio_utils.audio_io not found.") # Less relevant now

logger = logging.getLogger(__name__)

# --- Key definitions moved to config ---

# --- Main Rendering Function ---


def render_step_2(wizard):
    """
    Renders the UI for Step 2: Background Selection.
    Stores the choice and settings, does NOT generate/load audio.

    Args:
        wizard: An instance of the QuickWizard class.
    """
    st.subheader("Step 2: Choose Background Sound (Optional)")
    st.write("Select a background sound to mix with your affirmations.")

    # --- Initialization handled by initialize_wizard_state ---
    # if BG_CHOICE_KEY not in st.session_state: st.session_state[BG_CHOICE_KEY] = DEFAULT_BG_CHOICE ... etc

    # --- Background Selection ---
    choice_options = ["None", "Upload Music/Sound", "Generate Noise"]  # UI labels
    try:
        current_choice_label = st.session_state.get(
            BG_CHOICE_LABEL_KEY, DEFAULT_BG_CHOICE_LABEL
        )
        current_choice_index = choice_options.index(current_choice_label)
    except ValueError:
        current_choice_index = 0  # Default to 'None'
        st.session_state[BG_CHOICE_LABEL_KEY] = DEFAULT_BG_CHOICE_LABEL
        st.session_state[BG_CHOICE_KEY] = DEFAULT_BG_CHOICE

    selected_label = st.radio(
        "Background Sound Source:",
        options=choice_options,
        index=current_choice_index,
        key="wizard_bg_choice_radio",  # Widget key
        horizontal=True,
        help="Choose 'None', upload your own file, or generate noise.",
        on_change=wizard.sync_background_choice,  # Callback uses constant keys internally
        args=(choice_options,),
    )

    # --- Conditional UI based on Choice ---
    current_choice = st.session_state.get(BG_CHOICE_KEY, DEFAULT_BG_CHOICE)

    # 1. Upload Option
    if current_choice == "upload":
        st.markdown("**Upload Audio File:**")
        uploaded_file = st.file_uploader(
            "Choose a WAV, MP3, or FLAC file:",
            type=["wav", "mp3", "flac"],
            key="wizard_bg_uploader",
            help="Upload your background music or soundscape.",
            on_change=wizard.clear_background_upload_state,  # Callback uses constant keys internally
        )

        if uploaded_file is not None:
            if st.session_state.get(BG_UPLOADED_FILE_KEY) != uploaded_file:
                st.session_state[BG_UPLOADED_FILE_KEY] = uploaded_file
                logger.info(f"Stored uploaded file object: {uploaded_file.name}")
                st.session_state[BG_NOISE_TYPE_KEY] = NOISE_TYPES[0]  # Reset noise type
                st.success(f"File '{uploaded_file.name}' ready for processing later.")
        elif st.session_state.get(BG_UPLOADED_FILE_KEY) is not None:
            st.info(
                f"Using previously uploaded file: {st.session_state[BG_UPLOADED_FILE_KEY].name}"
            )

    # 2. Generate Noise Option
    elif current_choice == "noise":
        st.markdown("**Generate Noise:**")
        # Use NOISE_TYPES list from config
        current_noise_type = st.session_state.get(BG_NOISE_TYPE_KEY, DEFAULT_NOISE_TYPE)
        try:
            current_noise_index = NOISE_TYPES.index(current_noise_type)
        except ValueError:
            current_noise_index = 0  # Default to first in list if state is invalid
            st.session_state[BG_NOISE_TYPE_KEY] = NOISE_TYPES[0]

        selected_noise = st.selectbox(
            "Select Noise Type:",
            options=NOISE_TYPES,  # Use list from config
            key="wizard_bg_noise_selector",
            index=current_noise_index,
            help="Choose the type of noise to generate.",
        )
        if selected_noise != current_noise_type:
            st.session_state[BG_NOISE_TYPE_KEY] = selected_noise
            logger.info(f"Background noise type set to: {selected_noise}")
            if st.session_state.get(BG_UPLOADED_FILE_KEY) is not None:
                st.session_state[BG_UPLOADED_FILE_KEY] = None
                logger.debug("Cleared previous file upload due to noise selection.")
            st.success(f"'{selected_noise}' selected for generation later.")
        else:
            st.info(f"'{current_noise_type}' selected.")

    # 3. None Option
    else:  # current_choice == "none" or default
        st.info("No background sound will be added.")
        # Cleanup logic handled by sync_background_choice callback

    st.divider()

    # --- Navigation ---
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])
    with col_nav_1:
        if st.button(
            "🏠 Back to Home", key="wizard_step2_home", use_container_width=True
        ):
            wizard._reset_wizard_state()
    with col_nav_2:
        if st.button(
            "⬅️ Back: Affirmations", key="wizard_step2_back", use_container_width=True
        ):
            wizard._go_to_step(1)
    with col_nav_3:
        if st.button(
            "Next: Frequency ➡️",
            key="wizard_step2_next",
            type="primary",
            use_container_width=True,
        ):
            wizard._go_to_step(3)
