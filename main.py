# main.py
# ==========================================
# MindMorph - Pro Subliminal Audio Editor
# Main Application Entry Point (Router)
# ==========================================

import logging
import os

import streamlit as st
import streamlit.components.v1 as components  # Import components
from PIL import Image

# --- Keep imports needed for initial setup ---
# <<< MODIFIED: Removed unused theme imports, ensured PATERON_URL is imported >>>
from config import FAVICON_PATH, GA_MEASUREMENT_ID, GOOGLE_FORM_URL, PATERON_URL
from utils import setup_logging

# --- Early Setup: Logging and Page Config ---
logger = logging.getLogger(__name__)

# --- Determine Theme for Page Config ---
# Theme preference is stored, but applied via OS/default unless CSS is used
selected_theme_name = st.session_state.get("selected_theme", "System")

# --- Configure Streamlit page FIRST ---
try:
    page_icon = Image.open(FAVICON_PATH)
except FileNotFoundError:
    logger.warning(f"Favicon not found at {FAVICON_PATH}. Using default.")
    page_icon = "🧠"
except Exception as e:
    logger.error(f"Error loading favicon: {e}")
    page_icon = "🧠"

st.set_page_config(
    layout="wide",
    page_title="MindMorph - Subliminal Editor",
    page_icon=page_icon,
)

# --- Now Setup Logging ---
setup_logging()
logger.info(f"Theme preference set to: {selected_theme_name}")


# --- ADD GOOGLE ANALYTICS TAG ---
if not GA_MEASUREMENT_ID or GA_MEASUREMENT_ID == "YOUR_GA_MEASUREMENT_ID_HERE":  # Check for placeholder
    logger.warning("Google Analytics Measurement ID is not set in config.py.")
else:
    google_analytics_code = f"""
        <script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){{dataLayer.push(arguments);}}
          gtag('js', new Date());
          gtag('config', '{GA_MEASUREMENT_ID}');
        </script>
    """
    components.html(google_analytics_code, height=0)
    logger.info(f"Injected Google Analytics tag for configured ID.")
# --- END GOOGLE ANALYTICS TAG ---


# --- Helper Function to Reset Advanced State ---
def reset_advanced_editor_state():
    """Clears session state keys specific to the advanced editor."""
    logger.info("Resetting Advanced Editor state.")
    keys_to_delete = [
        "app_state",
        "tts_generator",
        "project_handler",
        "ui_manager",
        "app_mode",
        "selected_track_id",
        "tracks",
        "project_loaded",
        "export_buffer",
        "preview_audio_data",
        "selected_workflow",
        # Expansion/Undo states
        "sidebar_expansion_result",
        "sidebar_expansion_truncated",
        "sidebar_affirm_original_text",
        "sidebar_affirm_text_pending_update",
        "sidebar_affirm_truncated_pending",
        # Wizard states
        "wizard_affirm_text_area",
        "wizard_original_affirmation_text",
        "wizard_affirm_text_pending_update",
        "wizard_affirm_truncated_pending",
        # Add other wizard keys if necessary
        "wizard_step",
        "wizard_affirmation_audio",
        "wizard_affirmation_sr",
        "wizard_affirmation_source",
        "wizard_background_choice",
        "wizard_background_choice_label",
        "wizard_background_audio",
        "wizard_background_sr",
        "wizard_background_volume",
        "wizard_background_noise_type",
        "wizard_frequency_choice",
        "wizard_frequency_audio",
        "wizard_frequency_sr",
        "wizard_frequency_volume",
        "wizard_output_filename",
        "wizard_export_format",
        "wizard_export_buffer",
        "wizard_export_error",
    ]
    # Keep theme selection during reset
    # keys_to_delete.append("selected_theme")

    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
            # logger.debug(f"Deleted session state key: {key}") # Optional debug

    logger.info("Advanced Editor state reset complete.")


# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit application."""
    logger.info("=====================================================")
    logger.info("MindMorph Application starting/rerunning.")
    logger.info("=====================================================")

    # --- Define Workflow Selection ---
    if "selected_workflow" not in st.session_state:
        st.session_state.selected_workflow = None

    # --- Show Selection Screen if no workflow chosen ---
    if st.session_state.selected_workflow is None:
        st.title("🧠 Welcome to MindMorph!")
        st.subheader("Choose how you want to create your subliminal audio:")
        st.markdown("---")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### ✨ Quick Create Wizard")
            st.markdown(
                """
                Get started quickly with a simple, step-by-step process.
                Ideal for beginners projects:
                - Use your own affirmations (text or audio file).
                - Add optional background sounds (noise or music).
                - Include optional frequency tones (binaural, etc.).
                - Fixed settings for speed/volume for easy creation.
                """
            )
            st.markdown("")
            if st.button("Start Wizard", key="start_wizard_button", use_container_width=True, type="primary"):
                st.session_state.selected_workflow = "wizard"
                from wizard_state import initialize_wizard_state

                initialize_wizard_state()
                st.rerun()

        with col2:
            st.markdown("### 🎚️ Advanced Editor")
            st.markdown(
                """
                Unlock full control over your subliminal creation.
                Best for experienced users or complex projects:
                - Manage multiple audio tracks simultaneously.
                - Fine-tune volume, speed, pitch, looping per track.
                - Apply advanced effects (e.g., ultrasonic shift).
                - Save and load your project files for later use.
                """
            )
            st.markdown("")
            if st.button("Open Advanced Editor", key="start_advanced_button", use_container_width=True):
                st.session_state.selected_workflow = "advanced"
                if "app_mode" not in st.session_state:
                    st.session_state.app_mode = "Easy"
                st.rerun()

        st.markdown("---")

        # --- ADD PATREON SUPPORT BUTTON ---
        st.markdown("#### ❤️ Support MindMorph")
        st.markdown("If you find MindMorph useful, please consider supporting its development on Patreon. Your support helps keep the tool free and allows for new features!")

        # Use PATERON_URL from config
        # Check if the URL is the placeholder or empty
        if not PATERON_URL or PATERON_URL == "YOUR_PATERON_URL_HERE":  # Adjust placeholder if needed
            logger.warning("Patreon URL not configured in config.py")
            # Optionally display a message or hide the button
            # st.warning("Patreon link not configured.")
        else:
            # Use columns to center the button or adjust its width
            col_support_1, col_support_2, col_support_3 = st.columns([1, 1.5, 1])  # Adjust ratios as needed
            with col_support_2:
                st.link_button(
                    "💖 Join Patreon",
                    url=PATERON_URL,  # Use the imported constant
                    help="Support MindMorph development (opens in new tab).",
                    use_container_width=True,
                    type="secondary",  # Or "primary" if you want more emphasis
                )
        st.markdown("---")
        # --- END PATREON SUPPORT BUTTON ---

    # --- Run Selected Workflow ---
    elif st.session_state.selected_workflow == "wizard":
        logger.info("Running Quick Create Wizard workflow.")
        from quick_wizard import QuickWizard
        from tts_generator import TTSGenerator

        if "tts_generator_wizard" not in st.session_state:
            st.session_state.tts_generator_wizard = TTSGenerator()

        wizard = QuickWizard(st.session_state.tts_generator_wizard)
        wizard.render_wizard()  # Wizard UI takes over main area

    elif st.session_state.selected_workflow == "advanced":
        logger.info("Running Advanced Editor workflow.")
        from app_state import AppState
        from project_handler import ProjectHandler
        from tts_generator import TTSGenerator
        from ui_manager import UIManager

        # Initialize Core Components
        if "app_state" not in st.session_state:
            st.session_state.app_state = AppState()
        if "tts_generator" not in st.session_state:
            st.session_state.tts_generator = TTSGenerator()
        if "project_handler" not in st.session_state:
            st.session_state.project_handler = ProjectHandler(st.session_state.app_state)
        if "ui_manager" not in st.session_state:
            st.session_state.ui_manager = UIManager(st.session_state.app_state, st.session_state.tts_generator)

        # --- Render Top Bar for Advanced Editor ---
        st.title("🧠 MindMorph - Advanced Editor")

        header_cols = st.columns([4, 1])
        with header_cols[0]:
            # Mode Selector
            if "app_mode" not in st.session_state:
                st.session_state.app_mode = "Easy"
            mode_options = ["Easy", "Advanced"]
            try:
                current_mode_index = mode_options.index(st.session_state.app_mode)
            except ValueError:
                current_mode_index = 0

            selected_mode = st.radio(
                "Editor Mode:",
                options=mode_options,
                index=current_mode_index,
                key="mode_selector_radio",
                horizontal=True,
                help="Choose between a simplified view (Easy) or access to all features (Advanced).",
            )
            st.caption("Easy mode simplifies the interface; Advanced mode shows all track controls.")

            if selected_mode != st.session_state.app_mode:
                logger.info(f"Advanced editor mode changed from '{st.session_state.app_mode}' to '{selected_mode}'")
                st.session_state.app_mode = selected_mode
                if "export_buffer" in st.session_state:
                    del st.session_state.export_buffer
                if "preview_audio_data" in st.session_state:
                    del st.session_state.preview_audio_data
                st.rerun()

        with header_cols[1]:
            # Back to Home Button
            if st.button("🏠 Back to Home", key="advanced_back_home", help="Exit Advanced Editor and return to workflow selection.", use_container_width=True):
                reset_advanced_editor_state()
                st.rerun()

        st.markdown("---")

        # Render Main UI Sections using UIManager
        st.session_state.ui_manager.render_ui(mode=st.session_state.app_mode)

    else:
        logger.warning(f"Invalid selected_workflow state: {st.session_state.selected_workflow}. Resetting.")
        st.session_state.selected_workflow = None
        st.rerun()

    logger.info("--- End of main function render cycle ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("A critical error occurred in the main execution block.")
        st.error("An unexpected error occurred. Please check the application logs or try reloading the page.")
