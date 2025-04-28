# main.py
# ==========================================
# MindMorph - Pro Subliminal Audio Editor
# Main Application Entry Point (Refactored)
# ==========================================

import logging
import os  # Keep for path checks if any remain

import streamlit as st
from PIL import Image

# --- Keep imports needed for initial setup ---
from config import FAVICON_PATH
from utils import setup_logging

# --- Early Setup: Logging and Page Config ---
setup_logging()
logger = logging.getLogger(__name__)  # Get logger for this main module

# Configure Streamlit page
try:
    page_icon = Image.open(FAVICON_PATH)
except FileNotFoundError:
    logger.warning(f"Favicon not found at {FAVICON_PATH}. Using default.")
    page_icon = "🧠"  # Default emoji icon
except Exception as e:
    logger.error(f"Error loading favicon: {e}")
    page_icon = "🧠"

st.set_page_config(layout="wide", page_title="MindMorph - Pro Subliminal Editor", page_icon=page_icon)


# --- Main Application Logic ---


def main():
    """Main function to run the Streamlit application."""
    logger.info("=====================================================")
    logger.info("MindMorph Application starting/rerunning.")
    logger.info("=====================================================")

    # --- Import Core Components and Handlers inside main() ---
    from app_state import AppState
    from project_handler import ProjectHandler  # Import new handler
    from tts_generator import TTSGenerator
    from ui_manager import UIManager
    from welcome_handler import display_welcome_message  # Import new handler

    # --- Initialize Core Components ---
    app_state = AppState()
    tts_generator = TTSGenerator()
    ui_manager = UIManager(app_state, tts_generator)
    project_handler = ProjectHandler(app_state, tts_generator)  # Instantiate handler

    # --- Handle Project Loading ---
    project_handler.load_project()  # Call method from handler

    # --- Initial Welcome Message ---
    # display_welcome_message returns True if message was shown, False otherwise
    welcome_active = display_welcome_message()

    # --- Main UI Rendering (only if welcome message dismissed) ---
    if not welcome_active:
        st.title("🧠 MindMorph - Subliminal Audio Editor")  # Show title only after welcome

        # --- Mode Selector ---
        if "app_mode" not in st.session_state:
            st.session_state.app_mode = "Easy"
            logger.info(f"App mode initialized to: {st.session_state.app_mode}")

        mode_options = ["Easy", "Advanced"]
        try:
            current_mode_index = mode_options.index(st.session_state.app_mode)
        except ValueError:
            logger.warning(f"Invalid mode '{st.session_state.app_mode}'. Defaulting to Easy.")
            st.session_state.app_mode = "Easy"
            current_mode_index = 0

        selected_mode = st.radio(
            "Select Editor Mode:",
            options=mode_options,
            index=current_mode_index,
            key="mode_selector_radio",
            horizontal=True,
            help="Easy mode hides complex features, Advanced shows all options.",
        )
        if selected_mode != st.session_state.app_mode:
            logger.info(f"App mode changed from '{st.session_state.app_mode}' to '{selected_mode}'")
            st.session_state.app_mode = selected_mode
            # Clear potentially mode-dependent cached data
            if "export_buffer" in st.session_state:
                del st.session_state.export_buffer
            if "preview_audio_data" in st.session_state:
                del st.session_state.preview_audio_data
            st.rerun()

        st.markdown("---")

        # --- Render UI Sections using UIManager ---
        ui_manager.render_ui()

    else:
        logger.debug("Welcome message active, skipping main UI render.")

    logger.info("--- End of main function render cycle ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("A critical error occurred in the main execution block.")
        st.error("An unexpected error occurred. Please check the application logs or try reloading the page.")
        # st.exception(e) # Optionally show traceback in UI for debugging
