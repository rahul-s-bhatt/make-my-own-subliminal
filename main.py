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

# --- Set Page Config FIRST ---
st.set_page_config(layout="wide", page_title="MindMorph - Subliminal Editor", page_icon=page_icon)

# --- ADD GOOGLE ANALYTICS TAG ---
# Replace 'YOUR_GA_MEASUREMENT_ID' with your actual Google Analytics Measurement ID
GA_MEASUREMENT_ID = "G-B5LWHH5H7N"

# Google Analytics gtag.js code snippet
# This injects the necessary JavaScript into the app's HTML head on each run.
# It initializes Google Analytics tracking.
google_analytics_code = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());

      gtag('config', '{GA_MEASUREMENT_ID}');
    </script>
"""
# Inject the code into the app's <head>
# Using st.markdown with unsafe_allow_html=True is a common way,
# ensure it's placed where it runs on every interaction, like here.
# Note: st.html() injects into the body, but early injection works for GA.
# For guaranteed head injection, Streamlit Components or custom templates might be needed,
# but this method is standard for direct script modification.
components.html(google_analytics_code, height=0)  # Use components.html to inject raw HTML/JS
logger.info(f"Injected Google Analytics tag for ID: {GA_MEASUREMENT_ID}")
# --- END GOOGLE ANALYTICS TAG ---


# --- Helper Function to Reset Advanced State ---
def reset_advanced_editor_state():
    """Clears session state keys specific to the advanced editor."""
    logger.info("Resetting Advanced Editor state.")
    keys_to_delete = [
        "app_state",
        "tts_generator",  # Reset TTS generator instance if created per session
        "project_handler",
        "ui_manager",
        "app_mode",
        "selected_track_id",  # Example key, add others managed by AppState/UIManager
        "tracks",  # Example key from AppState
        "project_loaded",  # Example key
        "export_buffer",
        "preview_audio_data",
        # Add any other session state keys specific to the advanced editor
        "selected_workflow",  # Also reset workflow selection
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
            logger.debug(f"Deleted advanced editor session state key: {key}")

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
        st.markdown("---")  # Add a separator

        col1, col2 = st.columns(2, gap="large")  # Add gap between columns

        with col1:
            st.markdown("### ✨ Quick Create Wizard")
            # --- Adjusted Description ---
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
            st.markdown("")  # Add some space
            if st.button("Start Wizard", key="start_wizard_button", use_container_width=True, type="primary"):
                st.session_state.selected_workflow = "wizard"
                # Initialize wizard state if needed
                from wizard_state import initialize_wizard_state

                initialize_wizard_state()
                st.rerun()

        with col2:
            st.markdown("### 🎚️ Advanced Editor")
            # --- Adjusted Description ---
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
            st.markdown("")  # Add some space
            if st.button("Open Advanced Editor", key="start_advanced_button", use_container_width=True):
                st.session_state.selected_workflow = "advanced"
                if "app_mode" not in st.session_state:
                    st.session_state.app_mode = "Easy"  # Default advanced mode
                st.rerun()

        st.markdown("---")  # Add another separator

    # --- Run Selected Workflow ---
    elif st.session_state.selected_workflow == "wizard":
        logger.info("Running Quick Create Wizard workflow.")
        from quick_wizard import QuickWizard
        from tts_generator import TTSGenerator

        if "tts_generator_wizard" not in st.session_state:
            st.session_state.tts_generator_wizard = TTSGenerator()

        wizard = QuickWizard(st.session_state.tts_generator_wizard)
        wizard.render_wizard()

    elif st.session_state.selected_workflow == "advanced":
        logger.info("Running Advanced Editor workflow.")
        from app_state import AppState
        from project_handler import ProjectHandler
        from tts_generator import TTSGenerator
        from ui_manager import UIManager

        # Initialize Core Components for Advanced Mode using session state
        if "app_state" not in st.session_state:
            st.session_state.app_state = AppState()
        if "tts_generator" not in st.session_state:
            st.session_state.tts_generator = TTSGenerator()
        if "project_handler" not in st.session_state:
            # <<< MODIFIED: Removed tts_generator argument >>>
            st.session_state.project_handler = ProjectHandler(st.session_state.app_state)
        if "ui_manager" not in st.session_state:
            # Pass the reset function to the UI manager if needed, or keep it here
            st.session_state.ui_manager = UIManager(st.session_state.app_state, st.session_state.tts_generator)

        # Handle Project Loading (Project loading is currently removed/disabled)
        # st.session_state.project_handler.load_project() # Keep commented out if loading is disabled

        # --- Render Top Bar for Advanced Editor (Improved Layout) ---
        st.title("🧠 MindMorph - Advanced Editor")

        header_cols = st.columns([4, 3])  # Adjust ratios as needed
        with header_cols[0]:
            # Mode Selector with Explanation
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
                # label_visibility="collapsed", # Keep label for clarity
                help="Choose between a simplified view (Easy) or access to all features (Advanced).",
            )
            st.caption("Easy mode simplifies the interface; Advanced mode shows all track controls.")  # Explanation

            if selected_mode != st.session_state.app_mode:
                logger.info(f"Advanced editor mode changed from '{st.session_state.app_mode}' to '{selected_mode}'")
                st.session_state.app_mode = selected_mode
                # Clear potentially mode-dependent cached data
                if "export_buffer" in st.session_state:
                    del st.session_state.export_buffer
                if "preview_audio_data" in st.session_state:
                    del st.session_state.preview_audio_data
                st.rerun()

        with header_cols[1]:
            # "Back to Home" Button - aligned better
            st.markdown('<div style="height: 2.5em;"></div>', unsafe_allow_html=True)  # Add vertical space to align with radio button
            if st.button("🏠 Back to Home", key="advanced_back_home", help="Exit Advanced Editor and return to workflow selection.", use_container_width=True):
                reset_advanced_editor_state()  # Call the reset function
                st.rerun()  # Rerun to show the home screen

        st.markdown("---")

        # --- Render Main UI Sections using UIManager ---
        # Pass the current mode to the UI Manager so it can adjust the display
        st.session_state.ui_manager.render_ui(mode=st.session_state.app_mode)

    else:
        # Should not happen, but reset if state is invalid
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
