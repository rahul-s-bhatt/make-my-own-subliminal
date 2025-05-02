# main.py
# ==========================================
# MindMorph - Pro Subliminal Audio Editor
# Main Application Entry Point (Router)
# ==========================================

import atexit  # Added for cleanup registration
import logging
import os  # Potentially needed for cleanup tasks
import signal  # Added for signal handling
import sys  # Added for sys.exit

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# --- Keep imports needed for initial setup ---
from config import FAVICON_PATH, GA_MEASUREMENT_ID, PATERON_URL
from utils import setup_logging

# --- Early Setup: Logging and Page Config ---
# Setup logging first so subsequent messages are captured
setup_logging()
logger = logging.getLogger(__name__)  # Get logger after setup
logger.info("-----------------------------------------------------")
logger.info("Application starting up...")
logger.info("-----------------------------------------------------")

selected_theme_name = st.session_state.get("selected_theme", "System")
try:
    page_icon = Image.open(FAVICON_PATH)
except Exception:
    page_icon = "🧠"  # Fallback

st.set_page_config(
    layout="wide",
    page_title="MindMorph - Subliminal Editor",
    page_icon=page_icon,
)
# setup_logging() # Moved logging setup earlier
logger.info(f"Theme preference set to: {selected_theme_name}")


# --- ADD GOOGLE ANALYTICS TAG ---
if GA_MEASUREMENT_ID and GA_MEASUREMENT_ID != "YOUR_GA_MEASUREMENT_ID_HERE":
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
    logger.info(f"Injected Google Analytics tag.")
else:
    logger.warning("Google Analytics Measurement ID is not set.")
# --- END GOOGLE ANALYTICS TAG ---


# --- Graceful Shutdown Setup ---
def cleanup_resources():
    """Function to clean up resources on shutdown."""
    logger.info("-----------------------------------------------------")
    logger.info("Application shutting down. Performing cleanup...")
    logger.info("-----------------------------------------------------")
    # Add any specific cleanup tasks here, e.g.:
    # - Closing database connections
    # - Deleting specific temporary files if not handled automatically
    # - Saving any critical final state (though complex with Streamlit)
    # Example: Clean up temporary files if you manage them manually
    # temp_dir = st.session_state.get("temp_directory")
    # if temp_dir and os.path.exists(temp_dir):
    #     try:
    #         import shutil
    #         shutil.rmtree(temp_dir)
    #         logger.info(f"Cleaned up temporary directory: {temp_dir}")
    #     except Exception as e:
    #         logger.error(f"Error cleaning up temp directory {temp_dir}: {e}")
    print("MindMorph cleanup finished.", flush=True)  # Also print to stdout for cloud logs


def handle_shutdown_signal(signum, frame):
    """Signal handler for SIGTERM and SIGINT."""
    logger.warning(f"Received signal {signum}. Initiating graceful shutdown.")
    cleanup_resources()
    sys.exit(0)  # Exit gracefully


# Register the cleanup function to run on normal exit or unhandled exceptions
atexit.register(cleanup_resources)

# Register signal handlers for graceful shutdown requests
try:
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    logger.info("Registered signal handlers for SIGTERM and SIGINT.")
except ValueError:
    logger.warning("Could not set signal handlers (possibly running in a non-main thread).")
except AttributeError:
    logger.warning("Signal handling not available on this platform (e.g., Windows without WSL).")
# --- End Graceful Shutdown Setup ---


# --- Helper Function to Reset Advanced State ---
def reset_advanced_editor_state():
    """Clears session state keys specific to the advanced editor."""
    logger.info("Resetting Advanced Editor state.")
    if "advanced_processing_active" in st.session_state:
        st.session_state.advanced_processing_active = False

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
        "wizard_step",
        "wizard_affirmation_text",
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
        "wizard_original_affirmation_text",
        "wizard_affirm_text_pending_update",
        "wizard_affirm_truncated_pending",
        # Processing flag
        "advanced_processing_active",
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            try:
                del st.session_state[key]
            except Exception as e:
                logger.warning(f"Could not delete session state key '{key}': {e}")
    # Clear cache as well, might help ensure clean state
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("Advanced Editor state reset complete and caches cleared.")


# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit application."""
    logger.info("MindMorph Application starting/rerunning main function.")

    # --- Define Workflow Selection ---
    if "selected_workflow" not in st.session_state:
        st.session_state.selected_workflow = None

    # --- Show Selection Screen if no workflow chosen ---
    if st.session_state.selected_workflow is None:
        st.title("🧠 Welcome to MindMorph!")
        st.subheader("Choose how you want to create your subliminal audio:")
        st.markdown("---")

        col1, col2 = st.columns(2, gap="large")

        with col1:  # Quick Create Wizard Button
            st.markdown("### ✨ Quick Create Wizard")
            st.markdown(
                """
                Get started quickly with a simple, step-by-step process.
                - Use your own affirmations (text or audio file).
                - Add optional background sounds (noise or music).
                - Include optional frequency tones (binaural, etc.).
                - Uses new high-quality offline TTS (Piper).
                """
            )
            st.markdown("")
            if st.button("Start Wizard", key="start_wizard_button", use_container_width=True, type="primary"):
                st.session_state.selected_workflow = "wizard"
                st.rerun()

        with col2:  # Advanced Editor Button
            st.markdown("### 🎚️ Advanced Editor")
            st.markdown(
                """
                Unlock full control over your subliminal creation.
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
                    st.session_state.app_mode = "Easy"  # Default mode
                st.rerun()

        st.markdown("---")

        # --- ADD PATREON SUPPORT BUTTON ---
        st.markdown("#### ❤️ Support MindMorph")
        st.markdown("If you find MindMorph useful, please consider supporting its development on Patreon.")
        if PATERON_URL and PATERON_URL != "YOUR_PATERON_URL_HERE":
            col_support_1, col_support_2, col_support_3 = st.columns([1, 1.5, 1])
            with col_support_2:
                st.link_button("💖 Join Patreon", url=PATERON_URL, help="Support MindMorph development.", use_container_width=True, type="secondary")
        else:
            logger.warning("Patreon URL not configured.")
        st.markdown("---")
        # --- END PATREON SUPPORT BUTTON ---

    # --- Run Selected Workflow ---
    elif st.session_state.selected_workflow == "wizard":
        logger.info("Running Quick Create Wizard workflow.")
        try:
            # Import QuickWizard here to avoid loading if not needed
            from wizard_steps.quick_wizard import QuickWizard

            # Instantiate QuickWizard - it now handles its own TTS setup
            wizard = QuickWizard()
            wizard.render_wizard()
        except ImportError as e_import:
            logger.exception("Failed to import Quick Wizard components.")
            st.error(f"Error loading Quick Wizard module: {e_import}. Check file structure.", icon="🚨")
            if st.button("Return to Home"):
                reset_advanced_editor_state()
                st.rerun()
        except Exception as e:
            logger.exception("Failed to initialize or render Quick Wizard.")
            st.error(f"Failed to start Quick Wizard: {e}. Check logs and TTS config.", icon="🔥")
            if st.button("Return to Home"):
                reset_advanced_editor_state()
                st.rerun()

    elif st.session_state.selected_workflow == "advanced":
        logger.info("Running Advanced Editor workflow.")
        try:
            # Import necessary components for advanced editor
            from app_state import AppState
            from project_handler import ProjectHandler
            from tts.piper_tts import PiperTTSGenerator  # Import specific TTS
            from ui_manager import UIManager

            # Initialize Core Components (using session state to persist)
            if "app_state" not in st.session_state:
                st.session_state.app_state = AppState()
            # --- Instantiate PiperTTSGenerator ---
            # Use cache_resource within PiperTTSGenerator's _load_piper_voice
            if "tts_generator" not in st.session_state:
                logger.info("Initializing PiperTTSGenerator for Advanced Editor.")
                try:
                    # This might fail if config paths are wrong or models can't load
                    st.session_state.tts_generator = PiperTTSGenerator()
                    logger.info("PiperTTSGenerator initialized for Advanced Editor.")
                except Exception as e_tts:
                    logger.exception("Failed to initialize PiperTTSGenerator.")
                    st.error(f"TTS Engine Error: {e_tts}. Check model paths in config.", icon="🗣️")
                    # Allow app to continue without TTS if possible, or provide way back
                    st.session_state.tts_generator = None  # Indicate TTS is unavailable
                    # Optionally, force back to home:
                    # reset_advanced_editor_state(); st.rerun()

            if "project_handler" not in st.session_state:
                st.session_state.project_handler = ProjectHandler(st.session_state.app_state)
            if "ui_manager" not in st.session_state:
                # Pass the potentially None TTS generator
                st.session_state.ui_manager = UIManager(st.session_state.app_state, st.session_state.get("tts_generator"))

            # --- Render Top Bar for Advanced Editor ---
            st.title("🧠 MindMorph - Advanced Editor")
            header_cols = st.columns([4, 1])
            with header_cols[0]:  # Mode Selector
                if "app_mode" not in st.session_state:
                    st.session_state.app_mode = "Easy"
                mode_options = ["Easy", "Advanced"]
                try:
                    current_mode_index = mode_options.index(st.session_state.app_mode)
                except ValueError:
                    current_mode_index = 0  # Default to Easy if state is invalid
                    st.session_state.app_mode = "Easy"
                selected_mode = st.radio("Editor Mode:", options=mode_options, index=current_mode_index, key="mode_selector_radio", horizontal=True)
                st.caption("Easy mode simplifies the interface; Advanced mode shows all track controls.")
                if selected_mode != st.session_state.app_mode:
                    logger.info(f"Advanced editor mode changed to '{selected_mode}'")
                    st.session_state.app_mode = selected_mode
                    # No need to clear export/preview buffers on mode change, just rerun
                    st.rerun()
            with header_cols[1]:  # Back to Home Button
                if st.button("🏠 Back to Home", key="advanced_back_home", help="Exit Advanced Editor.", use_container_width=True):
                    reset_advanced_editor_state()
                    st.rerun()
            st.markdown("---")

            # Render Main UI Sections using UIManager
            st.session_state.ui_manager.render_ui(mode=st.session_state.app_mode)

        except ImportError as e_import:
            logger.exception("Failed to import Advanced Editor components.")
            st.error(f"Error loading Advanced Editor module: {e_import}. Check file structure.", icon="🚨")
            if st.button("Return to Home"):
                reset_advanced_editor_state()
                st.rerun()
        except Exception as e:
            logger.exception("Failed to initialize or render Advanced Editor.")
            st.error(f"Failed to start Advanced Editor: {e}. Check logs.", icon="🔥")
            if st.button("Return to Home"):
                reset_advanced_editor_state()
                st.rerun()

    else:  # Invalid state
        logger.warning(f"Invalid selected_workflow state: {st.session_state.selected_workflow}. Resetting.")
        st.session_state.selected_workflow = None
        st.rerun()

    logger.debug("--- End of main function render cycle ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch errors that might happen during initialization outside main()
        # or very early in main() before specific error handling is set up.
        logger.exception("A critical error occurred during main execution.")
        # Display error prominently if possible (might not work if Streamlit itself fails)
        try:
            st.error(f"A critical error stopped the application: {e}", icon="💥")
            st.warning("Please check the application logs for more details.")
            # Optionally add a button to reset everything
            if st.button("Attempt Reset Application State"):
                reset_advanced_editor_state()  # Try resetting state
                st.rerun()
        except Exception as display_error:
            # Fallback if st.error fails
            print(f"CRITICAL ERROR: {e}", file=sys.stderr)
            print(f"Could not display error in Streamlit: {display_error}", file=sys.stderr)
