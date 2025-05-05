# main.py
# ==========================================
# MindMorph - Pro Subliminal Audio Editor
# Main Application Entry Point (Router)
# STEP 3 OPTIMIZED: Pass tts_generator to UIManager
# ==========================================

import atexit
import logging
import signal
import sys

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from config import FAVICON_PATH, GA_MEASUREMENT_ID, PATERON_URL
from utils import setup_logging

# --- Early Setup ---
setup_logging()
logger = logging.getLogger(__name__)
logger.info("-----------------------------------------------------")
logger.info("Application starting up...")
logger.info("-----------------------------------------------------")

selected_theme_name = st.session_state.get("selected_theme", "System")
try:
    page_icon = Image.open(FAVICON_PATH)
except Exception:
    page_icon = "🧠"
st.set_page_config(
    layout="wide", page_title="MindMorph - Subliminal Editor", page_icon=page_icon
)
logger.info(f"Theme preference set to: {selected_theme_name}")

# --- Google Analytics ---
if GA_MEASUREMENT_ID and GA_MEASUREMENT_ID != "YOUR_GA_MEASUREMENT_ID_HERE":
    google_analytics_code = f"""<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script><script>window.dataLayer = window.dataLayer || []; function gtag(){{dataLayer.push(arguments);}} gtag('js', new Date()); gtag('config', '{GA_MEASUREMENT_ID}');</script>"""
    components.html(google_analytics_code, height=0)
    # st.markdown(google_analytics_code, unsafe_allow_html=True) # Duplicate? components.html should suffice
    logger.info(f"Injected Google Analytics tag.")
else:
    logger.warning("Google Analytics Measurement ID is not set.")


# --- Graceful Shutdown ---
def cleanup_resources():
    logger.info("-----------------------------------------------------")
    logger.info("Application shutting down. Performing cleanup...")
    logger.info("-----------------------------------------------------")
    # Add specific cleanup tasks here if needed
    print("MindMorph cleanup finished.", flush=True)


def handle_shutdown_signal(signum, frame):
    logger.warning(f"Received signal {signum}. Initiating graceful shutdown.")
    cleanup_resources()
    sys.exit(0)


atexit.register(cleanup_resources)
try:
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    logger.info("Registered signal handlers for SIGTERM and SIGINT.")
except (ValueError, AttributeError):
    logger.warning("Could not set signal handlers.")


# --- Reset State Helper ---
def reset_advanced_editor_state():
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
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("Advanced Editor state reset complete and caches cleared.")


# --- Main Application Logic ---
def main():
    logger.info("MindMorph Application starting/rerunning main function.")
    if "selected_workflow" not in st.session_state:
        st.session_state.selected_workflow = None

    # --- Selection Screen ---
    if st.session_state.selected_workflow is None:
        st.title("🧠 Welcome to MindMorph!")
        st.subheader("Choose how you want to create your subliminal audio:")
        st.markdown("---")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("### ✨ Quick Create Wizard")
            st.markdown(
                "- Simple, step-by-step process.\n- Use text or audio affirmations.\n- Optional background sounds/frequencies.\n- High-quality offline TTS (Piper)."
            )
            if st.button(
                "Start Wizard",
                key="start_wizard_button",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.selected_workflow = "wizard"
                st.rerun()
        with col2:
            st.markdown("### 🎚️ Advanced Editor")
            st.markdown(
                "- Full control with multiple tracks.\n- Fine-tune volume, speed, pitch, etc.\n- Advanced effects (ultrasonic shift).\n- Save/Load projects."
            )
            if st.button(
                "Open Advanced Editor",
                key="start_advanced_button",
                use_container_width=True,
            ):
                st.session_state.selected_workflow = "advanced"
                if "app_mode" not in st.session_state:
                    st.session_state.app_mode = "Easy"
                st.rerun()
        st.markdown("---")
        # Patreon Button
        st.markdown(
            '<div style="text-align: center;"><h4>❤️ Support MindMorph</h4><p>Consider supporting development on Patreon.</p></div>',
            unsafe_allow_html=True,
        )
        if PATERON_URL and PATERON_URL != "YOUR_PATERON_URL_HERE":
            st.markdown(
                f'<div style="text-align: center; margin-top: 10px; margin-bottom: 10px;"><a href="{PATERON_URL}" target="_blank"><button style="padding: 10px 20px; background-color: #f0f2f6; color: #31333F; border: 1px solid #ced4da; border-radius: 0.25rem; cursor: pointer; font-weight: bold;">💖 Join Patreon</button></a></div>',
                unsafe_allow_html=True,
            )
        else:
            logger.warning("Patreon URL not configured.")
        st.markdown("---")

    # --- Run Quick Wizard ---
    elif st.session_state.selected_workflow == "wizard":
        logger.info("Running Quick Create Wizard workflow.")
        try:
            from wizard_steps.quick_wizard import QuickWizard

            wizard = QuickWizard()  # Handles its own TTS setup
            wizard.render_wizard()
        except ImportError as e_import:
            logger.exception("Failed to import Quick Wizard components.")
            st.error(f"Error loading Quick Wizard module: {e_import}.", icon="🚨")
            if st.button("Return to Home"):
                reset_advanced_editor_state()
                st.rerun()
        except Exception as e:
            logger.exception("Failed to initialize or render Quick Wizard.")
            st.error(f"Failed to start Quick Wizard: {e}.", icon="🔥")
            if st.button("Return to Home"):
                reset_advanced_editor_state()
                st.rerun()

    # --- Run Advanced Editor ---
    elif st.session_state.selected_workflow == "advanced":
        logger.info("Running Advanced Editor workflow.")
        try:
            # Import necessary components
            from app_state import AppState
            from project_handler import ProjectHandler
            from tts.piper_tts import PiperTTSGenerator
            from ui_manager import UIManager  # Import UIManager

            # Initialize Core Components
            if "app_state" not in st.session_state:
                st.session_state.app_state = AppState()
            if "tts_generator" not in st.session_state:
                logger.info("Initializing PiperTTSGenerator for Advanced Editor.")
                try:
                    st.session_state.tts_generator = PiperTTSGenerator()
                    logger.info("PiperTTSGenerator initialized for Advanced Editor.")
                except Exception as e_tts:
                    logger.exception("Failed to initialize PiperTTSGenerator.")
                    st.error(f"TTS Engine Error: {e_tts}. Check model paths.", icon="🗣️")
                    st.session_state.tts_generator = None  # Indicate TTS unavailable

            if "project_handler" not in st.session_state:
                st.session_state.project_handler = ProjectHandler(
                    st.session_state.app_state
                )

            # --- MODIFIED: Pass tts_generator to UIManager ---
            if "ui_manager" not in st.session_state:
                st.session_state.ui_manager = UIManager(
                    app_state=st.session_state.app_state,
                    tts_generator=st.session_state.get(
                        "tts_generator"
                    ),  # Pass the generator instance (can be None)
                )
            # --- End Modification ---

            # Render Top Bar
            st.title("🧠 MindMorph - Advanced Editor")
            header_cols = st.columns([4, 1])
            with header_cols[0]:
                if "app_mode" not in st.session_state:
                    st.session_state.app_mode = "Easy"
                mode_options = ["Easy", "Advanced"]
                try:
                    current_mode_index = mode_options.index(st.session_state.app_mode)
                except ValueError:
                    current_mode_index = 0
                    st.session_state.app_mode = "Easy"
                selected_mode = st.radio(
                    "Editor Mode:",
                    options=mode_options,
                    index=current_mode_index,
                    key="mode_selector_radio",
                    horizontal=True,
                )
                st.caption("Easy mode simplifies; Advanced shows all controls.")
                if selected_mode != st.session_state.app_mode:
                    logger.info(f"Advanced editor mode changed to '{selected_mode}'")
                    st.session_state.app_mode = selected_mode
                    st.rerun()
            with header_cols[1]:
                if st.button(
                    "🏠 Back to Home",
                    key="advanced_back_home",
                    help="Exit Advanced Editor.",
                    use_container_width=True,
                ):
                    reset_advanced_editor_state()
                    st.rerun()
            st.markdown("---")

            # Render Main UI using UIManager
            st.session_state.ui_manager.render_ui(mode=st.session_state.app_mode)

        except ImportError as e_import:
            logger.exception("Failed to import Advanced Editor components.")
            st.error(f"Error loading Advanced Editor module: {e_import}.", icon="🚨")
            if st.button("Return to Home"):
                reset_advanced_editor_state()
                st.rerun()
        except Exception as e:
            logger.exception("Failed to initialize or render Advanced Editor.")
            st.error(f"Failed to start Advanced Editor: {e}.", icon="🔥")
            if st.button("Return to Home"):
                reset_advanced_editor_state()
                st.rerun()

    # --- Invalid State ---
    else:
        logger.warning(
            f"Invalid selected_workflow state: {st.session_state.selected_workflow}. Resetting."
        )
        st.session_state.selected_workflow = None
        st.rerun()

    logger.debug("--- End of main function render cycle ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("A critical error occurred during main execution.")
        try:
            st.error(f"A critical error stopped the application: {e}", icon="💥")
            st.warning("Please check the application logs for more details.")
            if st.button("Attempt Reset Application State"):
                reset_advanced_editor_state()
                st.rerun()
        except Exception as display_error:
            print(f"CRITICAL ERROR: {e}", file=sys.stderr)
            print(
                f"Could not display error in Streamlit: {display_error}",
                file=sys.stderr,
            )
