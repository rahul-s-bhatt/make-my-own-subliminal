# main.py
# ==========================================
# MindMorph - Pro Subliminal Audio Editor
# Main Application Entry Point (Router)
# Refactored: Auto-Create Subliminal UI moved to auto_subliminal.auto_subliminal_ui
# ==========================================

import atexit
import logging
import os
import signal
import sys

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from config import FAVICON_PATH, GA_MEASUREMENT_ID, GLOBAL_SR, PATERON_URL, PIPER_VOICE_CONFIG_PATH, PIPER_VOICE_MODEL_PATH
from utils import setup_logging

# --- MODIFIED: Import the Auto-Create Subliminal UI function from its new location ---
try:
    from auto_subliminal.auto_subliminal_ui import render_auto_subliminal_workflow

    AUTO_SUB_UI_AVAILABLE = True
except ImportError as e:
    AUTO_SUB_UI_AVAILABLE = False
    # Initialize logger early if not already done by setup_logging, for this specific error
    if not logging.getLogger(__name__).hasHandlers():  # Check if logger is already configured
        logging.basicConfig(level=logging.INFO)  # Basic config if not set up
    logging.getLogger(__name__).error(
        f"Failed to import render_auto_subliminal_workflow from auto_subliminal.auto_subliminal_ui: {e}. Auto-Create feature will be unavailable.", exc_info=True
    )

    # Define a fallback if import fails, so Streamlit doesn't crash entirely at import time
    def render_auto_subliminal_workflow(send_ga_event_func=None):
        st.error("Auto-Create Subliminal feature module could not be loaded. Please check the installation and logs (especially import paths).", icon="🚨")
        if st.button("Return to Home"):  # Provide a way out
            if "selected_workflow" in st.session_state:
                st.session_state.selected_workflow = None
            st.rerun()


# --- Early Setup ---
setup_logging()  # Ensure this is called before any logger usage if not done above
logger = logging.getLogger(__name__)  # Now get the configured logger
logger.info("-----------------------------------------------------")
logger.info("Application starting up...")
logger.info("-----------------------------------------------------")

selected_theme_name = st.session_state.get("selected_theme", "System")
try:
    page_icon = Image.open(FAVICON_PATH)
except Exception:
    page_icon = "🧠"
st.set_page_config(layout="wide", page_title="MindMorph - Subliminal Editor", page_icon=page_icon)
logger.info(f"Theme preference set to: {selected_theme_name}")

# --- Google Analytics ---
GA_ENABLED = GA_MEASUREMENT_ID and GA_MEASUREMENT_ID != "YOUR_GA_MEASUREMENT_ID_HERE"
if GA_ENABLED:
    google_analytics_code = f"""<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script><script>window.dataLayer = window.dataLayer || []; function gtag(){{dataLayer.push(arguments);}} gtag('js', new Date()); gtag('config', '{GA_MEASUREMENT_ID}');</script>"""
    components.html(google_analytics_code, height=0)
    st.markdown(google_analytics_code, unsafe_allow_html=True)
else:
    logger.warning("Google Analytics Measurement ID is not set.")


def send_ga_event(event_name: str, event_params: dict):
    if GA_ENABLED:
        event_js = f"""<script>if(typeof gtag === 'function') {{ gtag('event', '{event_name}', {event_params}); console.log("GA Event Sent: {event_name}", {event_params}); }} else {{ console.error('gtag function not found for event: {event_name}'); }}</script>"""
        try:
            components.html(event_js, height=0)
        except Exception as e:
            logger.error(f"Failed to inject GA event script for '{event_name}': {e}")
    else:
        logger.debug(f"GA Disabled: Event '{event_name}' not sent.")


# --- Graceful Shutdown ---
def cleanup_resources():
    logger.info("-----------------------------------------------------")
    logger.info("Application shutting down. Performing cleanup...")
    logger.info("-----------------------------------------------------")
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
    logger.warning("Could not set all signal handlers (this is often normal on Windows).")


# --- State Reset Functions ---
def reset_all_workflow_states():
    logger.info("Resetting ALL workflow states.")
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
        "sidebar_expansion_result",
        "sidebar_expansion_truncated",
        "sidebar_affirm_original_text",
        "sidebar_affirm_text_pending_update",
        "sidebar_affirm_truncated_pending",
        "advanced_processing_active",
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
        "auto_sub_topic",
        "auto_sub_results",
        "auto_sub_processing",
        "auto_sub_show_results",
        "selected_workflow",
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            try:
                del st.session_state[key]
            except Exception as e:
                logger.warning(f"Could not delete session state key '{key}': {e}")

    st.cache_data.clear()
    # Clear specific cached resources if you know their function names, or all.
    # If get_auto_subliminal_components is defined in auto_subliminal_ui, clearing it here might be tricky
    # unless you import it. st.cache_resource.clear() is broader.
    st.cache_resource.clear()

    logger.info("All workflow states reset and caches cleared.")
    st.session_state.selected_workflow = None


# --- Main Application Logic ---
def main():
    logger.info("MindMorph Application starting/rerunning main function.")
    if "selected_workflow" not in st.session_state:
        st.session_state.selected_workflow = None

    if st.session_state.selected_workflow is None:
        st.title("🧠 Welcome to MindMorph!")
        st.subheader("Choose how you want to create your subliminal audio:")
        st.markdown("---")

        col1, col2, col3 = st.columns(3, gap="large")

        with col1:  # Wizard
            st.markdown("### ✨ Quick Create Wizard")
            st.markdown("- Simple, step-by-step process.\n- Use text or audio affirmations.\n- Optional background sounds/frequencies.\n- High-quality offline TTS (Piper).")
            if st.button("Start Wizard", key="start_wizard_button", use_container_width=True, type="primary"):
                send_ga_event("select_workflow", {"workflow": "wizard", "event_category": "engagement", "event_label": "wizard_selected"})
                st.session_state.selected_workflow = "wizard"
                st.rerun()
        with col2:  # Advanced Editor
            st.markdown("### 🎚️ Advanced Editor")
            st.markdown("- Full control with multiple tracks.\n- Fine-tune volume, speed, pitch, etc.\n- Advanced effects (ultrasonic shift).\n- Save/Load projects.")
            if st.button("Open Advanced Editor", key="start_advanced_button", use_container_width=True):
                send_ga_event("select_workflow", {"workflow": "advanced", "event_category": "engagement", "event_label": "advanced_selected"})
                st.session_state.selected_workflow = "advanced"
                if "app_mode" not in st.session_state:
                    st.session_state.app_mode = "Easy"
                st.rerun()
        with col3:  # Auto-Create
            st.markdown("### 🚀 Auto-Create Subliminal")
            st.markdown("- Enter a topic, get a full subliminal!\n- Auto-generates affirmations.\n- Includes background sound.\n- _(Experimental Beta Feature)_")
            if st.button("Start Auto-Create", key="start_auto_create_button", use_container_width=True, disabled=not AUTO_SUB_UI_AVAILABLE):
                if AUTO_SUB_UI_AVAILABLE:
                    send_ga_event("select_workflow", {"workflow": "auto_subliminal", "event_category": "engagement", "event_label": "auto_subliminal_selected"})
                    st.session_state.selected_workflow = "auto_subliminal"
                    # Initialize states (can also be done in the workflow function if preferred)
                    st.session_state.auto_sub_topic = ""
                    st.session_state.auto_sub_results = None
                    st.session_state.auto_sub_processing = False
                    st.session_state.auto_sub_show_results = False
                    st.rerun()
                else:
                    st.error("Auto-Create feature is currently unavailable due to a loading error. Please check logs.")

        st.markdown("---")
        st.markdown('<div style="text-align: center;"><h4>❤️ Support MindMorph</h4><p>Consider supporting development on Patreon.</p></div>', unsafe_allow_html=True)
        if PATERON_URL and PATERON_URL != "YOUR_PATERON_URL_HERE":
            st.markdown(
                f'<div style="text-align: center; margin-top: 10px; margin-bottom: 10px;"><a href="{PATERON_URL}" target="_blank"><button style="padding: 10px 20px; background-color: #f0f2f6; color: #31333F; border: 1px solid #ced4da; border-radius: 0.25rem; cursor: pointer; font-weight: bold;">💖 Join Patreon</button></a></div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")

    elif st.session_state.selected_workflow == "wizard":
        logger.info("Running Quick Create Wizard workflow.")
        try:
            from wizard_steps.quick_wizard import QuickWizard

            wizard = QuickWizard()
            wizard.render_wizard()
        except ImportError as e_import:  # Catch import error for wizard too
            logger.exception("Failed to import Quick Wizard components.")
            st.error(f"Error loading Quick Wizard module: {e_import}.", icon="🚨")
            if st.button("Return to Home"):
                reset_all_workflow_states()
                st.rerun()
        except Exception as e:
            logger.exception("Failed to initialize or render Quick Wizard.")
            st.error(f"Failed to start Quick Wizard: {e}.", icon="🔥")
            if st.button("Return to Home"):
                reset_all_workflow_states()
                st.rerun()

    elif st.session_state.selected_workflow == "advanced":
        logger.info("Running Advanced Editor workflow.")
        try:
            from app_state import AppState
            from project_handler import ProjectHandler
            from tts.piper_tts import PiperTTSGenerator
            from ui_manager import UIManager

            if "app_state" not in st.session_state:
                st.session_state.app_state = AppState()

            if "tts_generator" not in st.session_state or st.session_state.tts_generator is None:
                logger.info("Initializing PiperTTSGenerator for Advanced Editor.")
                try:
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
                    piper_model_path_abs = PIPER_VOICE_MODEL_PATH
                    if not os.path.isabs(piper_model_path_abs):
                        piper_model_path_abs = os.path.join(project_root, piper_model_path_abs)
                    piper_config_path_abs = PIPER_VOICE_CONFIG_PATH
                    if not os.path.isabs(piper_config_path_abs):
                        piper_config_path_abs = os.path.join(project_root, piper_config_path_abs)

                    if not os.path.exists(piper_model_path_abs) or not os.path.exists(piper_config_path_abs):
                        st.error("Piper TTS model/config for Advanced Editor not found.", icon="🗣️")
                        st.session_state.tts_generator = None
                    else:
                        st.session_state.tts_generator = PiperTTSGenerator(model_path=piper_model_path_abs, config_path=piper_config_path_abs, target_sr=GLOBAL_SR)
                except Exception as e_tts:
                    logger.exception("Failed to initialize PiperTTSGenerator for Advanced Editor.")
                    st.error(f"TTS Engine Error (Advanced Editor): {e_tts}.", icon="🗣️")
                    st.session_state.tts_generator = None

            if "project_handler" not in st.session_state:
                st.session_state.project_handler = ProjectHandler(st.session_state.app_state)
            # Ensure tts_generator is passed to UIManager, even if it's None
            current_tts_gen = st.session_state.get("tts_generator")
            if "ui_manager" not in st.session_state or st.session_state.ui_manager.tts_generator is not current_tts_gen:
                st.session_state.ui_manager = UIManager(app_state=st.session_state.app_state, tts_generator=current_tts_gen)

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
                selected_mode = st.radio("Editor Mode:", options=mode_options, index=current_mode_index, key="mode_selector_radio", horizontal=True)
                st.caption("Easy mode simplifies; Advanced shows all controls.")
                if selected_mode != st.session_state.app_mode:
                    st.session_state.app_mode = selected_mode
                    st.rerun()
            with header_cols[1]:
                if st.button("🏠 Back to Home", key="advanced_back_home", help="Exit Advanced Editor.", use_container_width=True):
                    reset_all_workflow_states()
                    st.rerun()
            st.markdown("---")
            st.session_state.ui_manager.render_ui(mode=st.session_state.app_mode)

        except ImportError as e_import:
            logger.exception("Failed to import Advanced Editor components.")
            st.error(f"Error loading Advanced Editor module: {e_import}.", icon="🚨")
            if st.button("Return to Home"):
                reset_all_workflow_states()
                st.rerun()
        except Exception as e:
            logger.exception("Failed to initialize or render Advanced Editor.")
            st.error(f"Failed to start Advanced Editor: {e}.", icon="🔥")
            if st.button("Return to Home"):
                reset_all_workflow_states()
                st.rerun()

    elif st.session_state.selected_workflow == "auto_subliminal":
        logger.info("Running Auto-Create Subliminal workflow.")
        if AUTO_SUB_UI_AVAILABLE:
            try:
                render_auto_subliminal_workflow(send_ga_event_func=send_ga_event)
            except Exception as e:
                logger.exception("An error occurred within the Auto-Create Subliminal workflow function.")
                st.error(f"An unexpected error occurred in Auto-Create: {e}", icon="🔥")
                if st.button("Return to Home"):
                    reset_all_workflow_states()
                    st.rerun()
        else:  # AUTO_SUB_UI_AVAILABLE is False, error already logged at import
            # The fallback render_auto_subliminal_workflow defined at the top will handle this
            render_auto_subliminal_workflow()

    else:  # Invalid state
        logger.warning(f"Invalid selected_workflow state: {st.session_state.selected_workflow}. Resetting to home.")
        st.session_state.selected_workflow = None
        st.rerun()

    logger.debug("--- End of main function render cycle ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("A critical error occurred during main execution.")
        try:
            st.error(f"A critical error stopped the application: {e}", icon="💥")
            st.warning("Please check the application logs for more details.")
            if st.button("Attempt Reset Application State"):
                reset_all_workflow_states()
                st.rerun()
        except Exception as display_error:
            print(f"CRITICAL ERROR: {e}", file=sys.stderr)
            print(f"Could not display error in Streamlit: {display_error}", file=sys.stderr)
