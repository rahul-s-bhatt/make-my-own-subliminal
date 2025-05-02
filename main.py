# main.py
# ==========================================
# MindMorph - Pro Subliminal Audio Editor
# Main Application Entry Point (Router)
# ==========================================

import logging

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# --- Keep imports needed for initial setup ---
from config import FAVICON_PATH, GA_MEASUREMENT_ID, PATERON_URL
from utils import setup_logging

# --- Early Setup: Logging and Page Config ---
logger = logging.getLogger(__name__)
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
setup_logging()
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


# --- Helper Function to Reset Advanced State ---
# (Keep reset_advanced_editor_state function as is, it already clears relevant keys)
def reset_advanced_editor_state():
    """Clears session state keys specific to the advanced editor."""
    logger.info("Resetting Advanced Editor state.")
    # --- ADDED: Ensure processing flag is reset ---
    # This ensures the flag is False before we potentially delete it
    if "advanced_processing_active" in st.session_state:
        st.session_state.advanced_processing_active = False
    # --- END ADDED ---

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
        # Expansion/Undo states (if used by advanced editor sidebar)
        "sidebar_expansion_result",
        "sidebar_expansion_truncated",
        "sidebar_affirm_original_text",
        "sidebar_affirm_text_pending_update",
        "sidebar_affirm_truncated_pending",
        # Wizard states (clear them when switching back from advanced)
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
        # --- ADDED: Add the flag to the list to be deleted if it exists ---
        # This ensures it's removed when switching away from the advanced editor
        "advanced_processing_active",
        # --- END ADDED ---
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
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
            if st.button(
                "Start Wizard",
                key="start_wizard_button",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.selected_workflow = "wizard"
                # Initialization now happens within QuickWizard class
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

        # --- ADD PATREON SUPPORT BUTTON ---
        st.markdown("#### ❤️ Support MindMorph")
        st.markdown(
            "If you find MindMorph useful, please consider supporting its development on Patreon."
        )
        if PATERON_URL and PATERON_URL != "YOUR_PATERON_URL_HERE":
            col_support_1, col_support_2, col_support_3 = st.columns([1, 1.5, 1])
            with col_support_2:
                st.link_button(
                    "💖 Join Patreon",
                    url=PATERON_URL,
                    help="Support MindMorph development.",
                    use_container_width=True,
                    type="secondary",
                )
        else:
            logger.warning("Patreon URL not configured.")
        st.markdown("---")
        # --- END PATREON SUPPORT BUTTON ---

    # --- Run Selected Workflow ---
    elif st.session_state.selected_workflow == "wizard":
        logger.info("Running Quick Create Wizard workflow.")
        # Import QuickWizard here
        from wizard_steps.quick_wizard import QuickWizard

        try:
            # Instantiate QuickWizard - it now handles its own TTS setup
            wizard = QuickWizard()
            wizard.render_wizard()  # Wizard UI takes over main area
        except Exception as e:
            logger.exception("Failed to initialize or render Quick Wizard.")
            st.error(
                f"Failed to start Quick Wizard: {e}. Please check logs and TTS model configuration."
            )
            # Add button to go back home
            if st.button("Return to Home"):
                reset_advanced_editor_state()  # Use same reset function
                st.rerun()

    elif st.session_state.selected_workflow == "advanced":
        logger.info("Running Advanced Editor workflow.")
        # Import necessary components for advanced editor
        from app_state import AppState
        from project_handler import ProjectHandler

        # --- Import the NEW TTS Generator ---
        from tts.piper_tts import PiperTTSGenerator
        from ui_manager import UIManager

        try:
            # Initialize Core Components
            if "app_state" not in st.session_state:
                st.session_state.app_state = AppState()
            # --- Instantiate PiperTTSGenerator for Advanced Editor ---
            if "tts_generator" not in st.session_state:
                logger.info("Initializing PiperTTSGenerator for Advanced Editor.")
                # This might fail if config paths are wrong or models can't load
                st.session_state.tts_generator = PiperTTSGenerator()
                logger.info("PiperTTSGenerator initialized for Advanced Editor.")
            # --- End TTS Instantiation ---
            if "project_handler" not in st.session_state:
                st.session_state.project_handler = ProjectHandler(
                    st.session_state.app_state
                )
            if "ui_manager" not in st.session_state:
                # Pass the instantiated TTS generator
                st.session_state.ui_manager = UIManager(
                    st.session_state.app_state, st.session_state.tts_generator
                )

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
                    current_mode_index = 0
                selected_mode = st.radio(
                    "Editor Mode:",
                    options=mode_options,
                    index=current_mode_index,
                    key="mode_selector_radio",
                    horizontal=True,
                )
                st.caption(
                    "Easy mode simplifies the interface; Advanced mode shows all track controls."
                )
                if selected_mode != st.session_state.app_mode:
                    logger.info(f"Advanced editor mode changed to '{selected_mode}'")
                    st.session_state.app_mode = selected_mode
                    # Clear potentially mode-specific states
                    if "export_buffer" in st.session_state:
                        del st.session_state.export_buffer
                    if "preview_audio_data" in st.session_state:
                        del st.session_state.preview_audio_data
                    st.rerun()
            with header_cols[1]:  # Back to Home Button
                if st.button(
                    "🏠 Back to Home",
                    key="advanced_back_home",
                    help="Exit Advanced Editor.",
                    use_container_width=True,
                ):
                    reset_advanced_editor_state()
                    st.rerun()
            st.markdown("---")

            # Render Main UI Sections using UIManager
            st.session_state.ui_manager.render_ui(mode=st.session_state.app_mode)

        except Exception as e:
            logger.exception("Failed to initialize or render Advanced Editor.")
            st.error(
                f"Failed to start Advanced Editor: {e}. Please check logs and TTS model configuration."
            )
            # Add button to go back home
            if st.button("Return to Home"):
                reset_advanced_editor_state()
                st.rerun()

    else:  # Invalid state
        logger.warning(
            f"Invalid selected_workflow state: {st.session_state.selected_workflow}. Resetting."
        )
        st.session_state.selected_workflow = None
        st.rerun()

    logger.info("--- End of main function render cycle ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch errors that might happen during initialization (like TTS loading)
        logger.exception("A critical error occurred during main execution.")
        # Display error prominently if possible
        st.error(f"A critical error stopped the application: {e}")
        st.warning(
            "Please check the application logs for more details and ensure configuration (e.g., TTS model paths) is correct."
        )
        # Optionally add a button to reset everything
        if st.button("Attempt Reset"):
            reset_advanced_editor_state()  # Try resetting state
            st.rerun()
