# sidebar_manager.py
# ==========================================
# Main Sidebar UI Orchestrator for MindMorph
# ==========================================

import logging
import os

import streamlit as st
from PIL import Image

# Import necessary components from other modules
from app_state import AppState

# Import config/utils as needed by this class
from config import LOGO_PATH
from sidebar.sidebar_generators import SidebarGenerators
from sidebar.sidebar_uploader import SidebarUploader  # Keep this import
from tts.base_tts import BaseTTSGenerator

logger = logging.getLogger(__name__)


class SidebarManager:
    """
    Orchestrates rendering the sidebar by coordinating sub-managers.
    """

    # --- MODIFIED TYPE HINT ---
    def __init__(self, app_state: AppState, tts_generator: BaseTTSGenerator):
        # --- END MODIFIED TYPE HINT ---
        """
        Initializes the SidebarManager and its sub-managers.

        Args:
            app_state: An instance of the AppState class.
            tts_generator: An instance of a TTS generator (conforming to BaseTTSGenerator).
        """
        self.app_state = app_state
        # Pass the TTS generator instance (now BaseTTSGenerator) to SidebarUploader
        self.uploader = SidebarUploader(app_state, tts_generator)
        self.generators = SidebarGenerators(app_state)
        logger.debug("SidebarManager initialized with uploader and generators.")

    # --- Main Rendering Method ---

    def render_sidebar(self):
        """Renders the entire sidebar UI by calling sub-managers and rendering save/load."""
        with st.sidebar:
            # --- Logo and Title ---
            if os.path.exists(LOGO_PATH):
                try:
                    logo_image = Image.open(LOGO_PATH)
                    st.image(logo_image, width=200)
                except Exception as e:
                    logger.warning(f"Could not load logo image from {LOGO_PATH}: {e}")
                    st.header("MindMorph")
            else:
                st.header("MindMorph")
            st.caption("Subliminal Audio Editor")
            st.markdown("---")

            # --- Step 1: Add Audio Layers Section ---
            st.markdown("### STEP 1: Add Audio Layers")
            st.caption("Use the options below to add sounds to your project.")

            # --- Call Sub-manager Rendering Methods ---
            # SidebarUploader receives the TTS instance via self.uploader
            self.uploader.render_uploader()
            st.divider()
            self.uploader.render_affirmation_inputs()  # This method within SidebarUploader uses the TTS instance
            st.divider()
            self.generators.render_frequency_generators()
            st.divider()
            self.generators.render_background_generators()
            st.markdown("---")

            # --- Project Save/Load Section (REMOVED) ---
            # ...

            # --- ADD FEEDBACK BUTTON ---
            st.markdown("### 💬 Feedback")
            st.caption("Help us improve MindMorph!")
            google_form_url = "https://forms.gle/eXGtvAzEoEZCHpK69"  # Example URL
            if google_form_url == "YOUR_GOOGLE_FORM_URL_HERE":
                st.warning("Feedback form URL not configured.")
            else:
                st.link_button(
                    "📝 Provide Feedback",
                    url=google_form_url,
                    help="Opens feedback form in a new tab.",
                    use_container_width=True,
                    type="secondary",
                )
                logger.debug(f"Rendered feedback button linking to: {google_form_url}")
            st.markdown("---")
            # --- END FEEDBACK BUTTON ---

            # --- Final Info Message ---
            st.info("Edit track details and effects in the main panel.")
