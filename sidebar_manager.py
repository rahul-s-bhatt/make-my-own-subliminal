# sidebar_manager.py
# ==========================================
# Main Sidebar UI Orchestrator for MindMorph
# ==========================================

import json
import logging
import os
from typing import Any, Dict, cast  # Keep cast if needed elsewhere, maybe not

import numpy as np
import streamlit as st
from PIL import Image

# Import necessary components from other modules
from app_state import AppState, TrackDataDict, TrackType

# Import config/utils as needed by this class
from config import (
    LOGO_PATH,
    PROJECT_FILE_VERSION,  # Might remove if save/load is gone
)
from sidebar_generators import SidebarGenerators
from sidebar_uploader import SidebarUploader
from tts_generator import TTSGenerator

logger = logging.getLogger(__name__)


class SidebarManager:
    """
    Orchestrates rendering the sidebar by coordinating sub-managers.
    Handles Save/Load functionality directly (Now Removed).
    """

    def __init__(self, app_state: AppState, tts_generator: TTSGenerator):
        """
        Initializes the SidebarManager and its sub-managers.

        Args:
            app_state: An instance of the AppState class.
            tts_generator: An instance of the TTSGenerator class (passed to uploader).
        """
        self.app_state = app_state
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
            self.uploader.render_uploader()
            st.divider()
            self.uploader.render_affirmation_inputs()
            st.divider()
            self.generators.render_frequency_generators()
            st.divider()
            self.generators.render_background_generators()
            st.markdown("---")

            # --- Project Save/Load Section (REMOVED) ---
            # st.subheader("💾 Project")
            # self._render_save_load() # <<< Call REMOVED >>>
            # st.markdown("---")

            # --- Final Info Message ---
            st.info("Edit track details and effects in the main panel.")

    # --- Private Rendering Method for Save/Load (REMOVED) ---

    # def _render_save_load(self):
    #     """Renders project save and load components in the sidebar."""
    #     # ... (Entire method content commented out or deleted) ...
    #     pass # Keep method definition if just commenting out content, or remove entirely
