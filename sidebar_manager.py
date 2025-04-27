# sidebar_manager.py
# ==========================================
# Main Sidebar UI Orchestrator for MindMorph
# ==========================================

import json
import logging
import os
from typing import Any, Dict

import numpy as np  # Keep if needed by save/load or other parts
import streamlit as st
from PIL import Image

# Import necessary components from other modules
from app_state import AppState, TrackData, TrackType

# Import config/utils as needed by this class (e.g., for save/load)
from config import (
    LOGO_PATH,
    # Add other config constants if directly used by _render_save_load
    PROJECT_FILE_VERSION,
)
from sidebar_generators import SidebarGenerators

# Import the sub-managers we created
from sidebar_uploader import SidebarUploader
from tts_generator import TTSGenerator  # Needed to pass to SidebarUploader

# from utils import some_utility_if_needed_here

# Get a logger for this module
logger = logging.getLogger(__name__)


class SidebarManager:
    """
    Orchestrates rendering the sidebar by coordinating sub-managers.
    Handles Save/Load functionality directly.
    """

    def __init__(self, app_state: AppState, tts_generator: TTSGenerator):
        """
        Initializes the SidebarManager and its sub-managers.

        Args:
            app_state: An instance of the AppState class.
            tts_generator: An instance of the TTSGenerator class (passed to uploader).
        """
        self.app_state = app_state
        # Initialize sub-managers, passing necessary dependencies
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
                    st.header("MindMorph")  # Fallback to text header
            else:
                st.header("MindMorph")
            st.caption("Subliminal Audio Editor")
            st.markdown("---")

            # --- Step 1: Add Audio Layers Section ---
            st.markdown("### STEP 1: Add Audio Layers")
            st.caption("Use the options below to add sounds to your project.")

            # --- Call Sub-manager Rendering Methods ---
            # Render Uploaders (Audio + Affirmations/TTS)
            self.uploader.render_uploader()
            st.divider()
            self.uploader.render_affirmation_inputs()
            st.divider()

            # Render Generators (Frequencies + Noise)
            self.generators.render_frequency_generators()
            st.divider()
            self.generators.render_background_generators()
            st.markdown("---")

            # --- Project Save/Load Section (Handled by this class) ---
            st.subheader("💾 Project")
            self._render_save_load()  # Call private method of this class
            st.markdown("---")

            # --- Final Info Message ---
            st.info("Edit track details and effects in the main panel.")

    # --- Private Rendering Method for Save/Load (Kept in this class) ---

    def _render_save_load(self):
        """Renders project save and load components in the sidebar."""
        st.markdown("**Save/Load Project**")
        st.caption("Save project setup (track settings, sources) to a `.mindmorph` file. Audio data itself is NOT saved and needs regeneration or re-upload.")

        # --- Save Project ---
        project_data_str = ""
        tracks = self.app_state.get_all_tracks()
        save_disabled = not bool(tracks)  # Disable save if no tracks

        if tracks:
            try:
                serializable_tracks = {}
                for track_id, track_data in tracks.items():
                    # Create a copy to avoid modifying the original state
                    save_data = track_data.copy()
                    # Remove non-serializable or large/runtime data
                    save_data.pop("original_audio", None)
                    save_data.pop("preview_temp_file_path", None)
                    save_data.pop("preview_settings_hash", None)
                    # Ensure source_type is present (should be by default)
                    if "source_type" not in save_data:
                        save_data["source_type"] = "unknown"  # Fallback

                    serializable_tracks[track_id] = save_data

                # Create the final project file content structure
                project_file_content = {"version": PROJECT_FILE_VERSION, "tracks": serializable_tracks}
                # Convert to JSON string
                project_data_str = json.dumps(project_file_content, indent=2)

            except Exception as e:
                logger.error(f"Error preparing project data for saving: {e}")
                st.error("Could not prepare project data for saving.")
                save_disabled = True  # Disable save on error

        st.download_button(
            label="💾 Save Project File",
            data=project_data_str if project_data_str else "",  # Ensure data is string
            file_name="my_subliminal_project.mindmorph",
            mime="application/json",
            key="sidebar_save_project_button",  # Keep key consistent if needed elsewhere
            help="Saves the current track list and settings (excluding audio data).",
            use_container_width=True,
            disabled=save_disabled,
        )

        # --- Load Project ---
        # Use a different key for load uploader to avoid conflicts
        load_uploader_key = "sidebar_load_project_uploader"  # Keep key consistent
        uploaded_project_file = st.file_uploader(
            "⬆️ Load Project File (.mindmorph)",
            type=["mindmorph", "json"],  # Allow both extensions
            key=load_uploader_key,
            accept_multiple_files=False,
            help="Load a previously saved project configuration. This will replace the current project.",
            # No on_change needed here, handled by main.py checking session_state flag
        )
        if uploaded_project_file is not None:
            # Check if this file is different from the one potentially already processed
            # This avoids triggering load multiple times if user interacts elsewhere
            # Use file_id for potentially more robust checking than just name/size
            current_file_id = getattr(uploaded_project_file, "file_id", uploaded_project_file.id)  # Adapt based on Streamlit version/object
            if st.session_state.get("uploaded_project_file_id") != current_file_id:
                st.session_state.uploaded_project_file_data = uploaded_project_file.getvalue()
                st.session_state.project_load_requested = True  # Flag for main loop
                st.session_state.uploaded_project_file_id = current_file_id  # Store ID
                logger.info(f"Project file uploaded via sidebar: {uploaded_project_file.name}. Requesting load.")
                st.rerun()  # Rerun to trigger loading logic in main
