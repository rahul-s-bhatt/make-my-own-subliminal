# track_editor_manager.py
# ==========================================
# Track Editor UI Management for MindMorph (Orchestrator)
# ==========================================

import logging
import os
from typing import Any, Dict

import streamlit as st
from PIL import Image

# Import necessary components from other modules
from app_state import AppState, TrackData, TrackID, TrackType
from config import FAVICON_PATH, TRACK_TYPE_OTHER  # Only need constants used directly here
from track_metadata_ui import TrackMetadataUI

# Import the new UI component classes
from track_preview_ui import TrackPreviewUI

logger = logging.getLogger(__name__)


class TrackEditorManager:
    """Orchestrates rendering the main track editor panel using sub-components."""

    def __init__(self, app_state: AppState):
        """
        Initializes the TrackEditorManager and its UI sub-components.

        Args:
            app_state: An instance of the AppState class.
        """
        self.app_state = app_state
        # Instantiate the UI components
        self.preview_ui = TrackPreviewUI(app_state)
        self.metadata_ui = TrackMetadataUI(app_state)
        logger.debug("TrackEditorManager initialized with UI sub-components.")

    # --- Main Rendering Method ---

    def render_tracks_editor(self, mode: str = "Easy"):
        """
        Renders the main editor area, handling empty state and track iteration.

        Args:
            mode (str): The current editor mode ("Easy" or "Advanced").
        """
        st.header("🎚️ Tracks Editor")
        tracks = self.app_state.get_all_tracks()

        # --- Handle Empty State ---
        if not tracks:
            # <<< MODIFIED: Removed check for 'welcome_message_shown' >>>
            # Always show this message if no tracks exist.
            col_icon, col_text = st.columns([1, 5])
            with col_icon:
                st.markdown("<br/>", unsafe_allow_html=True)
                # Display icon or fallback
                if os.path.exists(FAVICON_PATH):
                    try:
                        st.image(Image.open(FAVICON_PATH), width=80)
                    except Exception:
                        st.markdown("🧠", unsafe_allow_html=True)
                else:
                    st.markdown("🧠", unsafe_allow_html=True)
            with col_text:
                st.subheader("✨ Let's Create Your Subliminal!")
                st.markdown("Your project is empty. Use the **sidebar on the left** (👈) to add your first audio layer.")
                st.markdown("- **Upload** your own audio files (music, voice).")
                st.markdown("- Generate **Affirmations** from text or a file.")
                st.markdown("- Add background **Noise** (White, Pink, Brown).")
                # Use the passed-in mode for conditional display
                if mode == "Easy":
                    st.markdown("- Add **Frequency Presets** for focus, relaxation, etc.")
                else:  # Advanced mode
                    st.markdown("- Add specific **Frequencies/Tones** or use Presets.")
            st.markdown("---")
            st.info("Once you add a track, its editor controls will appear here.")
            return  # Stop rendering if no tracks

        # --- Render Tracks ---
        st.caption(f"Current Mode: **{mode}** | Tracks: {len(tracks)}")
        st.markdown("Adjust settings for each track below. Click **'Update Preview'** inside a track's panel to refresh its 60s preview with the latest settings applied.")
        st.divider()

        track_ids_to_delete = []
        logger.debug(f"Rendering editor for {len(tracks)} tracks in {mode} mode.")

        track_ids = list(tracks.keys())  # Get keys before iterating
        for track_id in track_ids:
            track_data = self.app_state.get_track(track_id)  # Get fresh data each iteration
            if track_data is None:
                logger.warning(f"Track {track_id} not found during editor rendering, likely deleted.")
                continue  # Skip if track was deleted during rerun

            # Determine expander label
            track_name = track_data.get("name", "Unnamed Track")
            track_type_str = track_data.get("track_type", TRACK_TYPE_OTHER)
            track_type_icon = track_type_str.split(" ")[0] if " " in track_type_str else "⚪"
            expander_label = f"{track_type_icon} Track: **{track_name}**"
            if track_data.get("source_type") == "upload" and track_data.get("original_audio") is None:
                expander_label += "  ⚠️ Missing Source File"
            expander_label += f"  (`...{track_id[-6:]}`)"  # Add partial ID

            # Render track within an expander
            with st.expander(expander_label, expanded=True):
                logger.debug(f"Rendering expander for: '{track_name}' ({track_id}), Type: {track_type_str}")
                col_main, col_controls = st.columns([3, 1])  # Adjust ratios if needed

                # Delegate rendering to sub-components
                # Preview UI likely doesn't need the mode
                self.preview_ui.render_preview_column(track_id, track_data, col_main)
                # Pass 'mode' to metadata_ui
                deleted = self.metadata_ui.render_metadata_column(track_id, track_data, col_controls, mode=mode)

                if deleted:
                    track_ids_to_delete.append(track_id)

        # --- Process Deletions ---
        if track_ids_to_delete:
            deleted_count = 0
            for tid in track_ids_to_delete:
                if self.app_state.delete_track(tid):  # delete_track handles preview file cleanup
                    deleted_count += 1
            if deleted_count > 0:
                st.toast(f"Deleted {deleted_count} track(s).")
                logger.info(f"Processed deletion of {deleted_count} tracks.")
                st.rerun()  # Rerun to update the UI immediately after deletion
