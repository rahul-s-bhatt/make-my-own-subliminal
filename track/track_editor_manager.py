# track_editor_manager.py
# ==========================================
# Track Editor UI Management for MindMorph (Orchestrator)
# STEP 3 OPTIMIZED: Call get_or_generate_preview_snippet before rendering track details.
# IMPORT FIX: Corrected import path for audio_loader.
# ==========================================

import logging
import os
from typing import Optional

import streamlit as st
from PIL import Image

# Import necessary components and types
from app_state import AppState, TrackDataDict

# --- CORRECTED IMPORT PATH ---
from audio_utils.audio_loader import get_or_generate_preview_snippet
from config import FAVICON_PATH, TRACK_TYPE_OTHER
from track.track_metadata_ui import TrackMetadataUI
from track.track_preview_ui import TrackPreviewUI

# --- END CORRECTION ---
from tts.base_tts import BaseTTSGenerator

logger = logging.getLogger(__name__)


class TrackEditorManager:
    """Orchestrates rendering the main track editor panel using sub-components."""

    def __init__(self, app_state: AppState, tts_generator: Optional[BaseTTSGenerator]):
        """
        Initializes the TrackEditorManager and its UI sub-components.

        Args:
            app_state: An instance of the AppState class.
            tts_generator: The TTS generator instance (can be None).
        """
        self.app_state = app_state
        self.tts_generator = tts_generator  # Store the TTS generator
        self.preview_ui = TrackPreviewUI(app_state)
        self.metadata_ui = TrackMetadataUI(app_state)
        logger.debug(
            "TrackEditorManager initialized with UI sub-components and TTS generator."
        )

    def render_tracks_editor(self, mode: str = "Easy"):
        """
        Renders the main editor area, handling empty state and track iteration.
        Ensures preview snippets are generated/loaded before rendering track details.

        Args:
            mode (str): The current editor mode ("Easy" or "Advanced").
        """
        st.header("🎚️ Tracks Editor")
        tracks = self.app_state.get_all_tracks()

        # --- Handle Empty State (Remains the same) ---
        if not tracks:
            col_icon, col_text = st.columns([1, 5])
            with col_icon:
                st.markdown("<br/>", unsafe_allow_html=True)
                try:
                    st.image(Image.open(FAVICON_PATH), width=80)
                except Exception:
                    st.markdown("🧠", unsafe_allow_html=True)
            with col_text:
                st.subheader("✨ Let's Create Your Subliminal!")
                st.markdown(
                    "Use the **sidebar on the left** (👈) to add your first audio layer:"
                )
                st.markdown("- **Upload** audio files (music, voice).")
                st.markdown("- Generate **Affirmations** from text.")
                st.markdown("- Add background **Noise** (White, Pink, Brown).")
                if mode == "Easy":
                    st.markdown("- Add **Frequency Presets**.")
                else:
                    st.markdown("- Add specific **Frequencies/Tones** or Presets.")
            st.markdown("---")
            st.info("Track editor controls will appear here once tracks are added.")
            return

        # --- Render Tracks ---
        st.caption(f"Current Mode: **{mode}** | Tracks: {len(tracks)}")
        st.markdown(
            "Adjust settings for each track below. Previews load automatically."
        )
        st.divider()

        track_ids_to_delete = []
        logger.debug(f"Rendering editor for {len(tracks)} tracks in {mode} mode.")

        track_ids = list(tracks.keys())
        for track_id in track_ids:
            # --- Ensure snippet exists before rendering expander ---
            snippet_tuple = get_or_generate_preview_snippet(
                track_id=track_id,
                app_state=self.app_state,
                tts_generator=self.tts_generator,
            )
            # Refresh track_data *after* the call
            track_data: Optional[TrackDataDict] = self.app_state.get_track(track_id)

            if track_data is None:
                logger.warning(
                    f"Track {track_id} not found after snippet check, likely deleted."
                )
                continue

            # --- Determine expander label (Improved Status) ---
            track_name = track_data.get("name", "Unnamed Track")
            track_type_str = track_data.get("track_type", TRACK_TYPE_OTHER)
            track_type_icon = (
                track_type_str.split(" ")[0] if " " in track_type_str else "⚪"
            )
            expander_label = f"{track_type_icon} Track: **{track_name}**"

            source_info = track_data.get("source_info")
            snippet_status = ""
            if source_info and source_info.get("type") == "upload":
                temp_path = source_info.get("temp_file_path")
                # Ensure temp_path is treated as string for os.path.exists check
                if not temp_path or not os.path.exists(str(temp_path)):
                    snippet_status = f"  ⚠️ Missing Source: {source_info.get('original_filename', 'Unknown')}"
                elif track_data.get("audio_snippet") is None:
                    if snippet_tuple is None:
                        snippet_status = "  ❌ Preview Failed"
                    else:
                        snippet_status = "  ⏳ Preview Pending"
            elif track_data.get("audio_snippet") is None:
                if snippet_tuple is None:
                    snippet_status = "  ❌ Preview Failed"
                else:
                    snippet_status = "  ⏳ Preview Pending"

            expander_label += snippet_status
            expander_label += f"  (`...{track_id[-6:]}`)"
            # --- End Expander Label Logic ---

            # Render track within an expander
            with st.expander(expander_label, expanded=True):
                logger.debug(
                    f"Rendering expander for: '{track_name}' ({track_id}), Type: {track_type_str}"
                )
                col_main, col_controls = st.columns([3, 1])

                # Delegate rendering to sub-components
                self.preview_ui.render_preview_column(track_id, track_data, col_main)
                deleted = self.metadata_ui.render_metadata_column(
                    track_id, track_data, col_controls, mode=mode
                )

                if deleted:
                    track_ids_to_delete.append(track_id)

        # --- Process Deletions (Remains the same) ---
        if track_ids_to_delete:
            deleted_count = 0
            for tid in track_ids_to_delete:
                if self.app_state.delete_track(tid):
                    deleted_count += 1
            if deleted_count > 0:
                st.toast(f"Deleted {deleted_count} track(s).")
                logger.info(f"Processed deletion of {deleted_count} tracks.")
                st.rerun()
