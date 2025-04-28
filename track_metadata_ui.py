# track_metadata_ui.py
# ==========================================
# Track Metadata Column UI for MindMorph Editor
# ==========================================

import logging
from typing import Any, Dict

import streamlit as st

# Import necessary components from other modules
from app_state import AppState, TrackData, TrackID, TrackType
from config import TRACK_TYPE_OTHER, TRACK_TYPES

logger = logging.getLogger(__name__)


class TrackMetadataUI:
    """Handles rendering the metadata and mixing controls column for a track."""

    def __init__(self, app_state: AppState):
        """
        Initializes the TrackMetadataUI.

        Args:
            app_state: An instance of the AppState class.
        """
        self.app_state = app_state
        logger.debug("TrackMetadataUI initialized.")

    def render_metadata_column(self, track_id: TrackID, track_data: TrackData, column: st.delta_generator.DeltaGenerator) -> bool:
        """
        Renders the side controls for a track (Name, Type, Mute, Solo, Delete).

        Args:
            track_id: The ID of the track being rendered.
            track_data: The data dictionary for the track.
            column: The Streamlit column container to render into.

        Returns:
            bool: True if the delete button was clicked, False otherwise.
        """
        delete_clicked = False
        with column:
            try:
                st.markdown("**Track Details**")
                current_name = track_data.get("name", "Unnamed Track")
                new_name = st.text_input("Track Name", value=current_name, key=f"name_{track_id}", help="Rename this track.")
                if new_name != current_name and new_name.strip():
                    self.app_state.update_track_param(track_id, "name", new_name.strip())
                    st.rerun()  # Rerun to update expander label

                current_type = track_data.get("track_type", TRACK_TYPE_OTHER)
                try:
                    current_index = TRACK_TYPES.index(current_type)
                except ValueError:
                    current_index = TRACK_TYPES.index(TRACK_TYPE_OTHER)  # Default if type is somehow invalid
                new_type = st.selectbox("Track Type", TRACK_TYPES, index=current_index, key=f"type_{track_id}", help="Categorize this layer.")
                if new_type != current_type:
                    self.app_state.update_track_param(track_id, "track_type", new_type)
                    st.rerun()  # Rerun to update expander label

                st.caption("Mixing Controls")
                ms_col1, ms_col2 = st.columns(2)
                with ms_col1:
                    mute_value = track_data.get("mute", False)
                    mute = st.checkbox("Mute", value=mute_value, key=f"mute_{track_id}", help="Silence track in mix.")
                    if mute != mute_value:
                        self.app_state.update_track_param(track_id, "mute", mute)
                        # No rerun needed for mute/solo, affects next mix/preview
                with ms_col2:
                    solo_value = track_data.get("solo", False)
                    solo = st.checkbox("Solo", value=solo_value, key=f"solo_{track_id}", help="Isolate track(s) in mix.")
                    if solo != solo_value:
                        self.app_state.update_track_param(track_id, "solo", solo)
                        # No rerun needed for mute/solo

                st.markdown("---")
                if st.button("🗑️ Delete Track", key=f"delete_{track_id}", help="Permanently remove track.", type="secondary", use_container_width=True):
                    delete_clicked = True
                    logger.info(f"Delete button clicked for track {track_id} ('{track_data.get('name', 'N/A')}')")

            except Exception as e:
                logger.exception(f"Error rendering metadata column for track {track_id}")
                st.error(f"Error displaying track controls: {e}")

        return delete_clicked
