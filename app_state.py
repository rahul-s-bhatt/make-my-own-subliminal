# app_state.py
# ==========================================
# Application State Management for MindMorph
# MODIFIED: Optimized track adding to store only source_info initially.
# ==========================================

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, cast

import numpy as np
import streamlit as st

from audio_utils.audio_state_definitions import (
    AudioData,
    SampleRate,
    SourceInfo,
    SourceInfoUpload,
    TrackDataDict,
    TrackID,
)
from config import MAX_TRACK_LIMIT, TRACK_TYPE_OTHER

logger = logging.getLogger(__name__)


class AppState:
    """
    Manages the application's track data dictionary using Streamlit's session state.

    Handles adding, deleting, updating, and retrieving track data structures.
    Stores only source information initially, loading/generating audio snippets on demand.
    Manages temporary files associated with tracks.
    """

    STATE_KEY = "mindmorph_tracks_state_v1"
    TrackDataDict = TrackDataDict

    def __init__(self):
        """Initializes the AppState manager."""
        if self.STATE_KEY not in st.session_state:
            logger.info(
                f"Initializing new application state under key '{self.STATE_KEY}'."
            )
            st.session_state[self.STATE_KEY]: Dict[TrackID, TrackDataDict] = {}
        else:
            logger.debug(
                f"Using existing application state from key '{self.STATE_KEY}'."
            )

    def _get_tracks_dict(self) -> Dict[TrackID, TrackDataDict]:
        """Safely retrieves the tracks dictionary from session state."""
        if self.STATE_KEY not in st.session_state:
            st.session_state[self.STATE_KEY] = {}
        return st.session_state[self.STATE_KEY]

    def get_all_tracks(self) -> Dict[TrackID, TrackDataDict]:
        """Returns a copy of the entire tracks dictionary."""
        return self._get_tracks_dict().copy()

    def get_track(self, track_id: TrackID) -> Optional[TrackDataDict]:
        """Retrieves a copy of the data structure for a specific track ID."""
        track_data = self._get_tracks_dict().get(track_id)
        return track_data.copy() if track_data else None

    def get_track_snippet(self, track_id: TrackID) -> Optional[AudioData]:
        """Retrieves the cached audio snippet for a specific track, if available."""
        # Directly access state for potentially large snippet to avoid copy if not needed
        track_data = self._get_tracks_dict().get(track_id)
        if track_data:
            # Return the snippet itself (might be large, avoid copy if possible)
            return track_data.get(
                "audio_snippet"
            )  # Returns None if not yet generated/cached
        return None

    # --- MODIFIED: add_track no longer takes audio_snippet or sr directly ---
    def add_track(
        self,
        source_info: SourceInfo,
        initial_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrackID]:
        """
        Adds a new track data structure to the application state, storing only
        source information initially. Audio snippets are generated/loaded later.

        Args:
            source_info: Dictionary containing information to reload/regenerate audio.
            initial_params: Optional dictionary with other initial parameters (name, type, etc.).

        Returns:
            The unique TrackID assigned to the newly added track, or None if limit reached.
        Raises:
            ValueError: If source_info is invalid.
        """
        tracks = self._get_tracks_dict()
        if len(tracks) >= MAX_TRACK_LIMIT:
            logger.warning(f"Cannot add track. Limit of {MAX_TRACK_LIMIT} reached.")
            st.error(f"Track limit ({MAX_TRACK_LIMIT}) reached.", icon="🚫")
            return None

        if not isinstance(source_info, dict) or "type" not in source_info:
            raise ValueError("Invalid source_info provided (must be dict with 'type').")

        track_id = str(uuid.uuid4())
        logger.info(
            f"Adding new track ID: {track_id}, Source Type: {source_info['type']}"
        )

        # Define default structure - audio_snippet and sr are initially None
        new_track: TrackDataDict = {
            "id": track_id,
            "audio_snippet": None,  # Snippet will be loaded/generated on demand
            "source_info": source_info,
            "sr": None,  # Sample rate determined when snippet is loaded/generated
            "name": "New Track",
            "track_type": TRACK_TYPE_OTHER,
            "volume": 1.0,
            "mute": False,
            "solo": False,
            "speed_factor": 1.0,
            "pitch_shift": 0.0,
            "pan": 0.0,
            "filter_type": "off",
            "filter_cutoff": 8000.0,
            "loop_to_fit": False,
            "reverse_audio": False,
            "ultrasonic_shift": False,
            "preview_temp_file_path": None,  # Keep for track-specific preview audio *file* if needed
            "preview_settings_hash": None,
            "update_counter": 0,
            "start_time_s": 0.0,
            "end_time_s": None,
            # Add any other keys defined in your TrackDataDict type hint
        }

        # Override defaults with initial_params
        if initial_params:
            for key, value in initial_params.items():
                if key in new_track:
                    new_track[key] = value
                else:
                    logger.warning(
                        f"Ignoring unknown initial parameter '{key}' for track {track_id}"
                    )

        tracks[track_id] = new_track
        logger.info(
            f"Successfully added track ID: {track_id} (Source: {source_info['type']}). Snippet pending."
        )
        return track_id

    # --- NEW: Method to update the snippet and SR once generated/loaded ---
    def update_track_snippet(
        self,
        track_id: TrackID,
        audio_snippet: Optional[AudioData],
        sr: Optional[SampleRate],
    ):
        """
        Updates the cached audio snippet and sample rate for a track.
        Called after the snippet is generated/loaded on demand.
        """
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            tracks[track_id]["audio_snippet"] = audio_snippet
            tracks[track_id]["sr"] = sr
            logger.debug(
                f"Updated snippet (size: {audio_snippet.shape if audio_snippet is not None else 'None'}) and SR ({sr}) for track {track_id}"
            )
            # Optionally increment update counter if UI needs immediate refresh based on snippet
            # self.increment_update_counter(track_id)
        else:
            logger.warning(
                f"Attempted to update snippet for non-existent track ID: {track_id}"
            )

    def delete_track(self, track_id: TrackID) -> bool:
        """
        Deletes a track from the state and cleans up associated preview and temporary upload files.
        """
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            track_data = tracks[track_id]
            track_name = track_data.get("name", "N/A")
            preview_path = track_data.get("preview_temp_file_path")
            source_info = track_data.get("source_info")

            # Cleanup Preview File (if used)
            if (
                preview_path
                and isinstance(preview_path, str)
                and os.path.exists(preview_path)
            ):
                try:
                    os.remove(preview_path)
                    logger.info(
                        f"Deleted preview temp file '{preview_path}' for track {track_id} ('{track_name}')"
                    )
                except OSError as e:
                    logger.warning(
                        f"Failed to delete preview temp file '{preview_path}' for track {track_id}: {e}"
                    )

            # Cleanup Temporary Upload File
            if isinstance(source_info, dict) and source_info.get("type") == "upload":
                upload_info = cast(SourceInfoUpload, source_info)
                temp_upload_path = upload_info.get("temp_file_path")
                if (
                    temp_upload_path
                    and isinstance(temp_upload_path, str)
                    and os.path.exists(temp_upload_path)
                ):
                    try:
                        os.remove(temp_upload_path)
                        logger.info(
                            f"Deleted temporary upload file '{temp_upload_path}' for track {track_id} ('{track_name}')"
                        )
                    except OSError as e:
                        logger.warning(
                            f"Failed to delete temporary upload file '{temp_upload_path}' for track {track_id}: {e}"
                        )

            del tracks[track_id]
            logger.info(
                f"Deleted track ID: {track_id}, Name: '{track_name}' from state."
            )
            return True
        else:
            logger.warning(f"Attempted to delete non-existent track ID: {track_id}")
            return False

    def update_track_param(self, track_id: TrackID, param_name: str, value: Any):
        """
        Updates a specific parameter for a given track data structure.
        Invalidates the preview hash if a parameter affecting the preview changes.
        """
        tracks = self._get_tracks_dict()
        if track_id not in tracks:
            logger.warning(
                f"Attempted to update parameter '{param_name}' for non-existent track ID: {track_id}"
            )
            return

        # Prevent modification of core/managed data via this method
        protected_params = [
            "id",
            "audio_snippet",  # Managed by update_track_snippet
            "source_info",
            "sr",  # Managed by update_track_snippet
            "update_counter",
            "preview_temp_file_path",  # Managed separately if needed
            "preview_settings_hash",  # Managed separately
        ]
        if param_name in protected_params:
            logger.error(
                f"Attempted to update protected/managed parameter '{param_name}' via update_track_param. Ignoring."
            )
            return

        current_value = tracks[track_id].get(param_name)
        needs_update = False

        # Handle potential type differences and floating point comparisons
        try:
            if isinstance(current_value, float) and isinstance(value, (float, int)):
                if not np.isclose(current_value, float(value)):
                    needs_update = True
                    value = float(value)
            elif type(current_value) != type(value):
                if param_name == "pitch_shift" and isinstance(value, int):
                    value = float(value)
                    if not np.isclose(current_value, value):
                        needs_update = True
                elif current_value != value:
                    needs_update = True
            elif current_value != value:
                needs_update = True
        except Exception:
            if current_value != value:
                needs_update = True  # Fallback

        if not needs_update:
            return

        logger.debug(
            f"Updating parameter '{param_name}' for track {track_id} from '{current_value}' to '{value}'."
        )
        tracks[track_id][param_name] = value

        # Invalidate preview hash if relevant parameter changes
        preview_affecting_params = [
            "volume",
            "speed_factor",
            "pitch_shift",
            "pan",
            "filter_type",
            "filter_cutoff",
            "loop_to_fit",
            "reverse_audio",
            "ultrasonic_shift",
            "start_time_s",
            "end_time_s",
        ]
        # --- ALSO INVALIDATE SNIPPET if source-affecting params change ---
        # Example: If TTS text changes, the snippet needs regeneration
        source_affecting_params = [
            "tts_text",
            "gen_noise_type",
            "gen_freq_left",
            "gen_freq_right",
            "gen_freq",
            "gen_carrier_freq",
            "gen_pulse_freq",
        ]  # Add relevant keys from SourceInfo types

        invalidate_snippet = False
        if param_name in preview_affecting_params:
            if tracks[track_id].get("preview_settings_hash") is not None:
                logger.debug(
                    f"Preview parameter '{param_name}' changed, invalidating preview hash for track {track_id}."
                )
                tracks[track_id]["preview_settings_hash"] = None
                tracks[track_id][
                    "preview_temp_file_path"
                ] = None  # Clear any associated preview *file*
                # Decide if the snippet itself should be cleared too. Usually yes for effects.
                invalidate_snippet = True

        # If a parameter within source_info changes (e.g., TTS text), invalidate snippet
        # This requires checking if param_name corresponds to a key within source_info
        source_info = tracks[track_id].get("source_info", {})
        if isinstance(source_info, dict) and param_name in source_info:
            logger.debug(
                f"Source parameter '{param_name}' changed, invalidating snippet for track {track_id}."
            )
            invalidate_snippet = True
            # Also invalidate preview hash as source changed
            tracks[track_id]["preview_settings_hash"] = None
            tracks[track_id]["preview_temp_file_path"] = None

        if invalidate_snippet and tracks[track_id].get("audio_snippet") is not None:
            logger.debug(f"Invalidating cached audio snippet for track {track_id}.")
            tracks[track_id]["audio_snippet"] = None
            tracks[track_id]["sr"] = None  # Also clear SR as it might change

        # Handle specific parameter interactions (e.g., ultrasonic vs pitch)
        if param_name == "ultrasonic_shift" and value is True:
            if not np.isclose(tracks[track_id].get("pitch_shift", 0.0), 0.0):
                tracks[track_id]["pitch_shift"] = 0.0
        elif param_name == "pitch_shift" and not np.isclose(value, 0.0):
            if tracks[track_id].get("ultrasonic_shift", False) is True:
                tracks[track_id]["ultrasonic_shift"] = False

        self.increment_update_counter(track_id)

    def increment_update_counter(self, track_id: TrackID):
        """Increments the update counter for a track to help trigger UI refreshes."""
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            current_counter = tracks[track_id].get("update_counter", 0)
            tracks[track_id]["update_counter"] = current_counter + 1
        else:
            logger.warning(
                f"Attempted increment update counter for non-existent track ID: {track_id}"
            )

    def get_loaded_track_names(self) -> List[str]:
        """Returns a list of names of all currently loaded tracks."""
        tracks = self.get_all_tracks()
        return [t.get("name", f"Track {i + 1}") for i, t in enumerate(tracks.values())]

    def clear_all_tracks(self):
        """Removes all tracks from the state and cleans up associated files."""
        logger.info("Clearing all tracks from application state.")
        all_tracks = self.get_all_tracks()
        deleted_count = 0
        for track_id in list(all_tracks.keys()):
            if self.delete_track(track_id):
                deleted_count += 1
        logger.info(f"Cleared {deleted_count} tracks.")
        st.session_state[self.STATE_KEY] = {}

    # Method to update the preview file path and hash (Kept for potential future use)
    def update_track_preview_file(
        self, track_id: TrackID, file_path: Optional[str], settings_hash: Optional[str]
    ):
        """Updates the preview file path and settings hash for a track."""
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            old_path = tracks[track_id].get("preview_temp_file_path")
            if old_path and old_path != file_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except OSError as e:
                    logger.warning(
                        f"Could not clean up old preview file {old_path}: {e}"
                    )

            tracks[track_id]["preview_temp_file_path"] = file_path
            tracks[track_id]["preview_settings_hash"] = settings_hash
            logger.debug(
                f"Updated preview file path/hash for track {track_id}. Hash: {settings_hash}"
            )
        else:
            logger.warning(
                f"Attempted to update preview file for non-existent track ID: {track_id}"
            )
