# app_state.py
# ==========================================
# Application State Management for MindMorph
# ==========================================

import logging
import os
import tempfile  # Still needed for cleanup in delete_track
import uuid

# --- MODIFIED: Added 'cast' to the import ---
from typing import Any, Dict, List, Optional, Tuple, cast

# --- END MODIFIED ---
import numpy as np
import streamlit as st

# Import definitions from the new definitions file
from audio_utils.audio_state_definitions import (
    AudioData,
    SampleRate,  # Added SampleRate for type hints if needed elsewhere
    SourceInfo,
    SourceInfoUpload,  # Needed for cast check
    TrackDataDict,
    TrackID,
)

# Import constants and default parameters from config.py
# Assuming MAX_TRACK_LIMIT is defined in config, otherwise define here
from config import GLOBAL_SR, MAX_TRACK_LIMIT, TRACK_TYPE_OTHER

# Get a logger for this module
logger = logging.getLogger(__name__)


class AppState:
    """
    Manages the application's track data dictionary using Streamlit's session state.

    Handles adding, deleting, updating, and retrieving track data structures.
    Does NOT handle audio loading/regeneration itself (see audio_loader.py).
    Manages temporary files associated with tracks.
    """

    # The key used to store the track dictionary in st.session_state
    STATE_KEY = "mindmorph_tracks_state_v1"  # Using a versioned key
    # Expose TrackDataDict for type hinting in other modules if needed
    TrackDataDict = TrackDataDict

    def __init__(self):
        """
        Initializes the AppState manager. Ensures the state dictionary exists
        in st.session_state and performs basic validation/cleanup.
        """
        if self.STATE_KEY not in st.session_state:
            logger.info(f"Initializing new application state under key '{self.STATE_KEY}'.")
            st.session_state[self.STATE_KEY]: Dict[TrackID, TrackDataDict] = {}
        else:
            logger.debug(f"Using existing application state from key '{self.STATE_KEY}'.")
            # self._validate_and_clean_state() # Optional: Run validation on init

    def _validate_and_clean_state(self):
        """
        (Optional) Iterates through existing tracks in the state, ensuring they have
        required keys and cleaning up preview file paths if inconsistent.
        Removes deprecated keys if migrating from older state versions.
        """
        # This function can be complex and depends on previous state structures.
        # Implement as needed for state migrations or validation.
        logger.debug("Validation/cleanup skipped in this version.")
        pass

    def _get_tracks_dict(self) -> Dict[TrackID, TrackDataDict]:
        """Safely retrieves the tracks dictionary from session state."""
        # Ensure the state key exists, initialize if not (defensive)
        if self.STATE_KEY not in st.session_state:
            st.session_state[self.STATE_KEY] = {}
        return st.session_state[self.STATE_KEY]

    def get_all_tracks(self) -> Dict[TrackID, TrackDataDict]:
        """Returns a copy of the entire tracks dictionary."""
        # Return a copy to prevent external modification of the state dict
        return self._get_tracks_dict().copy()

    def get_track(self, track_id: TrackID) -> Optional[TrackDataDict]:
        """Retrieves a copy of the data structure for a specific track ID."""
        track_data = self._get_tracks_dict().get(track_id)
        # Return a copy to prevent external modification
        return track_data.copy() if track_data else None

    def get_track_snippet(self, track_id: TrackID) -> Optional[AudioData]:
        """Retrieves the audio snippet for a specific track."""
        # Directly access state for potentially large snippet to avoid copy if not needed
        track_data = self._get_tracks_dict().get(track_id)
        if track_data:
            # Return the snippet itself (might be large, avoid copy if possible)
            return track_data.get("audio_snippet")
        return None

    def add_track(
        self,
        audio_snippet: Optional[AudioData],
        source_info: SourceInfo,
        sr: int,
        initial_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrackID]:
        """
        Adds a new track data structure to the application state.

        Args:
            audio_snippet: The initial audio data snippet (NumPy array) or None.
            source_info: Dictionary containing information to reload/regenerate full audio.
            sr: The sample rate of the audio snippet.
            initial_params: Optional dictionary with other initial parameters (name, type, etc.).

        Returns:
            The unique TrackID assigned to the newly added track, or None if limit reached.
        Raises:
            ValueError: If source_info is invalid or essential data is missing.
        """
        tracks = self._get_tracks_dict()
        if len(tracks) >= MAX_TRACK_LIMIT:
            logger.warning(f"Cannot add track. Limit of {MAX_TRACK_LIMIT} reached.")
            st.error(f"Track limit ({MAX_TRACK_LIMIT}) reached.", icon="🚫")
            return None

        if not isinstance(source_info, dict) or "type" not in source_info:
            raise ValueError("Invalid source_info provided (must be dict with 'type').")
        if not isinstance(sr, int) or sr <= 0:
            logger.warning(f"Invalid sample rate {sr} provided for new track. Using GLOBAL_SR {GLOBAL_SR}.")
            sr = GLOBAL_SR  # Use default if invalid
        if audio_snippet is not None:
            if not isinstance(audio_snippet, np.ndarray):
                logger.error("Invalid audio_snippet provided (must be numpy array or None). Setting to None.")
                audio_snippet = None
            elif audio_snippet.size == 0:
                logger.warning("Provided audio_snippet is empty.")
                # Allow adding track even with empty snippet, might be intended

        track_id = str(uuid.uuid4())
        logger.info(f"Adding new track ID: {track_id}, Source Type: {source_info['type']}")

        # Define default structure based on TrackDataDict annotations
        # Ensure all keys from the type hint are present
        new_track: TrackDataDict = {
            "id": track_id,  # Add track_id to the dict itself
            "audio_snippet": audio_snippet,
            "source_info": source_info,
            "sr": sr,
            "name": "New Track",
            "track_type": TRACK_TYPE_OTHER,
            "volume": 1.0,
            "mute": False,
            "solo": False,
            "speed_factor": 1.0,
            "pitch_shift": 0.0,  # Use float for pitch
            "pan": 0.0,
            "filter_type": "off",
            "filter_cutoff": 8000.0,
            "loop_to_fit": False,
            "reverse_audio": False,
            "ultrasonic_shift": False,
            "preview_temp_file_path": None,  # Path for preview audio file
            "preview_settings_hash": None,  # Hash of settings used for preview
            "update_counter": 0,  # Counter to trigger UI updates
            # Add other potential keys from TrackDataDict if defined
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
                    logger.warning(f"Ignoring unknown initial parameter '{key}' for track {track_id}")

        # Add to session state
        tracks[track_id] = new_track  # Add to the dict retrieved from state
        # No need to reassign st.session_state[self.STATE_KEY] if tracks is a direct reference

        logger.info(f"Successfully added track ID: {track_id}, Name: '{new_track['name']}', Type: {new_track['track_type']}")
        return track_id

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

            # Cleanup Preview File
            if preview_path and isinstance(preview_path, str) and os.path.exists(preview_path):
                try:
                    os.remove(preview_path)
                    logger.info(f"Deleted preview temp file '{preview_path}' for track {track_id} ('{track_name}')")
                except OSError as e:
                    logger.warning(f"Failed to delete preview temp file '{preview_path}' for track {track_id}: {e}")

            # Cleanup Temporary Upload File
            # Use type guard for safer access
            if isinstance(source_info, dict) and source_info.get("type") == "upload":
                # --- Use cast here after checking type ---
                upload_info = cast(SourceInfoUpload, source_info)
                # --- End use cast ---
                temp_upload_path = upload_info.get("temp_file_path")
                if temp_upload_path and isinstance(temp_upload_path, str) and os.path.exists(temp_upload_path):
                    try:
                        os.remove(temp_upload_path)
                        logger.info(f"Deleted temporary upload file '{temp_upload_path}' for track {track_id} ('{track_name}')")
                    except OSError as e:
                        logger.warning(f"Failed to delete temporary upload file '{temp_upload_path}' for track {track_id}: {e}")

            # Remove the track from the session state dictionary
            del tracks[track_id]  # Delete from the dict retrieved from state
            logger.info(f"Deleted track ID: {track_id}, Name: '{track_name}' from state.")
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
            logger.warning(f"Attempted to update parameter '{param_name}' for non-existent track ID: {track_id}")
            return

        # Prevent modification of core data via this method
        if param_name in ["id", "audio_snippet", "source_info", "sr", "update_counter", "preview_temp_file_path", "preview_settings_hash"]:
            logger.error(f"Attempted to update protected parameter '{param_name}' via update_track_param. Ignoring.")
            return

        # Check if the parameter name is valid according to the type hint
        # This requires TrackDataDict to be properly defined with annotations
        # If TrackDataDict is just a Dict, this check might not be possible/useful
        # if param_name not in TrackDataDict.__annotations__:
        #     logger.warning(f"Attempted to update potentially invalid parameter '{param_name}' for track {track_id}. Allowing for flexibility.")
        # return # Or allow update if flexibility is needed

        current_value = tracks[track_id].get(param_name)
        needs_update = False

        # Handle potential type differences and floating point comparisons
        try:
            if isinstance(current_value, float) and isinstance(value, (float, int)):
                if not np.isclose(current_value, float(value)):
                    needs_update = True
                    value = float(value)  # Ensure value is float
            elif type(current_value) != type(value):
                # Attempt type conversion if reasonable (e.g., int to float for pitch)
                if param_name == "pitch_shift" and isinstance(value, int):
                    value = float(value)
                    if not np.isclose(current_value, value):
                        needs_update = True
                elif current_value != value:  # Fallback to direct comparison if types differ and no specific conversion
                    needs_update = True
            elif current_value != value:
                needs_update = True
        except Exception:
            # Fallback comparison if type checks fail
            if current_value != value:
                needs_update = True

        if not needs_update:
            # logger.debug(f"Parameter '{param_name}' for track {track_id} already has value '{value}'. No update needed.")
            return

        logger.debug(f"Updating parameter '{param_name}' for track {track_id} from '{current_value}' to '{value}'.")
        tracks[track_id][param_name] = value  # Update the dict retrieved from state

        # --- Invalidate preview hash if relevant parameter changes ---
        # Define parameters that affect the audio output for preview
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
            "end_time_s",  # Start/end times affect snippet processing
        ]
        if param_name in preview_affecting_params:
            if tracks[track_id].get("preview_settings_hash") is not None:
                logger.debug(f"Parameter '{param_name}' changed, invalidating preview hash for track {track_id}.")
                tracks[track_id]["preview_settings_hash"] = None
                # Also clear the preview file path as it's now stale
                tracks[track_id]["preview_temp_file_path"] = None

        # --- Handle specific parameter interactions ---
        # Ensure ultrasonic and pitch shift are mutually exclusive
        if param_name == "ultrasonic_shift" and value is True:
            if not np.isclose(tracks[track_id].get("pitch_shift", 0.0), 0.0):
                logger.debug(f"Ultrasonic shift enabled for track {track_id}, resetting regular pitch shift.")
                tracks[track_id]["pitch_shift"] = 0.0
                tracks[track_id]["preview_settings_hash"] = None  # Invalidate preview
        elif param_name == "pitch_shift" and not np.isclose(value, 0.0):
            if tracks[track_id].get("ultrasonic_shift", False) is True:
                logger.debug(f"Regular pitch shift set for track {track_id}, disabling ultrasonic shift.")
                tracks[track_id]["ultrasonic_shift"] = False
                tracks[track_id]["preview_settings_hash"] = None  # Invalidate preview

        # Increment update counter to signal UI changes if needed
        self.increment_update_counter(track_id)

    def increment_update_counter(self, track_id: TrackID):
        """Increments the update counter for a track to help trigger UI refreshes."""
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            current_counter = tracks[track_id].get("update_counter", 0)
            tracks[track_id]["update_counter"] = current_counter + 1
            # logger.debug(f"Incremented update counter for track {track_id} to {current_counter + 1}")
        else:
            logger.warning(f"Attempted increment update counter for non-existent track ID: {track_id}")

    def get_loaded_track_names(self) -> List[str]:
        """Returns a list of names of all currently loaded tracks."""
        tracks = self.get_all_tracks()
        return [t.get("name", f"Track {i + 1}") for i, t in enumerate(tracks.values())]

    def clear_all_tracks(self):
        """Removes all tracks from the state and cleans up associated files."""
        logger.info("Clearing all tracks from application state.")
        all_tracks = self.get_all_tracks()  # Get a copy of keys
        deleted_count = 0
        for track_id in list(all_tracks.keys()):  # Iterate over keys from the copy
            if self.delete_track(track_id):  # delete_track modifies the original state dict
                deleted_count += 1
        logger.info(f"Cleared {deleted_count} tracks.")
        # Ensure the state dictionary itself is empty after deletion loop
        st.session_state[self.STATE_KEY] = {}

    # Method to update the preview file path and hash
    def update_track_preview_file(self, track_id: TrackID, file_path: Optional[str], settings_hash: Optional[str]):
        """Updates the preview file path and settings hash for a track."""
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            # Clean up old file if path changes and old path exists
            old_path = tracks[track_id].get("preview_temp_file_path")
            if old_path and old_path != file_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                    logger.info(f"Cleaned up old preview file on update: {old_path}")
                except OSError as e:
                    logger.warning(f"Could not clean up old preview file {old_path} on update: {e}")

            tracks[track_id]["preview_temp_file_path"] = file_path
            tracks[track_id]["preview_settings_hash"] = settings_hash
            logger.debug(f"Updated preview file path/hash for track {track_id}. Hash: {settings_hash}")
        else:
            logger.warning(f"Attempted to update preview file for non-existent track ID: {track_id}")
