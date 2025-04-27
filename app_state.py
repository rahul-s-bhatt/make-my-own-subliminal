# app_state.py
# ==========================================
# Application State Management for MindMorph
# ==========================================

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

# Import constants and default parameters from config.py
from config import GLOBAL_SR, TRACK_TYPE_OTHER, get_default_track_params

# Define type hints used within this module
TrackID = str
TrackData = Dict[str, Any]  # Using Dict temporarily, might be replaced by a dataclass later
TrackType = str

# Get a logger for this module
logger = logging.getLogger(__name__)


class AppState:
    """
    Manages the application's track data using Streamlit's session state.

    Handles adding, deleting, updating, and retrieving track information,
    including managing associated preview files and state consistency.
    Uses a non-destructive approach for audio data (stores original only).
    """

    # The key used to store the track dictionary in st.session_state
    STATE_KEY = "mindmorph_tracks_state_v1"  # Renamed for clarity

    def __init__(self):
        """
        Initializes the AppState manager. Ensures the state dictionary exists
        in st.session_state and cleans up/validates existing track data.
        """
        if self.STATE_KEY not in st.session_state:
            logger.info(f"Initializing new application state under key '{self.STATE_KEY}'.")
            st.session_state[self.STATE_KEY] = {}
        else:
            logger.debug(f"Using existing application state from key '{self.STATE_KEY}'.")
            self._validate_and_clean_state()

    def _validate_and_clean_state(self):
        """
        Iterates through existing tracks in the state, ensuring they have all
        required default parameters and cleaning up outdated fields or files.
        """
        logger.debug("Validating and cleaning existing track state.")
        tracks_dict = self.get_all_tracks()
        default_params = get_default_track_params()

        for track_id, track_data in list(tracks_dict.items()):
            state_updated = False
            # Ensure all default keys exist
            for key, default_value in default_params.items():
                if key not in track_data:
                    logger.warning(f"Track '{track_data.get('name', track_id)}' missing key '{key}'. Adding default value.")
                    st.session_state[self.STATE_KEY][track_id][key] = default_value
                    state_updated = True

            # Remove deprecated keys if they exist (e.g., 'processed_audio')
            if "processed_audio" in track_data:
                logger.debug(f"Removing deprecated 'processed_audio' key for track {track_id}.")
                del st.session_state[self.STATE_KEY][track_id]["processed_audio"]
                state_updated = True

            # Clean up preview file path if the hash is missing (indicates inconsistency)
            if track_data.get("preview_settings_hash") is None:
                old_path = track_data.get("preview_temp_file_path")
                if old_path:
                    logger.warning(f"Preview hash missing for track {track_id}. Invalidating preview file path.")
                    st.session_state[self.STATE_KEY][track_id]["preview_temp_file_path"] = None
                    state_updated = True
                    # Attempt to delete the orphaned preview file
                    if os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                            logger.info(f"Cleaned up orphaned preview file: {old_path}")
                        except OSError as e:
                            logger.warning(f"Could not clean up orphaned preview file {old_path}: {e}")

            if state_updated:
                logger.debug(f"State updated during validation for track {track_id}.")

    def _get_tracks_dict(self) -> Dict[TrackID, TrackData]:
        """Safely retrieves the tracks dictionary from session state."""
        return st.session_state.get(self.STATE_KEY, {})

    def get_all_tracks(self) -> Dict[TrackID, TrackData]:
        """Returns a copy of the entire tracks dictionary."""
        # Return a copy to prevent external modification of the state dict directly
        return self._get_tracks_dict().copy()

    def get_track(self, track_id: TrackID) -> Optional[TrackData]:
        """
        Retrieves the data for a specific track ID.

        Args:
            track_id: The unique identifier of the track.

        Returns:
            A copy of the track data dictionary, or None if the track ID is not found.
        """
        track_data = self._get_tracks_dict().get(track_id)
        # Return a copy to prevent external modification of the state dict directly
        return track_data.copy() if track_data else None

    def add_track(self, track_data: TrackData, track_type: TrackType = TRACK_TYPE_OTHER) -> TrackID:
        """
        Adds a new track to the application state.

        Generates a unique ID for the track. Ensures all default parameters are present.

        Args:
            track_data: A dictionary containing initial parameters for the track.
                        Must include 'original_audio' (can be None for uploads) and 'sr'.
                        Other parameters will default if not provided.
            track_type: The category/type of the track (e.g., Affirmation, Background).

        Returns:
            The unique TrackID assigned to the newly added track.

        Raises:
            ValueError: If track_data is not a dictionary or essential keys are missing.
        """
        if not isinstance(track_data, dict):
            raise ValueError("Invalid track_data provided: must be a dictionary.")

        # Generate a unique ID for the new track
        track_id = str(uuid.uuid4())
        logger.info(f"Attempting to add new track with generated ID: {track_id}")

        # Start with default parameters
        default_params = get_default_track_params()
        final_track_data = default_params.copy()

        # Update with provided parameters, ensuring type safety where possible
        final_track_data.update(track_data)

        # --- Specific Validations and Defaults ---
        # Ensure essential audio info is present or handled
        if "original_audio" not in track_data:
            # Allow 'original_audio' to be None initially, especially for uploads
            if track_data.get("source_type") != "upload":
                logger.warning(f"Track {track_id} added without 'original_audio'. Source: {track_data.get('source_type')}")
            final_track_data["original_audio"] = None  # Explicitly set to None if missing
        elif not isinstance(final_track_data["original_audio"], (np.ndarray, type(None))):
            logger.error(f"Invalid type for 'original_audio' in track {track_id}. Setting to None.")
            final_track_data["original_audio"] = None

        # Ensure sample rate is set, default to GLOBAL_SR if missing or invalid
        if not isinstance(final_track_data.get("sr"), int) or final_track_data.get("sr", 0) <= 0:
            logger.warning(f"Invalid or missing 'sr' for track {track_id}. Defaulting to {GLOBAL_SR}Hz.")
            final_track_data["sr"] = GLOBAL_SR

        # Set track type
        final_track_data["track_type"] = track_type

        # Ensure preview state is reset for the new track
        final_track_data["preview_temp_file_path"] = None
        final_track_data["preview_settings_hash"] = None
        final_track_data["update_counter"] = 0  # Initialize update counter

        # Add the finalized track data to the session state dictionary
        st.session_state[self.STATE_KEY][track_id] = final_track_data

        logger.info(f"Successfully added track ID: {track_id}, Name: '{final_track_data.get('name', 'N/A')}', Type: {track_type}, Source: {final_track_data.get('source_type')}")

        return track_id

    def delete_track(self, track_id: TrackID) -> bool:
        """
        Deletes a track from the state and cleans up its associated preview temp file.

        Args:
            track_id: The unique identifier of the track to delete.

        Returns:
            True if the track was found and deleted, False otherwise.
        """
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            track_name = tracks[track_id].get("name", "N/A")
            preview_path = tracks[track_id].get("preview_temp_file_path")

            # Attempt to delete the associated preview file from disk
            if preview_path and isinstance(preview_path, str) and os.path.exists(preview_path):
                try:
                    os.remove(preview_path)
                    logger.info(f"Deleted preview temp file '{preview_path}' for track {track_id} ('{track_name}')")
                except OSError as e:
                    logger.warning(f"Failed to delete preview temp file '{preview_path}' for track {track_id}: {e}")
            elif preview_path:
                logger.debug(f"Preview file path '{preview_path}' for track {track_id} not found or invalid. No deletion needed.")

            # Remove the track from the session state dictionary
            del st.session_state[self.STATE_KEY][track_id]
            logger.info(f"Deleted track ID: {track_id}, Name: '{track_name}' from state.")
            return True
        else:
            logger.warning(f"Attempted to delete non-existent track ID: {track_id}")
            return False

    def update_track_param(self, track_id: TrackID, param_name: str, value: Any):
        """
        Updates a specific parameter for a given track.

        Invalidates the preview hash if a parameter affecting the preview changes.
        Handles interactions between 'ultrasonic_shift' and 'pitch_shift'.

        Args:
            track_id: The unique identifier of the track to update.
            param_name: The name of the parameter to update (must be a valid key).
            value: The new value for the parameter.
        """
        tracks = self._get_tracks_dict()
        if track_id not in tracks:
            logger.warning(f"Attempted to update parameter '{param_name}' for non-existent track ID: {track_id}")
            return

        # Check if the parameter exists in the default set (valid parameter)
        default_params = get_default_track_params()
        if param_name not in default_params:
            logger.warning(f"Attempted to update invalid parameter '{param_name}' for track {track_id}. Ignoring.")
            return

        current_value = tracks[track_id].get(param_name)

        # Only update if the value has actually changed
        # Use np.isclose for float comparisons if applicable
        needs_update = False
        if isinstance(current_value, float) and isinstance(value, (float, int)):
            if not np.isclose(current_value, float(value)):
                needs_update = True
        elif isinstance(current_value, np.ndarray) or isinstance(value, np.ndarray):
            # For numpy arrays (like original_audio), check if they are different objects or sizes
            # A more robust check might involve checking content hash if performance allows
            if not np.array_equal(current_value, value):  # Basic check
                needs_update = True
        elif current_value != value:
            needs_update = True

        if not needs_update:
            logger.debug(f"Parameter '{param_name}' for track {track_id} already has value '{value}'. No update needed.")
            return

        logger.debug(f"Updating parameter '{param_name}' for track {track_id} from '{current_value}' to '{value}'.")
        st.session_state[self.STATE_KEY][track_id][param_name] = value

        # --- Invalidate Preview Hash if relevant parameter changed ---
        # Define parameters that directly affect the audio preview generation
        preview_affecting_params = ["volume", "speed_factor", "pitch_shift", "pan", "filter_type", "filter_cutoff", "reverse_audio", "ultrasonic_shift", "original_audio"]
        if param_name in preview_affecting_params:
            logger.debug(f"Parameter '{param_name}' changed for track {track_id}, invalidating preview hash.")
            st.session_state[self.STATE_KEY][track_id]["preview_settings_hash"] = None

        # --- Handle specific parameter interactions ---
        # If ultrasonic shift is turned ON, reset regular pitch shift to 0
        if param_name == "ultrasonic_shift" and value is True:
            if st.session_state[self.STATE_KEY][track_id]["pitch_shift"] != 0:
                logger.debug(f"Ultrasonic shift enabled for {track_id}, disabling regular pitch shift (setting to 0).")
                st.session_state[self.STATE_KEY][track_id]["pitch_shift"] = 0.0
                # Also invalidate preview hash as pitch shift changed implicitly
                st.session_state[self.STATE_KEY][track_id]["preview_settings_hash"] = None

        # If regular pitch shift is set to non-zero, turn ultrasonic shift OFF
        elif param_name == "pitch_shift" and not np.isclose(value, 0.0):
            if st.session_state[self.STATE_KEY][track_id]["ultrasonic_shift"] is True:
                logger.debug(f"Regular pitch shift set for {track_id}, disabling ultrasonic shift.")
                st.session_state[self.STATE_KEY][track_id]["ultrasonic_shift"] = False
                # Also invalidate preview hash as ultrasonic shift changed implicitly
                st.session_state[self.STATE_KEY][track_id]["preview_settings_hash"] = None

    def increment_update_counter(self, track_id: TrackID):
        """
        Increments the update counter for a track.

        Used primarily to force Streamlit widgets (like audix) to re-render
        when underlying data changes but the widget's direct inputs haven't.

        Args:
            track_id: The unique identifier of the track.
        """
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            current_counter = tracks[track_id].get("update_counter", 0)
            new_counter = current_counter + 1
            st.session_state[self.STATE_KEY][track_id]["update_counter"] = new_counter
            logger.debug(f"Incremented update_counter for track {track_id} to {new_counter}")
        else:
            logger.warning(f"Attempted to increment update counter for non-existent track ID: {track_id}")

    def get_loaded_track_names(self) -> List[str]:
        """Returns a list of names of all currently loaded tracks."""
        tracks = self.get_all_tracks()
        return [t.get("name", "Unnamed Track") for t in tracks.values()]

    def clear_all_tracks(self):
        """Removes all tracks from the state and cleans up associated preview files."""
        logger.info("Clearing all tracks from application state.")
        all_tracks = self.get_all_tracks()  # Get a copy before iterating
        deleted_count = 0
        for track_id in list(all_tracks.keys()):  # Iterate over keys of the copy
            if self.delete_track(track_id):
                deleted_count += 1
        logger.info(f"Cleared {deleted_count} tracks.")
        # Ensure the state key itself is still an empty dict
        st.session_state[self.STATE_KEY] = {}
