# app_state.py
# ==========================================
# Application State Management for MindMorph
# ==========================================

import logging
import os
import tempfile  # Still needed for cleanup in delete_track
import uuid
from typing import Any, Dict, List, Optional, Tuple  # Keep basic types

import numpy as np
import streamlit as st

# <<< MODIFIED: Import definitions from the new definitions file >>>
from audio_utils.audio_state_definitions import (
    AudioData,
    SourceInfo,
    SourceInfoUpload,
    TrackDataDict,
    TrackID,
)

# Import constants and default parameters from config.py
from config import GLOBAL_SR, TRACK_TYPE_OTHER

# <<< REMOVED imports for audio generation/loading >>>
# from audio_utils.audio_io import load_audio
# from audio_utils.audio_generators import (...)
# from tts_generator import TTSGenerator


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
    STATE_KEY = "mindmorph_tracks_state_v1"
    # Expose TrackDataDict for type hinting in other modules if needed
    TrackDataDict = TrackDataDict

    def __init__(self):
        """
        Initializes the AppState manager. Ensures the state dictionary exists
        in st.session_state and performs basic validation/cleanup.
        """
        if self.STATE_KEY not in st.session_state:
            logger.info(
                f"Initializing new application state under key '{self.STATE_KEY}'."
            )
            st.session_state[self.STATE_KEY] = {}
        else:
            logger.debug(
                f"Using existing application state from key '{self.STATE_KEY}'."
            )
            self._validate_and_clean_state()

    def _validate_and_clean_state(self):
        """
        Iterates through existing tracks in the state, ensuring they have required keys
        and cleaning up preview file paths if inconsistent.
        Removes deprecated keys.
        """
        logger.debug("Validating and cleaning existing track state.")
        tracks_dict = self.get_all_tracks()
        required_keys = list(TrackDataDict.__annotations__.keys())

        for track_id, track_data in list(tracks_dict.items()):
            state_updated = False
            for key in required_keys:
                if key not in track_data:
                    logger.warning(
                        f"Track '{track_data.get('name', track_id)}' missing key '{key}'. State might be inconsistent."
                    )
                    # Attempting to add a default might be risky without knowing the type
                    # Consider adding basic defaults like None or 0 based on expected type if issues persist

            # Remove deprecated keys
            deprecated_keys = [
                "original_audio",
                "tts_text",
                "gen_duration",
                "gen_freq_left",
                "gen_freq_right",
                "gen_freq",
                "gen_carrier_freq",
                "gen_pulse_freq",
                "gen_noise_type",
                "gen_volume",
                "source_type",
                "original_filename",
            ]
            for key in deprecated_keys:
                if key in track_data:
                    logger.debug(
                        f"Removing deprecated key '{key}' for track {track_id}."
                    )
                    if (
                        track_id in st.session_state[self.STATE_KEY]
                    ):  # Check if track still exists
                        del st.session_state[self.STATE_KEY][track_id][key]
                        state_updated = True

            # Clean up preview file path if the hash is missing
            if track_data.get("preview_settings_hash") is None:
                old_path = track_data.get("preview_temp_file_path")
                if old_path:
                    logger.warning(
                        f"Preview hash missing for track {track_id}. Invalidating preview file path."
                    )
                    if (
                        track_id in st.session_state[self.STATE_KEY]
                    ):  # Check if track still exists
                        st.session_state[self.STATE_KEY][track_id][
                            "preview_temp_file_path"
                        ] = None
                        state_updated = True
                        if os.path.exists(old_path):
                            try:
                                os.remove(old_path)
                                logger.info(
                                    f"Cleaned up orphaned preview file: {old_path}"
                                )
                            except OSError as e:
                                logger.warning(
                                    f"Could not clean up orphaned preview file {old_path}: {e}"
                                )

            if state_updated:
                logger.debug(f"State updated during validation for track {track_id}.")

    def _get_tracks_dict(self) -> Dict[TrackID, TrackDataDict]:
        """Safely retrieves the tracks dictionary from session state."""
        return st.session_state.get(self.STATE_KEY, {})

    def get_all_tracks(self) -> Dict[TrackID, TrackDataDict]:
        """Returns a copy of the entire tracks dictionary."""
        return self._get_tracks_dict().copy()

    def get_track(self, track_id: TrackID) -> Optional[TrackDataDict]:
        """Retrieves the data structure for a specific track ID."""
        track_data = self._get_tracks_dict().get(track_id)
        return track_data.copy() if track_data else None

    def get_track_snippet(self, track_id: TrackID) -> Optional[AudioData]:
        """Retrieves the audio snippet for a specific track."""
        track_data = self.get_track(track_id)
        if track_data:
            return track_data.get("audio_snippet")
        return None

    def add_track(
        self,
        audio_snippet: Optional[AudioData],
        source_info: SourceInfo,
        sr: int,
        initial_params: Optional[Dict[str, Any]] = None,
    ) -> TrackID:
        """
        Adds a new track data structure to the application state.

        Args:
            audio_snippet: The initial audio data snippet (NumPy array) or None.
            source_info: Dictionary containing information to reload/regenerate full audio.
            sr: The sample rate of the audio snippet.
            initial_params: Optional dictionary with other initial parameters (name, type, etc.).

        Returns:
            The unique TrackID assigned to the newly added track.
        Raises:
            ValueError: If source_info is invalid or essential data is missing.
        """
        if not isinstance(source_info, dict) or "type" not in source_info:
            raise ValueError("Invalid source_info provided.")
        if not isinstance(sr, int) or sr <= 0:
            logger.warning(
                f"Invalid sample rate {sr} provided for new track. Defaulting to {GLOBAL_SR}."
            )
            sr = GLOBAL_SR
        if audio_snippet is not None and not isinstance(audio_snippet, np.ndarray):
            logger.error(
                "Invalid audio_snippet provided (must be numpy array or None). Setting snippet to None."
            )
            audio_snippet = None

        track_id = str(uuid.uuid4())
        logger.info(
            f"Attempting to add new track with generated ID: {track_id}, Source Type: {source_info['type']}"
        )

        # Define default structure based on TrackDataDict
        new_track: TrackDataDict = {
            "audio_snippet": audio_snippet,
            "source_info": source_info,
            "sr": sr,
            "name": "New Track",
            "track_type": TRACK_TYPE_OTHER,
            "volume": 1.0,
            "mute": False,
            "solo": False,
            "speed_factor": 1.0,
            "pitch_shift": 0,
            "pan": 0.0,
            "filter_type": "off",
            "filter_cutoff": 8000.0,
            "loop_to_fit": False,
            "reverse_audio": False,
            "ultrasonic_shift": False,
            "preview_temp_file_path": None,
            "preview_settings_hash": None,
            "update_counter": 0,
        }

        if initial_params:
            for key, value in initial_params.items():
                if key in new_track:
                    new_track[key] = value
                else:
                    logger.warning(
                        f"Ignoring unknown initial parameter '{key}' for track {track_id}"
                    )

        st.session_state[self.STATE_KEY][track_id] = new_track
        logger.info(
            f"Successfully added track ID: {track_id}, Name: '{new_track['name']}', Type: {new_track['track_type']}"
        )
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
            # Use type guard for safer access
            if isinstance(source_info, dict) and source_info.get("type") == "upload":
                # Cast to the specific type after checking 'type' for type checker happiness
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

            # Remove the track from the session state dictionary
            del st.session_state[self.STATE_KEY][track_id]
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

        if param_name in ["audio_snippet", "source_info", "sr"]:
            logger.error(
                f"Attempted to update protected parameter '{param_name}' via update_track_param. Ignoring."
            )
            return

        if param_name not in TrackDataDict.__annotations__:
            logger.warning(
                f"Attempted to update invalid parameter '{param_name}' for track {track_id}. Ignoring."
            )
            return

        current_value = tracks[track_id].get(param_name)
        needs_update = False
        if isinstance(current_value, float) and isinstance(value, (float, int)):
            if not np.isclose(current_value, float(value)):
                needs_update = True
        elif current_value != value:
            needs_update = True

        if not needs_update:
            return

        logger.debug(
            f"Updating parameter '{param_name}' for track {track_id} from '{current_value}' to '{value}'."
        )
        st.session_state[self.STATE_KEY][track_id][param_name] = value

        preview_affecting_params = [
            "volume",
            "speed_factor",
            "pitch_shift",
            "pan",
            "filter_type",
            "filter_cutoff",
            "reverse_audio",
            "ultrasonic_shift",
            "loop_to_fit",
        ]
        if param_name in preview_affecting_params:
            logger.debug(
                f"Parameter '{param_name}' changed, invalidating preview hash."
            )
            st.session_state[self.STATE_KEY][track_id]["preview_settings_hash"] = None

        # Handle specific parameter interactions
        if param_name == "ultrasonic_shift" and value is True:
            if st.session_state[self.STATE_KEY][track_id]["pitch_shift"] != 0:
                logger.debug(
                    f"Ultrasonic shift enabled, disabling regular pitch shift."
                )
                st.session_state[self.STATE_KEY][track_id]["pitch_shift"] = 0.0
                st.session_state[self.STATE_KEY][track_id][
                    "preview_settings_hash"
                ] = None
        elif param_name == "pitch_shift" and not np.isclose(value, 0.0):
            if st.session_state[self.STATE_KEY][track_id]["ultrasonic_shift"] is True:
                logger.debug(f"Regular pitch shift set, disabling ultrasonic shift.")
                st.session_state[self.STATE_KEY][track_id]["ultrasonic_shift"] = False
                st.session_state[self.STATE_KEY][track_id][
                    "preview_settings_hash"
                ] = None

    def increment_update_counter(self, track_id: TrackID):
        """Increments the update counter for a track."""
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            current_counter = tracks[track_id].get("update_counter", 0)
            st.session_state[self.STATE_KEY][track_id]["update_counter"] = (
                current_counter + 1
            )
        else:
            logger.warning(
                f"Attempted increment update counter for non-existent track ID: {track_id}"
            )

    def get_loaded_track_names(self) -> List[str]:
        """Returns a list of names of all currently loaded tracks."""
        tracks = self.get_all_tracks()
        return [t.get("name", "Unnamed Track") for t in tracks.values()]

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

    # <<< REMOVED get_full_audio method >>>
    # This logic is now in audio_loader.py
