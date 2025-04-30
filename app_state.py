# app_state.py
# ==========================================
# Application State Management for MindMorph
# ==========================================

import logging
import os
import tempfile  # For handling temporary upload files
import uuid
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import streamlit as st

from audio_generators import generate_binaural_beats, generate_isochronic_tones, generate_noise, generate_solfeggio_frequency

# Import functions needed for get_full_audio
from audio_io import load_audio

# Import constants and default parameters from config.py
from config import (
    GLOBAL_SR,
    TRACK_SNIPPET_DURATION_S,
    TRACK_TYPE_OTHER,
    get_default_track_params,  # Keep get_default_track_params for now, but modify its usage
)

# Assuming TTSGenerator can be instantiated or accessed when needed
# If TTSGenerator needs state or complex init, it might need to be passed to AppState
from tts_generator import TTSGenerator

# Define type hints used within this module
TrackID = str
AudioData = np.ndarray  # Assuming AudioData is just a numpy array
# <<< ADDED BACK: Ensure TrackType is defined and exported >>>
TrackType = str  # Define TrackType as a string alias


# Define structure for source information
class SourceInfoUpload(TypedDict):
    type: str  # 'upload'
    temp_file_path: Optional[str]  # Path to the temporarily stored full uploaded file (Optional before re-upload)
    original_filename: str


class SourceInfoTTS(TypedDict):
    type: str  # 'tts'
    text: str
    # Add any specific TTS params if needed, e.g., voice, speed (if not track params)
    # tts_params: Dict[str, Any]


class SourceInfoNoise(TypedDict):
    type: str  # 'noise'
    noise_type: str
    target_duration_s: Optional[float]  # Hint for regeneration length
    # Volume is a track param, not source info


class SourceInfoFrequency(TypedDict):
    type: str  # 'frequency'
    freq_type: str  # 'binaural', 'isochronic', 'solfeggio'
    target_duration_s: Optional[float]  # Hint for regeneration length
    # Specific params needed for generation
    f_left: Optional[float]
    f_right: Optional[float]
    freq: Optional[float]
    carrier: Optional[float]
    pulse: Optional[float]


SourceInfo = SourceInfoUpload | SourceInfoTTS | SourceInfoNoise | SourceInfoFrequency


# Define structure for track data dictionary
class TrackDataDict(TypedDict):
    # Core data
    audio_snippet: Optional[AudioData]  # Stores the 30s snippet
    source_info: Optional[SourceInfo]  # Info to reload/regen full audio
    sr: int
    # Metadata & Controls
    name: str
    track_type: TrackType  # Use the defined TrackType alias
    volume: float
    mute: bool
    solo: bool
    speed_factor: float
    pitch_shift: int  # Semitones
    pan: float  # -1.0 (L) to 1.0 (R)
    filter_type: str
    filter_cutoff: float
    loop_to_fit: bool
    reverse_audio: bool
    ultrasonic_shift: bool
    # Preview state
    preview_temp_file_path: Optional[str]
    preview_settings_hash: Optional[str]
    update_counter: int


# Get a logger for this module
logger = logging.getLogger(__name__)


class AppState:
    """
    Manages the application's track data using Streamlit's session state.

    Handles adding, deleting, updating, and retrieving track information.
    Stores short audio snippets in memory and information to reload/regenerate
    full audio for export. Manages temporary files for uploads.
    """

    # The key used to store the track dictionary in st.session_state
    STATE_KEY = "mindmorph_tracks_state_v1"
    # Expose TrackDataDict for type hinting in other modules if needed
    TrackDataDict = TrackDataDict

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
        required default parameters based on the new structure.
        Removes deprecated keys like 'original_audio'.
        """
        logger.debug("Validating and cleaning existing track state.")
        tracks_dict = self.get_all_tracks()
        # Define required keys for the new structure
        required_keys = list(TrackDataDict.__annotations__.keys())

        for track_id, track_data in list(tracks_dict.items()):
            state_updated = False
            # Ensure all required keys exist (using TrackDataDict definition)
            for key in required_keys:
                if key not in track_data:
                    # Need a way to get default value for the specific key
                    # This is complex without a proper default dict/dataclass
                    # For now, log a warning. Loading old state might be broken.
                    logger.warning(f"Track '{track_data.get('name', track_id)}' missing key '{key}'. State might be inconsistent.")
                    # Attempting to add a basic default might be needed, e.g. None
                    # st.session_state[self.STATE_KEY][track_id][key] = None # Be careful with this
                    # state_updated = True

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
                    logger.debug(f"Removing deprecated key '{key}' for track {track_id}.")
                    del st.session_state[self.STATE_KEY][track_id][key]
                    state_updated = True

            # Clean up preview file path if the hash is missing
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

    def _get_tracks_dict(self) -> Dict[TrackID, TrackDataDict]:
        """Safely retrieves the tracks dictionary from session state."""
        return st.session_state.get(self.STATE_KEY, {})

    def get_all_tracks(self) -> Dict[TrackID, TrackDataDict]:
        """Returns a copy of the entire tracks dictionary."""
        return self._get_tracks_dict().copy()

    def get_track(self, track_id: TrackID) -> Optional[TrackDataDict]:
        """
        Retrieves the data for a specific track ID.

        Args:
            track_id: The unique identifier of the track.

        Returns:
            A copy of the track data dictionary, or None if the track ID is not found.
        """
        track_data = self._get_tracks_dict().get(track_id)
        return track_data.copy() if track_data else None

    def get_track_snippet(self, track_id: TrackID) -> Optional[AudioData]:
        """Retrieves the audio snippet for a specific track."""
        track_data = self.get_track(track_id)
        if track_data:
            return track_data.get("audio_snippet")
        return None

    def add_track(self, audio_snippet: Optional[AudioData], source_info: SourceInfo, sr: int, initial_params: Optional[Dict[str, Any]] = None) -> TrackID:
        """
        Adds a new track to the application state using snippet and source info.

        Generates a unique ID for the track.

        Args:
            audio_snippet: The initial 30-second audio data (NumPy array) or None.
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
            logger.warning(f"Invalid sample rate {sr} provided for new track. Defaulting to {GLOBAL_SR}.")
            sr = GLOBAL_SR
        if audio_snippet is not None and not isinstance(audio_snippet, np.ndarray):
            logger.error("Invalid audio_snippet provided (must be numpy array or None). Setting snippet to None.")
            audio_snippet = None

        track_id = str(uuid.uuid4())
        logger.info(f"Attempting to add new track with generated ID: {track_id}, Source Type: {source_info['type']}")

        # Define default structure based on TrackDataDict
        new_track: TrackDataDict = {
            "audio_snippet": audio_snippet,
            "source_info": source_info,
            "sr": sr,
            "name": "New Track",
            "track_type": TRACK_TYPE_OTHER,  # Default type
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

        # Override defaults with any provided initial parameters
        if initial_params:
            for key, value in initial_params.items():
                if key in new_track:  # Only update keys that exist in our defined structure
                    new_track[key] = value
                else:
                    logger.warning(f"Ignoring unknown initial parameter '{key}' for track {track_id}")

        # Add the finalized track data to the session state dictionary
        st.session_state[self.STATE_KEY][track_id] = new_track

        logger.info(f"Successfully added track ID: {track_id}, Name: '{new_track['name']}', Type: {new_track['track_type']}")
        return track_id

    def delete_track(self, track_id: TrackID) -> bool:
        """
        Deletes a track from the state and cleans up associated preview and temporary upload files.

        Args:
            track_id: The unique identifier of the track to delete.

        Returns:
            True if the track was found and deleted, False otherwise.
        """
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            track_data = tracks[track_id]
            track_name = track_data.get("name", "N/A")
            preview_path = track_data.get("preview_temp_file_path")
            source_info = track_data.get("source_info")

            # --- Cleanup Preview File ---
            if preview_path and isinstance(preview_path, str) and os.path.exists(preview_path):
                try:
                    os.remove(preview_path)
                    logger.info(f"Deleted preview temp file '{preview_path}' for track {track_id} ('{track_name}')")
                except OSError as e:
                    logger.warning(f"Failed to delete preview temp file '{preview_path}' for track {track_id}: {e}")

            # --- Cleanup Temporary Upload File ---
            if source_info and source_info.get("type") == "upload":
                temp_upload_path = source_info.get("temp_file_path")
                if temp_upload_path and isinstance(temp_upload_path, str) and os.path.exists(temp_upload_path):
                    try:
                        os.remove(temp_upload_path)
                        logger.info(f"Deleted temporary upload file '{temp_upload_path}' for track {track_id} ('{track_name}')")
                    except OSError as e:
                        logger.warning(f"Failed to delete temporary upload file '{temp_upload_path}' for track {track_id}: {e}")

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
        Prevents direct modification of 'audio_snippet' or 'source_info'.

        Args:
            track_id: The unique identifier of the track to update.
            param_name: The name of the parameter to update (must be a valid key).
            value: The new value for the parameter.
        """
        tracks = self._get_tracks_dict()
        if track_id not in tracks:
            logger.warning(f"Attempted to update parameter '{param_name}' for non-existent track ID: {track_id}")
            return

        # Prevent modification of core data via this method
        if param_name in ["audio_snippet", "source_info", "sr"]:
            logger.error(f"Attempted to update protected parameter '{param_name}' for track {track_id} via update_track_param. Ignoring.")
            return

        # Check if the parameter exists in the defined structure
        if param_name not in TrackDataDict.__annotations__:
            logger.warning(f"Attempted to update invalid parameter '{param_name}' for track {track_id}. Ignoring.")
            return

        current_value = tracks[track_id].get(param_name)

        # Only update if the value has actually changed
        needs_update = False
        if isinstance(current_value, float) and isinstance(value, (float, int)):
            if not np.isclose(current_value, float(value)):
                needs_update = True
        # Cannot compare audio_snippet here as it's protected
        elif current_value != value:
            needs_update = True

        if not needs_update:
            # logger.debug(f"Parameter '{param_name}' for track {track_id} already has value '{value}'. No update needed.")
            return

        logger.debug(f"Updating parameter '{param_name}' for track {track_id} from '{current_value}' to '{value}'.")
        st.session_state[self.STATE_KEY][track_id][param_name] = value

        # --- Invalidate Preview Hash if relevant parameter changed ---
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
        ]  # Added loop_to_fit
        if param_name in preview_affecting_params:
            logger.debug(f"Parameter '{param_name}' changed for track {track_id}, invalidating preview hash.")
            st.session_state[self.STATE_KEY][track_id]["preview_settings_hash"] = None

        # --- Handle specific parameter interactions ---
        if param_name == "ultrasonic_shift" and value is True:
            if st.session_state[self.STATE_KEY][track_id]["pitch_shift"] != 0:
                logger.debug(f"Ultrasonic shift enabled for {track_id}, disabling regular pitch shift (setting to 0).")
                st.session_state[self.STATE_KEY][track_id]["pitch_shift"] = 0.0
                st.session_state[self.STATE_KEY][track_id]["preview_settings_hash"] = None
        elif param_name == "pitch_shift" and not np.isclose(value, 0.0):
            if st.session_state[self.STATE_KEY][track_id]["ultrasonic_shift"] is True:
                logger.debug(f"Regular pitch shift set for {track_id}, disabling ultrasonic shift.")
                st.session_state[self.STATE_KEY][track_id]["ultrasonic_shift"] = False
                st.session_state[self.STATE_KEY][track_id]["preview_settings_hash"] = None

    def increment_update_counter(self, track_id: TrackID):
        """Increments the update counter for a track."""
        tracks = self._get_tracks_dict()
        if track_id in tracks:
            current_counter = tracks[track_id].get("update_counter", 0)
            new_counter = current_counter + 1
            st.session_state[self.STATE_KEY][track_id]["update_counter"] = new_counter
            # logger.debug(f"Incremented update_counter for track {track_id} to {new_counter}")
        else:
            logger.warning(f"Attempted to increment update counter for non-existent track ID: {track_id}")

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
            if self.delete_track(track_id):  # delete_track now handles temp file cleanup
                deleted_count += 1
        logger.info(f"Cleared {deleted_count} tracks.")
        st.session_state[self.STATE_KEY] = {}

    # --- New Method to Get Full Audio ---
    def get_full_audio(self, track_id: TrackID, required_duration_samples: Optional[int] = None) -> Optional[Tuple[AudioData, int]]:
        """
        Loads or regenerates the FULL audio data for a track based on its source_info.

        Args:
            track_id: The ID of the track.
            required_duration_samples: The target duration in samples needed for looping
                                      generated tracks (noise, frequency). If None, uses
                                      a default or inherent duration.

        Returns:
            A tuple containing (AudioData, sample_rate), or None if loading/generation fails.
        """
        track_data = self.get_track(track_id)
        if not track_data:
            logger.error(f"get_full_audio: Track {track_id} not found.")
            return None

        source_info: Optional[SourceInfo] = track_data.get("source_info")
        if not source_info:
            logger.error(f"get_full_audio: Missing source_info for track {track_id}.")
            return None

        source_type = source_info.get("type")
        sr = track_data.get("sr", GLOBAL_SR)
        full_audio: Optional[AudioData] = None
        final_sr: int = sr

        logger.info(f"get_full_audio: Loading/Regenerating full audio for track {track_id} (Type: {source_type})")

        try:
            if source_type == "upload":
                temp_file_path = source_info.get("temp_file_path")
                if temp_file_path and os.path.exists(temp_file_path):
                    # Load full audio from the temporary file
                    full_audio, loaded_sr = load_audio(temp_file_path, target_sr=sr, duration=None)
                    if loaded_sr:
                        final_sr = loaded_sr
                else:
                    logger.error(f"get_full_audio: Temporary upload file not found for track {track_id} at path: {temp_file_path}")
                    st.error(f"Source file for track '{track_data.get('name')}' missing. Please remove and re-add the track.", icon="⚠️")
                    return None

            elif source_type == "tts":
                text = source_info.get("text")
                if text:
                    tts_gen = TTSGenerator()  # Assuming simple instantiation works
                    full_audio, loaded_sr = tts_gen.generate(text)
                    if loaded_sr:
                        final_sr = loaded_sr
                else:
                    logger.error(f"get_full_audio: Missing text in source_info for TTS track {track_id}")

            elif source_type == "noise":
                noise_type = source_info.get("noise_type")
                target_duration_s = source_info.get("target_duration_s")  # Get hint
                # Use required_duration if provided (for looping), else use hint or default
                duration_s = (required_duration_samples / sr) if required_duration_samples is not None and sr > 0 else target_duration_s
                if duration_s is None:
                    duration_s = 300  # Default if no hint and no requirement

                if noise_type:
                    full_audio = generate_noise(noise_type, duration_s, sr, volume=1.0)
                    final_sr = sr
                else:
                    logger.error(f"get_full_audio: Missing noise_type for noise track {track_id}")

            elif source_type == "frequency":
                freq_type = source_info.get("freq_type")
                target_duration_s = source_info.get("target_duration_s")  # Get hint
                duration_s = (required_duration_samples / sr) if required_duration_samples is not None and sr > 0 else target_duration_s
                if duration_s is None:
                    duration_s = 300  # Default if no hint and no requirement

                if freq_type:
                    gen_volume = 1.0
                    if freq_type == "binaural":
                        f_left = source_info.get("f_left")
                        f_right = source_info.get("f_right")
                        if f_left is not None and f_right is not None:
                            full_audio = generate_binaural_beats(duration_s, f_left, f_right, sr, gen_volume)
                    elif freq_type == "isochronic":
                        carrier = source_info.get("carrier")
                        pulse = source_info.get("pulse")
                        if carrier is not None and pulse is not None:
                            full_audio = generate_isochronic_tones(duration_s, carrier, pulse, sr, gen_volume)
                    elif freq_type == "solfeggio":
                        freq = source_info.get("freq")
                        if freq is not None:
                            full_audio = generate_solfeggio_frequency(duration_s, freq, sr, gen_volume)
                    else:
                        logger.error(f"get_full_audio: Unknown frequency type '{freq_type}' for track {track_id}")
                    final_sr = sr
                else:
                    logger.error(f"get_full_audio: Missing freq_type for frequency track {track_id}")

            else:
                logger.error(f"get_full_audio: Unknown source_type '{source_type}' for track {track_id}")

        except Exception as e:
            logger.exception(f"get_full_audio: Error loading/regenerating full audio for track {track_id} (Type: {source_type})")
            st.error(f"Error preparing audio for track '{track_data.get('name')}': {e}", icon="🔥")
            return None

        if full_audio is None:
            logger.error(f"get_full_audio: Failed to get full audio data for track {track_id}")
            return None

        logger.info(f"get_full_audio: Successfully retrieved full audio for track {track_id} ({len(full_audio) / final_sr:.2f}s)")
        return full_audio, final_sr
