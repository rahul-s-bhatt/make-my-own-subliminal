# project_handler.py
# ==========================================
# Project Loading Logic for MindMorph
# ==========================================

import json
import logging
from typing import TYPE_CHECKING

import streamlit as st

# --- Type Hinting ---
# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from app_state import AppState
    from audio_processing import AudioData
    from tts_generator import TTSGenerator

# Import necessary components from other modules
from audio_generators import (
    generate_binaural_beats,
    generate_isochronic_tones,
    generate_noise,
    generate_solfeggio_frequency,
)
from config import (
    GLOBAL_SR,
    PROJECT_FILE_VERSION,
    TRACK_TYPE_OTHER,
    get_default_track_params,
)

logger = logging.getLogger(__name__)


class ProjectHandler:
    """Handles loading project data from uploaded files."""

    def __init__(self, app_state: "AppState", tts_generator: "TTSGenerator"):
        """
        Initializes the ProjectHandler.

        Args:
            app_state: The application state manager instance.
            tts_generator: The TTS generator instance.
        """
        self.app_state = app_state
        self.tts_generator = tts_generator
        logger.debug("ProjectHandler initialized.")

    def load_project(self):
        """Loads project data from session state if requested."""
        logger.info("Checking for project load request.")
        if st.session_state.get("project_load_requested", False):
            logger.info("Project load requested. Processing uploaded file data.")
            loaded_data = st.session_state.get("uploaded_project_file_data")

            # Reset flags immediately
            st.session_state.project_load_requested = False
            st.session_state.uploaded_project_file_data = None  # Clear the loaded data

            if loaded_data:
                try:
                    # Decode and parse the JSON data
                    project_content = json.loads(loaded_data.decode("utf-8"))

                    # --- Validate Project Structure ---
                    if not isinstance(project_content, dict) or "tracks" not in project_content or "version" not in project_content:
                        raise ValueError("Invalid project file structure. Missing 'version' or 'tracks'.")

                    # Optional: Check version compatibility
                    loaded_version = project_content.get("version", "0.0")
                    if loaded_version != PROJECT_FILE_VERSION:
                        logger.warning(f"Loading project version {loaded_version}, current app version is {PROJECT_FILE_VERSION}. Compatibility not guaranteed.")
                        st.warning(f"Loading project from older version ({loaded_version}). Some settings might be lost or default.")

                    loaded_tracks_data = project_content.get("tracks", {})
                    if not isinstance(loaded_tracks_data, dict):
                        raise ValueError("Invalid 'tracks' data in project file.")

                    logger.info(f"Valid project structure found (Version: {loaded_version}). Clearing current state and loading {len(loaded_tracks_data)} tracks.")

                    # Clear existing tracks before loading
                    self.app_state.clear_all_tracks()

                    # --- Reconstruct Tracks ---
                    with st.spinner("Reconstructing project tracks..."):
                        tracks_needing_upload = []
                        for old_track_id, track_data in loaded_tracks_data.items():
                            if not isinstance(track_data, dict):
                                logger.warning(f"Skipping invalid track data entry (not a dict) for old ID {old_track_id}.")
                                continue

                            logger.debug(f"Loading track '{track_data.get('name', 'N/A')}' (Old ID: {old_track_id})")
                            source_type = track_data.get("source_type", "unknown")
                            track_type = track_data.get("track_type", TRACK_TYPE_OTHER)
                            reconstructed_audio: "AudioData" | None = None  # Use type hint

                            # --- Regenerate Audio Based on Source Type ---
                            try:
                                if source_type == "tts" and "tts_text" in track_data:
                                    logger.info(f"Regenerating TTS for track '{track_data.get('name')}'")
                                    reconstructed_audio, _ = self.tts_generator.generate(track_data["tts_text"])
                                elif source_type == "noise" and "gen_noise_type" in track_data:
                                    noise_type = track_data["gen_noise_type"]
                                    duration = track_data.get("gen_duration", 60)
                                    volume = track_data.get("gen_volume", 0.5)
                                    logger.info(f"Regenerating {noise_type} ({duration}s, vol={volume}) for track '{track_data.get('name')}'")
                                    reconstructed_audio = generate_noise(noise_type, duration, GLOBAL_SR, volume)
                                elif source_type == "binaural" and "gen_freq_left" in track_data:
                                    duration = track_data.get("gen_duration", 60)
                                    f_left = track_data["gen_freq_left"]
                                    f_right = track_data.get("gen_freq_right", f_left + 10.0)  # Default beat
                                    volume = track_data.get("gen_volume", 0.3)
                                    logger.info(f"Regenerating Binaural ({duration}s, L={f_left}, R={f_right}, vol={volume}) for track '{track_data.get('name')}'")
                                    reconstructed_audio = generate_binaural_beats(duration, f_left, f_right, GLOBAL_SR, volume)
                                elif source_type == "solfeggio" and "gen_freq" in track_data:
                                    duration = track_data.get("gen_duration", 60)
                                    freq = track_data["gen_freq"]
                                    volume = track_data.get("gen_volume", 0.3)
                                    logger.info(f"Regenerating Solfeggio ({duration}s, F={freq}, vol={volume}) for track '{track_data.get('name')}'")
                                    reconstructed_audio = generate_solfeggio_frequency(duration, freq, GLOBAL_SR, volume)
                                elif source_type == "isochronic" and "gen_carrier_freq" in track_data:
                                    duration = track_data.get("gen_duration", 60)
                                    carrier = track_data["gen_carrier_freq"]
                                    pulse = track_data.get("gen_pulse_freq", 10.0)  # Default pulse
                                    volume = track_data.get("gen_volume", 0.4)
                                    logger.info(f"Regenerating Isochronic ({duration}s, C={carrier}, P={pulse}, vol={volume}) for track '{track_data.get('name')}'")
                                    reconstructed_audio = generate_isochronic_tones(duration, carrier, pulse, GLOBAL_SR, volume)
                                elif source_type == "upload":
                                    filename = track_data.get("original_filename", "Unknown File")
                                    logger.warning(f"Track '{track_data.get('name')}' is an upload ('{filename}'). Audio data needs re-upload.")
                                    tracks_needing_upload.append(filename)
                                    reconstructed_audio = None
                                else:
                                    logger.warning(f"Unknown or missing source_type ('{source_type}') for track '{track_data.get('name')}'. Cannot reconstruct audio.")
                                    reconstructed_audio = None

                            except Exception as e_gen:
                                logger.error(f"Error regenerating audio for track '{track_data.get('name')}': {e_gen}")
                                st.warning(f"Could not regenerate audio for track '{track_data.get('name')}'.")
                                reconstructed_audio = None

                            # --- Add Track to State ---
                            track_data["original_audio"] = reconstructed_audio
                            track_data["sr"] = GLOBAL_SR
                            # Ensure all default keys exist
                            final_track_data_for_load = get_default_track_params()
                            final_track_data_for_load.update(track_data)

                            try:
                                self.app_state.add_track(final_track_data_for_load, track_type=track_type)
                            except ValueError as e_add:
                                logger.error(f"Failed to add loaded track '{track_data.get('name')}': {e_add}")
                                st.error(f"Failed to load track '{track_data.get('name')}'.")

                    st.success("Project loaded successfully!")
                    if tracks_needing_upload:
                        st.warning(f"Please re-upload the following audio file(s): {', '.join(tracks_needing_upload)}")

                except json.JSONDecodeError:
                    logger.error("Failed to decode project file. Invalid JSON.")
                    st.error("Failed to load project: Invalid project file format (not valid JSON).")
                except ValueError as e_val:
                    logger.error(f"Invalid project file content: {e_val}")
                    st.error(f"Failed to load project: {e_val}")
                except Exception as e:
                    logger.exception("An unexpected error occurred while loading the project.")
                    st.error(f"An error occurred while loading the project: {e}")
            else:
                logger.debug("No project load requested or no file data found.")
