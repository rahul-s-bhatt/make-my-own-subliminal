# project_handler.py
# ==========================================
# Project Loading Logic for MindMorph
# ==========================================

import json
import logging
from typing import TYPE_CHECKING, Optional

import streamlit as st

# --- Type Hinting ---
if TYPE_CHECKING:
    # <<< MODIFIED: Import types from definitions file >>>
    from app_state import AppState  # Keep this import
    from audio_utils.audio_state_definitions import SourceInfo

# Import necessary components from other modules
# <<< MODIFIED: Import SourceInfo types from definitions file >>>
from audio_utils.audio_state_definitions import (
    SourceInfoFrequency,
    SourceInfoNoise,
    SourceInfoTTS,
    SourceInfoUpload,
)
from config import GLOBAL_SR, PROJECT_FILE_VERSION

# <<< REMOVED import of SourceInfo types from app_state >>>
# from app_state import (SourceInfoUpload, SourceInfoTTS, SourceInfoNoise,
#                        SourceInfoFrequency)


logger = logging.getLogger(__name__)


class ProjectHandler:
    """Handles loading project data from uploaded files."""

    def __init__(self, app_state: "AppState"):
        """
        Initializes the ProjectHandler.

        Args:
            app_state: The application state manager instance.
        """
        self.app_state = app_state
        logger.debug("ProjectHandler initialized.")

    def load_project(self):
        """
        Loads project data from session state if requested.
        Reconstructs track state with source_info but without audio snippets.
        Attempts to infer source_type for older project files.
        Aborts loading and clears state if critical errors occur for any track.
        Prompts user to re-upload necessary files on success.
        Clears loading state and file uploader state after attempt.
        """
        logger.info("Checking for project load request.")
        load_requested = st.session_state.get("project_load_requested", False)
        loaded_data = st.session_state.get("uploaded_project_file_data")
        st.session_state.get("uploaded_project_file_id")
        load_uploader_key = "sidebar_load_project_uploader"

        if load_requested and loaded_data:
            logger.info("Project load requested. Processing uploaded file data.")
            load_successful = False  # Flag to track success
            load_errors_found = False  # Flag to track critical errors
            tracks_needing_upload = []  # Reset list for this load attempt
            temp_loaded_tracks = {}  # Store successfully processed tracks temporarily

            try:
                project_content = json.loads(loaded_data.decode("utf-8"))

                # --- Validation and Track Processing ---
                if (
                    not isinstance(project_content, dict)
                    or "tracks" not in project_content
                    or "version" not in project_content
                ):
                    raise ValueError(
                        "Invalid project file structure. Missing 'version' or 'tracks'."
                    )

                loaded_version = project_content.get("version", "0.0")
                if loaded_version != PROJECT_FILE_VERSION:
                    logger.warning(
                        f"Loading project version {loaded_version}, current app version is {PROJECT_FILE_VERSION}. Compatibility not guaranteed."
                    )
                    st.warning(
                        f"Loading project from older version ({loaded_version}). Some settings might be lost or default."
                    )

                loaded_tracks_data = project_content.get("tracks", {})
                if not isinstance(loaded_tracks_data, dict):
                    raise ValueError("Invalid 'tracks' data in project file.")

                logger.info(
                    f"Valid project structure found (Version: {loaded_version}). Preparing to load {len(loaded_tracks_data)} tracks."
                )

                with st.spinner("Validating project tracks..."):
                    for old_track_id, loaded_track_params in loaded_tracks_data.items():
                        if load_errors_found:
                            break

                        if not isinstance(loaded_track_params, dict):
                            logger.error(
                                f"Invalid track data entry (not a dict) for old ID {old_track_id}. Aborting load."
                            )
                            load_errors_found = True
                            st.error(
                                "Project load failed: Invalid track data found in file."
                            )
                            break

                        track_name = loaded_track_params.get(
                            "name", f"Track {old_track_id[:6]}"
                        )
                        logger.debug(
                            f"Validating track '{track_name}' (Old ID: {old_track_id})"
                        )

                        source_type = loaded_track_params.get("source_type", "unknown")
                        source_info: Optional["SourceInfo"] = None
                        sr = loaded_track_params.get("sr", GLOBAL_SR)

                        if source_type == "unknown":
                            logger.warning(
                                f"Track '{track_name}' has source_type 'unknown'. Attempting inference..."
                            )
                            if "original_filename" in loaded_track_params:
                                source_type = "upload"
                            elif "tts_text" in loaded_track_params:
                                source_type = "tts"
                            elif "gen_noise_type" in loaded_track_params:
                                source_type = "noise"
                            elif (
                                "gen_freq_left" in loaded_track_params
                                or "gen_freq_right" in loaded_track_params
                            ):
                                source_type = "binaural"
                            elif "gen_freq" in loaded_track_params:
                                source_type = "solfeggio"
                            elif (
                                "gen_carrier_freq" in loaded_track_params
                                or "gen_pulse_freq" in loaded_track_params
                            ):
                                source_type = "isochronic"
                            else:
                                logger.error(
                                    f"Could not infer source_type for track '{track_name}'. Aborting load."
                                )
                                load_errors_found = True
                                st.error(
                                    f"Project load failed: Could not determine source type for track '{track_name}'."
                                )
                                break

                        try:
                            # Construct SourceInfo
                            if source_type == "upload":
                                filename = loaded_track_params.get("original_filename")
                                if filename:
                                    tracks_needing_upload.append(filename)
                                    source_info = SourceInfoUpload(
                                        type="upload",
                                        temp_file_path=None,
                                        original_filename=filename,
                                    )
                                else:
                                    raise ValueError("Missing original_filename")
                            elif source_type == "tts":
                                text = loaded_track_params.get("tts_text")
                                if text:
                                    source_info = SourceInfoTTS(type="tts", text=text)
                                else:
                                    raise ValueError("Missing tts_text")
                            elif source_type == "noise":
                                noise_type = loaded_track_params.get("gen_noise_type")
                                target_duration = loaded_track_params.get(
                                    "gen_duration", 300
                                )
                                if noise_type:
                                    source_info = SourceInfoNoise(
                                        type="noise",
                                        noise_type=noise_type,
                                        target_duration_s=target_duration,
                                    )
                                else:
                                    raise ValueError("Missing gen_noise_type")
                            elif source_type in [
                                "binaural",
                                "binaural_preset",
                                "frequency",
                            ] and (
                                "gen_freq_left" in loaded_track_params
                                or "gen_freq_right" in loaded_track_params
                            ):
                                f_left = loaded_track_params.get("gen_freq_left")
                                f_right = loaded_track_params.get("gen_freq_right")
                                target_duration = loaded_track_params.get(
                                    "gen_duration", 300
                                )
                                if f_left is not None and f_right is not None:
                                    source_info = SourceInfoFrequency(
                                        type="frequency",
                                        freq_type="binaural",
                                        f_left=f_left,
                                        f_right=f_right,
                                        target_duration_s=target_duration,
                                        freq=None,
                                        carrier=None,
                                        pulse=None,
                                    )
                                else:
                                    raise ValueError("Missing binaural frequency data")
                            elif (
                                source_type
                                in ["solfeggio", "solfeggio_preset", "frequency"]
                                and "gen_freq" in loaded_track_params
                            ):
                                freq = loaded_track_params.get("gen_freq")
                                target_duration = loaded_track_params.get(
                                    "gen_duration", 300
                                )
                                if freq is not None:
                                    source_info = SourceInfoFrequency(
                                        type="frequency",
                                        freq_type="solfeggio",
                                        freq=freq,
                                        target_duration_s=target_duration,
                                        f_left=None,
                                        f_right=None,
                                        carrier=None,
                                        pulse=None,
                                    )
                                else:
                                    raise ValueError("Missing solfeggio frequency data")
                            elif source_type == "isochronic" and (
                                "gen_carrier_freq" in loaded_track_params
                                or "gen_pulse_freq" in loaded_track_params
                            ):
                                carrier = loaded_track_params.get("gen_carrier_freq")
                                pulse = loaded_track_params.get("gen_pulse_freq")
                                target_duration = loaded_track_params.get(
                                    "gen_duration", 300
                                )
                                if carrier is not None and pulse is not None:
                                    source_info = SourceInfoFrequency(
                                        type="frequency",
                                        freq_type="isochronic",
                                        carrier=carrier,
                                        pulse=pulse,
                                        target_duration_s=target_duration,
                                        f_left=None,
                                        f_right=None,
                                        freq=None,
                                    )
                                else:
                                    raise ValueError(
                                        "Missing isochronic frequency data"
                                    )
                            else:
                                raise ValueError(
                                    f"Unsupported source type '{source_type}'"
                                )
                        except Exception as e_info:
                            logger.exception(
                                f"Error reconstructing source_info for track '{track_name}': {e_info}"
                            )
                            load_errors_found = True
                            st.error(
                                f"Project load failed: Error processing track '{track_name}'."
                            )
                            break
                        if source_info is None:
                            logger.error(
                                f"Failed to create source_info for track '{track_name}'. Aborting load."
                            )
                            load_errors_found = True
                            st.error(
                                f"Project load failed: Could not process track '{track_name}'."
                            )
                            break

                        # Prepare Initial Parameters
                        initial_params = {}
                        allowed_keys = list(
                            self.app_state.TrackDataDict.__annotations__.keys()
                        )
                        exclude_keys = {
                            "audio_snippet",
                            "source_info",
                            "sr",
                            "preview_temp_file_path",
                            "preview_settings_hash",
                            "update_counter",
                        }
                        for key, value in loaded_track_params.items():
                            if key in allowed_keys and key not in exclude_keys:
                                initial_params[key] = value
                        temp_loaded_tracks[old_track_id] = {
                            "source_info": source_info,
                            "sr": sr,
                            "initial_params": initial_params,
                        }

                # --- Add Tracks to State ONLY if NO errors occurred ---
                if not load_errors_found:
                    logger.info(
                        "Validation complete. No critical errors found. Adding tracks to state."
                    )
                    self.app_state.clear_all_tracks()
                    for track_info in temp_loaded_tracks.values():
                        try:
                            self.app_state.add_track(
                                audio_snippet=None,
                                source_info=track_info["source_info"],
                                sr=track_info["sr"],
                                initial_params=track_info["initial_params"],
                            )
                        except Exception as e_add:
                            logger.exception(
                                f"Error adding validated track '{track_info['initial_params'].get('name')}': {e_add}"
                            )
                            st.error(
                                f"Error occurred while adding track '{track_info['initial_params'].get('name')}'. Load might be incomplete."
                            )
                            load_errors_found = True
                            break
                    if not load_errors_found:
                        st.success("Project loaded successfully!")
                        if tracks_needing_upload:
                            st.warning(
                                f"Please re-upload source file(s): {', '.join(tracks_needing_upload)}"
                            )
                            st.info(
                                "Tracks needing re-upload show 'Missing Source File'."
                            )
                        load_successful = True
                if load_errors_found:
                    logger.error(
                        "Errors found during project load. Clearing application state."
                    )
                    self.app_state.clear_all_tracks()

            # --- Exception Handling ---
            except json.JSONDecodeError:
                logger.error("Failed decode project file.")
                st.error("Load failed: Invalid project file format.")
                load_errors_found = True
            except ValueError as e_val:
                logger.error(f"Invalid project file content: {e_val}")
                st.error(f"Load failed: {e_val}")
                load_errors_found = True
            except Exception as e:
                logger.exception("Unexpected error loading project.")
                st.error(f"Error loading project: {e}")
                load_errors_found = True
            finally:
                # --- Always clear load request state ---
                logger.info("Clearing project load request state.")
                st.session_state.project_load_requested = False
                if "uploaded_project_file_data" in st.session_state:
                    del st.session_state.uploaded_project_file_data
                if "uploaded_project_file_id" in st.session_state:
                    del st.session_state.uploaded_project_file_id
                if load_uploader_key in st.session_state:
                    st.session_state[load_uploader_key] = None  # Clear uploader widget
                # Rerun only if load was fully successful
                if load_successful:
                    logger.info("Load successful, rerunning.")
                    st.rerun()
                else:
                    logger.info(
                        "Load unsuccessful or errors found, not rerunning via project handler."
                    )
        else:
            logger.debug("No project load requested or no file data found.")
