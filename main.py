# main.py
# ==========================================
# MindMorph - Pro Subliminal Audio Editor
# Main Application Entry Point (Refactored)
# ==========================================

import json
import logging
import os
import uuid

import streamlit as st
from PIL import Image

# Core Components
from app_state import AppState, TrackData, TrackType

# --- Updated Audio Imports ---
# Audio Generation Functions (needed for project loading)
from audio_generators import generate_binaural_beats, generate_isochronic_tones, generate_noise, generate_solfeggio_frequency  # <-- Import from audio_generators now

# Type Hint (can be defined centrally or imported from where it makes sense)
# Assuming AudioData is defined/used within audio_generators or audio_processing
from audio_processing import AudioData  # Or from audio_generators if defined there

# --- Local Imports ---
# Configuration and Constants
from config import FAVICON_PATH, GLOBAL_SR, PROJECT_FILE_VERSION, TRACK_TYPE_OTHER
from tts_generator import TTSGenerator
from ui_manager import UIManager

# Utility Functions
from utils import setup_logging  # Import the setup function

# --- End Updated Audio Imports ---

# --- Early Setup: Logging and Page Config ---
# Configure logging as the first step
setup_logging()
logger = logging.getLogger(__name__)  # Get logger for this main module

# Configure Streamlit page
try:
    page_icon = Image.open(FAVICON_PATH)
except FileNotFoundError:
    logger.warning(f"Favicon not found at {FAVICON_PATH}. Using default.")
    page_icon = "🧠"  # Default emoji icon
except Exception as e:
    logger.error(f"Error loading favicon: {e}")
    page_icon = "🧠"

st.set_page_config(layout="wide", page_title="MindMorph - Pro Subliminal Editor", page_icon=page_icon)

# --- Helper Functions ---


def show_welcome_message():
    """Displays the initial welcome message and mode explanation."""
    # Check if the message has already been shown/dismissed
    if "welcome_message_shown" not in st.session_state:
        with st.container(border=True):
            st.markdown("### 👋 Welcome to MindMorph!")
            st.markdown("Create custom subliminal audio by layering sounds and applying effects.")
            st.markdown("---")
            st.markdown("#### ✨ Choose Your Experience:")
            st.markdown("Use the **'Select Editor Mode'** option at the top of the main panel:")
            st.markdown("- **Easy Mode:** Simplified interface, perfect for getting started quickly.")
            st.markdown("- **Advanced Mode:** Access all features like detailed frequency generation and audio effects.")
            st.markdown("*(You can switch modes any time!)*")
            st.markdown("---")
            st.markdown("#### Quick Start Workflow:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("##### 1. Add Tracks ➕")
                st.markdown("Use the **sidebar** (👈).")
                st.caption("Upload, TTS, Noise, Freq.")
            with col2:
                st.markdown("##### 2. Edit Tracks 🎚️")
                st.markdown("Adjust **settings** below.")
                st.caption("Click 'Update Preview'!")
            with col3:
                st.markdown("##### 3. Mix & Export 🔊")
                st.markdown("Use **master controls** (bottom).")
                st.caption("Preview or Download")
            st.markdown("---")
            st.markdown("*(Click button below to hide this guide. Find details in Instructions at page bottom.)*")
            # Center the button
            button_cols = st.columns([1, 1.5, 1])  # Adjust ratios as needed
            with button_cols[1]:
                if st.button(
                    "Got it! Let's Start Creating ✨",
                    key="dismiss_welcome_button",  # Unique key
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.welcome_message_shown = True
                    logger.info("Welcome message dismissed by user.")
                    st.rerun()  # Rerun to hide the message immediately


def handle_project_load(app_state: AppState, tts_generator: TTSGenerator):
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
                app_state.clear_all_tracks()

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
                        reconstructed_audio: AudioData | None = None  # Use imported type hint

                        # --- Regenerate Audio Based on Source Type ---
                        try:
                            if source_type == "tts" and "tts_text" in track_data:
                                logger.info(f"Regenerating TTS for track '{track_data.get('name')}'")
                                reconstructed_audio, _ = tts_generator.generate(track_data["tts_text"])
                            elif source_type == "noise" and "gen_noise_type" in track_data:
                                noise_type = track_data["gen_noise_type"]
                                duration = track_data.get("gen_duration", 60)
                                volume = track_data.get("gen_volume", 0.5)
                                logger.info(f"Regenerating {noise_type} ({duration}s, vol={volume}) for track '{track_data.get('name')}'")
                                # Use function from audio_generators
                                reconstructed_audio = generate_noise(noise_type, duration, GLOBAL_SR, volume)
                            elif source_type == "binaural" and "gen_freq_left" in track_data:
                                duration = track_data.get("gen_duration", 60)
                                f_left = track_data["gen_freq_left"]
                                f_right = track_data.get("gen_freq_right", f_left + 10.0)  # Default beat
                                volume = track_data.get("gen_volume", 0.3)
                                logger.info(f"Regenerating Binaural ({duration}s, L={f_left}, R={f_right}, vol={volume}) for track '{track_data.get('name')}'")
                                # Use function from audio_generators
                                reconstructed_audio = generate_binaural_beats(duration, f_left, f_right, GLOBAL_SR, volume)
                            elif source_type == "solfeggio" and "gen_freq" in track_data:
                                duration = track_data.get("gen_duration", 60)
                                freq = track_data["gen_freq"]
                                volume = track_data.get("gen_volume", 0.3)
                                logger.info(f"Regenerating Solfeggio ({duration}s, F={freq}, vol={volume}) for track '{track_data.get('name')}'")
                                # Use function from audio_generators
                                reconstructed_audio = generate_solfeggio_frequency(duration, freq, GLOBAL_SR, volume)
                            elif source_type == "isochronic" and "gen_carrier_freq" in track_data:
                                duration = track_data.get("gen_duration", 60)
                                carrier = track_data["gen_carrier_freq"]
                                pulse = track_data.get("gen_pulse_freq", 10.0)  # Default pulse
                                volume = track_data.get("gen_volume", 0.4)
                                logger.info(f"Regenerating Isochronic ({duration}s, C={carrier}, P={pulse}, vol={volume}) for track '{track_data.get('name')}'")
                                # Use function from audio_generators
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
                        from config import get_default_track_params  # Import locally if needed

                        final_track_data_for_load = get_default_track_params()
                        final_track_data_for_load.update(track_data)

                        try:
                            app_state.add_track(final_track_data_for_load, track_type=track_type)
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


# --- Main Application Logic ---


def main():
    """Main function to run the Streamlit application."""
    logger.info("=====================================================")
    logger.info("MindMorph Application starting/rerunning.")
    logger.info("=====================================================")

    st.title("🧠 MindMorph - Subliminal Audio Editor")

    # --- Initialize Core Components ---
    app_state = AppState()
    tts_generator = TTSGenerator()
    ui_manager = UIManager(app_state, tts_generator)

    # --- Handle Project Loading ---
    handle_project_load(app_state, tts_generator)

    # --- Initial Welcome Message ---
    show_welcome_message()

    # --- Main UI Rendering (only if welcome message dismissed) ---
    if "welcome_message_shown" in st.session_state:
        # --- Mode Selector ---
        if "app_mode" not in st.session_state:
            st.session_state.app_mode = "Easy"
            logger.info(f"App mode initialized to: {st.session_state.app_mode}")

        mode_options = ["Easy", "Advanced"]
        try:
            current_mode_index = mode_options.index(st.session_state.app_mode)
        except ValueError:
            logger.warning(f"Invalid mode '{st.session_state.app_mode}'. Defaulting to Easy.")
            st.session_state.app_mode = "Easy"
            current_mode_index = 0

        selected_mode = st.radio(
            "Select Editor Mode:",
            options=mode_options,
            index=current_mode_index,
            key="mode_selector_radio",
            horizontal=True,
            help="Easy mode hides complex features, Advanced shows all options.",
        )
        if selected_mode != st.session_state.app_mode:
            logger.info(f"App mode changed from '{st.session_state.app_mode}' to '{selected_mode}'")
            st.session_state.app_mode = selected_mode
            if "export_buffer" in st.session_state:
                del st.session_state.export_buffer
            if "preview_audio_data" in st.session_state:
                del st.session_state.preview_audio_data
            st.rerun()

        st.markdown("---")

        # --- Render UI Sections using UIManager ---
        # This single call now handles sidebar, track editor, master controls, etc.
        ui_manager.render_ui()

    else:
        logger.debug("Welcome message active, skipping main UI render.")

    logger.info("--- End of main function render cycle ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("A critical error occurred in the main execution block.")
        st.error("An unexpected error occurred. Please check the application logs or try reloading the page.")
        # st.exception(e) # Optionally show traceback in UI for debugging
