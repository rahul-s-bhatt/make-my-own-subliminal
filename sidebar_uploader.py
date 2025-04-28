# sidebar_uploader.py
# ==========================================
# File Upload and TTS UI for MindMorph Sidebar
# ==========================================

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

# Import necessary components from other modules
from app_state import AppState, TrackData, TrackType
from audio_io import load_audio  # For audio uploads
from config import (
    GLOBAL_SR,
    MAX_AFFIRMATION_CHARS,
    MAX_AUDIO_DURATION_S,
    TRACK_TYPE_AFFIRMATION,
    TRACK_TYPE_BACKGROUND,
    TRACK_TYPE_OTHER,
    TRACK_TYPE_VOICE,
    get_default_track_params,
)
from tts_generator import TTSGenerator
from utils import read_text_file  # For affirmation file uploads

# Type hint (ensure consistency across files)
try:
    from audio_processing import AudioData
except ImportError:
    AudioData = np.ndarray


# Get a logger for this module
logger = logging.getLogger(__name__)

# Define the new maximum file size in bytes (15 MB)
MAX_UPLOAD_SIZE_MB = 15
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


class SidebarUploader:
    """Handles rendering file uploaders and related actions in the sidebar."""

    def __init__(self, app_state: AppState, tts_generator: TTSGenerator):
        """
        Initializes the SidebarUploader.

        Args:
            app_state: An instance of the AppState class.
            tts_generator: An instance of the TTSGenerator class.
        """
        self.app_state = app_state
        self.tts_generator = tts_generator
        # Define keys for file uploaders within this class scope
        self.audio_uploader_key = "sidebar_audio_file_uploader"
        self.affirmation_uploader_key = "sidebar_affirmation_file_uploader"
        logger.debug("SidebarUploader initialized.")

    # --- Callback Functions for File Uploaders ---

    def _handle_audio_upload(self):
        """Callback function to process uploaded audio files, including size validation."""
        logger.debug(f"Callback triggered: _handle_audio_upload (key: {self.audio_uploader_key})")
        uploaded_files = st.session_state.get(self.audio_uploader_key)

        if not uploaded_files:
            logger.debug("Audio upload callback triggered but no files found in session state.")
            return

        files_to_process = list(uploaded_files)
        files_processed_successfully = False

        current_tracks = self.app_state.get_all_tracks()
        current_track_filenames_with_audio = {
            tdata.get("original_filename") for tdata in current_tracks.values() if tdata.get("source_type") == "upload" and tdata.get("original_audio") is not None
        }
        current_tracks_missing_audio = {
            tdata.get("original_filename"): tid for tid, tdata in current_tracks.items() if tdata.get("source_type") == "upload" and tdata.get("original_audio") is None
        }

        for file in files_to_process:
            logger.info(f"Processing uploaded file via callback: {file.name} (Size: {file.size} bytes)")

            # --- File Size Validation ---
            if file.size > MAX_UPLOAD_SIZE_BYTES:
                logger.warning(f"Upload '{file.name}' rejected (callback): Size {file.size} bytes exceeds limit {MAX_UPLOAD_SIZE_BYTES} bytes.")
                st.error(f"❌ File '{file.name}' ({file.size / (1024 * 1024):.1f} MB) exceeds the {MAX_UPLOAD_SIZE_MB} MB limit.")
                continue  # Skip this file

            if file.name in current_track_filenames_with_audio:
                logger.info(f"Skipping upload (callback): Track for '{file.name}' already exists.")
                continue

            existing_track_id = current_tracks_missing_audio.get(file.name)

            try:
                # Load audio only after passing size check
                audio, sr = load_audio(file, target_sr=GLOBAL_SR)
            except Exception as e:
                logger.error(f"Error loading audio file {file.name} in callback: {e}")
                st.error(f"Failed to load {file.name}: {e}")
                continue

            if audio is not None and audio.size > 0:
                duration_seconds = len(audio) / sr if sr > 0 else 0
                if duration_seconds > MAX_AUDIO_DURATION_S:
                    logger.warning(f"Upload '{file.name}' rejected (callback): Duration {duration_seconds:.1f}s exceeds limit.")
                    st.error(f"❌ File '{file.name}' too long ({duration_seconds:.1f}s). Max is {MAX_AUDIO_DURATION_S // 60} min.")
                    continue

                files_processed_successfully = True

                if existing_track_id:
                    logger.info(f"Updating existing track '{self.app_state.get_track(existing_track_id).get('name')}' via callback.")
                    self.app_state.update_track_param(existing_track_id, "original_audio", audio)
                    self.app_state.update_track_param(existing_track_id, "sr", sr)
                    st.success(f"Re-loaded audio for track '{self.app_state.get_track(existing_track_id).get('name')}'")
                else:
                    track_params = get_default_track_params()
                    if any(keyword in file.name.lower() for keyword in ["voice", "record", "affirmation", "tts"]):
                        track_type = TRACK_TYPE_VOICE
                    elif any(keyword in file.name.lower() for keyword in ["music", "background", "mask", "ambient"]):
                        track_type = TRACK_TYPE_BACKGROUND
                    else:
                        track_type = TRACK_TYPE_OTHER

                    track_params.update({"original_audio": audio, "sr": sr, "name": file.name, "source_type": "upload", "original_filename": file.name})
                    new_track_id = self.app_state.add_track(track_params, track_type=track_type)
                    st.success(f"Loaded '{file.name}' as '{track_type}' ({duration_seconds:.1f}s)")

            elif audio is None:
                logger.error(f"Failed to load audio (callback): {file.name}")
            else:
                logger.warning(f"Skipped empty/invalid audio (callback): {file.name}")

        if files_processed_successfully:
            logger.debug(f"Audio upload callback finished processing. Filenames might remain in widget.")
        elif uploaded_files:
            logger.debug("No audio files processed successfully in callback.")

    # --- Rendering Methods for Uploader Sections ---

    def render_uploader(self):
        """Renders the audio file uploader component in the sidebar."""
        st.subheader("📁 Upload Audio File(s)")
        # Updated caption to reflect new limits
        st.caption(f"Upload music, voice, etc. (Max duration: {MAX_AUDIO_DURATION_S // 60} min, Max size: {MAX_UPLOAD_SIZE_MB} MB)")

        st.file_uploader(
            "Select audio files (.wav, .mp3)",  # Updated label
            type=["wav", "mp3"],  # Updated allowed types
            accept_multiple_files=True,
            key=self.audio_uploader_key,  # Use instance variable key
            label_visibility="collapsed",
            help=f"Select one or more WAV or MP3 files to add as tracks (Max {MAX_UPLOAD_SIZE_MB}MB each).",  # Updated help text
            on_change=self._handle_audio_upload,  # Assign the callback
        )

    def render_affirmation_inputs(self):
        """Renders the affirmation input options (TTS, File) in the sidebar."""
        st.subheader("🗣️ Add Affirmations")
        st.caption(f"Uses system default TTS voice. Max {MAX_AFFIRMATION_CHARS} chars.")

        tab1, tab2, tab3 = st.tabs(["Type Text", "Upload File", "Record Audio"])

        # --- Text Input Tab ---
        with tab1:
            st.caption("Type or paste affirmations below (one per line recommended).")
            affirmation_text = st.text_area(
                "Affirmation Text",
                height=150,
                key="sidebar_affirmation_text_area",
                label_visibility="collapsed",
                help="Enter the affirmations you want to convert to speech.",
                max_chars=MAX_AFFIRMATION_CHARS,
            )
            st.caption(f"{len(affirmation_text)} / {MAX_AFFIRMATION_CHARS} characters")

            if st.button(
                "Generate Affirmation Track", key="sidebar_generate_tts_from_text", use_container_width=True, type="primary", help="Convert the text above to a spoken audio track."
            ):
                # Read text directly from state when button is clicked
                text_to_generate = st.session_state.get("sidebar_affirmation_text_area", "")
                if not text_to_generate or not text_to_generate.strip():
                    st.warning("Please enter some text in the text area first.")
                elif len(text_to_generate) > MAX_AFFIRMATION_CHARS:
                    logger.warning(f"TTS Text input rejected: Length {len(text_to_generate)} exceeds limit {MAX_AFFIRMATION_CHARS}.")
                    st.error(f"❌ Text is too long ({len(text_to_generate)} chars). Max is {MAX_AFFIRMATION_CHARS}.")
                else:
                    default_name = "TTS Affirmations"
                    if len(text_to_generate) > 30:
                        default_name = f"TTS: {text_to_generate[:25]}..."
                    self._generate_tts_track(text_to_generate, default_name)  # Use helper in this class
                    # --- REMOVED state clearing attempt ---
                    # st.session_state.sidebar_affirmation_text_area = "" # REMOVED THIS LINE
                    st.rerun()  # Rerun needed after button click generation

        # --- File Upload Tab ---
        with tab2:
            st.caption("Upload a .txt or .docx file containing affirmations.")
            # Render the uploader WITHOUT on_change
            st.file_uploader(
                "Upload Affirmation File (.txt, .docx)",
                type=["txt", "docx"],
                key=self.affirmation_uploader_key,  # Use instance variable key
                label_visibility="collapsed",
                help="Select a text or Word document containing affirmations.",
                # on_change removed
            )
            # Add the button back
            if st.button(
                "Generate Track from File",
                key="sidebar_generate_tts_from_file_button",
                use_container_width=True,
                type="primary",
                help="Read the uploaded file and convert its text content to a spoken audio track.",
            ):
                # Get the file from session state when the button is clicked
                uploaded_file = st.session_state.get(self.affirmation_uploader_key)
                if uploaded_file:
                    file_processed = False
                    try:
                        text_from_file = read_text_file(uploaded_file)

                        if text_from_file is not None:
                            if not text_from_file.strip():
                                st.warning(f"File '{uploaded_file.name}' appears empty.")
                                logger.warning(f"File '{uploaded_file.name}' read as empty (button click).")
                            elif len(text_from_file) > MAX_AFFIRMATION_CHARS:
                                logger.warning(f"TTS File '{uploaded_file.name}' rejected (button click): Length {len(text_from_file)} > {MAX_AFFIRMATION_CHARS}.")
                                st.error(f"❌ Text in file '{uploaded_file.name}' is too long ({len(text_from_file)} chars). Max is {MAX_AFFIRMATION_CHARS}.")
                            else:
                                default_name = f"File Affirmations ({uploaded_file.name})"
                                self._generate_tts_track(text_from_file, default_name)  # Use helper in this class
                                file_processed = True  # Mark as successfully processed
                        # else: read_text_file shows error

                    except Exception as e:
                        logger.error(f"Error processing affirmation file {uploaded_file.name} on button click: {e}")
                        st.error(f"Failed to process affirmation file {uploaded_file.name}: {e}")

                    # --- Clear state AFTER processing on button click ---
                    if file_processed:
                        try:
                            logger.debug(f"Button click processed file, clearing state for key: {self.affirmation_uploader_key}")
                            st.session_state[self.affirmation_uploader_key] = None
                            st.rerun()  # Rerun AFTER clearing state and adding track
                        except Exception as e_clear:
                            logger.error(f"Error trying to clear affirmation uploader state on button click: {e_clear}")

                else:
                    st.warning("Please upload a .txt or .docx file first.")

        # --- Record Audio Tab ---
        with tab3:
            st.caption("Record your own voice directly in the browser.")
            st.info("🎙️ Audio recording feature coming soon!")
            st.markdown("For now, please record using other software and use the 'Upload Audio File(s)' option.")
            st.button("Start Recording", key="sidebar_start_recording", disabled=True, use_container_width=True)

    # --- Helper Method Specific to this Class ---

    def _generate_tts_track(self, text_content: str, track_name: str):
        """
        Helper method to generate TTS audio and add it as a new track.
        Called by text input button or affirmation file upload callback.

        Args:
            text_content: The text to synthesize.
            track_name: The default name for the new track.
        """
        logger.info(f"SidebarUploader: Generating TTS track '{track_name}'")
        try:
            with st.spinner(f"Generating '{track_name}'..."):
                audio, sr = self.tts_generator.generate(text_content)

            if audio is not None and sr is not None:
                track_params = get_default_track_params()
                track_params.update(
                    {
                        "original_audio": audio,
                        "sr": sr,
                        "name": track_name,
                        "source_type": "tts",
                        "tts_text": text_content,
                    }
                )
                track_id = self.app_state.add_track(track_params, track_type=TRACK_TYPE_AFFIRMATION)
                st.success(f"'{track_name}' track generated (ID: {track_id[:6]})!")
                st.toast("Affirmation track added!", icon="✅")
            else:
                logger.error("TTS generation returned None within SidebarUploader.")
        except Exception as e:
            logger.exception(f"Error during TTS track generation within SidebarUploader for '{track_name}'.")
            st.error(f"Failed to create TTS track '{track_name}': {e}")
