# sidebar_uploader.py
# ==========================================
# File Upload and TTS UI for MindMorph Sidebar
# ==========================================

import logging
import os
import tempfile  # <<< Added for temporary file handling
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

# Import necessary components from other modules
# <<< Updated AppState imports >>>
from app_state import AppState, SourceInfoTTS, SourceInfoUpload, TrackDataDict, TrackType
from audio_io import load_audio, save_audio_to_temp_file  # <<< Added save_audio_to_temp_file (though maybe not needed here)
from config import (
    GLOBAL_SR,
    MAX_AFFIRMATION_CHARS,
    MAX_AUDIO_DURATION_S,
    TRACK_SNIPPET_DURATION_S,  # <<< Added snippet duration
    TRACK_TYPE_AFFIRMATION,
    TRACK_TYPE_BACKGROUND,
    TRACK_TYPE_OTHER,
    TRACK_TYPE_VOICE,
    # get_default_track_params, # No longer used directly here
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
        """
        Callback function to process uploaded audio files.
        Saves full file temporarily, loads snippet, adds track with source info.
        Includes size validation.
        """
        logger.debug(f"Callback triggered: _handle_audio_upload (key: {self.audio_uploader_key})")
        uploaded_files = st.session_state.get(self.audio_uploader_key)

        if not uploaded_files:
            logger.debug("Audio upload callback triggered but no files found in session state.")
            return

        files_to_process = list(uploaded_files)
        files_processed_successfully = False

        current_tracks = self.app_state.get_all_tracks()
        # Check based on original filename stored in source_info now
        current_track_filenames = {
            tdata["source_info"]["original_filename"]
            for tid, tdata in current_tracks.items()
            if tdata.get("source_info") and tdata["source_info"].get("type") == "upload" and "original_filename" in tdata["source_info"]
        }
        # Logic for replacing missing audio might need adjustment based on how project load works

        for file in files_to_process:
            logger.info(f"Processing uploaded file via callback: {file.name} (Size: {file.size} bytes)")

            # --- File Size Validation ---
            if file.size > MAX_UPLOAD_SIZE_BYTES:
                logger.warning(f"Upload '{file.name}' rejected (callback): Size {file.size} bytes exceeds limit {MAX_UPLOAD_SIZE_BYTES} bytes.")
                st.error(f"❌ File '{file.name}' ({file.size / (1024 * 1024):.1f} MB) exceeds the {MAX_UPLOAD_SIZE_MB} MB limit.")
                continue  # Skip this file

            if file.name in current_track_filenames:
                logger.info(f"Skipping upload (callback): Track for '{file.name}' seems to already exist based on filename.")
                st.warning(f"⚠️ Track with filename '{file.name}' already exists. Skipping duplicate upload.")
                continue

            temp_file_path = None
            audio_snippet = None
            snippet_sr = None

            try:
                # 1. Save the FULL uploaded file to a temporary location
                # Use a context manager for safety if possible, or ensure cleanup
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                    tmp.write(file.getvalue())
                    temp_file_path = tmp.name
                logger.info(f"Saved uploaded file '{file.name}' temporarily to: {temp_file_path}")

                # 2. Load only the SNIPPET from the temporary file
                audio_snippet, snippet_sr = load_audio(temp_file_path, target_sr=GLOBAL_SR, duration=TRACK_SNIPPET_DURATION_S)

            except Exception as e:
                logger.error(f"Error saving temp file or loading snippet for {file.name}: {e}", exc_info=True)
                st.error(f"Failed to process {file.name}: {e}")
                # Clean up temp file if created
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        logger.warning(f"Failed to clean up temp file {temp_file_path} after error.")
                continue  # Skip this file

            # 3. Add Track to AppState if snippet loaded successfully
            if audio_snippet is not None and snippet_sr is not None and audio_snippet.size > 0:
                # Check duration (based on snippet - might not be accurate for full file!)
                # A full file duration check might require loading without duration first,
                # which defeats the purpose. Rely on size limit for now.
                # duration_seconds = len(audio_snippet) / snippet_sr if snippet_sr > 0 else 0
                # We might skip duration check here or accept it's less precise

                files_processed_successfully = True

                # Determine track type based on filename (heuristic)
                if any(keyword in file.name.lower() for keyword in ["voice", "record", "affirmation", "tts"]):
                    track_type = TRACK_TYPE_VOICE
                elif any(keyword in file.name.lower() for keyword in ["music", "background", "mask", "ambient"]):
                    track_type = TRACK_TYPE_BACKGROUND
                else:
                    track_type = TRACK_TYPE_OTHER

                # Prepare source info and initial params
                source_info: SourceInfoUpload = {
                    "type": "upload",
                    "temp_file_path": temp_file_path,  # Store path to the FULL temp file
                    "original_filename": file.name,
                }
                initial_params = {
                    "name": os.path.splitext(file.name)[0],  # Use filename without extension as name
                    "track_type": track_type,
                }

                # Add track to state
                new_track_id = self.app_state.add_track(audio_snippet=audio_snippet, source_info=source_info, sr=snippet_sr, initial_params=initial_params)
                duration_seconds = len(audio_snippet) / snippet_sr if snippet_sr > 0 else 0
                st.success(f"Loaded '{file.name}' as '{track_type}' (Snippet: {duration_seconds:.1f}s)")

            elif audio_snippet is None:
                logger.error(f"Failed to load audio snippet (callback): {file.name}")
                st.error(f"Failed to load audio data from {file.name}.")
                # Clean up temp file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        logger.warning(f"Failed to clean up temp file {temp_file_path} after load failure.")
            else:  # Snippet size is 0
                logger.warning(f"Skipped empty/invalid audio snippet (callback): {file.name}")
                st.warning(f"Audio file '{file.name}' appears empty or could not be processed.")
                # Clean up temp file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        logger.warning(f"Failed to clean up temp file {temp_file_path} for empty audio.")

        if files_processed_successfully:
            logger.debug(f"Audio upload callback finished processing.")
            # Optionally clear the uploader widget state if desired, though might cause issues
            # st.session_state[self.audio_uploader_key] = []
        elif uploaded_files:
            logger.debug("No audio files processed successfully in callback.")

    # --- Rendering Methods for Uploader Sections ---

    def render_uploader(self):
        """Renders the audio file uploader component in the sidebar."""
        st.subheader("📁 Upload Audio File(s)")
        # Updated caption to reflect new limits
        st.caption(f"Upload music, voice, etc. (Max duration: {MAX_AUDIO_DURATION_S // 60} min approx, Max size: {MAX_UPLOAD_SIZE_MB} MB)")

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
                    # <<< Call updated helper method >>>
                    self._generate_tts_track(text_to_generate, default_name)
                    # Don't clear text area automatically
                    st.rerun()

        # --- File Upload Tab ---
        with tab2:
            st.caption("Upload a .txt or .docx file containing affirmations.")
            st.file_uploader(
                "Upload Affirmation File (.txt, .docx)",
                type=["txt", "docx"],
                key=self.affirmation_uploader_key,
                label_visibility="collapsed",
                help="Select a text or Word document containing affirmations.",
            )
            if st.button(
                "Generate Track from File",
                key="sidebar_generate_tts_from_file_button",
                use_container_width=True,
                type="primary",
                help="Read the uploaded file and convert its text content to a spoken audio track.",
            ):
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
                                # <<< Call updated helper method >>>
                                self._generate_tts_track(text_from_file, default_name)
                                file_processed = True
                    except Exception as e:
                        logger.error(f"Error processing affirmation file {uploaded_file.name} on button click: {e}")
                        st.error(f"Failed to process affirmation file {uploaded_file.name}: {e}")

                    if file_processed:
                        try:
                            logger.debug(f"Button click processed file, clearing state for key: {self.affirmation_uploader_key}")
                            st.session_state[self.affirmation_uploader_key] = None
                            st.rerun()
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

    # <<< MODIFIED: To handle snippet and source_info >>>
    def _generate_tts_track(self, text_content: str, track_name: str):
        """
        Helper method to generate TTS audio, extract snippet, and add track with source info.

        Args:
            text_content: The text to synthesize.
            track_name: The default name for the new track.
        """
        logger.info(f"SidebarUploader: Generating TTS track '{track_name}'")
        audio_snippet: Optional[AudioData] = None
        snippet_sr: Optional[int] = None

        try:
            with st.spinner(f"Generating '{track_name}'..."):
                # 1. Generate FULL TTS audio
                full_audio, full_sr = self.tts_generator.generate(text_content)

            if full_audio is not None and full_sr is not None and full_audio.size > 0:
                # 2. Extract Snippet
                snippet_length_samples = int(TRACK_SNIPPET_DURATION_S * full_sr)
                if len(full_audio) > snippet_length_samples:
                    audio_snippet = full_audio[:snippet_length_samples]
                else:
                    audio_snippet = full_audio  # Use full audio if shorter than snippet duration
                snippet_sr = full_sr

                # 3. Prepare source_info and initial_params
                source_info: SourceInfoTTS = {
                    "type": "tts",
                    "text": text_content,
                    # Add tts_params here if needed
                }
                initial_params = {
                    "name": track_name,
                    "track_type": TRACK_TYPE_AFFIRMATION,
                }

                # 4. Add track to state
                track_id = self.app_state.add_track(audio_snippet=audio_snippet, source_info=source_info, sr=snippet_sr, initial_params=initial_params)
                st.success(f"'{track_name}' track generated (ID: {track_id[:6]})!")
                st.toast("Affirmation track added!", icon="✅")
            else:
                logger.error("TTS generation returned None or empty audio within SidebarUploader.")
                st.error(f"Failed to generate audio for '{track_name}'. TTS might have failed.")

        except Exception as e:
            logger.exception(f"Error during TTS track generation within SidebarUploader for '{track_name}'.")
            st.error(f"Failed to create TTS track '{track_name}': {e}")
