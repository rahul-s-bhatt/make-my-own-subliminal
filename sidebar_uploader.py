# sidebar_uploader.py
# ==========================================
# File Upload and TTS UI for MindMorph Sidebar
# ==========================================

import logging
import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

# Import necessary components from other modules
from app_state import AppState
from audio_io import load_audio
from audio_state_definitions import AudioData, SourceInfoTTS, SourceInfoUpload, TrackDataDict, TrackType
from config import (
    GLOBAL_SR,
    MAX_AFFIRMATION_CHARS,
    MAX_AUDIO_DURATION_S,
    MAX_TRACK_LIMIT,
    TRACK_SNIPPET_DURATION_S,
    TRACK_TYPE_AFFIRMATION,
    TRACK_TYPE_BACKGROUND,
    TRACK_TYPE_OTHER,
    TRACK_TYPE_VOICE,
)
from tts_generator import TTSGenerator
from utils import read_text_file

# Get a logger for this module
logger = logging.getLogger(__name__)

# Define the new maximum file size in bytes (15 MB)
MAX_UPLOAD_SIZE_MB = 15
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


class SidebarUploader:
    """Handles rendering file uploaders and related actions in the sidebar."""

    def __init__(self, app_state: AppState, tts_generator: TTSGenerator):
        """Initializes the SidebarUploader."""
        self.app_state = app_state
        self.tts_generator = tts_generator
        self.audio_uploader_key = "sidebar_audio_file_uploader"
        self.affirmation_uploader_key = "sidebar_affirmation_file_uploader"
        logger.debug("SidebarUploader initialized.")

    # --- Check Track Limit Helper ---
    def _check_track_limit(self, adding_count: int = 1) -> bool:
        """Checks if adding tracks would exceed the limit."""
        current_count = len(self.app_state.get_all_tracks())
        if current_count + adding_count > MAX_TRACK_LIMIT:
            logger.warning(f"Track limit ({MAX_TRACK_LIMIT}) reached. Cannot add {adding_count} more track(s). Current count: {current_count}")
            st.error(f"Cannot add more tracks. Maximum limit of {MAX_TRACK_LIMIT} reached.", icon="🚫")
            return False
        return True

    # --- Callback Functions for File Uploaders ---

    def _handle_audio_upload(self):
        """
        Callback function to process uploaded audio files.
        Saves full file temporarily, loads snippet, adds track with source info.
        Includes size and track limit validation.
        """
        logger.debug(f"Callback triggered: _handle_audio_upload (key: {self.audio_uploader_key})")
        uploaded_files = st.session_state.get(self.audio_uploader_key)
        if not uploaded_files:
            return

        files_to_process = list(uploaded_files)
        files_processed_successfully = False
        files_skipped_or_failed = 0

        # Check limit before processing
        if not self._check_track_limit(adding_count=len(files_to_process)):
            # Don't clear state here, just return. User needs to remove files manually.
            return

        current_tracks = self.app_state.get_all_tracks()
        current_track_filenames = {
            tdata["source_info"]["original_filename"]
            for _, tdata in current_tracks.items()
            if isinstance(tdata.get("source_info"), dict) and tdata["source_info"].get("type") == "upload" and "original_filename" in tdata["source_info"]
        }

        for file in files_to_process:
            if not self._check_track_limit(adding_count=1):
                break  # Stop if limit reached mid-batch

            logger.info(f"Processing uploaded file via callback: {file.name} (Size: {file.size} bytes)")
            if file.size > MAX_UPLOAD_SIZE_BYTES:
                logger.warning(f"Upload '{file.name}' rejected: Size exceeds limit.")
                st.error(f"❌ File '{file.name}' ({file.size / (1024 * 1024):.1f} MB) exceeds the {MAX_UPLOAD_SIZE_MB} MB limit.")
                files_skipped_or_failed += 1
                continue
            if file.name in current_track_filenames:
                logger.info(f"Skipping upload: Track for '{file.name}' already exists.")
                st.warning(f"⚠️ Track with filename '{file.name}' already exists. Skipping.")
                files_skipped_or_failed += 1
                continue

            temp_file_path = None
            audio_snippet = None
            snippet_sr = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                    tmp.write(file.getvalue())
                    temp_file_path = tmp.name
                logger.info(f"Saved uploaded file '{file.name}' temporarily to: {temp_file_path}")
                audio_snippet, snippet_sr = load_audio(temp_file_path, target_sr=GLOBAL_SR, duration=TRACK_SNIPPET_DURATION_S)
            except Exception as e:
                logger.error(f"Error saving temp file or loading snippet for {file.name}: {e}", exc_info=True)
                st.error(f"Failed to process {file.name}: {e}")
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        logger.warning(f"Failed cleanup temp file {temp_file_path} after error.")
                files_skipped_or_failed += 1
                continue

            if audio_snippet is not None and snippet_sr is not None and audio_snippet.size > 0:
                files_processed_successfully = True
                if any(k in file.name.lower() for k in ["voice", "record", "affirmation", "tts"]):
                    track_type = TRACK_TYPE_VOICE
                elif any(k in file.name.lower() for k in ["music", "background", "mask", "ambient"]):
                    track_type = TRACK_TYPE_BACKGROUND
                else:
                    track_type = TRACK_TYPE_OTHER
                source_info: SourceInfoUpload = {"type": "upload", "temp_file_path": temp_file_path, "original_filename": file.name}
                initial_params = {"name": os.path.splitext(file.name)[0], "track_type": track_type}
                new_track_id = self.app_state.add_track(audio_snippet=audio_snippet, source_info=source_info, sr=snippet_sr, initial_params=initial_params)
                duration_seconds = len(audio_snippet) / snippet_sr if snippet_sr > 0 else 0
                st.success(f"Loaded '{file.name}' as '{track_type}' (Snippet: {duration_seconds:.1f}s)")
                current_track_filenames.add(file.name)  # Add to set to prevent duplicates in same batch
            elif audio_snippet is None:
                logger.error(f"Failed to load audio snippet: {file.name}")
                st.error(f"Failed to load audio data from {file.name}.")
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        logger.warning(f"Failed cleanup temp file {temp_file_path} after load failure.")
                files_skipped_or_failed += 1
            else:
                logger.warning(f"Skipped empty/invalid audio snippet: {file.name}")
                st.warning(f"Audio file '{file.name}' appears empty or could not be processed.")
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        logger.warning(f"Failed cleanup temp file {temp_file_path} for empty audio.")
                files_skipped_or_failed += 1

        # <<< REMOVED line that caused the error >>>
        # st.session_state[self.audio_uploader_key] = [] # DO NOT DO THIS

        # Log summary
        if files_processed_successfully:
            logger.debug(f"Audio upload callback finished. Processed successfully: {len(files_to_process) - files_skipped_or_failed}, Skipped/Failed: {files_skipped_or_failed}")
        elif uploaded_files:
            logger.debug(f"Audio upload callback finished. Skipped/Failed all {len(files_to_process)} files.")

        # Trigger a rerun if tracks were successfully added to update the main UI
        if files_processed_successfully:
            st.rerun()

    # --- Rendering Methods for Uploader Sections ---

    def render_uploader(self):
        """Renders the audio file uploader component in the sidebar."""
        st.subheader("📁 Upload Audio File(s)")
        st.caption(f"Upload music, voice, etc. (Max duration: {MAX_AUDIO_DURATION_S // 60} min approx, Max size: {MAX_UPLOAD_SIZE_MB} MB)")
        limit_reached = len(self.app_state.get_all_tracks()) >= MAX_TRACK_LIMIT
        if limit_reached:
            st.warning(f"Track limit ({MAX_TRACK_LIMIT}) reached.", icon="⚠️")
        st.file_uploader(
            "Select audio files (.wav, .mp3)",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            key=self.audio_uploader_key,
            label_visibility="collapsed",
            help=f"Select WAV/MP3 files (Max {MAX_UPLOAD_SIZE_MB}MB each). Limit: {MAX_TRACK_LIMIT} tracks total.",
            on_change=self._handle_audio_upload,
            disabled=limit_reached,
        )

    def render_affirmation_inputs(self):
        """Renders the affirmation input options (TTS, File) in the sidebar."""
        st.subheader("🗣️ Add Affirmations")
        st.caption(f"Uses system default TTS voice. Max {MAX_AFFIRMATION_CHARS} chars.")
        limit_reached = len(self.app_state.get_all_tracks()) >= MAX_TRACK_LIMIT
        if limit_reached:
            st.warning(f"Track limit ({MAX_TRACK_LIMIT}) reached.", icon="⚠️")
        tab1, tab2, tab3 = st.tabs(["Type Text", "Upload File", "Record Audio"])
        with tab1:
            st.caption("Type or paste affirmations below (one per line recommended).")
            affirmation_text = st.text_area(
                "Affirmation Text",
                height=150,
                key="sidebar_affirmation_text_area",
                label_visibility="collapsed",
                help="Enter affirmations.",
                max_chars=MAX_AFFIRMATION_CHARS,
                disabled=limit_reached,
            )
            st.caption(f"{len(affirmation_text)} / {MAX_AFFIRMATION_CHARS} characters")
            if st.button(
                "Generate Affirmation Track",
                key="sidebar_generate_tts_from_text",
                use_container_width=True,
                type="primary",
                help="Convert text to spoken audio track.",
                disabled=limit_reached,
            ):
                if not self._check_track_limit(adding_count=1):
                    return
                text_to_generate = st.session_state.get("sidebar_affirmation_text_area", "")
                if not text_to_generate or not text_to_generate.strip():
                    st.warning("Please enter some text.")
                elif len(text_to_generate) > MAX_AFFIRMATION_CHARS:
                    st.error(f"❌ Text too long ({len(text_to_generate)} chars). Max {MAX_AFFIRMATION_CHARS}.")
                else:
                    default_name = f"TTS: {text_to_generate[:25]}..." if len(text_to_generate) > 30 else "TTS Affirmations"
                    self._generate_tts_track(text_to_generate, default_name)
                    # Rerun handled by helper if successful
        with tab2:
            st.caption("Upload a .txt or .docx file containing affirmations.")
            st.file_uploader(
                "Upload Affirmation File (.txt, .docx)",
                type=["txt", "docx"],
                key=self.affirmation_uploader_key,
                label_visibility="collapsed",
                help="Select text/Word document.",
                disabled=limit_reached,
            )
            if st.button(
                "Generate Track from File",
                key="sidebar_generate_tts_from_file_button",
                use_container_width=True,
                type="primary",
                help="Read file and convert text to spoken audio.",
                disabled=limit_reached,
            ):
                if not self._check_track_limit(adding_count=1):
                    return
                uploaded_file = st.session_state.get(self.affirmation_uploader_key)
                if uploaded_file:
                    file_processed = False
                    try:
                        text_from_file = read_text_file(uploaded_file)
                        if text_from_file is not None:
                            if not text_from_file.strip():
                                st.warning(f"File '{uploaded_file.name}' appears empty.")
                            elif len(text_from_file) > MAX_AFFIRMATION_CHARS:
                                st.error(f"❌ Text in file '{uploaded_file.name}' too long ({len(text_from_file)} chars). Max {MAX_AFFIRMATION_CHARS}.")
                            else:
                                self._generate_tts_track(text_from_file, f"File Affirmations ({uploaded_file.name})")
                                file_processed = True  # Assume helper handles success/rerun
                    except Exception as e:
                        logger.error(f"Error processing affirmation file {uploaded_file.name}: {e}")
                        st.error(f"Failed to process file {uploaded_file.name}: {e}")
                    # Clear uploader only if processing started (even if helper failed)
                    if uploaded_file:  # Check again in case it was cleared by another process
                        try:
                            st.session_state[self.affirmation_uploader_key] = None
                            logger.debug(f"Cleared affirmation uploader state after button click processing for {uploaded_file.name}")
                            # Don't rerun here, let the helper handle it on success
                        except Exception as e_clear:
                            logger.error(f"Error clearing affirmation uploader state: {e_clear}")
                else:
                    st.warning("Please upload a .txt or .docx file first.")
        with tab3:
            st.caption("Record your own voice directly in the browser.")
            st.info("🎙️ Audio recording feature coming soon!")
            st.markdown("For now, please record using other software and use the 'Upload Audio File(s)' option.")
            st.button("Start Recording", key="sidebar_start_recording", disabled=True, use_container_width=True)

    # --- Helper Method Specific to this Class ---
    def _generate_tts_track(self, text_content: str, track_name: str):
        """Helper method to generate TTS audio, extract snippet, and add track."""
        if not self._check_track_limit(adding_count=1):
            return
        logger.info(f"SidebarUploader: Generating TTS track '{track_name}'")
        audio_snippet: Optional[AudioData] = None
        snippet_sr: Optional[int] = None
        success = False
        try:
            with st.spinner(f"Generating '{track_name}'..."):
                full_audio, full_sr = self.tts_generator.generate(text_content)
            if full_audio is not None and full_sr is not None and full_audio.size > 0:
                snippet_length_samples = int(TRACK_SNIPPET_DURATION_S * full_sr)
                audio_snippet = full_audio[:snippet_length_samples] if len(full_audio) > snippet_length_samples else full_audio
                snippet_sr = full_sr
                source_info: SourceInfoTTS = {"type": "tts", "text": text_content}
                initial_params = {"name": track_name, "track_type": TRACK_TYPE_AFFIRMATION}
                track_id = self.app_state.add_track(audio_snippet=audio_snippet, source_info=source_info, sr=snippet_sr, initial_params=initial_params)
                st.success(f"'{track_name}' track generated (ID: {track_id[:6]})!")
                st.toast("Affirmation track added!", icon="✅")
                success = True
            else:
                logger.error("TTS generation returned None or empty.")
                st.error(f"Failed to generate audio for '{track_name}'.")
        except Exception as e:
            logger.exception(f"Error during TTS track generation for '{track_name}'.")
            st.error(f"Failed to create TTS track '{track_name}': {e}")
        # Rerun only on success
        if success:
            st.rerun()
