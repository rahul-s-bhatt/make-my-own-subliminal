# sidebar_uploader.py
# ==========================================
# File Upload and TTS UI for MindMorph Sidebar
# ==========================================

import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import streamlit as st

# Import the affirmation expander function
from affirmation_expander import expand_affirmations

# Import necessary components from other modules
from app_state import AppState
from audio_utils.audio_io import load_audio
from audio_utils.audio_state_definitions import (
    AudioData,
    SourceInfoTTS,
    SourceInfoUpload,
)
from config import (
    GLOBAL_SR,
    MAX_AFFIRMATION_CHARS,
    MAX_AUDIO_DURATION_S,
    MAX_TRACK_LIMIT,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    TRACK_SNIPPET_DURATION_S,
    TRACK_TYPE_AFFIRMATION,
    TRACK_TYPE_BACKGROUND,
    TRACK_TYPE_OTHER,
    TRACK_TYPE_VOICE,
)
from tts.base_tts import BaseTTSGenerator
from utils import read_text_file

# Get a logger for this module
logger = logging.getLogger(__name__)


class SidebarUploader:
    """Handles rendering file uploaders and related actions in the sidebar."""

    # --- MODIFIED TYPE HINT ---
    def __init__(self, app_state: AppState, tts_generator: BaseTTSGenerator):
        # --- END MODIFIED TYPE HINT ---
        """
        Initializes the SidebarUploader.

        Args:
            app_state: An instance of the AppState class.
            tts_generator: An instance of a TTS generator (conforming to BaseTTSGenerator).
        """
        self.app_state = app_state
        self.tts_generator = tts_generator  # Store the passed-in generator instance
        self.audio_uploader_key = "sidebar_audio_file_uploader"
        self.affirmation_uploader_key = "sidebar_affirmation_file_uploader"
        # Session state keys for expansion results and main text area
        self.expansion_result_key = "sidebar_expansion_result"
        self.expansion_truncated_key = "sidebar_expansion_truncated"
        self.affirmation_text_area_key = "sidebar_affirmation_text_area"
        # Keys for Undo and Pending Update mechanism
        self.affirm_original_text_key = "sidebar_affirm_original_text"
        self.affirm_pending_update_key = "sidebar_affirm_text_pending_update"
        self.affirm_pending_truncated_key = "sidebar_affirm_truncated_pending"

        # Initialize expansion/pending/original state if not present
        # (Initialization logic remains the same)
        if self.expansion_result_key not in st.session_state:
            st.session_state[self.expansion_result_key] = None
        if self.expansion_truncated_key not in st.session_state:
            st.session_state[self.expansion_truncated_key] = False
        if self.affirmation_text_area_key not in st.session_state:
            st.session_state[self.affirmation_text_area_key] = ""
        if self.affirm_original_text_key not in st.session_state:
            st.session_state[self.affirm_original_text_key] = None
        if self.affirm_pending_update_key not in st.session_state:
            st.session_state[self.affirm_pending_update_key] = None
        if self.affirm_pending_truncated_key not in st.session_state:
            st.session_state[self.affirm_pending_truncated_key] = False

        logger.debug("SidebarUploader initialized.")

    # --- Check Track Limit Helper ---
    def _check_track_limit(self, adding_count: int = 1) -> bool:
        """Checks if adding tracks would exceed the limit."""
        # (remains the same)
        current_count = len(self.app_state.get_all_tracks())
        if current_count + adding_count > MAX_TRACK_LIMIT:
            logger.warning(
                f"Track limit ({MAX_TRACK_LIMIT}) reached. Cannot add {adding_count} more track(s)."
            )
            st.error(
                f"Cannot add more tracks. Maximum limit of {MAX_TRACK_LIMIT} reached.",
                icon="🚫",
            )
            return False
        return True

    # --- Callback Functions for File Uploaders ---
    def _handle_audio_upload(self):
        """
        Callback function to process uploaded audio files.
        """
        # (remains the same)
        logger.debug(
            f"Callback triggered: _handle_audio_upload (key: {self.audio_uploader_key})"
        )
        uploaded_files = st.session_state.get(self.audio_uploader_key)
        if not uploaded_files:
            return

        files_to_process = list(uploaded_files)
        files_processed_successfully = False
        files_skipped_or_failed = 0

        if not self._check_track_limit(adding_count=len(files_to_process)):
            return

        current_tracks = self.app_state.get_all_tracks()
        current_track_filenames = {
            tdata["source_info"]["original_filename"]
            for _, tdata in current_tracks.items()
            if isinstance(tdata.get("source_info"), dict)
            and tdata["source_info"].get("type") == "upload"
            and "original_filename" in tdata["source_info"]
        }

        for file in files_to_process:
            if not self._check_track_limit(adding_count=1):
                break
            logger.info(
                f"Processing uploaded file: {file.name} (Size: {file.size} bytes)"
            )
            if file.size > MAX_UPLOAD_SIZE_BYTES:
                logger.warning(f"Upload '{file.name}' rejected: Size exceeds limit.")
                st.error(
                    f"❌ File '{file.name}' ({file.size / (1024 * 1024):.1f} MB) exceeds {MAX_UPLOAD_SIZE_MB} MB limit."
                )
                files_skipped_or_failed += 1
                continue
            if file.name in current_track_filenames:
                logger.info(f"Skipping upload: Track for '{file.name}' already exists.")
                st.warning(
                    f"⚠️ Track with filename '{file.name}' already exists. Skipping."
                )
                files_skipped_or_failed += 1
                continue

            temp_file_path = None
            audio_snippet = None
            snippet_sr = None
            try:
                # Save temp file and load snippet (remains the same)
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(file.name)[1]
                ) as tmp:
                    tmp.write(file.getvalue())
                    temp_file_path = tmp.name
                logger.info(
                    f"Saved uploaded file '{file.name}' temporarily to: {temp_file_path}"
                )
                audio_snippet, snippet_sr = load_audio(
                    temp_file_path,
                    target_sr=GLOBAL_SR,
                    duration=TRACK_SNIPPET_DURATION_S,
                )
            except Exception as e:
                # Error handling (remains the same)
                logger.error(
                    f"Error saving temp file or loading snippet for {file.name}: {e}",
                    exc_info=True,
                )
                st.error(f"Failed to process {file.name}: {e}")
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        logger.warning(
                            f"Failed cleanup temp file {temp_file_path} after error."
                        )
                files_skipped_or_failed += 1
                continue

            # Process loaded snippet (remains the same)
            if (
                audio_snippet is not None
                and snippet_sr is not None
                and audio_snippet.size > 0
            ):
                files_processed_successfully = True
                # Determine track type (remains the same)
                if any(
                    k in file.name.lower()
                    for k in ["voice", "record", "affirmation", "tts"]
                ):
                    track_type = TRACK_TYPE_VOICE
                elif any(
                    k in file.name.lower()
                    for k in ["music", "background", "mask", "ambient"]
                ):
                    track_type = TRACK_TYPE_BACKGROUND
                else:
                    track_type = TRACK_TYPE_OTHER
                source_info: SourceInfoUpload = {
                    "type": "upload",
                    "temp_file_path": temp_file_path,
                    "original_filename": file.name,
                }
                initial_params = {
                    "name": os.path.splitext(file.name)[0],
                    "track_type": track_type,
                }
                # Add track (remains the same)
                new_track_id = self.app_state.add_track(
                    audio_snippet=audio_snippet,
                    source_info=source_info,
                    sr=snippet_sr,
                    initial_params=initial_params,
                )
                duration_seconds = (
                    len(audio_snippet) / snippet_sr if snippet_sr > 0 else 0
                )
                st.success(
                    f"Loaded '{file.name}' as '{track_type}' (Snippet: {duration_seconds:.1f}s)"
                )
                current_track_filenames.add(file.name)
                # Clear text states (remains the same)
                st.session_state[self.affirmation_text_area_key] = ""
                st.session_state[self.affirm_original_text_key] = None
                st.session_state[self.affirm_pending_update_key] = None
                st.session_state[self.affirm_pending_truncated_key] = False
                st.session_state[self.expansion_result_key] = None
                st.session_state[self.expansion_truncated_key] = False
            elif audio_snippet is None:
                # Error handling (remains the same)
                logger.error(f"Failed to load audio snippet: {file.name}")
                st.error(f"Failed to load audio data from {file.name}.")
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        logger.warning(
                            f"Failed cleanup temp file {temp_file_path} after load failure."
                        )
                files_skipped_or_failed += 1
            else:
                # Handle empty snippet (remains the same)
                logger.warning(f"Skipped empty/invalid audio snippet: {file.name}")
                st.warning(
                    f"Audio file '{file.name}' appears empty or could not be processed."
                )
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        logger.warning(
                            f"Failed cleanup temp file {temp_file_path} for empty audio."
                        )
                files_skipped_or_failed += 1

        # Logging and rerun logic (remains the same)
        if files_processed_successfully:
            logger.debug(
                f"Audio upload callback finished. Processed: {len(files_to_process) - files_skipped_or_failed}, Skipped/Failed: {files_skipped_or_failed}"
            )
        elif uploaded_files:
            logger.debug(
                f"Audio upload callback finished. Skipped/Failed all {len(files_to_process)} files."
            )
        if files_processed_successfully:
            st.rerun()

    # --- Rendering Methods for Uploader Sections ---

    def render_uploader(self):
        """Renders the audio file uploader component in the sidebar."""
        # (remains the same)
        st.subheader("📁 Upload Audio File(s)")
        st.caption(
            f"Upload music, voice, etc. (Max duration: {MAX_AUDIO_DURATION_S // 60} min approx, Max size: {MAX_UPLOAD_SIZE_MB} MB)"
        )
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
        # (remains the same, including expansion logic and text area callbacks)
        st.subheader("🗣️ Add Affirmations")
        st.caption(
            f"Uses high-quality offline TTS. Max {MAX_AFFIRMATION_CHARS} chars."
        )  # Updated caption
        limit_reached = len(self.app_state.get_all_tracks()) >= MAX_TRACK_LIMIT
        if limit_reached:
            st.warning(f"Track limit ({MAX_TRACK_LIMIT}) reached.", icon="⚠️")

        # Apply pending update (remains the same)
        if st.session_state.get(self.affirm_pending_update_key) is not None:
            logger.debug("Applying pending affirmation text update (Advanced Editor).")
            st.session_state[self.affirmation_text_area_key] = st.session_state[
                self.affirm_pending_update_key
            ]
            if st.session_state.get(self.affirm_pending_truncated_key):
                st.warning(
                    f"⚠️ Expanded text shortened to fit {MAX_AFFIRMATION_CHARS} chars.",
                    icon="✂️",
                )
            st.session_state[self.affirm_pending_update_key] = None
            st.session_state[self.affirm_pending_truncated_key] = False

        tab1, tab2, tab3 = st.tabs(["Type Text", "Upload File", "Record Audio"])
        with tab1:  # Type Text Tab
            st.caption("Type or paste affirmations below (one per line recommended).")
            affirmation_text_value = st.session_state.get(
                self.affirmation_text_area_key, ""
            )

            def clear_original_on_edit():  # Callback
                current_val = st.session_state.get(self.affirmation_text_area_key)
                original_val = st.session_state.get(self.affirm_original_text_key)
                if original_val is not None and current_val != original_val:
                    st.session_state[self.affirm_original_text_key] = None
                    st.session_state[self.expansion_result_key] = None
                    st.session_state[self.expansion_truncated_key] = False
                    logger.debug(
                        "Cleared original affirmation backup and display due to manual edit."
                    )

            # Text Area (remains the same)
            affirmation_text = st.text_area(
                "Affirmation Text",
                value=affirmation_text_value,
                height=150,
                key=self.affirmation_text_area_key,
                label_visibility="collapsed",
                help="Enter affirmations.",
                max_chars=MAX_AFFIRMATION_CHARS,
                disabled=limit_reached,
                on_change=clear_original_on_edit,
            )
            st.caption(
                f"{len(affirmation_text_value)} / {MAX_AFFIRMATION_CHARS} characters"
            )

            # Expansion/Undo Buttons (remain the same)
            expand_col, undo_col = st.columns(2)
            with expand_col:  # Expand
                expand_disabled = limit_reached or not affirmation_text_value.strip()
                if st.button(
                    "✨ Expand Affirmations",
                    key="sidebar_expand_affirmations",
                    disabled=expand_disabled,
                    use_container_width=True,
                ):
                    with st.spinner("Expanding affirmations..."):
                        try:
                            st.session_state[self.affirm_original_text_key] = (
                                affirmation_text_value
                            )
                            expanded_text, truncated = expand_affirmations(
                                base_text=affirmation_text_value,
                                max_chars=MAX_AFFIRMATION_CHARS,
                                multiplier=3,
                            )
                            st.session_state[self.expansion_result_key] = expanded_text
                            st.session_state[self.expansion_truncated_key] = truncated
                            logger.info(
                                f"Affirmation expansion complete (Advanced). Truncated: {truncated}"
                            )
                            st.session_state[self.affirm_pending_update_key] = None
                            st.session_state[self.affirm_pending_truncated_key] = False
                            st.rerun()
                        except Exception as e:
                            logger.error(
                                f"Error during affirmation expansion: {e}",
                                exc_info=True,
                            )
                            st.error(f"Failed to expand affirmations: {e}")
                            st.session_state[self.affirm_original_text_key] = None
                            st.session_state[self.expansion_result_key] = None
                            st.session_state[self.expansion_truncated_key] = False
            with undo_col:  # Undo
                undo_disabled = (
                    st.session_state.get(self.affirm_original_text_key) is None
                )
                if st.button(
                    "↩️ Undo Expansion",
                    key="sidebar_undo_expansion",
                    disabled=undo_disabled,
                    use_container_width=True,
                ):
                    original_text = st.session_state.get(self.affirm_original_text_key)
                    if original_text is not None:
                        st.session_state[self.affirm_pending_update_key] = original_text
                        st.session_state[self.affirm_pending_truncated_key] = False
                        st.session_state[self.affirm_original_text_key] = None
                        st.session_state[self.expansion_result_key] = None
                        st.session_state[self.expansion_truncated_key] = False
                        logger.info(
                            "User staged affirmation undo for next run (Advanced)."
                        )
                        st.rerun()
                    else:
                        logger.warning(
                            "Undo clicked but no original text found in state (Advanced)."
                        )

            # Display Expansion Results (remains the same)
            if st.session_state.get(self.expansion_result_key) is not None:
                st.markdown("**Suggested Expansions:**")
                if st.session_state[self.expansion_truncated_key]:
                    st.warning(
                        f"⚠️ Text shortened to fit {MAX_AFFIRMATION_CHARS} chars.",
                        icon="✂️",
                    )
                st.text_area(
                    "Expanded Affirmations Result",
                    value=st.session_state[self.expansion_result_key],
                    height=200,
                    key="sidebar_expansion_result_display",
                    label_visibility="collapsed",
                    help="Review expansions. Click 'Use Expanded Text' to replace original.",
                    disabled=True,
                )
                if st.button(
                    "✅ Use Expanded Text",
                    key="sidebar_use_expanded_text",
                    use_container_width=True,
                ):
                    st.session_state[self.affirm_pending_update_key] = st.session_state[
                        self.expansion_result_key
                    ]
                    st.session_state[self.affirm_pending_truncated_key] = (
                        st.session_state[self.expansion_truncated_key]
                    )
                    st.session_state[self.expansion_result_key] = None
                    st.session_state[self.expansion_truncated_key] = False
                    st.session_state[self.affirm_original_text_key] = None
                    logger.info(
                        "User staged expanded affirmations for main text area update (Advanced)."
                    )
                    st.rerun()

            # Generate Track Button (remains the same)
            st.markdown("---")
            if st.button(
                "▶️ Generate Track from Text Box",
                key="sidebar_generate_tts_from_text",
                use_container_width=True,
                type="primary",
                help="Convert text in the main box above to a spoken audio track.",
                disabled=limit_reached or not affirmation_text_value.strip(),
            ):
                if not self._check_track_limit(adding_count=1):
                    return
                text_to_generate = st.session_state.get(
                    self.affirmation_text_area_key, ""
                )
                if not text_to_generate or not text_to_generate.strip():
                    st.warning("Please enter some text.")
                elif len(text_to_generate) > MAX_AFFIRMATION_CHARS:
                    st.error(
                        f"❌ Text too long ({len(text_to_generate)} chars). Max {MAX_AFFIRMATION_CHARS}."
                    )
                else:
                    default_name = (
                        f"TTS: {text_to_generate[:25]}..."
                        if len(text_to_generate) > 30
                        else "TTS Affirmations"
                    )
                    generation_successful = self._generate_tts_track(
                        text_to_generate, default_name
                    )
                    if generation_successful:
                        # Stage text area clear
                        st.session_state[self.affirm_pending_update_key] = ""
                        st.session_state[self.affirm_pending_truncated_key] = False
                        # Clear other related states
                        st.session_state[self.expansion_result_key] = None
                        st.session_state[self.expansion_truncated_key] = False
                        st.session_state[self.affirm_original_text_key] = None
                        logger.info(
                            "Staged state clearing after successful TTS generation."
                        )
                        st.rerun()

        with tab2:  # Upload File Tab
            # (remains the same)
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
                    generation_successful = False
                    try:
                        text_from_file = read_text_file(uploaded_file)
                        if text_from_file is not None:
                            if not text_from_file.strip():
                                st.warning(
                                    f"File '{uploaded_file.name}' appears empty."
                                )
                            elif len(text_from_file) > MAX_AFFIRMATION_CHARS:
                                st.error(
                                    f"❌ Text in file '{uploaded_file.name}' too long ({len(text_from_file)} chars). Max {MAX_AFFIRMATION_CHARS}."
                                )
                            else:
                                # Clear other text states
                                st.session_state[self.affirmation_text_area_key] = ""
                                st.session_state[self.expansion_result_key] = None
                                st.session_state[self.expansion_truncated_key] = False
                                st.session_state[self.affirm_original_text_key] = None
                                st.session_state[self.affirm_pending_update_key] = None
                                st.session_state[self.affirm_pending_truncated_key] = (
                                    False
                                )
                                # Generate track
                                generation_successful = self._generate_tts_track(
                                    text_from_file,
                                    f"File Affirmations ({uploaded_file.name})",
                                )
                    except Exception as e:
                        logger.error(
                            f"Error processing affirmation file {uploaded_file.name}: {e}"
                        )
                        st.error(f"Failed to process file {uploaded_file.name}: {e}")
                    if uploaded_file:  # Clear uploader state
                        try:
                            st.session_state[self.affirmation_uploader_key] = None
                            logger.debug(
                                f"Cleared affirmation uploader state for {uploaded_file.name}"
                            )
                        except Exception as e_clear:
                            logger.error(
                                f"Error clearing affirmation uploader state: {e_clear}"
                            )
                    if generation_successful:
                        st.rerun()  # Rerun only on success
                else:
                    st.warning("Please upload a .txt or .docx file first.")

        with tab3:  # Record Audio Tab
            # (remains the same)
            st.caption("Record your own voice directly in the browser.")
            st.info("🎙️ Audio recording feature coming soon!")
            st.markdown(
                "For now, please record using other software and use the 'Upload Audio File(s)' option."
            )
            st.button(
                "Start Recording",
                key="sidebar_start_recording",
                disabled=True,
                use_container_width=True,
            )

    # --- Helper Method Specific to this Class ---
    def _generate_tts_track(self, text_content: str, track_name: str) -> bool:
        """
        Helper method to generate TTS audio, extract snippet, and add track.
        Returns True if track generation and addition was successful, False otherwise.
        """
        if not self._check_track_limit(adding_count=1):
            return False
        logger.info(
            f"SidebarUploader: Generating TTS track '{track_name}' using provided generator."
        )
        audio_snippet: Optional[AudioData] = None
        snippet_sr: Optional[int] = None
        success = False
        try:
            # Use the TTS generator instance passed during initialization
            with st.spinner(f"Generating '{track_name}'..."):
                # Call the standard generate method on the instance
                full_audio, full_sr = self.tts_generator.generate(text_content)

            # Process the result (remains the same)
            if full_audio is not None and full_sr is not None and full_audio.size > 0:
                snippet_length_samples = int(TRACK_SNIPPET_DURATION_S * full_sr)
                audio_snippet = (
                    full_audio[:snippet_length_samples]
                    if len(full_audio) > snippet_length_samples
                    else full_audio
                )
                snippet_sr = full_sr
                source_info: SourceInfoTTS = {"type": "tts", "text": text_content}
                initial_params = {
                    "name": track_name,
                    "track_type": TRACK_TYPE_AFFIRMATION,
                }
                track_id = self.app_state.add_track(
                    audio_snippet=audio_snippet,
                    source_info=source_info,
                    sr=snippet_sr,
                    initial_params=initial_params,
                )
                st.success(f"'{track_name}' track generated (ID: {track_id[:6]})!")
                st.toast("Affirmation track added!", icon="✅")
                success = True
            else:
                logger.error("TTS generation returned None or empty.")
                st.error(f"Failed to generate audio for '{track_name}'.")
        except Exception as e:
            logger.exception(f"Error during TTS track generation for '{track_name}'.")
            st.error(f"Failed to create TTS track '{track_name}': {e}")

        return success
