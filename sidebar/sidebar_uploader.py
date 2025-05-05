# sidebar_uploader.py
# ==========================================
# File Upload and TTS UI for MindMorph Sidebar
# STEP 1 OPTIMIZED: Removed immediate snippet loading/generation on track add.
# ==========================================

import logging
import os
import tempfile

import streamlit as st

from affirmation_expander import expand_affirmations
from app_state import AppState

# --- REMOVED: load_audio import (no longer needed here) ---
# from audio_utils.audio_io import load_audio
from audio_utils.audio_state_definitions import (
    AudioData,
)  # Keep for type hints if needed elsewhere, but not directly used now
from audio_utils.audio_state_definitions import SourceInfoTTS, SourceInfoUpload
from config import GLOBAL_SR  # Keep for potential future use or reference
from config import MAX_AUDIO_DURATION_S  # Keep for display text
from config import (
    TRACK_SNIPPET_DURATION_S,
)  # Keep for potential future use or reference
from config import (
    MAX_AFFIRMATION_CHARS,
    MAX_TRACK_LIMIT,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    TRACK_TYPE_AFFIRMATION,
    TRACK_TYPE_BACKGROUND,
    TRACK_TYPE_OTHER,
    TRACK_TYPE_VOICE,
)
from tts.base_tts import BaseTTSGenerator

logger = logging.getLogger(__name__)


class SidebarUploader:
    """Handles rendering file uploaders and related actions in the sidebar."""

    def __init__(self, app_state: AppState, tts_generator: BaseTTSGenerator):
        """Initializes the SidebarUploader."""
        self.app_state = app_state
        self.tts_generator = tts_generator  # Keep TTS generator for triggering generation later if needed from here
        self.audio_uploader_key = "sidebar_audio_file_uploader"
        self.affirmation_uploader_key = "sidebar_affirmation_file_uploader"
        self.expansion_result_key = "sidebar_expansion_result"
        self.expansion_truncated_key = "sidebar_expansion_truncated"
        self.affirmation_text_area_key = "sidebar_affirmation_text_area"
        self.affirm_original_text_key = "sidebar_affirm_original_text"
        self.affirm_pending_update_key = "sidebar_affirm_text_pending_update"
        self.affirm_pending_truncated_key = "sidebar_affirm_truncated_pending"

        # Initialize state (remains the same)
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

    def _check_track_limit(self, adding_count: int = 1) -> bool:
        """Checks if adding tracks would exceed the limit."""
        current_count = len(self.app_state.get_all_tracks())
        if current_count + adding_count > MAX_TRACK_LIMIT:
            logger.warning(f"Track limit ({MAX_TRACK_LIMIT}) reached.")
            st.error(f"Cannot add more tracks. Limit: {MAX_TRACK_LIMIT}.", icon="🚫")
            return False
        return True

    def _handle_audio_upload(self):
        """Callback to process uploaded audio files: Saves temp file, creates source_info, adds track state."""
        logger.debug(
            f"Callback triggered: _handle_audio_upload (key: {self.audio_uploader_key})"
        )
        uploaded_files = st.session_state.get(self.audio_uploader_key)
        if not uploaded_files:
            return

        files_to_process = list(uploaded_files)
        tracks_added_successfully = False
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
            # --- Size and Duplicate Checks (Remain the same) ---
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
            try:
                # --- Save temp file (Still necessary) ---
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(file.name)[1]
                ) as tmp:
                    tmp.write(file.getvalue())
                    temp_file_path = tmp.name
                logger.info(
                    f"Saved uploaded file '{file.name}' temporarily to: {temp_file_path}"
                )

                # --- REMOVED: Snippet loading ---
                # audio_snippet, snippet_sr = load_audio(...)

                # --- Add Track with Source Info Only ---
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

                # Create SourceInfo
                source_info: SourceInfoUpload = {
                    "type": "upload",
                    "temp_file_path": temp_file_path,
                    "original_filename": file.name,
                }
                # Prepare initial parameters
                initial_params = {
                    "name": os.path.splitext(file.name)[0],
                    "track_type": track_type,
                    # Set loop_to_fit based on type? Optional.
                    "loop_to_fit": track_type == TRACK_TYPE_BACKGROUND,
                }

                # Call app_state.add_track (New signature)
                new_track_id = self.app_state.add_track(
                    source_info=source_info,
                    initial_params=initial_params,
                )

                if new_track_id:
                    tracks_added_successfully = True
                    st.success(
                        f"Added '{file.name}' as '{track_type}'. Preview pending."
                    )
                    current_track_filenames.add(file.name)
                    # Clear text states (remains the same)
                    st.session_state[self.affirmation_text_area_key] = ""
                    st.session_state[self.affirm_original_text_key] = None
                    st.session_state[self.affirm_pending_update_key] = None
                    st.session_state[self.affirm_pending_truncated_key] = False
                    st.session_state[self.expansion_result_key] = None
                    st.session_state[self.expansion_truncated_key] = False
                else:
                    # Handle case where add_track failed (e.g., limit reached between check and add)
                    logger.error(
                        f"Failed to add track state for {file.name} after saving temp file."
                    )
                    st.error(
                        f"Failed to add track for {file.name}. Limit might be reached."
                    )
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            logger.info(
                                f"Cleaned up temp file {temp_file_path} after add_track failure."
                            )
                        except OSError:
                            logger.warning(
                                f"Failed cleanup temp file {temp_file_path} after add_track failure."
                            )
                    files_skipped_or_failed += 1

            except Exception as e:
                # Error handling for temp file saving or add_track call
                logger.error(f"Error processing upload {file.name}: {e}", exc_info=True)
                st.error(f"Failed to process {file.name}: {e}")
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logger.info(
                            f"Cleaned up temp file {temp_file_path} after error."
                        )
                    except OSError:
                        logger.warning(
                            f"Failed cleanup temp file {temp_file_path} after error."
                        )
                files_skipped_or_failed += 1
                continue

        # Logging and rerun logic
        if tracks_added_successfully:
            logger.debug(
                f"Audio upload callback finished. Added: {len(files_to_process) - files_skipped_or_failed}, Skipped/Failed: {files_skipped_or_failed}"
            )
            st.rerun()  # Rerun if any track was successfully added
        elif uploaded_files:
            logger.debug(
                f"Audio upload callback finished. Skipped/Failed all {len(files_to_process)} files."
            )
        # No rerun if nothing was added

    def render_uploader(self):
        """Renders the audio file uploader component in the sidebar."""
        # (UI remains the same)
        st.subheader("📁 Upload Audio File(s)")
        st.caption(f"Upload music, voice, etc. (Max size: {MAX_UPLOAD_SIZE_MB} MB)")
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
        # (UI and Expansion logic remain the same)
        st.subheader("🗣️ Add Affirmations")
        st.caption(f"Uses high-quality offline TTS. Max {MAX_AFFIRMATION_CHARS} chars.")
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

        tab1, tab3 = st.tabs(["Type Text", "Record Audio"])
        with tab1:  # Type Text Tab
            st.caption("Type or paste affirmations below (one per line recommended).")
            affirmation_text_value = st.session_state.get(
                self.affirmation_text_area_key, ""
            )

            def clear_original_on_edit():  # Callback (remains the same)
                current_val = st.session_state.get(self.affirmation_text_area_key)
                original_val = st.session_state.get(self.affirm_original_text_key)
                if original_val is not None and current_val != original_val:
                    st.session_state[self.affirm_original_text_key] = None
                    st.session_state[self.expansion_result_key] = None
                    st.session_state[self.expansion_truncated_key] = False
                    logger.debug(
                        "Cleared original affirmation backup due to manual edit."
                    )

            affirmation_text = st.text_area(  # Text Area (remains the same)
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
            with expand_col:
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
                            )
                            st.session_state[self.expansion_result_key] = expanded_text
                            st.session_state[self.expansion_truncated_key] = truncated
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
            with undo_col:
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
                        st.rerun()

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
                    help="Review expansions. Click 'Use Expanded Text'.",
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
                    st.rerun()

            # --- MODIFIED: Generate Track Button ---
            st.markdown("---")
            if st.button(
                "➕ Add Affirmation Track",  # Changed label slightly
                key="sidebar_generate_tts_from_text",
                use_container_width=True,
                type="primary",
                help="Add this text as an affirmation track. Audio will be generated later.",
                disabled=limit_reached or not affirmation_text_value.strip(),
            ):
                if not self._check_track_limit(adding_count=1):
                    return

                text_to_add = st.session_state.get(self.affirmation_text_area_key, "")
                if not text_to_add or not text_to_add.strip():
                    st.warning("Please enter some text.")
                elif len(text_to_add) > MAX_AFFIRMATION_CHARS:
                    st.error(
                        f"❌ Text too long ({len(text_to_add)} chars). Max {MAX_AFFIRMATION_CHARS}."
                    )
                else:
                    # --- Only create SourceInfo and add track state ---
                    logger.info(
                        f"SidebarUploader: Adding TTS track state for text: '{text_to_add[:50]}...'"
                    )
                    source_info: SourceInfoTTS = {"type": "tts", "text": text_to_add}
                    default_name = (
                        f"TTS: {text_to_add[:25]}..."
                        if len(text_to_add) > 30
                        else "TTS Affirmations"
                    )
                    initial_params = {
                        "name": default_name,
                        "track_type": TRACK_TYPE_AFFIRMATION,
                    }
                    # Call app_state.add_track (New signature)
                    track_id = self.app_state.add_track(
                        source_info=source_info,
                        initial_params=initial_params,
                    )
                    if track_id:
                        st.success(
                            f"'{default_name}' track added (ID: {track_id[:6]}). Preview pending."
                        )
                        st.toast("Affirmation track added!", icon="➕")
                        # Stage text area clear (remains same)
                        st.session_state[self.affirm_pending_update_key] = ""
                        st.session_state[self.affirm_pending_truncated_key] = False
                        st.session_state[self.expansion_result_key] = None
                        st.session_state[self.expansion_truncated_key] = False
                        st.session_state[self.affirm_original_text_key] = None
                        logger.info(
                            "Staged state clearing after adding TTS track state."
                        )
                        st.rerun()
                    else:
                        logger.error("Failed to add TTS track state.")
                        st.error(
                            f"Failed to add track for '{default_name}'. Limit might be reached."
                        )
                    # --- End of modification ---

        with tab3:  # Record Audio Tab (remains the same)
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

    # --- REMOVED: _generate_tts_track helper method (logic moved inline) ---
