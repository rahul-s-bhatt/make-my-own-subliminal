# ui_manager.py
# ==========================================
# Main UI Orchestrator for MindMorph
# ==========================================

import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict

import numpy as np
import streamlit as st

# Import necessary components from other modules
from app_state import AppState

# --- Updated Audio Imports ---
from audio_io import save_audio_to_bytesio
from audio_processing import AudioData, mix_tracks

# --- End Updated Audio Imports ---
from config import (
    GLOBAL_SR,
    MIX_PREVIEW_DURATION_S,
    # MIX_PREVIEW_PROCESSING_BUFFER_S, # No longer needed here
    PROJECT_FILE_VERSION,  # For instructions footer
)

# Import the sub-managers
from sidebar_manager import SidebarManager
from track_editor_manager import TrackEditorManager
from tts_generator import TTSGenerator

# Optional MP3 export dependency check
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Get a logger for this module
logger = logging.getLogger(__name__)


class UIManager:
    """
    Orchestrates the overall UI rendering by coordinating sub-managers.

    Handles rendering of master controls and instructions.
    """

    def __init__(self, app_state: AppState, tts_generator: TTSGenerator):
        """
        Initializes the UIManager and its sub-managers.

        Args:
            app_state: An instance of the AppState class.
            tts_generator: An instance of the TTSGenerator class.
        """
        self.app_state = app_state
        # Note: tts_generator is passed to SidebarManager where it's needed
        self.sidebar_manager = SidebarManager(app_state, tts_generator)
        self.track_editor_manager = TrackEditorManager(app_state)
        logger.debug("UIManager initialized with sub-managers.")

    # --- Main Rendering Method ---

    def render_ui(self, mode: str = "Easy"):  # Accept mode from main.py
        """
        Renders the complete user interface based on the selected mode.

        Args:
            mode (str): The current editor mode ("Easy" or "Advanced").
        """
        # Render sidebar using SidebarManager
        self.sidebar_manager.render_sidebar()

        # Render main panel sections
        # Mode selector and title are handled in main.py before calling this

        # Render track editor using TrackEditorManager, passing the mode
        # TrackEditorManager handles the empty state message internally
        self.track_editor_manager.render_tracks_editor(mode=mode)

        # Render master controls (managed by this class)
        self.render_master_controls()

        # Render master preview player (managed by this class)
        self.render_preview_audio_player()

        # Render instructions (managed by this class)
        self.render_instructions()

        # Footer (managed by this class or main.py)
        st.divider()
        st.caption(f"MindMorph Subliminal Editor | Version: {PROJECT_FILE_VERSION}")

    # --- Master Controls Rendering and Handling ---

    def render_master_controls(self):
        """Renders the master output controls (Preview, Export, Filename)."""
        st.divider()

        st.header("🔊 Master Output")
        st.caption("Preview the combined mix or export the final audio file.")
        st.markdown("---")  # Keep separator after header

        default_filename = "mindmorph_mix"
        export_filename_input = st.text_input(
            "Export Filename (no extension):",
            value=st.session_state.get("export_filename", default_filename),
            key="master_export_filename_input",  # Unique key
            help="Enter the desired name for the exported file (invalid chars will be removed).",
        )
        sanitized_filename = re.sub(r'[\\/*?:"<>|]', "", export_filename_input).strip()
        if not sanitized_filename:
            sanitized_filename = default_filename
        # Store sanitized name in session state for download button
        st.session_state.export_filename_sanitized = sanitized_filename

        # Display estimated duration if available (calculated during export)
        if "calculated_mix_duration_s" in st.session_state and st.session_state.calculated_mix_duration_s is not None:
            duration_s = st.session_state.calculated_mix_duration_s
            duration_str = f"{duration_s:.2f} seconds"
            if duration_s > 60:
                duration_str = f"{duration_s // 60:.0f} min {duration_s % 60:.1f} sec"
            st.info(f"Last Export Duration: **{duration_str}**")
        else:
            st.caption("Full mix duration will be calculated upon export.")

        master_cols = st.columns(2)
        with master_cols[0]:
            if st.button(
                f"🎧 Preview Mix ({MIX_PREVIEW_DURATION_S}s)",
                key="master_preview_mix_button",
                use_container_width=True,
                help="Generate and play the first few seconds of the final mix.",
            ):
                self._handle_preview_click()

        with master_cols[1]:
            export_formats = ["WAV"]
            help_text = "Export in WAV format (lossless, larger file size)."
            if PYDUB_AVAILABLE:
                export_formats.append("MP3")
                help_text = "Choose WAV (lossless, large) or MP3 (compressed, smaller - requires ffmpeg)."
            else:
                help_text += " MP3 export disabled (requires 'pydub' library and 'ffmpeg')."

            export_format = st.radio("Export Format:", export_formats, key="master_export_format_selection", horizontal=True, help=help_text)

            export_disabled = (export_format == "MP3" and not PYDUB_AVAILABLE) or not self.app_state.get_all_tracks()
            export_button_label = f"💾 Export Full Mix (.{export_format.lower()})"
            export_help = f"Generate the complete final mix as a .{export_format.lower()} file."
            if export_disabled and export_format == "MP3":
                export_help += " MP3 export disabled."
            elif export_disabled:
                export_help = "Add tracks before exporting."

            if st.button(export_button_label, key="master_export_mix_button", use_container_width=True, help=export_help, disabled=export_disabled):
                self._handle_export_click()

            # Display download button if buffer exists
            if "export_buffer" in st.session_state and st.session_state.export_buffer:
                file_ext = st.session_state.get("export_file_ext", "wav")
                download_filename = f"{st.session_state.get('export_filename_sanitized', default_filename)}.{file_ext}"
                mime_type = f"audio/{file_ext}" if file_ext == "wav" else "audio/mpeg"

                st.download_button(
                    label=f"⬇️ Download: {download_filename}",
                    data=st.session_state.export_buffer,
                    file_name=download_filename,
                    mime=mime_type,
                    key="master_download_export_button",
                    use_container_width=True,
                    help="Click to download the exported audio file.",
                    # Clear buffer on click to remove button after download starts
                    on_click=self._clear_export_buffer,
                )

    # <<< New helper method to clear export buffer >>>
    def _clear_export_buffer(self):
        """Clears export-related keys from session state."""
        if "export_buffer" in st.session_state:
            del st.session_state.export_buffer
        if "export_file_ext" in st.session_state:
            del st.session_state.export_file_ext
        if "calculated_mix_duration_s" in st.session_state:
            del st.session_state.calculated_mix_duration_s  # Clear duration too
        logger.info("Cleared export buffer state after download button click.")

    def _handle_preview_click(self):
        """Handles the logic when the 'Preview Mix' button is clicked."""
        logger.info("Master Preview Mix button clicked.")
        tracks = self.app_state.get_all_tracks()

        # Clear previous preview data
        if "preview_audio_data" in st.session_state:
            del st.session_state.preview_audio_data
        # Don't clear calculated_mix_duration_s on preview click

        if not tracks:
            st.warning("No tracks loaded to preview.")
            return

        with st.spinner("Generating preview mix..."):
            try:
                # <<< MODIFIED: Pass app_state, remove preview_buffer_s >>>
                mix_preview, _ = mix_tracks(
                    app_state=self.app_state,  # Pass AppState instance
                    tracks_dict=tracks,
                    preview=True,
                    preview_duration_s=MIX_PREVIEW_DURATION_S,
                    # preview_buffer_s=MIX_PREVIEW_PROCESSING_BUFFER_S, # Removed argument
                    target_sr=GLOBAL_SR,
                )
                if mix_preview is not None and mix_preview.size > 0:
                    preview_buffer = save_audio_to_bytesio(mix_preview, GLOBAL_SR)
                    st.session_state.preview_audio_data = preview_buffer
                    logger.info("Preview mix generated successfully.")
                    st.rerun()  # Rerun to show the player immediately
                elif mix_preview is not None:
                    logger.warning("Preview mix generation resulted in empty audio.")
                    st.warning("Generated preview mix is empty.")
                else:
                    logger.error("Preview mix generation failed (returned None).")
                    st.error("Failed to generate preview mix.")
            except Exception as e:
                logger.exception("Error occurred during preview mix generation.")
                st.error(f"Failed to generate preview mix: {e}")
        # No explicit rerun here, done inside try block on success

    def _handle_export_click(self):
        """Handles the logic when the 'Export Mix' button is clicked."""
        logger.info("Master Export Mix button clicked.")
        tracks = self.app_state.get_all_tracks()
        export_format = st.session_state.get("master_export_format_selection", "WAV").lower()
        # export_filename_base = st.session_state.get("export_filename_sanitized", "mindmorph_mix") # Filename used by download button

        # Clear previous export/preview state
        self._clear_export_buffer()  # Use helper method
        if "preview_audio_data" in st.session_state:
            del st.session_state.preview_audio_data

        if not tracks:
            st.warning("No tracks loaded to export.")
            return

        # Generate Full Mix
        with st.spinner(f"Generating full mix ({export_format.upper()})... This may take time."):
            try:
                # <<< MODIFIED: Pass app_state >>>
                full_mix, final_mix_len_samples = mix_tracks(
                    app_state=self.app_state,  # Pass AppState instance
                    tracks_dict=tracks,
                    preview=False,
                    target_sr=GLOBAL_SR,
                )

                # Store calculated duration if available
                if final_mix_len_samples is not None and GLOBAL_SR > 0:
                    st.session_state.calculated_mix_duration_s = final_mix_len_samples / GLOBAL_SR
                    logger.info(f"Actual final mix duration (post-looping): {st.session_state.calculated_mix_duration_s:.2f}s")
                else:
                    st.session_state.calculated_mix_duration_s = None

                if full_mix is not None and full_mix.size > 0:
                    # Save to buffer based on format
                    if export_format == "wav":
                        export_buffer = save_audio_to_bytesio(full_mix, GLOBAL_SR)
                        if export_buffer:
                            st.session_state.export_buffer = export_buffer
                            st.session_state.export_file_ext = "wav"
                            logger.info("Full WAV mix generated and stored.")
                        else:
                            raise ValueError("Failed to save WAV mix to buffer.")
                    elif export_format == "mp3" and PYDUB_AVAILABLE:
                        try:
                            logger.info("Converting full mix to MP3...")
                            # Ensure audio is in correct range [-1, 1] before scaling
                            full_mix_clipped = np.clip(full_mix, -1.0, 1.0)
                            audio_int16 = (full_mix_clipped * 32767).astype(np.int16)
                            channels = 2 if audio_int16.ndim > 1 and audio_int16.shape[1] == 2 else 1
                            segment = AudioSegment(data=audio_int16.tobytes(), sample_width=audio_int16.dtype.itemsize, frame_rate=GLOBAL_SR, channels=channels)
                            if channels == 1:
                                segment = segment.set_channels(2)
                                logger.info("Converted mono mix to stereo for MP3 export.")

                            mp3_buffer = BytesIO()
                            segment.export(mp3_buffer, format="mp3", bitrate="192k")
                            mp3_buffer.seek(0)
                            st.session_state.export_buffer = mp3_buffer
                            st.session_state.export_file_ext = "mp3"
                            logger.info("Full MP3 mix generated and stored.")
                        except Exception as e_mp3:
                            logger.exception("Failed to export mix as MP3 using pydub.")
                            st.error(f"MP3 Export Failed: {e_mp3}. Ensure ffmpeg is installed and accessible.")
                            self._clear_export_buffer()  # Clear buffer on MP3 error
                    else:  # Handle unsupported format or missing library
                        if export_format == "mp3":
                            logger.error("MP3 export selected, but 'pydub' library is missing.")
                            st.error("MP3 export requires 'pydub' and 'ffmpeg'. Please install them or choose WAV.")
                        else:
                            logger.error(f"Unsupported export format '{export_format}' selected.")
                            st.error(f"Export format '{export_format}' is not supported.")
                        self._clear_export_buffer()

                elif full_mix is not None:
                    logger.warning("Full mix generation resulted in empty audio.")
                    st.warning("Generated mix is empty. Check track settings.")
                    self._clear_export_buffer()
                else:
                    logger.error("Full mix generation failed (returned None).")
                    st.error("Failed to generate the final mix.")
                    self._clear_export_buffer()
            except Exception as e:
                logger.exception("Error occurred during full mix generation or export.")
                st.error(f"Failed to generate or export mix: {e}")
                self._clear_export_buffer()

        # Rerun to display download button if export was successful
        if "export_buffer" in st.session_state and st.session_state.export_buffer:
            st.rerun()

    def render_preview_audio_player(self):
        """Displays the master preview audio player if preview data exists."""
        if "preview_audio_data" in st.session_state and st.session_state.preview_audio_data:
            logger.debug("Rendering master preview audio player.")
            st.markdown("**Mix Preview:**")
            try:
                st.session_state.preview_audio_data.seek(0)
                st.audio(st.session_state.preview_audio_data, format="audio/wav")
            except Exception as e:
                logger.error(f"Error playing preview audio: {e}")
                st.error("Could not play generated preview audio.")
            # Clear after displaying to prevent re-showing on next rerun
            # Maybe only clear if explicitly requested or after download? Let's keep it for now.
            # Consider adding a "Clear Preview" button if needed.
            # del st.session_state.preview_audio_data

    def render_instructions(self):
        """Renders the instructions expander."""
        tracks_exist = bool(self.app_state.get_all_tracks())
        st.divider()
        with st.expander("📖 Show Instructions & Notes", expanded=not tracks_exist):
            st.markdown("""
            **Welcome to MindMorph!** Create custom subliminal audio...

            **Workflow:**
            1.  **✨ Select Mode (Top):** Easy or Advanced.
            2.  **➕ Add Tracks (Sidebar):** Upload, TTS, Noise, Frequencies.
            3.  **🎚️ Edit Tracks (Main Panel):** Adjust settings. Click `⚙️ Update Preview`.
            4.  **🔊 Mix & Export (Bottom Panel):** Preview, Export, Download.

            **Tips:**
            * Subliminal: High speed (4x-10x), low volume (0.05-0.15). Try '⚡ Quick...' button.
            * Advanced: '🔊 Ultrasonic Shift' for silent (experimental).
            * Advanced: `🔁 Loop` for short sounds.
            * Save/Load project via sidebar. Re-upload audio when loading.
            """)  # Keep instructions concise
