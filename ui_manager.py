# ui_manager.py
# ==========================================
# Main UI Orchestrator for MindMorph
# ==========================================

import logging
import re
from io import BytesIO

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Import necessary components from other modules
from app_state import AppState
from audio_utils.audio_io import save_audio_to_bytesio
from audio_utils.audio_mixers import mix_tracks  # Keep this import
from config import GLOBAL_SR, MIX_PREVIEW_DURATION_S, PROJECT_FILE_VERSION

# Import the sub-managers
from sidebar.sidebar_manager import SidebarManager
from track.track_editor_manager import TrackEditorManager
from tts.base_tts import BaseTTSGenerator  # Use base class

# Optional MP3 export dependency check
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = logging.getLogger(__name__)


class UIManager:
    """
    Orchestrates the overall UI rendering by coordinating sub-managers.
    Handles rendering of master controls and instructions.
    """

    def __init__(self, app_state: AppState, tts_generator: BaseTTSGenerator):
        """Initializes the UIManager and its sub-managers."""
        self.app_state = app_state
        self.tts_generator = tts_generator  # Store the TTS generator instance
        self.sidebar_manager = SidebarManager(app_state, tts_generator)
        self.track_editor_manager = TrackEditorManager(app_state)
        if "advanced_processing_active" not in st.session_state:
            st.session_state.advanced_processing_active = False
        logger.debug("UIManager initialized with sub-managers.")

    # --- Main Rendering Method --- (Keep as is)
    def render_ui(self, mode: str = "Easy"):  # ... (no changes)
        self.sidebar_manager.render_sidebar()
        self.track_editor_manager.render_tracks_editor(mode=mode)
        self.render_master_controls()
        self.render_preview_audio_player()
        self.render_instructions()
        st.divider()
        st.caption(f"MindMorph Subliminal Editor | Version: {PROJECT_FILE_VERSION}")

    # --- Master Controls Rendering and Handling --- (Keep as is)
    def render_master_controls(self):  # ... (no changes here, button logic is correct)
        st.divider()
        st.header("🔊 Master Output")
        st.caption("Preview the combined mix or export the final audio file.")
        st.markdown("---")
        # Filename Input
        default_filename = "mindmorph_mix"
        export_filename_input = st.text_input(
            "Export Filename (no extension):",
            value=st.session_state.get("export_filename", default_filename),
            key="master_export_filename_input",
        )
        sanitized_filename = re.sub(r'[\\/*?:"<>|]', "", export_filename_input).strip()
        if not sanitized_filename:
            sanitized_filename = default_filename
        st.session_state.export_filename_sanitized = sanitized_filename
        # Display Duration
        if (
            "calculated_mix_duration_s" in st.session_state
            and st.session_state.calculated_mix_duration_s is not None
        ):
            duration_s = st.session_state.calculated_mix_duration_s
            duration_str = f"{duration_s:.2f}s"
            if duration_s > 60:
                duration_str = f"{duration_s // 60:.0f} min {duration_s % 60:.1f} sec"
            st.info(f"Last Export Duration: **{duration_str}**")
        else:
            st.caption("Full mix duration will be calculated upon export.")
        master_cols = st.columns(2)
        is_processing = st.session_state.get("advanced_processing_active", False)
        with master_cols[0]:  # Preview Button
            if st.button(
                f"🎧 Preview Mix ({MIX_PREVIEW_DURATION_S}s)",
                key="master_preview_mix_button",
                use_container_width=True,
                help="Generate and play the first few seconds of the final mix.",
                disabled=is_processing,
            ):
                if not is_processing:
                    self._handle_preview_click()
        with master_cols[1]:  # Export Section
            export_formats = ["WAV"]
            help_text = "Export in WAV format (lossless, larger file size)."
            if PYDUB_AVAILABLE:
                export_formats.append("MP3")
                help_text = "Choose WAV (lossless, large) or MP3 (compressed, smaller - requires ffmpeg)."
            else:
                help_text += (
                    " MP3 export disabled (requires 'pydub' library and 'ffmpeg')."
                )
            export_format = st.radio(
                "Export Format:",
                export_formats,
                key="master_export_format_selection",
                horizontal=True,
                help=help_text,
                disabled=is_processing,
            )
            # Export Button Logic
            mp3_unavailable = export_format == "MP3" and not PYDUB_AVAILABLE
            no_tracks = not self.app_state.get_all_tracks()
            export_disabled = is_processing or no_tracks or mp3_unavailable
            export_tooltip = ""
            generate_button_label = f"💾 Export Full Mix (.{export_format.lower()})"
            if is_processing:
                export_tooltip = "Processing... Please wait."
                generate_button_label = "⏳ Processing..."
            elif no_tracks:
                export_tooltip = "Add tracks before exporting."
            elif mp3_unavailable:
                export_tooltip = "MP3 export requires 'pydub' and 'ffmpeg'."
            else:
                export_tooltip = f"Generate the complete final mix as a .{export_format.lower()} file."
            if st.button(
                generate_button_label,
                key="master_export_mix_button",
                use_container_width=True,
                help=export_tooltip,
                disabled=export_disabled,
                type="primary",
            ):
                st.session_state.advanced_processing_active = True
                logger.info("Set advanced_processing_active flag to True.")
                self._clear_export_buffer()
                if "preview_audio_data" in st.session_state:
                    del st.session_state.preview_audio_data
                st.rerun()
            # Perform processing if flag was just set
            if st.session_state.get("advanced_processing_active", False):
                if st.session_state.get("export_buffer") is None:
                    logger.info(
                        "Processing flag is True, starting advanced export process..."
                    )
                    self._handle_export_click()
                    logger.info(
                        "Advanced processing finished, triggering rerun to display results."
                    )
                    st.rerun()
            # Download button logic
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
                    on_click=self._clear_export_buffer,
                )
            elif (
                "export_error_message" in st.session_state
                and st.session_state.export_error_message
            ):
                st.error(f"Export Failed: {st.session_state.export_error_message}")
                st.session_state.export_error_message = None

    def _clear_export_buffer(self):  # ... (no changes)
        keys_to_clear = [
            "export_buffer",
            "export_file_ext",
            "calculated_mix_duration_s",
            "export_error_message",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        logger.info("Cleared export buffer state.")

    def _handle_preview_click(self):  # ... (no changes needed here)
        logger.info("Master Preview Mix button clicked.")
        tracks = self.app_state.get_all_tracks()
        if "preview_audio_data" in st.session_state:
            del st.session_state.preview_audio_data
        if not tracks:
            st.warning("No tracks loaded to preview.")
            return
        with st.spinner("Generating preview mix..."):
            try:
                # --- Pass tts_generator=None or handle optional arg in mix_tracks if needed ---
                # Since preview doesn't regenerate, passing TTS isn't strictly necessary,
                # but if mix_tracks signature requires it, pass self.tts_generator.
                # Assuming mix_tracks handles preview without needing TTS for now.
                mix_preview, _ = mix_tracks(
                    tracks_dict=tracks,
                    tts_generator=self.tts_generator,  # Pass it even for preview for consistent signature
                    preview=True,
                    preview_duration_s=MIX_PREVIEW_DURATION_S,
                    target_sr=GLOBAL_SR,
                )
                if mix_preview is not None and mix_preview.size > 0:
                    preview_buffer = save_audio_to_bytesio(mix_preview, GLOBAL_SR)
                    st.session_state.preview_audio_data = preview_buffer
                    logger.info("Preview mix generated successfully.")
                    st.rerun()
                elif mix_preview is not None:
                    logger.warning("Preview mix is empty.")
                    st.warning("Generated preview is empty.")
                else:
                    logger.error("Preview mix generation failed.")
                    st.error("Failed to generate preview.")
            except Exception as e:
                logger.exception("Error during preview mix.")
                st.error(f"Preview failed: {e}")

    def _handle_export_click(self):
        """Handles the logic when the 'Export Mix' button is clicked."""
        logger.info("Advanced Editor: _handle_export_click started.")
        st.session_state.export_error_message = None
        export_successful = False
        try:
            tracks = self.app_state.get_all_tracks()
            export_format = st.session_state.get(
                "master_export_format_selection", "WAV"
            ).lower()
            if not tracks:
                st.session_state.export_error_message = "No tracks loaded to export."
                logger.warning(st.session_state.export_error_message)
                return

            # --- MODIFIED: Pass self.tts_generator to mix_tracks ---
            full_mix, final_mix_len_samples = mix_tracks(
                tracks_dict=tracks,
                tts_generator=self.tts_generator,  # Pass the stored instance
                preview=False,
                target_sr=GLOBAL_SR,
            )
            # --- END MODIFIED ---

            # Calculate duration (keep as is)
            if final_mix_len_samples is not None and GLOBAL_SR > 0:
                st.session_state.calculated_mix_duration_s = (
                    final_mix_len_samples / GLOBAL_SR
                )
                logger.info(
                    f"Actual final mix duration: {st.session_state.calculated_mix_duration_s:.2f}s"
                )
            else:
                st.session_state.calculated_mix_duration_s = None

            # Export logic (keep as is)
            if full_mix is not None and full_mix.size > 0:
                if export_format == "wav":
                    export_buffer = save_audio_to_bytesio(full_mix, GLOBAL_SR)
                    if export_buffer and export_buffer.getbuffer().nbytes > 0:
                        st.session_state.export_buffer = export_buffer
                        st.session_state.export_file_ext = "wav"
                        logger.info("Full WAV mix generated.")
                        export_successful = True
                    else:
                        raise ValueError("Failed to save WAV mix to buffer.")
                elif export_format == "mp3" and PYDUB_AVAILABLE:
                    try:  # MP3 conversion logic (keep as is)
                        logger.info("Converting full mix to MP3...")
                        full_mix_clipped = np.clip(full_mix, -1.0, 1.0)
                        audio_int16 = (full_mix_clipped * 32767).astype(np.int16)
                        channels = 2
                        if audio_int16.ndim == 1:
                            channels = 1
                        elif audio_int16.ndim == 2 and audio_int16.shape[1] == 1:
                            channels = 1
                            audio_int16 = audio_int16.flatten()
                        segment = AudioSegment(
                            data=audio_int16.tobytes(),
                            sample_width=audio_int16.dtype.itemsize,
                            frame_rate=GLOBAL_SR,
                            channels=channels,
                        )
                        if channels == 1:
                            segment = segment.set_channels(2)
                        mp3_buffer = BytesIO()
                        segment.export(mp3_buffer, format="mp3", bitrate="192k")
                        mp3_buffer.seek(0)
                        if mp3_buffer.getbuffer().nbytes > 0:
                            st.session_state.export_buffer = mp3_buffer
                            st.session_state.export_file_ext = "mp3"
                            logger.info("Full MP3 mix generated.")
                            export_successful = True
                        else:
                            raise ValueError("MP3 export resulted in empty buffer.")
                    except Exception as e_mp3:
                        logger.exception("Failed to export mix as MP3.")
                        st.session_state.export_error_message = (
                            f"MP3 Export Failed: {e_mp3}. Ensure ffmpeg is installed."
                        )
                elif export_format == "mp3":
                    st.session_state.export_error_message = (
                        "MP3 requires 'pydub' and 'ffmpeg'."
                    )
                else:
                    st.session_state.export_error_message = (
                        f"Unsupported format '{export_format}'."
                    )
            elif full_mix is not None:
                st.session_state.export_error_message = "Generated mix is empty."
            else:
                st.session_state.export_error_message = (
                    "Failed to generate the final mix."
                )

        except Exception as e:
            logger.exception("Error during full mix generation or export.")
            st.session_state.export_error_message = (
                f"Failed to generate or export mix: {e}"
            )
        finally:
            st.session_state.advanced_processing_active = False
            logger.info("Reset advanced_processing_active flag to False.")
            # GA Event Sending
            if export_successful and not st.session_state.get("export_error_message"):
                logger.info(
                    f"Export successful. Sending 'mix_exported' event to GA for format: {export_format}"
                )
                ga_event_js = f"""<script>if (typeof gtag === 'function') {{gtag('event', 'mix_exported', {{'event_category': 'engagement', 'event_label': '{export_format.upper()}', 'value': 1}}); console.log("GA Event Sent: mix_exported ({export_format.upper()})");}} else {{console.error("gtag function not found.");}}</script>"""
                try:
                    components.html(ga_event_js, height=0)
                except Exception as e_comp:
                    logger.error(f"Failed to inject GA event script: {e_comp}")

    def render_preview_audio_player(self):  # ... (no changes)
        if (
            "preview_audio_data" in st.session_state
            and st.session_state.preview_audio_data
        ):
            logger.debug("Rendering master preview audio player.")
            st.markdown("**Mix Preview:**")
            try:
                st.session_state.preview_audio_data.seek(0)
                st.audio(st.session_state.preview_audio_data, format="audio/wav")
            except Exception as e:
                logger.error(f"Error playing preview audio: {e}")
                st.error("Could not play preview.")

    def render_instructions(self):  # ... (no changes)
        tracks_exist = bool(self.app_state.get_all_tracks())
        st.divider()
        with st.expander("📖 Show Instructions & Notes", expanded=not tracks_exist):
            st.markdown(""" **Welcome to MindMorph!** ... (instructions) ... """)
