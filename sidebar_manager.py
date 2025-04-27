# sidebar_manager.py
# ==========================================
# Sidebar UI Management for MindMorph
# ==========================================

import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st
from PIL import Image

# Import necessary components from other modules
from app_state import AppState, TrackData, TrackType
from audio_utils import AudioData, generate_binaural_beats, generate_isochronic_tones, generate_noise, generate_solfeggio_frequency, load_audio
from config import (
    ASSETS_DIR,
    GLOBAL_SR,
    LOGO_PATH,
    MAX_AFFIRMATION_CHARS,
    MAX_AUDIO_DURATION_S,
    PROJECT_FILE_VERSION,
    TRACK_TYPE_AFFIRMATION,
    TRACK_TYPE_BACKGROUND,
    TRACK_TYPE_FREQUENCY,
    TRACK_TYPE_OTHER,
    TRACK_TYPE_VOICE,
    TRACK_TYPES,  # Needed? Maybe not directly
    get_default_track_params,  # Import the function itself
)
from tts_generator import TTSGenerator
from utils import read_text_file

# Get a logger for this module
logger = logging.getLogger(__name__)


class SidebarManager:
    """Handles rendering all components within the Streamlit sidebar."""

    def __init__(self, app_state: AppState, tts_generator: TTSGenerator):
        """
        Initializes the SidebarManager.

        Args:
            app_state: An instance of the AppState class.
            tts_generator: An instance of the TTSGenerator class.
        """
        self.app_state = app_state
        self.tts_generator = tts_generator
        logger.debug("SidebarManager initialized.")

    # --- Main Rendering Method ---

    def render_sidebar(self):
        """Renders the entire sidebar UI."""
        with st.sidebar:
            # Logo and Title
            if os.path.exists(LOGO_PATH):
                try:
                    logo_image = Image.open(LOGO_PATH)
                    st.image(logo_image, width=200)
                except Exception as e:
                    logger.warning(f"Could not load logo image from {LOGO_PATH}: {e}")
                    st.header("MindMorph")  # Fallback to text header
            else:
                st.header("MindMorph")
            st.caption("Subliminal Audio Editor")
            st.markdown("---")

            # Step 1: Add Audio Layers Section
            st.markdown("### STEP 1: Add Audio Layers")
            st.caption("Use the options below to add sounds to your project.")

            # Render individual sidebar components
            self._render_uploader()
            st.divider()
            self._render_affirmation_inputs()
            st.divider()
            self._render_frequency_generators()
            st.divider()
            self._render_background_generators()
            st.markdown("---")

            # Project Save/Load Section
            st.subheader("💾 Project")
            self._render_save_load()
            st.markdown("---")

            st.info("Edit track details and effects in the main panel.")

    # --- Private Rendering Methods for Sidebar Sections ---

    def _render_uploader(self):
        """Renders the file uploader component in the sidebar."""
        st.subheader("📁 Upload Audio File(s)")
        st.caption(f"Upload music, voice, etc. (Max duration: {MAX_AUDIO_DURATION_S // 60} min)")

        uploaded_files = st.file_uploader(
            "Select audio files (.wav, .mp3, .ogg, .flac)",
            type=["wav", "mp3", "ogg", "flac"],
            accept_multiple_files=True,
            key="sidebar_audio_file_uploader",  # Unique key for sidebar uploader
            label_visibility="collapsed",
            help="Select one or more audio files to add as tracks.",
        )

        if uploaded_files:
            current_tracks = self.app_state.get_all_tracks()
            current_track_filenames_with_audio = {
                tdata.get("original_filename") for tdata in current_tracks.values() if tdata.get("source_type") == "upload" and tdata.get("original_audio") is not None
            }
            current_tracks_missing_audio = {
                tdata.get("original_filename"): tid for tid, tdata in current_tracks.items() if tdata.get("source_type") == "upload" and tdata.get("original_audio") is None
            }
            files_processed_this_run = set()

            for file in uploaded_files:
                if file.name in files_processed_this_run:
                    continue

                logger.info(f"Processing uploaded file via sidebar: {file.name}")

                if file.name in current_track_filenames_with_audio:
                    logger.info(f"Skipping sidebar upload: Track for '{file.name}' already exists with audio data.")
                    files_processed_this_run.add(file.name)
                    continue

                existing_track_id = current_tracks_missing_audio.get(file.name)

                with st.spinner(f"Loading {file.name}..."):
                    audio, sr = load_audio(file, target_sr=GLOBAL_SR)  # From audio_utils

                if audio is not None and audio.size > 0:
                    duration_seconds = len(audio) / sr if sr > 0 else 0
                    if duration_seconds > MAX_AUDIO_DURATION_S:
                        logger.warning(f"Sidebar Upload '{file.name}' rejected: Duration {duration_seconds:.1f}s exceeds limit {MAX_AUDIO_DURATION_S}s.")
                        st.error(f"❌ File '{file.name}' is too long ({duration_seconds:.1f}s). Max is {MAX_AUDIO_DURATION_S // 60} min.")
                        files_processed_this_run.add(file.name)
                        continue

                    if existing_track_id:
                        logger.info(
                            f"Updating existing track '{self.app_state.get_track(existing_track_id).get('name')}' ({existing_track_id}) with uploaded audio for {file.name}."
                        )
                        self.app_state.update_track_param(existing_track_id, "original_audio", audio)
                        self.app_state.update_track_param(existing_track_id, "sr", sr)
                        st.success(f"Re-loaded audio for track '{self.app_state.get_track(existing_track_id).get('name')}'")
                        st.rerun()
                    else:
                        track_params = get_default_track_params()
                        if any(keyword in file.name.lower() for keyword in ["voice", "record", "affirmation", "tts"]):
                            track_type = TRACK_TYPE_VOICE
                        elif any(keyword in file.name.lower() for keyword in ["music", "background", "mask", "ambient"]):
                            track_type = TRACK_TYPE_BACKGROUND
                        else:
                            track_type = TRACK_TYPE_OTHER

                        track_params.update(
                            {
                                "original_audio": audio,
                                "sr": sr,
                                "name": file.name,
                                "source_type": "upload",
                                "original_filename": file.name,
                            }
                        )
                        new_track_id = self.app_state.add_track(track_params, track_type=track_type)
                        st.success(f"Loaded '{file.name}' as '{track_type}' ({duration_seconds:.1f}s)")
                        current_track_filenames_with_audio.add(file.name)

                elif audio is None:
                    logger.error(f"Failed to load audio from sidebar upload: {file.name}")
                else:
                    logger.warning(f"Skipped empty/invalid sidebar audio upload: {file.name}")
                    st.warning(f"Skipped empty or invalid audio file: {file.name}")

                files_processed_this_run.add(file.name)

    def _render_affirmation_inputs(self):
        """Renders the affirmation input options (TTS, File) in the sidebar."""
        st.subheader("🗣️ Add Affirmations")
        st.caption(f"Uses system default TTS voice. Max {MAX_AFFIRMATION_CHARS} chars.")

        tab1, tab2, tab3 = st.tabs(["Type Text", "Upload File", "Record Audio"])

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
                if not affirmation_text or not affirmation_text.strip():
                    st.warning("Please enter some text in the text area first.")
                elif len(affirmation_text) > MAX_AFFIRMATION_CHARS:
                    logger.warning(f"TTS Text input rejected: Length {len(affirmation_text)} exceeds limit {MAX_AFFIRMATION_CHARS}.")
                    st.error(f"❌ Text is too long ({len(affirmation_text)} chars). Max is {MAX_AFFIRMATION_CHARS}.")
                else:
                    default_name = "TTS Affirmations"
                    if len(affirmation_text) > 30:
                        default_name = f"TTS: {affirmation_text[:25]}..."
                    self._generate_tts_track(affirmation_text, default_name)  # Use helper

        with tab2:
            st.caption("Upload a .txt or .docx file containing affirmations.")
            uploaded_affirmation_file = st.file_uploader(
                "Upload Affirmation File (.txt, .docx)",
                type=["txt", "docx"],
                key="sidebar_affirmation_file_uploader",
                label_visibility="collapsed",
                help="Select a text or Word document containing affirmations.",
            )
            if st.button(
                "Generate Track from File",
                key="sidebar_generate_tts_from_file",
                use_container_width=True,
                type="primary",
                help="Read the uploaded file and convert its text content to a spoken audio track.",
            ):
                if uploaded_affirmation_file:
                    with st.spinner(f"Reading {uploaded_affirmation_file.name}..."):
                        text_from_file = read_text_file(uploaded_affirmation_file)  # From utils

                    if text_from_file is not None:
                        if not text_from_file.strip():
                            st.warning(f"File '{uploaded_affirmation_file.name}' appears empty.")
                            logger.warning(f"File '{uploaded_affirmation_file.name}' read as empty.")
                        elif len(text_from_file) > MAX_AFFIRMATION_CHARS:
                            logger.warning(f"TTS File '{uploaded_affirmation_file.name}' rejected: Length {len(text_from_file)} exceeds limit {MAX_AFFIRMATION_CHARS}.")
                            st.error(f"❌ Text in file '{uploaded_affirmation_file.name}' is too long ({len(text_from_file)} chars). Max is {MAX_AFFIRMATION_CHARS}.")
                        else:
                            default_name = f"File Affirmations ({uploaded_affirmation_file.name})"
                            self._generate_tts_track(text_from_file, default_name)  # Use helper
                    # else: read_text_file shows error

                else:
                    st.warning("Please upload a .txt or .docx file first.")

        with tab3:
            st.caption("Record your own voice directly in the browser.")
            st.info("🎙️ Audio recording feature coming soon!")
            st.markdown("For now, please record using other software and use the 'Upload Audio File(s)' option.")
            st.button("Start Recording", key="sidebar_start_recording", disabled=True, use_container_width=True)

    def _render_frequency_generators(self):
        """Renders frequency/tone generation options based on app mode."""
        st.subheader("🧠✨ Add Frequencies / Tones")
        app_mode = st.session_state.get("app_mode", "Easy")

        if app_mode == "Easy":
            available_gen_types = ["Presets"]
            st.caption("Use presets for common frequency patterns.")
        else:  # Advanced mode
            available_gen_types = ["Binaural Beats", "Solfeggio Tones", "Isochronic Tones", "Presets"]
            st.caption("Generate specific tones or use presets.")

        gen_type_key = "sidebar_frequency_gen_type"
        default_gen_type = "Presets"
        if gen_type_key in st.session_state and st.session_state[gen_type_key] in available_gen_types:
            default_gen_type = st.session_state[gen_type_key]
        try:
            default_index = available_gen_types.index(default_gen_type)
        except ValueError:
            default_index = 0

        gen_type = st.radio("Select Frequency Type:", available_gen_types, index=default_index, key=gen_type_key, horizontal=True, label_visibility="collapsed")

        if gen_type == "Presets":
            self._render_frequency_presets()
        elif gen_type == "Binaural Beats" and app_mode == "Advanced":
            self._render_binaural_generator()
        elif gen_type == "Solfeggio Tones" and app_mode == "Advanced":
            self._render_solfeggio_generator()
        elif gen_type == "Isochronic Tones" and app_mode == "Advanced":
            self._render_isochronic_generator()

    def _render_frequency_presets(self):
        """Renders the UI for selecting and generating frequency presets."""
        st.markdown("<small>Generate common frequency patterns like Alpha for focus or Delta for sleep.</small>", unsafe_allow_html=True)
        preset_options = {
            "Focus (Alpha 10Hz Binaural)": {"type": "binaural", "f_left": 200.0, "f_right": 210.0},
            "Relaxation (Theta 5Hz Binaural)": {"type": "binaural", "f_left": 150.0, "f_right": 155.0},
            "Deep Sleep (Delta 2Hz Binaural)": {"type": "binaural", "f_left": 100.0, "f_right": 102.0},
            "Love Frequency (Solfeggio 528Hz)": {"type": "solfeggio", "freq": 528.0},
            "Miracle Tone (Solfeggio 417Hz)": {"type": "solfeggio", "freq": 417.0},
        }
        preset_name = st.selectbox("Select Preset:", list(preset_options.keys()), key="sidebar_freq_preset_select")
        preset_data = preset_options[preset_name]

        cols_preset = st.columns(2)
        with cols_preset[0]:
            preset_duration = st.number_input("Duration (s)", 1, 7200, 60, 1, key="sidebar_preset_duration", help="Duration in seconds.")
        with cols_preset[1]:
            preset_vol = st.slider("Volume", 0.0, 1.0, 0.2, 0.05, key="sidebar_preset_volume", help="Volume (usually kept low).")

        if st.button(f"Generate '{preset_name}' Track", key="sidebar_generate_preset"):
            with st.spinner("Generating preset frequency..."):
                audio: Optional[AudioData] = None
                gen_params: Dict[str, Any] = {}
                default_track_name = preset_name

                if preset_data["type"] == "binaural":
                    audio = generate_binaural_beats(preset_duration, preset_data["f_left"], preset_data["f_right"], GLOBAL_SR, preset_vol)
                    gen_params = {
                        "source_type": "binaural_preset",
                        "gen_duration": preset_duration,
                        "gen_freq_left": preset_data["f_left"],
                        "gen_freq_right": preset_data["f_right"],
                        "gen_volume": preset_vol,
                    }
                elif preset_data["type"] == "solfeggio":
                    audio = generate_solfeggio_frequency(preset_duration, preset_data["freq"], GLOBAL_SR, preset_vol)
                    gen_params = {"source_type": "solfeggio_preset", "gen_duration": preset_duration, "gen_freq": preset_data["freq"], "gen_volume": preset_vol}

                if audio is not None and audio.size > 0:
                    track_params = get_default_track_params()
                    track_params.update({"original_audio": audio, "sr": GLOBAL_SR, "name": default_track_name})
                    track_params.update(gen_params)
                    new_track_id = self.app_state.add_track(track_params, track_type=TRACK_TYPE_FREQUENCY)
                    st.success(f"'{default_track_name}' track generated (ID: {new_track_id[:6]})!")
                elif audio is not None:
                    st.warning("Generated preset audio was empty.")
                else:
                    st.error("Failed to generate audio for the selected preset.")

    def _render_binaural_generator(self):
        """Renders UI for generating custom Binaural Beats (Advanced Mode)."""
        st.markdown("<small>Generates stereo tones potentially inducing brainwave states (requires headphones).</small>", unsafe_allow_html=True)
        bb_cols = st.columns(2)
        with bb_cols[0]:
            bb_duration = st.number_input("Duration (s)", 1, 7200, 60, 1, key="sidebar_bb_duration", help="Length in seconds.")
            bb_fleft = st.number_input("Left Freq (Hz)", 20.0, 1000.0, 200.0, 0.1, format="%.1f", key="sidebar_bb_freq_left", help="Left ear frequency.")
        with bb_cols[1]:
            bb_vol = st.slider("Volume##BB", 0.0, 1.0, 0.3, 0.05, key="sidebar_bb_volume", help="Loudness (0.0 to 1.0). Usually kept low.")
            bb_fright = st.number_input("Right Freq (Hz)", 20.0, 1000.0, 210.0, 0.1, format="%.1f", key="sidebar_bb_freq_right", help="Right ear frequency.")

        beat_freq = abs(bb_fleft - bb_fright)
        st.caption(f"Resulting Beat Frequency: {beat_freq:.1f} Hz")

        if st.button("Generate Binaural Track", key="sidebar_generate_bb", help="Create track with these settings."):
            with st.spinner("Generating Binaural Beats..."):
                audio = generate_binaural_beats(bb_duration, bb_fleft, bb_fright, GLOBAL_SR, bb_vol)
            if audio is not None and audio.size > 0:
                track_params = get_default_track_params()
                default_name = f"Binaural {bb_fleft:.1f}Hz L / {bb_fright:.1f}Hz R ({beat_freq:.1f}Hz Beat)"
                track_params.update(
                    {
                        "original_audio": audio,
                        "sr": GLOBAL_SR,
                        "name": default_name,
                        "source_type": "binaural",
                        "gen_duration": bb_duration,
                        "gen_freq_left": bb_fleft,
                        "gen_freq_right": bb_fright,
                        "gen_volume": bb_vol,
                    }
                )
                new_track_id = self.app_state.add_track(track_params, track_type=TRACK_TYPE_FREQUENCY)
                st.success(f"'{default_name}' generated (ID: {new_track_id[:6]})!")
            elif audio is not None:
                st.warning("Generated binaural audio was empty.")
            else:
                st.error("Failed to generate binaural beats.")

    def _render_solfeggio_generator(self):
        """Renders UI for generating Solfeggio Tones (Advanced Mode)."""
        st.markdown("<small>Generates pure tones based on historical Solfeggio frequencies.</small>", unsafe_allow_html=True)
        freqs = [174.0, 285.0, 396.0, 417.0, 528.0, 639.0, 741.0, 852.0, 963.0]
        freq_labels = {f: f"{f} Hz" for f in freqs}

        cols = st.columns(2)
        with cols[0]:
            freq = st.selectbox("Frequency (Hz)", freqs, index=4, key="sidebar_solf_freq", format_func=lambda x: freq_labels.get(x, f"{x} Hz"), help="Select Solfeggio frequency.")
        with cols[1]:
            duration = st.number_input("Duration (s)##Solf", 1, 7200, 60, 1, key="sidebar_solf_duration", help="Length in seconds.")

        vol = st.slider("Volume##Solf", 0.0, 1.0, 0.3, 0.05, key="sidebar_solf_volume", help="Loudness (0.0 to 1.0). Usually kept low.")

        if st.button("Generate Solfeggio Track", key="sidebar_generate_solf", help="Create track with this tone."):
            with st.spinner("Generating Solfeggio Tone..."):
                audio = generate_solfeggio_frequency(duration, freq, GLOBAL_SR, vol)
            if audio is not None and audio.size > 0:
                track_params = get_default_track_params()
                default_name = f"Solfeggio {freq}Hz"
                track_params.update(
                    {"original_audio": audio, "sr": GLOBAL_SR, "name": default_name, "source_type": "solfeggio", "gen_duration": duration, "gen_freq": freq, "gen_volume": vol}
                )
                new_track_id = self.app_state.add_track(track_params, track_type=TRACK_TYPE_FREQUENCY)
                st.success(f"'{default_name}' generated (ID: {new_track_id[:6]})!")
            elif audio is not None:
                st.warning("Generated Solfeggio audio was empty.")
            else:
                st.error("Failed to generate Solfeggio tone.")

    def _render_isochronic_generator(self):
        """Renders UI for generating Isochronic Tones (Advanced Mode)."""
        st.markdown("<small>Generates rhythmic pulses of a single tone (headphones not required).</small>", unsafe_allow_html=True)
        iso_cols = st.columns(2)
        with iso_cols[0]:
            iso_duration = st.number_input("Duration (s)##Iso", 1, 7200, 60, 1, key="sidebar_iso_duration", help="Length in seconds.")
            iso_carrier = st.number_input("Carrier Freq (Hz)", 20.0, 1000.0, 150.0, 0.1, format="%.1f", key="sidebar_iso_carrier", help="The base tone frequency being pulsed.")
        with iso_cols[1]:
            iso_vol = st.slider("Volume##Iso", 0.0, 1.0, 0.4, 0.05, key="sidebar_iso_volume", help="Loudness (0.0 to 1.0).")
            iso_pulse = st.number_input(
                "Pulse Freq (Hz)", 0.1, 40.0, 10.0, 0.1, format="%.1f", key="sidebar_iso_pulse", help="How many times per second the tone pulses (e.g., 10Hz for Alpha)."
            )

        if st.button("Generate Isochronic Track", key="sidebar_generate_iso", help="Create track with these settings."):
            with st.spinner("Generating Isochronic Tones..."):
                audio = generate_isochronic_tones(iso_duration, iso_carrier, iso_pulse, GLOBAL_SR, iso_vol)
            if audio is not None and audio.size > 0:
                track_params = get_default_track_params()
                default_name = f"Isochronic {iso_carrier:.1f}Hz / {iso_pulse:.1f}Hz Pulse"
                track_params.update(
                    {
                        "original_audio": audio,
                        "sr": GLOBAL_SR,
                        "name": default_name,
                        "source_type": "isochronic",
                        "gen_duration": iso_duration,
                        "gen_carrier_freq": iso_carrier,
                        "gen_pulse_freq": iso_pulse,
                        "gen_volume": iso_vol,
                    }
                )
                new_track_id = self.app_state.add_track(track_params, track_type=TRACK_TYPE_FREQUENCY)
                st.success(f"'{default_name}' generated (ID: {new_track_id[:6]})!")
            elif audio is not None:
                st.warning("Generated Isochronic audio was empty.")
            else:
                st.error("Failed to generate Isochronic tones.")

    def _render_background_generators(self):
        """Renders options for generating background noise."""
        st.subheader("🎵 Add Background Noise")
        noise_options = ["White Noise", "Pink Noise", "Brown Noise"]
        noise_type = st.selectbox("Select Noise Type:", noise_options, key="sidebar_noise_type", help="Choose a type of background noise.")

        cols_noise = st.columns(2)
        with cols_noise[0]:
            noise_duration = st.number_input("Duration (s)##Noise", 10, 7200, 300, 10, key="sidebar_noise_duration", help="Length in seconds. Will loop if needed.")
        with cols_noise[1]:
            noise_vol = st.slider("Volume##Noise", 0.0, 1.0, 0.5, 0.05, key="sidebar_noise_volume", help="Loudness of the noise.")

        if st.button(f"Generate {noise_type} Track", key="sidebar_generate_noise"):
            with st.spinner(f"Generating {noise_type}..."):
                audio = generate_noise(noise_type, noise_duration, GLOBAL_SR, noise_vol)  # From audio_utils
                if audio is not None and audio.size > 0:
                    track_params = get_default_track_params()
                    default_name = f"{noise_type}"
                    track_params.update(
                        {
                            "original_audio": audio,
                            "sr": GLOBAL_SR,
                            "name": default_name,
                            "loop_to_fit": True,
                            "source_type": "noise",
                            "gen_noise_type": noise_type,
                            "gen_duration": noise_duration,
                            "gen_volume": noise_vol,
                        }
                    )
                    new_track_id = self.app_state.add_track(track_params, track_type=TRACK_TYPE_BACKGROUND)
                    st.success(f"'{default_name}' track generated (ID: {new_track_id[:6]})!")
                elif audio is not None:
                    st.warning(f"Generated {noise_type} was empty.")
                else:
                    logger.error(f"Failed to generate {noise_type}.")  # generate_noise shows error

    def _render_save_load(self):
        """Renders project save and load components in the sidebar."""
        st.markdown("**Save/Load Project**")
        st.caption("Save project setup (track settings, sources) to a `.mindmorph` file. Audio data itself is NOT saved.")

        # --- Save Project ---
        project_data_str = ""
        tracks = self.app_state.get_all_tracks()
        save_disabled = not bool(tracks)

        if tracks:
            try:
                serializable_tracks = {}
                for track_id, track_data in tracks.items():
                    save_data = track_data.copy()
                    # Remove non-serializable or large/runtime data
                    save_data.pop("original_audio", None)
                    save_data.pop("preview_temp_file_path", None)
                    save_data.pop("preview_settings_hash", None)
                    if "source_type" not in save_data:
                        save_data["source_type"] = "unknown"
                    serializable_tracks[track_id] = save_data

                project_file_content = {"version": PROJECT_FILE_VERSION, "tracks": serializable_tracks}
                project_data_str = json.dumps(project_file_content, indent=2)
            except Exception as e:
                logger.error(f"Error preparing project data for saving: {e}")
                st.error("Could not prepare project data for saving.")
                save_disabled = True

        st.download_button(
            label="💾 Save Project File",
            data=project_data_str if project_data_str else "",
            file_name="my_subliminal_project.mindmorph",
            mime="application/json",
            key="sidebar_save_project_button",
            help="Saves the current track list and settings (excluding audio data).",
            use_container_width=True,
            disabled=save_disabled,
        )

        # --- Load Project ---
        uploaded_project_file = st.file_uploader(
            "⬆️ Load Project File (.mindmorph)",
            type=["mindmorph", "json"],
            key="sidebar_load_project_uploader",
            accept_multiple_files=False,
            help="Load a previously saved project configuration. This will replace the current project.",
        )
        if uploaded_project_file is not None:
            st.session_state.uploaded_project_file_data = uploaded_project_file.getvalue()
            st.session_state.project_load_requested = True  # Flag for main loop
            logger.info(f"Project file uploaded via sidebar: {uploaded_project_file.name}. Requesting load.")
            st.rerun()  # Rerun to trigger loading logic in main

    # --- Helper Methods Specific to Sidebar Actions ---

    def _generate_tts_track(self, text_content: str, track_name: str):
        """
        Helper method (specific to sidebar) to generate TTS audio and add it as a new track.

        Args:
            text_content: The text to synthesize.
            track_name: The default name for the new track.
        """
        logger.info(f"Sidebar: Generating TTS track '{track_name}'")
        try:
            audio, sr = self.tts_generator.generate(text_content)  # Use TTSGenerator instance
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
                logger.error("Sidebar TTS generation returned None.")
                # TTSGenerator likely showed an error already
        except Exception as e:
            logger.exception(f"Error during sidebar TTS track generation for '{track_name}'.")
            st.error(f"Failed to create TTS track '{track_name}': {e}")
