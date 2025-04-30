# quick_wizard.py
# ==========================================
# Quick Create Wizard Orchestrator for MindMorph
# ==========================================

import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# Import necessary components from other modules
from audio_generators import (
    generate_binaural_beats,
    generate_isochronic_tones,
    generate_noise,
    generate_solfeggio_frequency,
)
from audio_io import load_audio, save_audio_to_bytesio

# <<< MODIFIED: Import mix_wizard_tracks from audio_mixers >>>
from audio_mixers import mix_wizard_tracks

# <<< Type hint for AudioData >>>
try:
    from audio_effects_pipeline import AudioData
except ImportError:
    AudioData = np.ndarray  # Fallback

from config import (
    GLOBAL_SR,
    MAX_AFFIRMATION_CHARS,
    MAX_AUDIO_DURATION_S,
    PROJECT_FILE_VERSION,
)
from tts_generator import TTSGenerator
from utils import read_text_file

# Import wizard state management
from wizard_state import initialize_wizard_state, reset_wizard_state

# Import step rendering functions
from wizard_steps import step_1_affirmations, step_2_background, step_3_frequency, step_4_export

# Optional MP3 export dependency check
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


logger = logging.getLogger(__name__)

# --- Constants for Wizard ---
WIZARD_AFFIRMATION_SPEED = 10.0
WIZARD_AFFIRMATION_VOLUME = 0.10


class QuickWizard:
    """Manages the state and UI rendering orchestration for the Quick Create Wizard."""

    def __init__(self, tts_generator: TTSGenerator):
        """Initializes the QuickWizard."""
        self.tts_generator = tts_generator
        initialize_wizard_state()
        logger.debug("QuickWizard initialized and state ensured.")

    # --- State Synchronization Callbacks ---
    def sync_affirmation_text(self):
        st.session_state.wizard_affirmation_text = st.session_state.get("wizard_affirm_text_area", "")
        if st.session_state.get("wizard_affirmation_source") == "text":
            st.session_state.wizard_affirmation_audio = None
            st.session_state.wizard_affirmation_sr = None
        logger.debug("Synced affirmation text state.")

    def clear_affirmation_upload_state(self):
        if st.session_state.get("wizard_affirmation_source") == "upload":
            st.session_state.wizard_affirmation_audio = None
            st.session_state.wizard_affirmation_sr = None
            st.session_state.wizard_affirmation_source = None
            st.session_state.wizard_affirmation_text = ""
            logger.debug("Cleared affirmation upload state.")

    def sync_background_choice(self, choice_options: List[str]):
        selected_label = st.session_state.get("wizard_bg_choice_radio")
        if not selected_label:
            return
        new_choice = "none"
        if selected_label == "Upload Music/Sound":
            new_choice = "upload"
        elif selected_label == "Generate Noise":
            new_choice = "noise"
        if new_choice != st.session_state.get("wizard_background_choice"):
            st.session_state.wizard_background_audio = None
            st.session_state.wizard_background_sr = None
            logger.debug(f"Background choice changed to {new_choice}, cleared audio.")
        st.session_state.wizard_background_choice = new_choice
        st.session_state.wizard_background_choice_label = selected_label

    def clear_background_upload_state(self):
        if st.session_state.get("wizard_background_choice") == "upload":
            st.session_state.wizard_background_audio = None
            st.session_state.wizard_background_sr = None
            logger.debug("Cleared background upload state.")

    # --- Navigation and Reset ---
    def _reset_wizard_state(self):
        reset_wizard_state()
        st.rerun()

    def _go_to_step(self, step: int):
        if 1 <= step <= 4:
            st.session_state.wizard_step = step
            logger.debug(f"Navigating to wizard step {step}")
            st.rerun()
        else:
            logger.warning(f"Invalid step navigation requested: {step}")

    # --- Core Processing Logic ---
    def _process_and_export(self):
        """Handles the final processing and export logic for the wizard."""
        logger.info("Starting wizard export process.")
        st.session_state.wizard_export_buffer = None
        st.session_state.wizard_export_error = None
        affirmation_audio_data = st.session_state.get("wizard_affirmation_audio")
        affirmation_sr = st.session_state.get("wizard_affirmation_sr")
        background_audio_data = st.session_state.get("wizard_background_audio")
        background_sr = st.session_state.get("wizard_background_sr")
        background_volume = st.session_state.get("wizard_background_volume", 0.7)
        frequency_audio_data = st.session_state.get("wizard_frequency_audio")
        frequency_sr = st.session_state.get("wizard_frequency_sr")
        frequency_volume = st.session_state.get("wizard_frequency_volume", 0.2)
        export_format = st.session_state.get("wizard_export_format", "wav").lower()
        affirmation_tuple = (affirmation_audio_data, affirmation_sr) if affirmation_audio_data is not None and affirmation_sr is not None else None
        background_tuple = (background_audio_data, background_sr) if background_audio_data is not None and background_sr is not None else None
        frequency_tuple = (frequency_audio_data, frequency_sr) if frequency_audio_data is not None and frequency_sr is not None else None
        if affirmation_tuple is None:
            st.session_state.wizard_export_error = "Affirmation audio is missing."
            logger.error("Wizard export failed: Missing affirmation audio.")
            return
        logger.info("Wizard mixing tracks...")
        spinner_msg = f"Generating final mix ({export_format.upper()})..."
        with st.spinner(spinner_msg):
            try:
                # Call mix_wizard_tracks from audio_mixers
                full_mix = mix_wizard_tracks(
                    affirmation_audio=affirmation_tuple,
                    background_audio=background_tuple,
                    frequency_audio=frequency_tuple,
                    affirmation_speed=WIZARD_AFFIRMATION_SPEED,
                    affirmation_volume=WIZARD_AFFIRMATION_VOLUME,
                    background_volume=background_volume,
                    frequency_volume=frequency_volume,
                    target_sr=GLOBAL_SR,
                )
                if full_mix is None or full_mix.size == 0:
                    raise ValueError("Mixing process resulted in empty or None audio.")
                mix_duration_s = len(full_mix) / GLOBAL_SR if GLOBAL_SR > 0 else 0
                logger.info(f"Mixing successful. Final mix length: {mix_duration_s:.2f} seconds.")
                if export_format == "wav":
                    export_buffer = save_audio_to_bytesio(full_mix, GLOBAL_SR)
                    if export_buffer and export_buffer.getbuffer().nbytes > 0:
                        st.session_state.wizard_export_buffer = export_buffer
                        logger.info("Wizard WAV mix generated.")
                    else:
                        raise ValueError("Failed to save WAV mix to buffer (empty buffer).")
                elif export_format == "mp3" and PYDUB_AVAILABLE:
                    try:
                        logger.info("Wizard converting full mix to MP3...")
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
                        if mp3_buffer.getbuffer().nbytes > 0:
                            st.session_state.wizard_export_buffer = mp3_buffer
                            logger.info("Wizard MP3 mix generated.")
                        else:
                            raise ValueError("MP3 export resulted in an empty buffer.")
                    except Exception as e_mp3:
                        logger.exception("Wizard failed to export mix as MP3.")
                        st.session_state.wizard_export_error = f"MP3 Export Failed: {e_mp3}. Ensure ffmpeg is installed."
                elif export_format == "mp3" and not PYDUB_AVAILABLE:
                    st.session_state.wizard_export_error = "MP3 export requires 'pydub' and 'ffmpeg'."
                    logger.error(st.session_state.wizard_export_error)
                else:
                    st.session_state.wizard_export_error = f"Unsupported export format '{export_format}'."
                    logger.error(st.session_state.wizard_export_error)
            except Exception as e:
                logger.exception("Error during wizard mix/save.")
                st.session_state.wizard_export_error = f"Processing Failed: {e}"

    # --- Main Rendering Method ---
    def render_wizard(self):
        """Renders the current step of the wizard by calling the appropriate step function."""
        st.title("✨ MindMorph Quick Create Wizard")
        initialize_wizard_state()
        step = st.session_state.get("wizard_step", 1)
        steps_display = ["Affirmations", "Background", "Frequency", "Export"]
        progress_step = max(1, min(step, len(steps_display)))
        try:
            st.progress((progress_step) / len(steps_display), text=f"Step {progress_step}: {steps_display[progress_step - 1]}")
        except IndexError:
            st.progress(0.0)
            logger.warning(f"Progress bar step index out of range: {progress_step}")
        if step == 1:
            step_1_affirmations.render_step_1(self)
        elif step == 2:
            step_2_background.render_step_2(self)
        elif step == 3:
            step_3_frequency.render_step_3(self)
        elif step == 4:
            step_4_export.render_step_4(self)
        else:
            st.error("Invalid wizard step detected. Resetting.")
            logger.error(f"Invalid wizard step: {step}. Resetting.")
            self._reset_wizard_state()
