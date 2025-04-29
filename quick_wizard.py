# quick_wizard.py
# ==========================================
# Quick Create Wizard Orchestrator for MindMorph
# ==========================================

import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple  # Added List

import numpy as np
import streamlit as st

# Import necessary components from other modules
from audio_generators import (
    generate_binaural_beats,
    generate_isochronic_tones,
    generate_noise,
    generate_solfeggio_frequency,  # Needed for export
)
from audio_io import load_audio, save_audio_to_bytesio  # Needed for export
from audio_processing import mix_tracks  # Wizard will call the main mix function
from config import (
    GLOBAL_SR,
    MAX_AFFIRMATION_CHARS,
    MAX_AUDIO_DURATION_S,
    PROJECT_FILE_VERSION,
    TRACK_TYPE_AFFIRMATION,
    TRACK_TYPE_BACKGROUND,
    TRACK_TYPE_FREQUENCY,
    get_default_track_params,  # Needed for export
)
from tts_generator import TTSGenerator
from utils import read_text_file  # Keep if needed, maybe not directly here

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

# --- Constants for Wizard (can be moved to a wizard_config.py) ---
WIZARD_AFFIRMATION_SPEED = 10.0  # Fixed speed for affirmations
WIZARD_AFFIRMATION_VOLUME = 0.10  # Fixed (low) volume for affirmations
# Other constants like WIZARD_MAX_UPLOAD_SIZE_MB are now in step modules where used


class QuickWizard:
    """Manages the state and UI rendering orchestration for the Quick Create Wizard."""

    def __init__(self, tts_generator: TTSGenerator):
        """
        Initializes the QuickWizard.

        Args:
            tts_generator: An instance of the TTSGenerator.
        """
        self.tts_generator = tts_generator
        # Initialize state using the dedicated function
        initialize_wizard_state()
        logger.debug("QuickWizard initialized and state ensured.")

    # --- State Synchronization Callbacks (used by step renderers) ---

    def sync_affirmation_text(self):
        """Callback to update affirmation text state from text_area."""
        st.session_state.wizard_affirmation_text = st.session_state.get("wizard_affirm_text_area", "")
        # If text is updated, clear any previously generated/uploaded audio for text source
        if st.session_state.get("wizard_affirmation_source") == "text":
            st.session_state.wizard_affirmation_audio = None
            st.session_state.wizard_affirmation_sr = None
            # Keep source as 'text' but require regeneration
        logger.debug("Synced affirmation text state.")

    def clear_affirmation_upload_state(self):
        """Callback to clear affirmation audio state when file uploader changes."""
        # This is called when a file is uploaded OR removed.
        # We only clear if the source was 'upload'. If it was 'text', keep the generated audio.
        if st.session_state.get("wizard_affirmation_source") == "upload":
            st.session_state.wizard_affirmation_audio = None
            st.session_state.wizard_affirmation_sr = None
            st.session_state.wizard_affirmation_source = None  # Reset source
            st.session_state.wizard_affirmation_text = ""  # Clear text reference
            logger.debug("Cleared affirmation upload state due to file uploader change.")

    def sync_background_choice(self, choice_options: List[str]):
        """Callback to update background choice and clear audio if needed."""
        selected_label = st.session_state.get("wizard_bg_choice_radio")
        if not selected_label:
            return  # Should not happen

        new_choice = "none"  # Default
        if selected_label == "Upload Music/Sound":
            new_choice = "upload"
        elif selected_label == "Generate Noise":
            new_choice = "noise"

        # If the choice actually changed, clear the audio/sr
        if new_choice != st.session_state.get("wizard_background_choice"):
            st.session_state.wizard_background_audio = None
            st.session_state.wizard_background_sr = None
            logger.debug(f"Background choice changed to {new_choice}, cleared audio.")

        st.session_state.wizard_background_choice = new_choice
        st.session_state.wizard_background_choice_label = selected_label  # Store label for radio index

    def clear_background_upload_state(self):
        """Callback to clear background audio state when file uploader changes."""
        if st.session_state.get("wizard_background_choice") == "upload":
            st.session_state.wizard_background_audio = None
            st.session_state.wizard_background_sr = None
            logger.debug("Cleared background upload state due to file uploader change.")

    # --- Navigation and Reset ---

    def _reset_wizard_state(self):
        """Resets the wizard state using the dedicated function."""
        reset_wizard_state()
        # No need to call initialize here, reset does it.
        # Rerun to go back to the home screen (handled by main.py checking selected_workflow)
        st.rerun()

    def _go_to_step(self, step: int):
        """Updates the wizard step in session state and reruns."""
        if 1 <= step <= 4:  # Basic validation
            st.session_state.wizard_step = step
            logger.debug(f"Navigating to wizard step {step}")
            st.rerun()
        else:
            logger.warning(f"Invalid step navigation requested: {step}")

    # --- Core Processing Logic ---

    def _process_and_export(self):
        """Handles the final processing and export logic for the wizard."""
        logger.info("Starting wizard export process.")
        st.session_state.wizard_export_buffer = None  # Clear previous buffer/error
        st.session_state.wizard_export_error = None

        # --- Gather Audio Data from Session State ---
        affirmation_audio = st.session_state.get("wizard_affirmation_audio")
        affirmation_sr = st.session_state.get("wizard_affirmation_sr")
        background_audio = st.session_state.get("wizard_background_audio")
        background_sr = st.session_state.get("wizard_background_sr")
        background_volume = st.session_state.get("wizard_background_volume", 0.7)
        frequency_audio = st.session_state.get("wizard_frequency_audio")
        frequency_sr = st.session_state.get("wizard_frequency_sr")
        frequency_volume = st.session_state.get("wizard_frequency_volume", 0.2)
        export_format = st.session_state.get("wizard_export_format", "wav").lower()

        if affirmation_audio is None or affirmation_sr is None:
            st.session_state.wizard_export_error = "Affirmation audio is missing. Cannot export."
            logger.error("Wizard export failed: Missing affirmation audio.")
            return

        # --- Construct Tracks Dictionary for Mixing ---
        tracks_dict = {}
        track_id_counter = 0

        # 1. Affirmation Track (with fixed processing)
        affirm_params = get_default_track_params()
        affirm_params.update(
            {
                "original_audio": affirmation_audio,
                "sr": affirmation_sr,
                "name": "Wizard Affirmations",
                "track_type": TRACK_TYPE_AFFIRMATION,
                "volume": WIZARD_AFFIRMATION_VOLUME,  # Fixed low volume
                "speed_factor": WIZARD_AFFIRMATION_SPEED,  # Fixed high speed
                "pitch_shift": 0.0,  # Ensure no pitch shift
                "ultrasonic_shift": False,
                "loop_to_fit": True,  # Always loop affirmations
            }
        )
        tracks_dict[f"wizard_track_{track_id_counter}"] = affirm_params
        track_id_counter += 1
        logger.debug("Added affirmation track to mix dict.")

        # 2. Background Track (Optional)
        if background_audio is not None and background_sr is not None:
            bg_params = get_default_track_params()
            bg_params.update(
                {
                    "original_audio": background_audio,
                    "sr": background_sr,
                    "name": "Wizard Background",
                    "track_type": TRACK_TYPE_BACKGROUND,
                    "volume": background_volume,  # User chosen volume
                    "loop_to_fit": True,  # Always loop background
                }
            )
            tracks_dict[f"wizard_track_{track_id_counter}"] = bg_params
            track_id_counter += 1
            logger.debug("Added background track to mix dict.")

        # 3. Frequency Track (Optional)
        if frequency_audio is not None and frequency_sr is not None:
            freq_params = get_default_track_params()
            freq_params.update(
                {
                    "original_audio": frequency_audio,
                    "sr": frequency_sr,
                    "name": "Wizard Frequency",
                    "track_type": TRACK_TYPE_FREQUENCY,
                    "volume": frequency_volume,  # User chosen volume
                    "loop_to_fit": True,  # Always loop frequency
                }
            )
            tracks_dict[f"wizard_track_{track_id_counter}"] = freq_params
            track_id_counter += 1
            logger.debug("Added frequency track to mix dict.")

        # --- Mix Tracks ---
        logger.info(f"Wizard mixing {len(tracks_dict)} tracks.")
        spinner_msg = f"Generating final mix ({export_format.upper()}). This may take a moment..."
        with st.spinner(spinner_msg):
            try:
                # Ensure mix_tracks is robust and handles potential errors
                full_mix, final_mix_len_samples = mix_tracks(
                    tracks_dict,
                    preview=False,  # Generate full mix
                    target_sr=GLOBAL_SR,
                )

                if full_mix is None or full_mix.size == 0:
                    raise ValueError("Mixing process resulted in empty audio.")

                logger.info(f"Mixing successful. Final mix length: {final_mix_len_samples / GLOBAL_SR:.2f} seconds.")

                # --- Save to Buffer ---
                if export_format == "wav":
                    export_buffer = save_audio_to_bytesio(full_mix, GLOBAL_SR)
                    if export_buffer:
                        st.session_state.wizard_export_buffer = export_buffer
                        logger.info("Wizard WAV mix generated and stored in buffer.")
                    else:
                        raise ValueError("Failed to save WAV mix to buffer.")
                elif export_format == "mp3" and PYDUB_AVAILABLE:
                    try:
                        logger.info("Wizard converting full mix to MP3...")
                        # Ensure audio is in correct range [-1, 1] before scaling
                        full_mix = np.clip(full_mix, -1.0, 1.0)
                        audio_int16 = (full_mix * 32767).astype(np.int16)
                        # Ensure stereo if not already (pydub might handle mono, but explicit is safer)
                        channels = 2 if audio_int16.ndim > 1 and audio_int16.shape[1] == 2 else 1

                        segment = AudioSegment(data=audio_int16.tobytes(), sample_width=audio_int16.dtype.itemsize, frame_rate=GLOBAL_SR, channels=channels)
                        # If mono, convert to stereo for wider compatibility
                        if channels == 1:
                            segment = segment.set_channels(2)
                            logger.info("Converted mono mix to stereo for MP3 export.")

                        mp3_buffer = BytesIO()
                        segment.export(mp3_buffer, format="mp3", bitrate="192k")  # Consider making bitrate configurable?
                        mp3_buffer.seek(0)
                        st.session_state.wizard_export_buffer = mp3_buffer
                        logger.info("Wizard MP3 mix generated and stored in buffer.")
                    except Exception as e_mp3:
                        logger.exception("Wizard failed to export mix as MP3 using pydub.")
                        st.session_state.wizard_export_error = f"MP3 Export Failed: {e_mp3}. Ensure ffmpeg is installed and accessible in system PATH."
                elif export_format == "mp3" and not PYDUB_AVAILABLE:
                    st.session_state.wizard_export_error = "MP3 export selected, but 'pydub' library is missing."
                    logger.error(st.session_state.wizard_export_error)
                else:
                    st.session_state.wizard_export_error = f"Unsupported export format '{export_format}' requested."
                    logger.error(st.session_state.wizard_export_error)

            except Exception as e:
                logger.exception("Error during wizard mix generation or saving.")
                st.session_state.wizard_export_error = f"Processing Failed: {e}"

    # --- Main Rendering Method ---

    def render_wizard(self):
        """Renders the current step of the wizard by calling the appropriate step function."""
        st.title("✨ MindMorph Quick Create Wizard")

        # Ensure state is initialized (might be redundant if __init__ always runs, but safe)
        initialize_wizard_state()

        step = st.session_state.get("wizard_step", 1)

        # Simple progress indicator
        steps_display = ["Affirmations", "Background", "Frequency", "Export"]
        # Ensure step is within valid range for progress calculation
        progress_step = max(1, min(step, len(steps_display)))
        try:
            st.progress((progress_step) / len(steps_display), text=f"Step {progress_step}: {steps_display[progress_step - 1]}")
        except IndexError:
            st.progress(0.0)  # Should not happen with validation
            logger.warning(f"Progress bar step index out of range: {progress_step}")

        # Call the appropriate rendering function, passing self
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
            logger.error(f"Invalid wizard step in render_wizard: {step}. Resetting state.")
            self._reset_wizard_state()  # This will trigger a rerun
