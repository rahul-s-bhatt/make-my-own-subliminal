# quick_wizard.py
# ==========================================
# Quick Create Wizard Orchestrator for MindMorph
# ==========================================

import logging
import math
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st

# Import necessary components from other modules
from audio_utils.audio_io import save_audio_to_bytesio
from audio_utils.audio_mixers import mix_wizard_tracks

# Import wizard state management
from config import (
    GLOBAL_SR,
    QUICK_SUBLIMINAL_PRESET_SPEED,
    QUICK_SUBLIMINAL_PRESET_VOLUME,
)

# Import the new Piper TTS Generator
from tts.piper_tts import PiperTTSGenerator

# Import step rendering functions
from wizard_steps import (
    step_1_affirmations,
    step_2_background,
    step_3_frequency,
    step_4_export,
)
from wizard_steps.wizard_state import initialize_wizard_state, reset_wizard_state

# Type hint for AudioData
try:
    from audio_utils.audio_effects_pipeline import AudioData
except ImportError:
    AudioData = np.ndarray  # Fallback

# Optional MP3 export dependency check
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


logger = logging.getLogger(__name__)


# Define the AudioTuple type hint more explicitly
AudioTuple = Optional[Tuple[AudioData, int]]


class QuickWizard:
    """Manages the state and UI rendering orchestration for the Quick Create Wizard."""

    def __init__(self):
        """Initializes the QuickWizard."""
        try:
            self.tts_generator = PiperTTSGenerator()
            logger.info("PiperTTSGenerator initialized successfully for QuickWizard.")
        except Exception as e:
            logger.exception(
                "CRITICAL: Failed to initialize PiperTTSGenerator in QuickWizard."
            )
            raise RuntimeError(f"Failed to initialize TTS engine: {e}") from e
        initialize_wizard_state()
        logger.debug("QuickWizard initialized and state ensured.")

    # --- State Synchronization Callbacks --- (Keep as is)
    def sync_affirmation_text(self):  # ... (no changes needed)
        st.session_state.wizard_affirmation_text = st.session_state.get(
            "wizard_affirm_text_area", ""
        )
        if st.session_state.get("wizard_affirmation_source") == "text":
            st.session_state.wizard_affirmation_audio = None
            st.session_state.wizard_affirmation_sr = None
        logger.debug("Synced affirmation text state.")

    def clear_affirmation_upload_state(self):
        pass

    def sync_background_choice(
        self, choice_options: List[str]
    ):  # ... (no changes needed)
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
        pass

    # --- Navigation and Reset --- (Keep as is)
    def _reset_wizard_state(self):  # ... (no changes needed)
        reset_wizard_state()
        st.rerun()

    def _go_to_step(self, step: int):  # ... (no changes needed)
        if 1 <= step <= 4:
            st.session_state.wizard_step = step
            logger.debug(f"Navigating to wizard step {step}")
            st.rerun()
        else:
            logger.warning(f"Invalid step navigation requested: {step}")

    # --- Helper Function for Looping Audio --- (Keep as is)
    def _loop_audio_to_length(
        self, audio_data: AudioData, target_length: int
    ) -> AudioData:  # ... (no changes needed)
        current_length = audio_data.shape[0]
        if current_length == target_length:
            return audio_data
        elif current_length < target_length:
            num_repeats = math.ceil(target_length / current_length)
            if audio_data.ndim == 1:
                looped_audio = np.tile(audio_data, num_repeats)
            else:
                looped_audio = np.tile(audio_data, (num_repeats, 1))
            return looped_audio[:target_length]
        else:
            return audio_data[:target_length]

    # --- Core Processing Logic ---
    def _process_and_export(self):
        """Handles the final processing and export logic for the wizard."""
        logger.info("Starting wizard export process.")
        st.session_state.wizard_export_buffer = None
        st.session_state.wizard_export_error = None

        # --- ADDED: Wrap processing in try...finally to reset lock ---
        try:
            # Get base audio data and sample rates
            affirmation_audio_data: Optional[AudioData] = st.session_state.get(
                "wizard_affirmation_audio"
            )
            affirmation_sr: Optional[int] = st.session_state.get(
                "wizard_affirmation_sr"
            )
            background_audio_data: Optional[AudioData] = st.session_state.get(
                "wizard_background_audio"
            )
            background_sr: Optional[int] = st.session_state.get("wizard_background_sr")
            frequency_audio_data: Optional[AudioData] = st.session_state.get(
                "wizard_frequency_audio"
            )
            frequency_sr: Optional[int] = st.session_state.get("wizard_frequency_sr")

            # Get volume settings
            background_volume: float = st.session_state.get(
                "wizard_background_volume", 0.7
            )
            frequency_volume: float = st.session_state.get(
                "wizard_frequency_volume", 0.2
            )

            # Validate Affirmation Audio
            if affirmation_audio_data is None or affirmation_sr is None:
                st.session_state.wizard_export_error = "Affirmation audio is missing."
                logger.error("Wizard export failed: Missing affirmation audio.")
                return  # Exit early

            # Determine Affirmation Speed/Volume
            apply_quick_settings: bool = st.session_state.get(
                "wizard_apply_quick_settings", True
            )
            if apply_quick_settings:
                eff_affirm_speed: float = QUICK_SUBLIMINAL_PRESET_SPEED
                eff_affirm_volume: float = QUICK_SUBLIMINAL_PRESET_VOLUME
                logger.info("Applying Quick Wizard preset settings (Speed/Volume).")
            else:
                eff_affirm_speed = 1.0
                eff_affirm_volume = 1.0
                logger.info("Using original speed/volume for affirmations.")

            export_format: str = st.session_state.get(
                "wizard_export_format", "wav"
            ).lower()

            # Prepare Tracks for Mixing
            try:
                if affirmation_sr != GLOBAL_SR:
                    logger.warning(
                        f"Affirmation SR ({affirmation_sr}) != GLOBAL_SR ({GLOBAL_SR})."
                    )
                target_length_samples = affirmation_audio_data.shape[0]
                logger.info(
                    f"Target mix length: {target_length_samples / GLOBAL_SR:.2f}s"
                )
                affirmation_tuple: AudioTuple = (affirmation_audio_data, GLOBAL_SR)
                background_tuple: AudioTuple = None
                if background_audio_data is not None and background_sr is not None:
                    if background_sr != GLOBAL_SR:
                        logger.warning(f"Background SR mismatch.")
                    looped_background = self._loop_audio_to_length(
                        background_audio_data, target_length_samples
                    )
                    background_tuple = (looped_background, GLOBAL_SR)
                    logger.info(f"Background audio prepared.")
                frequency_tuple: AudioTuple = None
                if frequency_audio_data is not None and frequency_sr is not None:
                    if frequency_sr != GLOBAL_SR:
                        logger.warning(f"Frequency SR mismatch.")
                    looped_frequency = self._loop_audio_to_length(
                        frequency_audio_data, target_length_samples
                    )
                    frequency_tuple = (looped_frequency, GLOBAL_SR)
                    logger.info(f"Frequency audio prepared.")
            except Exception as e_prep:
                logger.exception("Error preparing tracks for mixing.")
                st.session_state.wizard_export_error = (
                    f"Track Preparation Failed: {e_prep}"
                )
                return  # Exit early

            # Mix Tracks
            logger.info("Wizard mixing tracks...")
            try:
                full_mix = mix_wizard_tracks(
                    affirmation_audio=affirmation_tuple,
                    background_audio=background_tuple,
                    frequency_audio=frequency_tuple,
                    affirmation_speed=eff_affirm_speed,
                    affirmation_volume=eff_affirm_volume,
                    background_volume=background_volume,
                    frequency_volume=frequency_volume,
                    target_sr=GLOBAL_SR,
                )
                if full_mix is None or full_mix.size == 0:
                    raise ValueError("Mixing resulted in empty audio.")
                mix_duration_s = len(full_mix) / GLOBAL_SR if GLOBAL_SR > 0 else 0
                logger.info(f"Mixing successful. Final length: {mix_duration_s:.2f}s.")

                # Export to selected format
                export_buffer = None
                if export_format == "wav":
                    logger.info("Saving final mix to WAV buffer...")
                    export_buffer = save_audio_to_bytesio(full_mix, GLOBAL_SR)
                    if export_buffer and export_buffer.getbuffer().nbytes > 0:
                        buffer_size = export_buffer.getbuffer().nbytes
                        expected_min_size = full_mix.shape[0] * 2 * 2 * 0.9
                        logger.info(f"Generated WAV buffer size: {buffer_size} bytes.")
                        if buffer_size < expected_min_size:
                            logger.error(
                                f"WAV buffer size ({buffer_size}) seems too small!"
                            )
                            st.session_state.wizard_export_error = (
                                "Generated WAV buffer was unexpectedly small."
                            )
                            export_buffer = None
                        else:
                            st.session_state.wizard_export_buffer = export_buffer
                            logger.info("Wizard WAV mix buffer generated.")
                    else:
                        logger.error("save_audio_to_bytesio failed (WAV).")
                        st.session_state.wizard_export_error = (
                            "Failed to save WAV buffer."
                        )
                        export_buffer = None
                elif export_format == "mp3" and PYDUB_AVAILABLE:
                    # MP3 Export Logic (keep existing)
                    try:
                        logger.info("Wizard converting full mix to MP3...")
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
                            buffer_size = mp3_buffer.getbuffer().nbytes
                            logger.info(
                                f"Generated MP3 export buffer size: {buffer_size} bytes."
                            )
                            if buffer_size < 1000:
                                logger.error(
                                    f"MP3 buffer size ({buffer_size}) seems too small!"
                                )
                                st.session_state.wizard_export_error = (
                                    "Generated MP3 buffer was unexpectedly small."
                                )
                                export_buffer = None
                            else:
                                st.session_state.wizard_export_buffer = mp3_buffer
                                logger.info("Wizard MP3 mix buffer generated.")
                                export_buffer = mp3_buffer
                        else:
                            logger.error("MP3 export resulted in empty buffer.")
                            st.session_state.wizard_export_error = (
                                "MP3 export failed (empty buffer)."
                            )
                            export_buffer = None
                    except Exception as e_mp3:
                        logger.exception("Wizard failed to export mix as MP3.")
                        st.session_state.wizard_export_error = (
                            f"MP3 Export Failed: {e_mp3}."
                        )
                        export_buffer = None
                # Handle other cases (MP3 unavailable, unsupported format)
                elif export_format == "mp3":
                    st.session_state.wizard_export_error = (
                        "MP3 requires 'pydub' and 'ffmpeg'."
                    )
                else:
                    st.session_state.wizard_export_error = (
                        f"Unsupported format '{export_format}'."
                    )
                if export_buffer is None and not st.session_state.get(
                    "wizard_export_error"
                ):
                    st.session_state.wizard_export_error = (
                        "Export failed for unknown reason."  # Fallback error
                    )

            except Exception as e_mix_save:
                logger.exception("Error during wizard mix/save.")
                st.session_state.wizard_export_error = (
                    f"Processing Failed: {e_mix_save}"
                )

        # --- ADDED: finally block to reset the processing flag ---
        finally:
            st.session_state.wizard_processing_active = False
            logger.info("Reset wizard_processing_active flag to False.")
            # No st.rerun() here, it happens in step_4_export after this function returns
        # --- END ADDED ---

    # --- Main Rendering Method --- (Keep as is)
    def render_wizard(self):  # ... (no changes needed)
        st.title("✨ MindMorph Quick Create Wizard")
        initialize_wizard_state()
        if not hasattr(self, "tts_generator") or self.tts_generator is None:
            st.error("TTS Engine failed to initialize. Check logs/config.")
            return
        step = st.session_state.get("wizard_step", 1)
        steps_display = ["Affirmations", "Background", "Frequency", "Export"]
        progress_step = max(1, min(step, len(steps_display)))
        try:
            st.progress(
                (progress_step) / len(steps_display),
                text=f"Step {progress_step}: {steps_display[progress_step - 1]}",
            )
        except IndexError:
            st.progress(0.0)
            logger.warning(f"Progress bar index out of range: {progress_step}")
        if step == 1:
            step_1_affirmations.render_step_1(self)
        elif step == 2:
            step_2_background.render_step_2(self)
        elif step == 3:
            step_3_frequency.render_step_3(self)
        elif step == 4:
            step_4_export.render_step_4(self)
        else:
            st.error("Invalid wizard step. Resetting.")
            logger.error(f"Invalid step: {step}.")
            self._reset_wizard_state()
