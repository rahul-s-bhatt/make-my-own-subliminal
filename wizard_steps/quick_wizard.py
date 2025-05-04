# quick_wizard.py (Refactored - Uses Config)
# ==========================================
# Quick Create Wizard Orchestrator for MindMorph
# Imports constants and defaults from quick_wizard_config.py
# ==========================================

import gc
import logging
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# Import necessary components from other modules
try:
    from audio_utils.audio_io import save_audio_to_bytesio

    AUDIO_IO_AVAILABLE = True
except ImportError:
    AUDIO_IO_AVAILABLE = False
    logging.warning("audio_utils.audio_io.save_audio_to_bytesio not found.")

# Import wizard state management & config
try:
    # Main app config
    from config import GLOBAL_SR, QUICK_SUBLIMINAL_PRESET_SPEED
except ImportError as e:
    logging.error(f"Failed to import from main config: {e}")
    GLOBAL_SR = 22050
    QUICK_SUBLIMINAL_PRESET_SPEED = 2.0

# Import state initialization and reset functions from wizard_state
try:
    from .wizard_state import initialize_wizard_state, reset_wizard_state
except ImportError:
    logging.error("Failed to import state management functions from .wizard_state")

    def initialize_wizard_state():
        logging.error("Dummy initialize_wizard_state called.")

    def reset_wizard_state():
        logging.error("Dummy reset_wizard_state called.")


# Import ALL constants from the new wizard config file
try:
    from .quick_wizard_config import *  # Import all constants

    logging.info("Successfully imported constants from .quick_wizard_config")
except ImportError as e:
    logging.error(
        f"CRITICAL: Failed to import constants from .quick_wizard_config: {e}"
    )
    # Define fallbacks for essential keys if import fails, otherwise app will crash
    AFFIRM_APPLY_SPEED_KEY = "wizard_apply_speed_change"
    AFFIRMATION_TEXT_KEY = "wizard_affirmation_text"
    BG_CHOICE_KEY = "wizard_background_choice"
    BG_UPLOADED_FILE_KEY = "wizard_background_uploaded_file"
    BG_NOISE_TYPE_KEY = "wizard_background_noise_type"
    FREQ_CHOICE_KEY = "wizard_frequency_choice"
    FREQ_PARAMS_KEY = "wizard_frequency_params"
    AFFIRMATION_VOLUME_KEY = "wizard_affirmation_volume"
    BG_VOLUME_KEY = "wizard_background_volume"
    FREQ_VOLUME_KEY = "wizard_frequency_volume"
    EXPORT_FORMAT_KEY = "wizard_export_format"
    EXPORT_BUFFER_KEY = "wizard_export_buffer"
    EXPORT_ERROR_KEY = "wizard_export_error"
    PREVIEW_BUFFER_KEY = "wizard_preview_buffer"
    PREVIEW_ERROR_KEY = "wizard_preview_error"
    WIZARD_PROCESSING_ACTIVE_KEY = "wizard_processing_active"
    WIZARD_STEP_KEY = "wizard_step"
    LEGACY_AFFIRM_AUDIO_KEY = "wizard_affirmation_audio"
    LEGACY_AFFIRM_SR_KEY = "wizard_affirmation_sr"
    LEGACY_BG_AUDIO_KEY = "wizard_background_audio"
    LEGACY_BG_SR_KEY = "wizard_background_sr"
    LEGACY_FREQ_AUDIO_KEY = "wizard_frequency_audio"
    LEGACY_FREQ_SR_KEY = "wizard_frequency_sr"
    # Fallback for lists/defaults might be needed too
    NOISE_TYPES = ["White Noise", "Pink Noise", "Brown Noise"]


# Import the Piper TTS Generator
try:
    from tts.piper_tts import PiperTTSGenerator
except ImportError:
    PiperTTSGenerator = None
    logging.error("Failed to import PiperTTSGenerator.")

# Import step rendering functions
try:
    from . import (
        step_1_affirmations,
        step_2_background,
        step_3_frequency,
        step_4_export,
    )
except ImportError as e:
    logging.error(f"Failed to import step modules relatively: {e}")
    step_1_affirmations = step_2_background = step_3_frequency = step_4_export = None


# Import the Audio Processor
try:
    from .wizard_audio_processor import AudioData, AudioTuple, WizardAudioProcessor

    AUDIO_PROCESSOR_AVAILABLE = True
    logging.info("Successfully imported WizardAudioProcessor relatively.")
except ImportError as e:
    AUDIO_PROCESSOR_AVAILABLE = False
    logging.error(
        f"CRITICAL: Failed relative import of WizardAudioProcessor: {e}. Audio functions disabled."
    )
    AudioData = np.ndarray
    AudioTuple = Optional[Tuple[AudioData, int]]  # type: ignore

    class WizardAudioProcessor:  # type: ignore
        def __init__(self):
            logger.error("Dummy WizardAudioProcessor initialized.")

        def load_uploaded_audio(self, *args, **kwargs):
            raise RuntimeError("WAProc not available.")

        def generate_noise_audio(self, *args, **kwargs):
            raise RuntimeError("WAProc not available.")

        def generate_frequency_audio(self, *args, **kwargs):
            raise RuntimeError("WAProc not available.")

        def generate_preview_mix(self, *args, **kwargs):
            raise RuntimeError("WAProc not available.")

        def process_and_export(self, *args, **kwargs):
            raise RuntimeError("WAProc not available.")


logger = logging.getLogger(__name__)


# --- Caching Functions ---
@st.cache_resource(show_spinner="Initializing TTS Engine...")
def get_tts_generator() -> Optional[PiperTTSGenerator]:
    if PiperTTSGenerator is None:
        logging.error("Cannot init TTS: Class not imported.")
        return None
    logger.info("Attempting to create and cache PiperTTSGenerator instance.")
    try:
        return PiperTTSGenerator()
    except Exception as e:
        logger.exception("CRITICAL: Failed to create PiperTTSGenerator.")
        return None


@st.cache_resource(show_spinner="Initializing Audio Processor...")
def get_audio_processor() -> Optional[WizardAudioProcessor]:
    if not AUDIO_PROCESSOR_AVAILABLE:
        logger.error("Cannot init Audio Processor: Class import failed.")
        return None
    logger.info("Attempting to create and cache WizardAudioProcessor instance.")
    try:
        return WizardAudioProcessor()
    except Exception as e:
        logger.exception("CRITICAL: Failed to create WizardAudioProcessor.")
        return None


# --- Main Wizard Class ---
class QuickWizard:
    """Manages the state and UI rendering orchestration for the Quick Create Wizard."""

    def __init__(self):
        """Initializes the QuickWizard."""
        self.tts_generator: Optional[PiperTTSGenerator] = None
        self.audio_processor: Optional[WizardAudioProcessor] = None
        initialization_error = None
        try:
            initialize_wizard_state()  # Ensure state is initialized first
            self.tts_generator = get_tts_generator()
            self.audio_processor = get_audio_processor()
            logger.info("Retrieved TTS Generator and Audio Processor instances.")
            if self.tts_generator is None:
                initialization_error = "TTS Generator failed."
            if self.audio_processor is None:
                error_msg = "Audio Processor failed."
                initialization_error = (
                    f"{initialization_error} {error_msg}"
                    if initialization_error
                    else error_msg
                )
            if initialization_error:
                logger.critical(f"Init Failed: {initialization_error}")
        except Exception as e:
            initialization_error = f"Unexpected init error: {e}"
            logger.exception(f"CRITICAL: {initialization_error}")
            self.tts_generator = None
            self.audio_processor = None
        self._initialization_error = initialization_error
        # Use constants for keys when checking state
        if PREVIEW_BUFFER_KEY not in st.session_state:
            st.session_state[PREVIEW_BUFFER_KEY] = None
        if PREVIEW_ERROR_KEY not in st.session_state:
            st.session_state[PREVIEW_ERROR_KEY] = None
        if EXPORT_BUFFER_KEY not in st.session_state:
            st.session_state[EXPORT_BUFFER_KEY] = None
        if EXPORT_ERROR_KEY not in st.session_state:
            st.session_state[EXPORT_ERROR_KEY] = None
        if WIZARD_PROCESSING_ACTIVE_KEY not in st.session_state:
            st.session_state[WIZARD_PROCESSING_ACTIVE_KEY] = False
        logger.debug("QuickWizard initialization attempt finished.")

    # --- State Synchronization Callbacks ---
    # Use constants for keys
    def sync_affirmation_text(self):
        st.session_state[AFFIRMATION_TEXT_KEY] = st.session_state.get(
            "wizard_affirm_text_area", ""
        )  # Widget key kept separate
        cleared = st.session_state.pop(LEGACY_AFFIRM_AUDIO_KEY, None) is not None
        cleared |= st.session_state.pop(LEGACY_AFFIRM_SR_KEY, None) is not None
        if cleared:
            logger.debug("Cleared legacy affirmation audio/sr state.")
            gc.collect()
        logger.debug("Synced affirmation text state.")

    def clear_affirmation_upload_state(self):
        self.sync_affirmation_text()

    def sync_background_choice(self, choice_options: List[str]):
        selected_label = st.session_state.get("wizard_bg_choice_radio")
        if not selected_label:
            return
        new_choice = "none"
        if selected_label == "Upload Music/Sound":
            new_choice = "upload"
        elif selected_label == "Generate Noise":
            new_choice = "noise"
        current_choice = st.session_state.get(BG_CHOICE_KEY)
        if new_choice != current_choice:
            logger.info(
                f"BG choice changed: '{current_choice}'->'{new_choice}'. Updating."
            )
            st.session_state[BG_CHOICE_KEY] = new_choice
            st.session_state[BG_CHOICE_LABEL_KEY] = selected_label
            cleared = st.session_state.pop(LEGACY_BG_AUDIO_KEY, None) is not None
            cleared |= st.session_state.pop(LEGACY_BG_SR_KEY, None) is not None
            if cleared:
                logger.debug("Cleared legacy background audio/sr state.")
            if new_choice == "upload":
                st.session_state[BG_NOISE_TYPE_KEY] = NOISE_TYPES[0]
                logger.debug("Reset noise type.")
            elif new_choice == "noise":
                st.session_state.pop(BG_UPLOADED_FILE_KEY, None)
                logger.debug("Cleared uploaded file object.")
            else:
                st.session_state.pop(BG_UPLOADED_FILE_KEY, None)
                st.session_state[BG_NOISE_TYPE_KEY] = NOISE_TYPES[0]
                logger.debug("Cleared upload & reset noise type.")
            gc.collect()

    def clear_background_upload_state(self):
        cleared = st.session_state.pop(LEGACY_BG_AUDIO_KEY, None) is not None
        cleared |= st.session_state.pop(LEGACY_BG_SR_KEY, None) is not None
        cleared |= st.session_state.pop(BG_UPLOADED_FILE_KEY, None) is not None
        if cleared:
            logger.debug("Cleared legacy BG audio/sr/file object via callback.")
            gc.collect()

    # --- Navigation and Reset ---
    def _reset_wizard_state(self):
        # This function now calls the reset defined in wizard_state, which uses constants
        logger.info("Resetting wizard state via wizard_state.reset_wizard_state().")
        reset_wizard_state()
        # No st.rerun() needed here, reset_wizard_state should handle it if necessary

    def _go_to_step(self, step: int):
        # Use constant for step key
        if 1 <= step <= 4:
            current_step = st.session_state.get(WIZARD_STEP_KEY, 1)
            if step != current_step:
                st.session_state[WIZARD_STEP_KEY] = step
                logger.debug(f"Navigating from {current_step} to {step}")
                # Use constants for buffer/error keys
                st.session_state.pop(PREVIEW_BUFFER_KEY, None)
                st.session_state.pop(PREVIEW_ERROR_KEY, None)
                st.session_state.pop(EXPORT_BUFFER_KEY, None)
                st.session_state.pop(EXPORT_ERROR_KEY, None)
                st.rerun()
            else:
                logger.debug(f"Already on step {step}.")
        else:
            logger.warning(f"Invalid step navigation: {step}")

    # --- State Gathering Helper ---
    def _get_current_processing_state(self) -> Dict[str, Any]:
        """Gathers relevant wizard state values needed for audio processing."""
        # Use constants for keys and provide defaults
        speed_state_value = st.session_state.get(
            AFFIRM_APPLY_SPEED_KEY, DEFAULT_APPLY_SPEED
        )
        logger.info(
            f"DEBUG GET_STATE - Reading {AFFIRM_APPLY_SPEED_KEY}: {speed_state_value} (Type: {type(speed_state_value)})"
        )

        state = {
            # Affirmation Settings
            "wizard_apply_speed_change": speed_state_value,
            "wizard_speed_factor": QUICK_SUBLIMINAL_PRESET_SPEED,  # From main config
            # Background Settings
            "wizard_background_choice": st.session_state.get(
                BG_CHOICE_KEY, DEFAULT_BG_CHOICE
            ),
            "wizard_background_uploaded_file": st.session_state.get(
                BG_UPLOADED_FILE_KEY
            ),  # Default is None via init
            "wizard_background_noise_type": st.session_state.get(
                BG_NOISE_TYPE_KEY, DEFAULT_NOISE_TYPE
            ),
            # Frequency Settings
            "wizard_frequency_choice": st.session_state.get(
                FREQ_CHOICE_KEY, DEFAULT_FREQ_CHOICE
            ),
            "wizard_frequency_params": st.session_state.get(
                FREQ_PARAMS_KEY, DEFAULT_FREQ_PARAMS
            ),
            # Volume Settings
            "wizard_affirmation_volume": st.session_state.get(
                AFFIRMATION_VOLUME_KEY, DEFAULT_AFFIRMATION_VOLUME
            ),
            "wizard_background_volume": st.session_state.get(
                BG_VOLUME_KEY, DEFAULT_BG_VOLUME
            ),
            "wizard_frequency_volume": st.session_state.get(
                FREQ_VOLUME_KEY, DEFAULT_FREQ_VOLUME
            ),
            # Export Format
            "wizard_export_format": st.session_state.get(
                EXPORT_FORMAT_KEY, DEFAULT_EXPORT_FORMAT
            ),
        }
        logger.debug("Gathered current processing state.")
        return state

    # --- Dynamic Generation Helpers ---
    def _generate_affirmation_audio(self) -> Optional[AudioTuple]:
        # Use constant key
        affirmation_text = st.session_state.get(AFFIRMATION_TEXT_KEY, "").strip()
        if not affirmation_text:
            logger.warning("Affirmation text empty.")
            return None
        if not self.tts_generator:
            raise RuntimeError("TTS Engine not initialized.")
        logger.info("Generating affirmation audio dynamically...")
        try:
            start_time = time.time()
            audio, sr = self.tts_generator.generate(affirmation_text)
            logger.info(
                f"Affirmation audio generated in {time.time() - start_time:.2f}s."
            )
            if audio.size == 0 or sr != GLOBAL_SR:
                logger.error(f"TTS failed/wrong SR ({sr}).")
                return None
            return (audio, sr)
        except Exception as e:
            raise RuntimeError(f"Failed to generate affirmation audio: {e}") from e

    def _generate_or_load_background_audio(
        self,
        bg_choice: str,
        bg_noise_type: str,
        uploaded_file: Optional[Any],
        target_duration_hint: float,
    ) -> Optional[AudioTuple]:
        # Logic remains the same, relies on audio_processor methods
        if bg_choice == "none":
            logger.info("No background selected.")
            return None
        if not self.audio_processor:
            logger.error("Audio processor unavailable for BG.")
            return None
        bg_tuple: Optional[AudioTuple] = None
        try:
            if bg_choice == "upload":
                if uploaded_file:
                    logger.info(f"Loading uploaded BG: {uploaded_file.name}")
                    bg_tuple = self.audio_processor.load_uploaded_audio(uploaded_file)
                else:
                    logger.warning("BG choice 'upload' but no file found.")
            elif bg_choice == "noise":
                logger.info(f"Generating BG noise: {bg_noise_type}")
                bg_tuple = self.audio_processor.generate_noise_audio(
                    bg_noise_type, target_duration_hint, GLOBAL_SR
                )
            if bg_tuple and (
                not isinstance(bg_tuple[0], np.ndarray)
                or bg_tuple[0].size == 0
                or bg_tuple[1] != GLOBAL_SR
            ):
                logger.error(f"Dynamic BG processing invalid result for '{bg_choice}'.")
                bg_tuple = None
            elif bg_tuple:
                logger.info(f"BG audio '{bg_choice}' processed successfully.")
        except Exception as e:
            logger.exception(f"Error processing dynamic BG ({bg_choice}).")
            bg_tuple = None
        return bg_tuple

    def _generate_frequency_audio(
        self, freq_choice: str, freq_params: Dict[str, Any], target_duration_hint: float
    ) -> Optional[AudioTuple]:
        # Logic remains the same, relies on audio_processor method
        if freq_choice == "None":
            logger.info("No frequency selected.")
            return None
        if not self.audio_processor:
            logger.error("Audio processor unavailable for Freq.")
            return None
        freq_tuple: Optional[AudioTuple] = None
        try:
            logger.info(
                f"Generating frequency audio: {freq_choice} with params: {freq_params}"
            )
            freq_tuple = self.audio_processor.generate_frequency_audio(
                freq_choice, freq_params, target_duration_hint, GLOBAL_SR
            )
            if freq_tuple and (
                not isinstance(freq_tuple[0], np.ndarray)
                or freq_tuple[0].size == 0
                or freq_tuple[1] != GLOBAL_SR
            ):
                logger.error(
                    f"Dynamic Freq processing invalid result for '{freq_choice}'."
                )
                freq_tuple = None
            elif freq_tuple:
                logger.info(f"Freq audio '{freq_choice}' processed successfully.")
        except Exception as e:
            logger.exception(f"Error processing dynamic Freq ({freq_choice}).")
            freq_tuple = None
        return freq_tuple

    # --- Wrappers for Audio Processing ---
    def generate_preview(self, duration_seconds: int = 10):
        logger.info(
            f"QuickWizard: Requesting preview generation ({duration_seconds}s)."
        )
        # Use constants for keys
        st.session_state.pop(PREVIEW_BUFFER_KEY, None)
        st.session_state.pop(PREVIEW_ERROR_KEY, None)
        if not self.audio_processor or not self.tts_generator or not AUDIO_IO_AVAILABLE:
            st.session_state[PREVIEW_ERROR_KEY] = "Required components not available."
            logger.error(f"Preview failed: Missing components.")
            return
        affirmation_tuple: Optional[AudioTuple] = None
        background_tuple: Optional[AudioTuple] = None
        frequency_tuple: Optional[AudioTuple] = None
        preview_buffer: Optional[BytesIO] = None
        try:
            affirmation_tuple = self._generate_affirmation_audio()
            if affirmation_tuple is None:
                st.session_state[PREVIEW_ERROR_KEY] = (
                    "Failed to generate affirmation audio."
                )
                return
            affirmation_duration = (
                affirmation_tuple[0].shape[0] / GLOBAL_SR if affirmation_tuple else 0
            )
            processing_state = (
                self._get_current_processing_state()
            )  # Gets state using constants
            background_tuple = self._generate_or_load_background_audio(
                processing_state["wizard_background_choice"],
                processing_state["wizard_background_noise_type"],
                processing_state["wizard_background_uploaded_file"],
                affirmation_duration,
            )
            frequency_tuple = self._generate_frequency_audio(
                processing_state["wizard_frequency_choice"],
                processing_state["wizard_frequency_params"],
                affirmation_duration,
            )
            logger.info("Calling audio processor for preview mix...")
            preview_audio, preview_sr = self.audio_processor.generate_preview_mix(
                duration_seconds,
                affirmation_tuple,
                background_tuple,
                frequency_tuple,
                processing_state,
            )
            if preview_audio is not None and preview_sr is not None:
                preview_buffer = save_audio_to_bytesio(preview_audio, preview_sr)
                if preview_buffer and preview_buffer.getbuffer().nbytes > 0:
                    st.session_state[PREVIEW_BUFFER_KEY] = preview_buffer
                    logger.info("Preview generated.")
                else:
                    st.session_state[PREVIEW_ERROR_KEY] = (
                        "Preview failed (empty buffer)."
                    )
                    st.session_state.pop(PREVIEW_BUFFER_KEY, None)
            else:
                st.session_state[PREVIEW_ERROR_KEY] = (
                    "Preview failed (processor returned None)."
                )
        except Exception as e:
            logger.exception("Error during preview wrapper.")
            st.session_state[PREVIEW_ERROR_KEY] = f"Preview Error: {e}"
            st.session_state.pop(PREVIEW_BUFFER_KEY, None)
        finally:
            del affirmation_tuple, background_tuple, frequency_tuple
            if (
                "preview_buffer" in locals()
                and preview_buffer is not None
                and id(preview_buffer) != id(st.session_state.get(PREVIEW_BUFFER_KEY))
            ):
                del preview_buffer
            gc.collect()
            logger.debug("Cleaned up temp audio tuples for preview.")

    def process_and_export_audio(self):
        logger.info("QuickWizard: Requesting final processing and export.")
        # Use constants for keys
        st.session_state.pop(EXPORT_BUFFER_KEY, None)
        st.session_state.pop(EXPORT_ERROR_KEY, None)
        st.session_state[WIZARD_PROCESSING_ACTIVE_KEY] = True
        if not self.audio_processor or not self.tts_generator:
            st.session_state[EXPORT_ERROR_KEY] = "Required components not available."
            logger.error(f"Export failed: Missing components.")
            st.session_state[WIZARD_PROCESSING_ACTIVE_KEY] = False
            return
        affirmation_tuple: Optional[AudioTuple] = None
        background_tuple: Optional[AudioTuple] = None
        frequency_tuple: Optional[AudioTuple] = None
        try:
            affirmation_tuple = self._generate_affirmation_audio()
            if affirmation_tuple is None:
                st.session_state[EXPORT_ERROR_KEY] = (
                    "Failed to generate affirmation audio."
                )
                st.session_state[WIZARD_PROCESSING_ACTIVE_KEY] = False
                return
            affirmation_duration = (
                affirmation_tuple[0].shape[0] / GLOBAL_SR if affirmation_tuple else 0
            )
            processing_state = (
                self._get_current_processing_state()
            )  # Gets state using constants
            background_tuple = self._generate_or_load_background_audio(
                processing_state["wizard_background_choice"],
                processing_state["wizard_background_noise_type"],
                processing_state["wizard_background_uploaded_file"],
                affirmation_duration,
            )
            frequency_tuple = self._generate_frequency_audio(
                processing_state["wizard_frequency_choice"],
                processing_state["wizard_frequency_params"],
                affirmation_duration,
            )
            logger.info("Calling audio processor for final export...")
            export_buffer, error_message = self.audio_processor.process_and_export(
                affirmation_tuple, background_tuple, frequency_tuple, processing_state
            )
            if export_buffer and not error_message:
                st.session_state[EXPORT_BUFFER_KEY] = export_buffer
                logger.info("Export successful.")
                st.session_state.pop(PREVIEW_BUFFER_KEY, None)
            else:
                st.session_state[EXPORT_ERROR_KEY] = error_message or "Export failed."
                logger.error(f"Export failed: {st.session_state[EXPORT_ERROR_KEY]}")
        except Exception as e:
            logger.exception("Unhandled error during export wrapper.")
            st.session_state[EXPORT_ERROR_KEY] = f"Unexpected Export Error: {e}"
        finally:
            del affirmation_tuple, background_tuple, frequency_tuple
            gc.collect()
            logger.debug("Cleaned up temp audio tuples for export.")
            st.session_state[WIZARD_PROCESSING_ACTIVE_KEY] = False
            logger.info("Reset wizard_processing_active flag.")

    # --- Main Rendering Method ---
    def render_wizard(self):
        st.title("✨ MindMorph Quick Create Wizard")
        if self._initialization_error:
            logger.error(f"Render blocked by init error: {self._initialization_error}")
            st.error(f"FATAL ERROR: {self._initialization_error}. Wizard cannot start.")
            if st.button("Attempt Reset"):
                self._reset_wizard_state()
            return
        # Use constant for step key
        step = st.session_state.get(WIZARD_STEP_KEY, 1)
        steps_display = ["Affirmations", "Background", "Frequency", "Mix & Export"]
        progress_step = max(1, min(step, len(steps_display)))
        try:
            st.progress(
                (progress_step) / len(steps_display),
                text=f"Step {progress_step}: {steps_display[progress_step - 1]}",
            )
        except IndexError:
            st.progress(0.0)
            logger.error(f"Progress bar index error: step={step}")
        try:
            if step_1_affirmations and step == 1:
                step_1_affirmations.render_step_1(self)
            elif step_2_background and step == 2:
                step_2_background.render_step_2(self)
            elif step_3_frequency and step == 3:
                step_3_frequency.render_step_3(self)
            elif step_4_export and step == 4:
                step_4_export.render_step_4(self)
            elif step > 4 or step < 1:
                st.error(f"Internal Error: Invalid step ({step}). Resetting.")
                logger.error(f"Invalid step: {step}.")
                st.session_state[WIZARD_STEP_KEY] = 1
                st.rerun()
            else:
                st.error(f"Error: UI component for Step {step} failed to load.")
                logger.error(f"Cannot render Step {step}, module might be None.")
        except Exception as e_render:
            logger.exception(f"Error rendering UI for step {step}: {e_render}")
            st.error(f"An error occurred displaying Step {step}.")
            if st.button("Reset Wizard"):
                self._reset_wizard_state()
