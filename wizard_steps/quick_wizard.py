# quick_wizard.py (Refactored - Uses Config)
# ==========================================
# Quick Create Wizard Orchestrator for MindMorph
# Imports constants and defaults from quick_wizard_config.py
# Manages wizard flow, state interactions, and calls audio processing.
# ==========================================

import gc  # Garbage Collector interface
import logging
import time
from io import BytesIO  # For handling in-memory binary data (like audio buffers)
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # For numerical operations, especially with audio arrays
import streamlit as st  # The core Streamlit library for building the UI

# Import necessary components from other modules within the application
try:
    # Function to save audio data (numpy array) to a BytesIO buffer
    from audio_utils.audio_io import save_audio_to_bytesio

    AUDIO_IO_AVAILABLE = True
except ImportError:
    AUDIO_IO_AVAILABLE = False
    logging.warning("audio_utils.audio_io.save_audio_to_bytesio not found.")

# Import wizard state management & configuration constants
try:
    # Global application settings (like sample rate)
    from config import GLOBAL_SR, QUICK_SUBLIMINAL_PRESET_SPEED
except ImportError as e:
    # Fallback values if main config is not found (useful for isolated testing)
    logging.error(f"Failed to import from main config: {e}. Using fallbacks.")
    GLOBAL_SR = 22050  # Default sample rate
    QUICK_SUBLIMINAL_PRESET_SPEED = 2.0  # Default speed factor for affirmations

# Import state initialization and reset functions from the dedicated wizard state module
try:
    from .wizard_state import initialize_wizard_state, reset_wizard_state
except ImportError:
    # Provide dummy functions if the state module fails to import, preventing crashes
    logging.error(
        "Failed to import state management functions from .wizard_state. Using dummies."
    )

    def initialize_wizard_state():
        logging.error("Dummy initialize_wizard_state called.")

    def reset_wizard_state():
        logging.error("Dummy reset_wizard_state called.")


# Import ALL constants (keys, defaults, lists) from the wizard config file
try:
    # Using '*' imports all names defined in quick_wizard_config
    from .quick_wizard_config import *  # Includes WIZARD_PREVIEW_ACTIVE_KEY now

    logging.info("Successfully imported constants from .quick_wizard_config")
except ImportError as e:
    # Critical error if config cannot be imported, define essential fallbacks
    logging.error(
        f"CRITICAL: Failed to import constants from .quick_wizard_config: {e}. Defining fallbacks."
    )
    # Define fallback keys to prevent NameErrors later in the code
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
    WIZARD_PROCESSING_ACTIVE_KEY = "wizard_processing_active"  # Export flag
    WIZARD_PREVIEW_ACTIVE_KEY = "wizard_preview_active"  # Preview flag <--- FALLBACK
    WIZARD_STEP_KEY = "wizard_step"
    # Legacy keys might also need fallbacks if cleanup logic depends on them
    LEGACY_AFFIRM_AUDIO_KEY = "wizard_affirmation_audio"
    LEGACY_AFFIRM_SR_KEY = "wizard_affirmation_sr"
    LEGACY_BG_AUDIO_KEY = "wizard_background_audio"
    LEGACY_BG_SR_KEY = "wizard_background_sr"
    LEGACY_FREQ_AUDIO_KEY = "wizard_frequency_audio"
    LEGACY_FREQ_SR_KEY = "wizard_frequency_sr"
    # Fallback for lists/defaults might be needed too
    NOISE_TYPES = ["White Noise", "Pink Noise", "Brown Noise"]
    DEFAULT_STEP = 1
    DEFAULT_PROCESSING_ACTIVE = False  # <--- FALLBACK
    DEFAULT_PREVIEW_ACTIVE = False  # <--- FALLBACK
    DEFAULT_APPLY_SPEED = False
    DEFAULT_BG_CHOICE = "none"
    DEFAULT_BG_UPLOADED_FILE = None
    DEFAULT_NOISE_TYPE = "Brown Noise"
    DEFAULT_FREQ_CHOICE = "None"
    DEFAULT_FREQ_PARAMS = {}
    DEFAULT_AFFIRMATION_VOLUME = 1.0
    DEFAULT_BG_VOLUME = 0.7
    DEFAULT_FREQ_VOLUME = 0.2
    DEFAULT_EXPORT_FORMAT = "WAV"


# Import the Text-to-Speech (TTS) Generator class
try:
    from tts.piper_tts import PiperTTSGenerator
except ImportError:
    PiperTTSGenerator = None  # Set to None if import fails
    logging.error("Failed to import PiperTTSGenerator.")

# Import the rendering functions for each step of the wizard UI
try:
    # Relative imports for modules in the same directory (wizard_steps)
    from . import (
        step_1_affirmations,
        step_2_background,
        step_3_frequency,
        step_4_export,
    )
except ImportError as e:
    # Set modules to None if import fails, allows graceful degradation
    logging.error(f"Failed to import step modules relatively: {e}")
    step_1_affirmations = step_2_background = step_3_frequency = step_4_export = None


# Import the main Audio Processor class responsible for audio manipulation
try:
    from .wizard_audio_processor import AudioData, AudioTuple, WizardAudioProcessor

    AUDIO_PROCESSOR_AVAILABLE = True
    logging.info("Successfully imported WizardAudioProcessor relatively.")
except ImportError as e:
    # Critical failure if audio processor cannot be imported
    AUDIO_PROCESSOR_AVAILABLE = False
    logging.error(
        f"CRITICAL: Failed relative import of WizardAudioProcessor: {e}. Audio functions disabled."
    )
    # Define dummy types/classes to prevent NameErrors
    AudioData = np.ndarray
    AudioTuple = Optional[Tuple[AudioData, int]]  # type: ignore

    class WizardAudioProcessor:  # type: ignore
        """Dummy class used when the real WizardAudioProcessor fails to import."""

        def __init__(self):
            logger.error("Dummy WizardAudioProcessor initialized.")

        # Define dummy methods that raise errors if called
        def load_uploaded_audio(self, *args, **kwargs):
            raise RuntimeError("WizardAudioProcessor not available.")

        def generate_noise_audio(self, *args, **kwargs):
            raise RuntimeError("WizardAudioProcessor not available.")

        def generate_frequency_audio(self, *args, **kwargs):
            raise RuntimeError("WizardAudioProcessor not available.")

        def generate_preview_mix(self, *args, **kwargs):
            raise RuntimeError("WizardAudioProcessor not available.")

        def process_and_export(self, *args, **kwargs):
            raise RuntimeError("WizardAudioProcessor not available.")


# Setup logging
logger = logging.getLogger(__name__)

# --- Caching Functions ---
# Use Streamlit's caching to initialize potentially expensive resources once per session


@st.cache_resource(show_spinner="Initializing TTS Engine...")
def get_tts_generator() -> Optional[PiperTTSGenerator]:
    """
    Initializes and caches the PiperTTSGenerator instance.
    Returns None if the class wasn't imported or initialization fails.
    """
    if PiperTTSGenerator is None:
        logging.error("Cannot initialize TTS: PiperTTSGenerator class not imported.")
        return None
    logger.info("Attempting to create and cache PiperTTSGenerator instance.")
    try:
        tts_instance = PiperTTSGenerator()
        return tts_instance
    except Exception as e:
        logger.exception("CRITICAL: Failed to create PiperTTSGenerator instance.")
        return None


@st.cache_resource(show_spinner="Initializing Audio Processor...")
def get_audio_processor() -> Optional[WizardAudioProcessor]:
    """
    Initializes and caches the WizardAudioProcessor instance.
    Returns None if the class wasn't imported or initialization fails.
    """
    if not AUDIO_PROCESSOR_AVAILABLE:
        logger.error("Cannot initialize Audio Processor: Class import failed.")
        return None
    logger.info("Attempting to create and cache WizardAudioProcessor instance.")
    try:
        audio_processor_instance = WizardAudioProcessor()
        return audio_processor_instance
    except Exception as e:
        logger.exception("CRITICAL: Failed to create WizardAudioProcessor instance.")
        return None


# --- Main Wizard Class ---
class QuickWizard:
    """
    Manages the state, UI rendering orchestration, and audio processing calls
    for the multi-step Quick Create Wizard.
    """

    def __init__(self):
        """
        Initializes the QuickWizard. This involves:
        1. Ensuring the wizard's session state is initialized.
        2. Retrieving cached instances of the TTS generator and audio processor.
        3. Handling potential initialization errors.
        """
        self.tts_generator: Optional[PiperTTSGenerator] = None
        self.audio_processor: Optional[WizardAudioProcessor] = None
        initialization_error = None  # Flag to store any errors during init

        try:
            # Ensure session state variables are set up using the function from wizard_state
            initialize_wizard_state()  # This should set defaults for all keys

            # Retrieve (or create if first time) cached resources
            self.tts_generator = get_tts_generator()
            self.audio_processor = get_audio_processor()
            logger.info(
                "Retrieved TTS Generator and Audio Processor instances from cache/init."
            )

            # Check if retrieval failed and build an error message
            if self.tts_generator is None:
                initialization_error = "TTS Engine failed to initialize."
            if self.audio_processor is None:
                error_msg = "Audio Processor failed to initialize."
                initialization_error = (
                    f"{initialization_error} {error_msg}"
                    if initialization_error
                    else error_msg
                )

            # Log if any critical components failed
            if initialization_error:
                logger.critical(
                    f"QuickWizard Initialization Failed: {initialization_error}"
                )

        except Exception as e:
            # Catch any unexpected errors during the initialization process
            initialization_error = f"Unexpected initialization error: {e}"
            logger.exception(f"CRITICAL: {initialization_error}")
            self.tts_generator = None
            self.audio_processor = None

        # Store the initialization error status for later checks
        self._initialization_error = initialization_error

        # Ensure specific state keys used by the wizard exist (Safeguard)
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
        # Ensure the preview active key is initialized
        if WIZARD_PREVIEW_ACTIVE_KEY not in st.session_state:
            st.session_state[WIZARD_PREVIEW_ACTIVE_KEY] = False

        logger.debug("QuickWizard initialization attempt finished.")

    # --- State Synchronization Callbacks ---
    def sync_affirmation_text(self):
        """Callback to update affirmation text in state and clear old audio data."""
        st.session_state[AFFIRMATION_TEXT_KEY] = st.session_state.get(
            "wizard_affirm_text_area", ""
        )
        cleared = st.session_state.pop(LEGACY_AFFIRM_AUDIO_KEY, None) is not None
        cleared |= st.session_state.pop(LEGACY_AFFIRM_SR_KEY, None) is not None
        if cleared:
            logger.debug(
                "Cleared legacy affirmation audio/sr state due to text change."
            )
            gc.collect()
        logger.debug("Synced affirmation text state from text area.")

    def clear_affirmation_upload_state(self):
        """Placeholder or specific logic if affirmations could be uploaded."""
        self.sync_affirmation_text()
        logger.debug("Called clear_affirmation_upload_state (currently syncs text).")

    def sync_background_choice(self, choice_options: List[str]):
        """Callback for background choice radio button."""
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
        """Callback specifically for the file uploader's on_change."""
        cleared = st.session_state.pop(LEGACY_BG_AUDIO_KEY, None) is not None
        cleared |= st.session_state.pop(LEGACY_BG_SR_KEY, None) is not None
        cleared |= st.session_state.pop(BG_UPLOADED_FILE_KEY, None) is not None
        if cleared:
            logger.debug(
                "Cleared legacy BG audio/sr/file object via file uploader callback."
            )
            gc.collect()

    # --- Navigation and Reset ---
    def _reset_wizard_state(self):
        """Resets the entire wizard state."""
        logger.info("Resetting wizard state via wizard_state.reset_wizard_state().")
        reset_wizard_state()

    def _go_to_step(self, step: int):
        """Navigates the wizard to the specified step number."""
        if 1 <= step <= 4:
            current_step = st.session_state.get(WIZARD_STEP_KEY, DEFAULT_STEP)
            if step != current_step:
                st.session_state[WIZARD_STEP_KEY] = step
                logger.debug(f"Navigating wizard from step {current_step} to {step}")
                st.session_state.pop(PREVIEW_BUFFER_KEY, None)
                st.session_state.pop(PREVIEW_ERROR_KEY, None)
                st.session_state.pop(EXPORT_BUFFER_KEY, None)
                st.session_state.pop(EXPORT_ERROR_KEY, None)
                st.rerun()
            else:
                logger.debug(f"Already on step {step}, no navigation needed.")
        else:
            logger.warning(f"Invalid step navigation requested: {step}")

    # --- State Gathering Helper ---
    def _get_current_processing_state(self) -> Dict[str, Any]:
        """Gathers relevant wizard settings from session state."""
        speed_state_value = st.session_state.get(
            AFFIRM_APPLY_SPEED_KEY, DEFAULT_APPLY_SPEED
        )
        logger.debug(
            f"DEBUG GET_STATE - Reading {AFFIRM_APPLY_SPEED_KEY}: {speed_state_value} (Type: {type(speed_state_value)})"
        )
        state = {
            "wizard_apply_speed_change": speed_state_value,
            "wizard_speed_factor": QUICK_SUBLIMINAL_PRESET_SPEED,
            "wizard_background_choice": st.session_state.get(
                BG_CHOICE_KEY, DEFAULT_BG_CHOICE
            ),
            "wizard_background_uploaded_file": st.session_state.get(
                BG_UPLOADED_FILE_KEY, DEFAULT_BG_UPLOADED_FILE
            ),
            "wizard_background_noise_type": st.session_state.get(
                BG_NOISE_TYPE_KEY, DEFAULT_NOISE_TYPE
            ),
            "wizard_frequency_choice": st.session_state.get(
                FREQ_CHOICE_KEY, DEFAULT_FREQ_CHOICE
            ),
            "wizard_frequency_params": st.session_state.get(
                FREQ_PARAMS_KEY, DEFAULT_FREQ_PARAMS.copy()
            ),
            "wizard_affirmation_volume": st.session_state.get(
                AFFIRMATION_VOLUME_KEY, DEFAULT_AFFIRMATION_VOLUME
            ),
            "wizard_background_volume": st.session_state.get(
                BG_VOLUME_KEY, DEFAULT_BG_VOLUME
            ),
            "wizard_frequency_volume": st.session_state.get(
                FREQ_VOLUME_KEY, DEFAULT_FREQ_VOLUME
            ),
            "wizard_export_format": st.session_state.get(
                EXPORT_FORMAT_KEY, DEFAULT_EXPORT_FORMAT
            ),
        }
        logger.debug("Gathered current processing state from session state.")
        return state

    # --- Dynamic Generation Helpers ---
    def _generate_affirmation_audio(self) -> Optional[AudioTuple]:
        """Generates affirmation audio using the TTS engine."""
        affirmation_text = st.session_state.get(AFFIRMATION_TEXT_KEY, "").strip()
        if not affirmation_text:
            logger.warning("Cannot generate affirmation audio: Text is empty.")
            return None
        if not self.tts_generator:
            logger.error(
                "Cannot generate affirmation audio: TTS Engine not initialized."
            )
            raise RuntimeError("TTS Engine not initialized.")

        logger.info("Generating affirmation audio dynamically using TTS...")
        try:
            start_time = time.time()
            audio_data, sample_rate = self.tts_generator.generate(affirmation_text)
            duration = time.time() - start_time
            logger.info(
                f"Affirmation audio generated ({len(audio_data) / sample_rate:.2f}s) in {duration:.2f}s."
            )
            if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
                logger.error("TTS generation resulted in empty audio data.")
                return None
            if sample_rate != GLOBAL_SR:
                logger.error(
                    f"TTS generated audio with incorrect sample rate ({sample_rate} Hz vs expected {GLOBAL_SR} Hz)."
                )
                return None
            return (audio_data, sample_rate)
        except Exception as e:
            logger.exception("Failed to generate affirmation audio using TTS.")
            raise RuntimeError(f"Failed to generate affirmation audio: {e}") from e

    def _generate_or_load_background_audio(
        self,
        bg_choice: str,
        bg_noise_type: str,
        uploaded_file: Optional[Any],
        target_duration_hint: float,
    ) -> Optional[AudioTuple]:
        """Loads uploaded background audio or generates noise."""
        if bg_choice == "none":
            return None
        if not self.audio_processor:
            logger.error(
                "Cannot process background audio: Audio processor unavailable."
            )
            return None

        bg_tuple: Optional[AudioTuple] = None
        try:
            if bg_choice == "upload":
                if uploaded_file:
                    logger.info(
                        f"Loading uploaded background file: {uploaded_file.name}"
                    )
                    bg_tuple = self.audio_processor.load_uploaded_audio(uploaded_file)
                else:
                    logger.warning("BG choice 'upload' but no file object found.")
            elif bg_choice == "noise":
                logger.info(
                    f"Generating background noise: Type='{bg_noise_type}', Duration Hint={target_duration_hint:.2f}s"
                )
                bg_tuple = self.audio_processor.generate_noise_audio(
                    bg_noise_type, target_duration_hint, GLOBAL_SR
                )

            if bg_tuple and (
                not isinstance(bg_tuple[0], np.ndarray)
                or bg_tuple[0].size == 0
                or bg_tuple[1] != GLOBAL_SR
            ):
                logger.error(
                    f"Dynamic background processing returned invalid result for '{bg_choice}'."
                )
                bg_tuple = None
            elif bg_tuple:
                logger.info(f"Background audio ('{bg_choice}') processed successfully.")
        except Exception as e:
            logger.exception(
                f"Error processing dynamic background audio (Choice: {bg_choice})."
            )
            bg_tuple = None
        return bg_tuple

    def _generate_frequency_audio(
        self, freq_choice: str, freq_params: Dict[str, Any], target_duration_hint: float
    ) -> Optional[AudioTuple]:
        """Generates frequency-based audio."""
        if freq_choice == "None":
            return None
        if not self.audio_processor:
            logger.error(
                "Cannot generate frequency audio: Audio processor unavailable."
            )
            return None

        freq_tuple: Optional[AudioTuple] = None
        try:
            logger.info(
                f"Generating frequency audio: Type='{freq_choice}', Params={freq_params}, Duration Hint={target_duration_hint:.2f}s"
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
                    f"Dynamic frequency processing returned invalid result for '{freq_choice}'."
                )
                freq_tuple = None
            elif freq_tuple:
                logger.info(
                    f"Frequency audio ('{freq_choice}') generated successfully."
                )
        except Exception as e:
            logger.exception(
                f"Error processing dynamic frequency audio (Choice: {freq_choice})."
            )
            freq_tuple = None
        return freq_tuple

    # --- Wrappers for Audio Processing ---
    def generate_preview(self, duration_seconds: int = 10):
        """
        Generates a short preview mix. Called when WIZARD_PREVIEW_ACTIVE_KEY is True.
        Updates PREVIEW_BUFFER_KEY or PREVIEW_ERROR_KEY and resets the active key.
        """
        logger.info(
            f"QuickWizard: Starting actual preview generation ({duration_seconds}s)."
        )
        # Note: Clearing previous results and setting WIZARD_PREVIEW_ACTIVE_KEY=True
        # happened in the button click logic in step_4_export.py

        # Initialize tuples
        affirmation_tuple: Optional[AudioTuple] = None
        background_tuple: Optional[AudioTuple] = None
        frequency_tuple: Optional[AudioTuple] = None
        preview_buffer: Optional[BytesIO] = None

        try:
            # Check essential components again (belt and suspenders)
            if (
                not self.audio_processor
                or not self.tts_generator
                or not AUDIO_IO_AVAILABLE
            ):
                error_msg = "Required components not available for preview generation."
                st.session_state[PREVIEW_ERROR_KEY] = error_msg
                logger.error(f"Preview failed early: {error_msg}")
                raise RuntimeError(error_msg)  # Raise to trigger finally block

            # 1. Generate Affirmation Audio
            logger.info("Generating affirmations for preview...")
            affirmation_tuple = self._generate_affirmation_audio()
            if affirmation_tuple is None:
                st.session_state[PREVIEW_ERROR_KEY] = (
                    "Failed to generate affirmation audio for preview."
                )
                logger.error(st.session_state[PREVIEW_ERROR_KEY])
                raise RuntimeError("Affirmation generation failed during preview.")

            affirmation_duration = affirmation_tuple[0].shape[0] / GLOBAL_SR
            logger.info(
                f"Affirmation duration (for preview hint): {affirmation_duration:.2f}s"
            )

            # 2. Get Current Settings
            processing_state = self._get_current_processing_state()

            # 3. Generate/Load Background Audio
            logger.info("Processing background audio for preview...")
            background_tuple = self._generate_or_load_background_audio(
                processing_state["wizard_background_choice"],
                processing_state["wizard_background_noise_type"],
                processing_state["wizard_background_uploaded_file"],
                affirmation_duration,
            )

            # 4. Generate Frequency Audio
            logger.info("Processing frequency audio for preview...")
            frequency_tuple = self._generate_frequency_audio(
                processing_state["wizard_frequency_choice"],
                processing_state["wizard_frequency_params"],
                affirmation_duration,
            )

            # 5. Mix the Tracks using Audio Processor
            logger.info("Calling audio processor to generate preview mix...")
            preview_mix_result = self.audio_processor.generate_preview_mix(
                duration_seconds,
                affirmation_tuple,
                background_tuple,
                frequency_tuple,
                processing_state,
            )

            # 6. Process the Result
            if preview_mix_result is not None:
                preview_audio, preview_sr = preview_mix_result
                if preview_audio is not None and preview_sr is not None:
                    logger.info("Preview mix successful. Saving to buffer...")
                    preview_buffer = save_audio_to_bytesio(
                        preview_audio, preview_sr
                    )  # WAV default
                    if preview_buffer and preview_buffer.getbuffer().nbytes > 0:
                        st.session_state[PREVIEW_BUFFER_KEY] = preview_buffer
                        logger.info("Preview buffer stored successfully.")
                    else:
                        st.session_state[PREVIEW_ERROR_KEY] = (
                            "Preview generation failed (could not save to buffer)."
                        )
                        logger.error(st.session_state[PREVIEW_ERROR_KEY])
                        st.session_state.pop(PREVIEW_BUFFER_KEY, None)
                else:
                    st.session_state[PREVIEW_ERROR_KEY] = (
                        "Preview mixing failed (processor returned invalid audio/sr)."
                    )
                    logger.error(st.session_state[PREVIEW_ERROR_KEY])
            else:
                st.session_state[PREVIEW_ERROR_KEY] = (
                    "Preview mixing failed (processor returned None)."
                )
                logger.error(st.session_state[PREVIEW_ERROR_KEY])

        except Exception as e:
            # Catch any unexpected errors during the preview process
            logger.exception("Error occurred during generate_preview wrapper.")
            # Store error only if one wasn't already set specifically
            if not st.session_state.get(PREVIEW_ERROR_KEY):
                st.session_state[PREVIEW_ERROR_KEY] = f"Unexpected Preview Error: {e}"
            st.session_state.pop(
                PREVIEW_BUFFER_KEY, None
            )  # Ensure buffer is cleared on error

        finally:
            # --- Reset Preview Processing Flag ---
            st.session_state[WIZARD_PREVIEW_ACTIVE_KEY] = False
            logger.debug(f"Reset {WIZARD_PREVIEW_ACTIVE_KEY} to False (Finally Block).")

            # Cleanup temporary audio data
            del affirmation_tuple, background_tuple, frequency_tuple
            if (
                "preview_buffer" in locals()
                and preview_buffer is not None
                and id(preview_buffer) != id(st.session_state.get(PREVIEW_BUFFER_KEY))
            ):
                del preview_buffer
            gc.collect()
            logger.debug(
                "Cleaned up temporary audio tuples after preview generation attempt."
            )

    def process_and_export_audio(self):
        """
        Generates, mixes, and prepares the final audio. Called when WIZARD_PROCESSING_ACTIVE_KEY is True.
        Updates EXPORT_BUFFER_KEY or EXPORT_ERROR_KEY and resets the active key.
        """
        logger.info("QuickWizard: Starting actual audio processing and export.")
        # Note: Clearing buffers and setting WIZARD_PROCESSING_ACTIVE_KEY=True
        # happened in the button click logic in step_4_export.py

        # Initialize tuples
        affirmation_tuple: Optional[AudioTuple] = None
        background_tuple: Optional[AudioTuple] = None
        frequency_tuple: Optional[AudioTuple] = None

        try:
            # Check essential components
            if not self.audio_processor or not self.tts_generator:
                error_msg = "Required components not available for export."
                st.session_state[EXPORT_ERROR_KEY] = error_msg
                logger.error(f"Export failed early: {error_msg}")
                raise RuntimeError(error_msg)  # Trigger finally block

            # 1. Generate Affirmation Audio
            logger.info("Generating affirmations for export...")
            affirmation_tuple = self._generate_affirmation_audio()
            if affirmation_tuple is None:
                st.session_state[EXPORT_ERROR_KEY] = (
                    "Failed to generate affirmation audio for export."
                )
                logger.error(st.session_state[EXPORT_ERROR_KEY])
                raise RuntimeError("Affirmation generation failed during export.")

            affirmation_duration = affirmation_tuple[0].shape[0] / GLOBAL_SR
            logger.info(f"Affirmation duration: {affirmation_duration:.2f}s")

            # 2. Get Current Settings
            processing_state = self._get_current_processing_state()

            # 3. Generate/Load Background Audio
            logger.info("Processing background audio for export...")
            background_tuple = self._generate_or_load_background_audio(
                processing_state["wizard_background_choice"],
                processing_state["wizard_background_noise_type"],
                processing_state["wizard_background_uploaded_file"],
                affirmation_duration,
            )

            # 4. Generate Frequency Audio
            logger.info("Processing frequency audio for export...")
            frequency_tuple = self._generate_frequency_audio(
                processing_state["wizard_frequency_choice"],
                processing_state["wizard_frequency_params"],
                affirmation_duration,
            )

            # 5. Process and Export using Audio Processor
            logger.info("Calling audio processor method: process_and_export...")
            export_buffer, error_message = self.audio_processor.process_and_export(
                affirmation_tuple, background_tuple, frequency_tuple, processing_state
            )

            # 6. Update Session State with Results
            if export_buffer and not error_message:
                st.session_state[EXPORT_BUFFER_KEY] = export_buffer
                logger.info("Export successful. Buffer stored.")
                st.session_state.pop(PREVIEW_BUFFER_KEY, None)  # Clear outdated preview
                st.session_state.pop(PREVIEW_ERROR_KEY, None)
            else:
                st.session_state[EXPORT_ERROR_KEY] = (
                    error_message or "Export failed (unknown reason)."
                )
                logger.error(f"Export failed: {st.session_state[EXPORT_ERROR_KEY]}")
                st.session_state.pop(EXPORT_BUFFER_KEY, None)

        except Exception as e:
            # Catch unexpected errors
            logger.exception(
                "Unhandled error occurred during process_and_export_audio wrapper."
            )
            if not st.session_state.get(
                EXPORT_ERROR_KEY
            ):  # Avoid overwriting specific errors
                st.session_state[EXPORT_ERROR_KEY] = f"Unexpected Export Error: {e}"
            st.session_state.pop(EXPORT_BUFFER_KEY, None)

        finally:
            # --- Reset Export Processing Flag ---
            st.session_state[WIZARD_PROCESSING_ACTIVE_KEY] = False
            logger.debug(
                f"Reset {WIZARD_PROCESSING_ACTIVE_KEY} to False (Finally Block)."
            )

            # Cleanup temporary audio data
            del affirmation_tuple, background_tuple, frequency_tuple
            gc.collect()
            logger.debug("Cleaned up temporary audio tuples after export attempt.")

    # --- Main Rendering Method ---
    def render_wizard(self):
        """
        Renders the main wizard UI, including triggering processing if flags are set.
        """
        st.title("✨ MindMorph Quick Create Wizard")

        # Check for initialization errors
        if self._initialization_error:
            logger.error(
                f"Wizard rendering blocked by initialization error: {self._initialization_error}"
            )
            st.error(f"FATAL ERROR: {self._initialization_error}. Wizard cannot start.")
            if st.button("Attempt Reset"):
                self._reset_wizard_state()
                st.rerun()
            return

        # --- Check for Processing Triggers ---
        # These checks happen *before* rendering the step UI for the current run.
        # If a flag is True, the corresponding function is called, which will
        # perform the processing and reset the flag in its finally block.
        preview_triggered = st.session_state.get(WIZARD_PREVIEW_ACTIVE_KEY, False)
        export_triggered = st.session_state.get(WIZARD_PROCESSING_ACTIVE_KEY, False)

        if preview_triggered:
            logger.info(
                f"Detected {WIZARD_PREVIEW_ACTIVE_KEY}=True. Triggering preview generation."
            )
            with st.spinner("Generating preview... Please wait."):
                self.generate_preview()  # This will reset the flag internally
        # No rerun here, allow rendering to continue to show results/updated state

        # Check export trigger *only if* preview wasn't triggered in the same run
        # (Avoids running both if flags somehow get set simultaneously)
        elif export_triggered:
            logger.info(
                f"Detected {WIZARD_PROCESSING_ACTIVE_KEY}=True. Triggering export process."
            )
            with st.spinner("Generating final audio... Please wait."):
                self.process_and_export_audio()  # This will reset the flag internally
            # No rerun here, allow rendering to continue

        # --- Render Current Step UI ---
        step = st.session_state.get(WIZARD_STEP_KEY, DEFAULT_STEP)

        # Display progress bar
        steps_display = ["Affirmations", "Background", "Frequency", "Mix & Export"]
        progress_step = max(1, min(step, len(steps_display)))
        try:
            st.progress(
                (progress_step) / len(steps_display),
                text=f"Step {progress_step}: {steps_display[progress_step - 1]}",
            )
        except IndexError:
            st.progress(0.0)
            logger.error(f"Progress bar encountered invalid step index: step={step}")

        # Render the UI for the current step
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
                st.error(f"Internal Error: Invalid wizard step ({step}). Resetting.")
                logger.error(f"Invalid step number found: {step}. Resetting.")
                st.session_state[WIZARD_STEP_KEY] = DEFAULT_STEP
                st.rerun()
            else:
                st.error(f"Error: UI component for Step {step} could not be loaded.")
                logger.error(f"Cannot render Step {step}, module missing.")
        except Exception as e_render:
            logger.exception(f"Error rendering UI for step {step}: {e_render}")
            st.error(f"An error occurred displaying Step {step}.")
            if st.button("Reset Wizard"):
                self._reset_wizard_state()
                st.rerun()
