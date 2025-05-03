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
from audio_utils.audio_io import save_audio_to_bytesio  # Assuming this function exists
from audio_utils.audio_mixers import mix_wizard_tracks  # Assuming this function exists

# Import wizard state management
from config import (
    GLOBAL_SR,
    QUICK_SUBLIMINAL_PRESET_SPEED,
    # QUICK_SUBLIMINAL_PRESET_VOLUME, # No longer needed, volume comes from slider
)

# Import the new Piper TTS Generator
from tts.piper_tts import PiperTTSGenerator  # Assuming this exists

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
    # Assuming this path is correct relative to where quick_wizard.py is
    from audio_utils.audio_effects_pipeline import AudioData
except ImportError:
    # Fallback if the import path is different or module doesn't exist
    AudioData = np.ndarray
    logging.warning("Could not import AudioData type hint from audio_utils.audio_effects_pipeline. Using np.ndarray fallback.")


# Optional MP3 export dependency check
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub library not found. MP3 export will be disabled.")


logger = logging.getLogger(__name__)


# Define the AudioTuple type hint more explicitly
AudioTuple = Optional[Tuple[AudioData, int]]


class QuickWizard:
    """Manages the state and UI rendering orchestration for the Quick Create Wizard."""

    def __init__(self):
        """Initializes the QuickWizard."""
        try:
            # Ensure the TTS class is correctly referenced
            self.tts_generator = PiperTTSGenerator()
            logger.info("PiperTTSGenerator initialized successfully for QuickWizard.")
        except NameError:
            logger.exception("CRITICAL: PiperTTSGenerator class not found. Check imports.")
            # Optionally raise or handle this critical failure
            self.tts_generator = None  # Ensure it's None if init fails
            st.error("FATAL: TTS Engine class not found. Wizard cannot function.")
            # Consider raising RuntimeError to halt execution if TTS is essential
            # raise RuntimeError("Failed to find TTS engine class: PiperTTSGenerator")
        except Exception as e:
            logger.exception("CRITICAL: Failed to initialize PiperTTSGenerator in QuickWizard.")
            # Optionally raise or handle this critical failure
            self.tts_generator = None  # Ensure it's None if init fails
            st.error(f"FATAL: Failed to initialize TTS engine: {e}. Wizard cannot function.")
            # Consider raising RuntimeError
            # raise RuntimeError(f"Failed to initialize TTS engine: {e}") from e

        initialize_wizard_state()
        logger.debug("QuickWizard initialized and state ensured.")

    # --- State Synchronization Callbacks ---
    def sync_affirmation_text(self):
        st.session_state.wizard_affirmation_text = st.session_state.get("wizard_affirm_text_area", "")
        if st.session_state.get("wizard_affirmation_source") == "text":
            st.session_state.wizard_affirmation_audio = None
            st.session_state.wizard_affirmation_sr = None
            st.session_state.wizard_affirmation_volume = 1.0
        logger.debug("Synced affirmation text state.")

    def clear_affirmation_upload_state(self):
        # Placeholder if needed later
        pass

    def sync_background_choice(self, choice_options: List[str]):
        selected_label = st.session_state.get("wizard_bg_choice_radio")
        if not selected_label:
            return
        new_choice = "none"
        if selected_label == "Upload Music/Sound":
            new_choice = "upload"
        elif selected_label == "Generate Noise":
            new_choice = "noise"

        # Only clear audio if the actual choice type (none, upload, noise) changes
        current_choice = st.session_state.get("wizard_background_choice")
        if new_choice != current_choice:
            st.session_state.wizard_background_audio = None
            st.session_state.wizard_background_sr = None
            logger.debug(f"Background choice changed from '{current_choice}' to '{new_choice}', cleared related audio state.")
            # Reset volume only if changing *away* from a choice that uses volume
            # Or maybe always reset? Let's reset to default when choice changes.
            # from wizard_state import DEFAULT_BG_VOLUME # Need to import default
            # st.session_state.wizard_background_volume = DEFAULT_BG_VOLUME

        st.session_state.wizard_background_choice = new_choice
        st.session_state.wizard_background_choice_label = selected_label  # Store the display label too

    def clear_background_upload_state(self):
        # Placeholder if needed later
        pass

    # --- Navigation and Reset ---
    def _reset_wizard_state(self):
        reset_wizard_state()
        # No st.rerun() needed here, reset_wizard_state calls initialize which sets step to 1
        # The calling function (e.g., button click handler) might trigger rerun if needed.
        # However, typically resetting should force a rerun to go back to step 1 or home.
        st.rerun()

    def _go_to_step(self, step: int):
        if 1 <= step <= 4:
            st.session_state.wizard_step = step
            logger.debug(f"Navigating to wizard step {step}")
            st.rerun()  # Rerun is necessary to render the new step's UI
        else:
            logger.warning(f"Invalid step navigation requested: {step}")

    # --- Helper Function for Looping Audio ---
    def _loop_audio_to_length(self, audio_data: AudioData, target_length: int) -> AudioData:
        """Loops or truncates audio data to match the target length."""
        if not isinstance(audio_data, np.ndarray):
            logger.error(f"Invalid audio data type for looping: {type(audio_data)}")
            raise TypeError("audio_data must be a NumPy array for looping.")
        if target_length <= 0:
            logger.warning("Target length for looping is zero or negative. Returning empty array.")
            return np.array([], dtype=audio_data.dtype)

        current_length = audio_data.shape[0]
        if current_length == 0:
            logger.warning("Input audio for looping is empty. Returning empty array.")
            return np.array([], dtype=audio_data.dtype)

        if current_length == target_length:
            return audio_data
        elif current_length < target_length:
            num_repeats = math.ceil(target_length / current_length)
            if audio_data.ndim == 1:
                looped_audio = np.tile(audio_data, num_repeats)
            elif audio_data.ndim == 2:  # Assume stereo or multi-channel
                # Tile along the time axis (axis 0)
                looped_audio = np.tile(audio_data, (num_repeats, 1))
            else:
                logger.error(f"Unsupported audio dimensions for looping: {audio_data.ndim}")
                raise ValueError("Audio data must be 1D or 2D for looping.")
            # Truncate precisely to the target length
            return looped_audio[:target_length]
        else:  # current_length > target_length
            # Truncate to the target length
            return audio_data[:target_length]

    # --- Core Processing Logic ---
    def _process_and_export(self):
        """Handles the final processing and export logic for the wizard."""
        logger.info("Starting wizard export process.")
        st.session_state.wizard_export_buffer = None
        st.session_state.wizard_export_error = None
        processing_success = False  # Flag to track success

        try:
            # --- Get Audio Data and Sample Rates ---
            affirmation_audio_data: Optional[AudioData] = st.session_state.get("wizard_affirmation_audio")
            affirmation_sr: Optional[int] = st.session_state.get("wizard_affirmation_sr")
            background_audio_data: Optional[AudioData] = st.session_state.get("wizard_background_audio")
            background_sr: Optional[int] = st.session_state.get("wizard_background_sr")
            frequency_audio_data: Optional[AudioData] = st.session_state.get("wizard_frequency_audio")
            frequency_sr: Optional[int] = st.session_state.get("wizard_frequency_sr")

            # --- Get Volume Settings ---
            affirmation_volume_slider: float = st.session_state.get("wizard_affirmation_volume", 1.0)
            background_volume: float = st.session_state.get("wizard_background_volume", 0.7)
            frequency_volume: float = st.session_state.get("wizard_frequency_volume", 0.2)

            # --- Validate Affirmation Audio ---
            if affirmation_audio_data is None or affirmation_sr is None:
                st.session_state.wizard_export_error = "Affirmation audio is missing. Please go back to Step 1."
                logger.error("Wizard export failed: Missing affirmation audio.")
                return  # Exit early

            # --- Determine Affirmation Speed (Volume comes from slider) ---
            apply_quick_settings: bool = st.session_state.get("wizard_apply_quick_settings", True)
            eff_affirm_speed: float = QUICK_SUBLIMINAL_PRESET_SPEED if apply_quick_settings else 1.0
            eff_affirm_volume: float = affirmation_volume_slider  # Use volume from slider

            log_speed_msg = f"Applying Quick Wizard preset SPEED ({eff_affirm_speed}x)." if apply_quick_settings else f"Using original speed ({eff_affirm_speed}x)."
            logger.info(f"{log_speed_msg} Volume set by slider to {eff_affirm_volume:.2f}.")

            export_format: str = st.session_state.get("wizard_export_format", "WAV").lower()

            # --- Prepare Tracks for Mixing ---
            target_length_samples = 0
            try:
                # Ensure affirmation audio is valid numpy array
                if not isinstance(affirmation_audio_data, np.ndarray) or affirmation_audio_data.size == 0:
                    raise ValueError("Affirmation audio data is invalid or empty.")

                # Ensure affirmation audio uses GLOBAL_SR (should be handled by load/TTS)
                if affirmation_sr != GLOBAL_SR:
                    logger.warning(f"Affirmation SR ({affirmation_sr}) != GLOBAL_SR ({GLOBAL_SR}). This should be handled earlier.")
                    # Ideally, resample here if necessary, but it's better handled upstream.

                target_length_samples = affirmation_audio_data.shape[0]
                if target_length_samples == 0:
                    raise ValueError("Affirmation audio has zero length.")

                logger.info(f"Target mix length: {target_length_samples / GLOBAL_SR:.2f}s based on affirmation length.")

                # Ensure affirmation audio is stereo for consistency if mixer expects it
                if affirmation_audio_data.ndim == 1:
                    affirmation_audio_data = np.stack([affirmation_audio_data] * 2, axis=-1)
                    logger.debug("Converted mono affirmation to stereo for mixing.")
                elif affirmation_audio_data.ndim != 2:
                    raise ValueError(f"Unsupported affirmation audio dimensions: {affirmation_audio_data.ndim}")

                affirmation_tuple: AudioTuple = (affirmation_audio_data, GLOBAL_SR)

                # Prepare Background Audio
                background_tuple: AudioTuple = None
                if background_audio_data is not None and background_sr is not None:
                    if not isinstance(background_audio_data, np.ndarray) or background_audio_data.size == 0:
                        logger.warning("Background audio data is invalid or empty. Skipping.")
                    else:
                        if background_sr != GLOBAL_SR:
                            logger.warning(f"Background SR mismatch ({background_sr} vs {GLOBAL_SR}). Should be resampled earlier.")
                        if background_audio_data.ndim == 1:
                            background_audio_data = np.stack([background_audio_data] * 2, axis=-1)
                            logger.debug("Converted mono background to stereo.")
                        elif background_audio_data.ndim != 2:
                            logger.warning(f"Unsupported background audio dimensions: {background_audio_data.ndim}. Skipping.")
                            background_audio_data = None  # Skip if not 1D or 2D

                        if background_audio_data is not None:
                            looped_background = self._loop_audio_to_length(background_audio_data, target_length_samples)
                            background_tuple = (looped_background, GLOBAL_SR)
                            logger.info(f"Background audio prepared (looped/trimmed to {target_length_samples / GLOBAL_SR:.2f}s).")

                # Prepare Frequency Audio
                frequency_tuple: AudioTuple = None
                if frequency_audio_data is not None and frequency_sr is not None:
                    if not isinstance(frequency_audio_data, np.ndarray) or frequency_audio_data.size == 0:
                        logger.warning("Frequency audio data is invalid or empty. Skipping.")
                    else:
                        if frequency_sr != GLOBAL_SR:
                            logger.warning(f"Frequency SR mismatch ({frequency_sr} vs {GLOBAL_SR}). Should be resampled earlier.")
                        # Generators should produce stereo, but double-check
                        if frequency_audio_data.ndim == 1:
                            frequency_audio_data = np.stack([frequency_audio_data] * 2, axis=-1)
                            logger.warning("Frequency audio was mono, converted to stereo.")
                        elif frequency_audio_data.ndim != 2:
                            logger.warning(f"Unsupported frequency audio dimensions: {frequency_audio_data.ndim}. Skipping.")
                            frequency_audio_data = None  # Skip if not 1D or 2D

                        if frequency_audio_data is not None:
                            looped_frequency = self._loop_audio_to_length(frequency_audio_data, target_length_samples)
                            frequency_tuple = (looped_frequency, GLOBAL_SR)
                            logger.info(f"Frequency audio prepared (looped/trimmed to {target_length_samples / GLOBAL_SR:.2f}s).")

            except Exception as e_prep:
                logger.exception("Error preparing tracks for mixing.")
                st.session_state.wizard_export_error = f"Track Preparation Failed: {e_prep}"
                return  # Exit early

            # --- Mix Tracks ---
            logger.info("Wizard mixing tracks...")
            full_mix: Optional[AudioData] = None
            try:
                # Ensure the mixing function is available
                if "mix_wizard_tracks" not in globals() or not callable(mix_wizard_tracks):
                    raise NameError("Mixing function 'mix_wizard_tracks' is not available. Check imports.")

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
                if full_mix is None or not isinstance(full_mix, np.ndarray) or full_mix.size == 0:
                    raise ValueError("Mixing resulted in empty or invalid audio data.")

                mix_duration_s = len(full_mix) / GLOBAL_SR if GLOBAL_SR > 0 else 0
                logger.info(f"Mixing successful. Final length: {mix_duration_s:.2f}s.")

            except Exception as e_mix:
                logger.exception("Error during wizard track mixing.")
                st.session_state.wizard_export_error = f"Mixing Failed: {e_mix}"
                return  # Exit early

            # --- Export to selected format ---
            logger.info(f"Exporting final mix as {export_format.upper()}...")
            export_buffer = None
            try:
                # Ensure save function is available
                if "save_audio_to_bytesio" not in globals() or not callable(save_audio_to_bytesio):
                    raise NameError("Saving function 'save_audio_to_bytesio' is not available. Check imports.")

                if export_format == "wav":
                    # --- FIX: Remove the 'format' argument ---
                    export_buffer = save_audio_to_bytesio(full_mix, GLOBAL_SR)
                    # --- END FIX ---
                    if export_buffer and export_buffer.getbuffer().nbytes > 0:
                        buffer_size = export_buffer.getbuffer().nbytes
                        expected_min_size = full_mix.shape[0] * full_mix.shape[1] * 2 * 0.9  # Stereo, 16-bit PCM, 90% tolerance
                        logger.info(f"Generated WAV buffer size: {buffer_size} bytes.")
                        if buffer_size < expected_min_size and full_mix.shape[0] > 0:
                            logger.error(f"WAV buffer size ({buffer_size}) seems too small! Expected ~{expected_min_size * 1.1:.0f}")
                            st.session_state.wizard_export_error = "Generated WAV buffer was unexpectedly small."
                            export_buffer = None  # Invalidate buffer
                        else:
                            logger.info("Wizard WAV mix buffer generated successfully.")
                            processing_success = True  # Mark success
                    else:
                        logger.error("save_audio_to_bytesio returned empty buffer for WAV.")
                        st.session_state.wizard_export_error = "Failed to save WAV buffer (empty)."
                        export_buffer = None

                elif export_format == "mp3":
                    if not PYDUB_AVAILABLE:
                        st.session_state.wizard_export_error = "MP3 export requires 'pydub' and 'ffmpeg'."
                        logger.error("MP3 export requested but pydub is not available.")
                    else:
                        # Pydub Export Logic
                        logger.info("Converting full mix to MP3 using pydub...")
                        full_mix_clipped = np.clip(full_mix, -1.0, 1.0)
                        audio_int16 = (full_mix_clipped * 32767).astype(np.int16)

                        channels = audio_int16.shape[1] if audio_int16.ndim == 2 else 1
                        if channels != 2:
                            logger.warning(f"Mix has {channels} channels before MP3 export. Pydub might handle this, but stereo is expected.")
                            # Pydub might require stereo, ensure it is:
                            if audio_int16.ndim == 1:
                                audio_int16 = np.stack([audio_int16] * 2, axis=-1)
                            elif audio_int16.shape[1] == 1:
                                audio_int16 = np.concatenate([audio_int16, audio_int16], axis=1)
                            channels = 2  # Now it's stereo

                        segment = AudioSegment(
                            data=audio_int16.tobytes(),
                            sample_width=audio_int16.dtype.itemsize,  # Should be 2
                            frame_rate=GLOBAL_SR,
                            channels=channels,  # Should be 2
                        )
                        mp3_buffer = BytesIO()
                        segment.export(mp3_buffer, format="mp3", bitrate="192k")
                        mp3_buffer.seek(0)

                        if mp3_buffer.getbuffer().nbytes > 0:
                            buffer_size = mp3_buffer.getbuffer().nbytes
                            logger.info(f"Generated MP3 export buffer size: {buffer_size} bytes.")
                            if buffer_size < 1000 and full_mix.shape[0] > GLOBAL_SR:  # Basic check if > 1 sec
                                logger.error(f"MP3 buffer size ({buffer_size}) seems too small!")
                                st.session_state.wizard_export_error = "Generated MP3 buffer was unexpectedly small."
                                export_buffer = None
                            else:
                                logger.info("Wizard MP3 mix buffer generated successfully.")
                                export_buffer = mp3_buffer
                                processing_success = True  # Mark success
                        else:
                            logger.error("MP3 export resulted in empty buffer.")
                            st.session_state.wizard_export_error = "MP3 export failed (empty buffer)."
                            export_buffer = None
                else:
                    st.session_state.wizard_export_error = f"Unsupported export format requested: '{export_format}'."
                    logger.error(f"Unsupported export format: {export_format}")

            except NameError as e_func:
                logger.exception(f"A required function ({e_func}) was not found. Check imports.")
                st.session_state.wizard_export_error = f"Processing function error: {e_func}. Check application setup."
            except Exception as e_export:
                logger.exception(f"Error during wizard audio export ({export_format}).")
                st.session_state.wizard_export_error = f"Export Failed ({export_format.upper()}): {e_export}"
                export_buffer = None  # Ensure buffer is None on error

            # Final check: if processing was marked successful, assign buffer to state
            if processing_success and export_buffer:
                st.session_state.wizard_export_buffer = export_buffer
            elif not st.session_state.get("wizard_export_error"):
                # If no buffer and no specific error, set a generic error
                st.session_state.wizard_export_error = "Export process completed but no valid output buffer was generated."
                logger.error("Export finished without errors but buffer is missing.")

        except Exception as e_main:
            # Catch any unexpected errors in the main try block
            logger.exception("Unhandled error during wizard export process.")
            st.session_state.wizard_export_error = f"An unexpected error occurred: {e_main}"

        finally:
            # Reset the processing flag regardless of success or failure
            # This allows the user to try again if there was an error
            st.session_state.wizard_processing_active = False
            logger.info("Reset wizard_processing_active flag to False.")
            # No st.rerun() here; step_4_export handles rerunning after this returns

    # --- Main Rendering Method ---
    def render_wizard(self):
        """Renders the main wizard UI and steps."""
        st.title("✨ MindMorph Quick Create Wizard")

        # Ensure state is initialized ONCE per session ideally, but check here for robustness
        initialize_wizard_state()

        # Check if TTS initialization failed critically
        if not hasattr(self, "tts_generator") or self.tts_generator is None:
            st.error("TTS Engine failed to initialize. Cannot proceed. Please check logs/configuration.")
            # Optionally add a button to retry or go home
            if st.button("Go Home"):
                self._reset_wizard_state()
            return  # Stop rendering the wizard steps

        step = st.session_state.get("wizard_step", 1)
        steps_display = ["Affirmations", "Background", "Frequency", "Export"]
        # Ensure step is within valid range for display
        progress_step = max(1, min(step, len(steps_display)))

        # Display progress bar
        try:
            st.progress(
                (progress_step) / len(steps_display),
                text=f"Step {progress_step}: {steps_display[progress_step - 1]}",
            )
        except IndexError:
            # This should not happen with the max/min check above, but log if it does
            st.progress(0.0)
            logger.error(f"Progress bar index out of range: step={step}, progress_step={progress_step}")

        # Render the current step based on session state
        try:
            if step == 1:
                step_1_affirmations.render_step_1(self)
            elif step == 2:
                step_2_background.render_step_2(self)
            elif step == 3:
                step_3_frequency.render_step_3(self)
            elif step == 4:
                step_4_export.render_step_4(self)
            else:
                # Handle invalid step value gracefully
                st.error(f"Invalid wizard step encountered: {step}. Resetting to Step 1.")
                logger.error(f"Invalid step detected: {step}. Resetting wizard.")
                st.session_state.wizard_step = 1  # Reset step to 1
                st.rerun()  # Rerun to show step 1
                # Alternatively, call reset: self._reset_wizard_state()
        except Exception as e_render:
            logger.exception(f"Error rendering wizard step {step}: {e_render}")
            st.error(f"An error occurred while rendering Step {step}. Please try again or reset the wizard.")
            # Add a reset button for the user
            if st.button("Reset Wizard"):
                self._reset_wizard_state()
