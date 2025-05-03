# quick_wizard.py
# ==========================================
# Quick Create Wizard Orchestrator for MindMorph (Added debug logging for mixing)
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
)

# Import the Piper TTS Generator
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
            self.tts_generator = PiperTTSGenerator()
            logger.info("PiperTTSGenerator initialized successfully for QuickWizard.")
        except NameError:
            logger.exception("CRITICAL: PiperTTSGenerator class not found. Check imports.")
            self.tts_generator = None
            st.error("FATAL: TTS Engine class not found. Wizard cannot function.")
        except Exception as e:
            logger.exception("CRITICAL: Failed to initialize PiperTTSGenerator in QuickWizard.")
            self.tts_generator = None
            st.error(f"FATAL: Failed to initialize TTS engine: {e}. Wizard cannot function.")

        initialize_wizard_state()
        # Ensure preview state variables exist
        if "wizard_preview_buffer" not in st.session_state:
            st.session_state.wizard_preview_buffer = None
        if "wizard_preview_error" not in st.session_state:
            st.session_state.wizard_preview_error = None
        logger.debug("QuickWizard initialized and state ensured.")

    # --- State Synchronization Callbacks ---
    def sync_affirmation_text(self):
        st.session_state.wizard_affirmation_text = st.session_state.get("wizard_affirm_text_area", "")
        if st.session_state.get("wizard_affirmation_source") == "text":
            st.session_state.wizard_affirmation_audio = None
            st.session_state.wizard_affirmation_sr = None
        logger.debug("Synced affirmation text state.")

    def clear_affirmation_upload_state(self):
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

        current_choice = st.session_state.get("wizard_background_choice")
        if new_choice != current_choice:
            st.session_state.wizard_background_audio = None
            st.session_state.wizard_background_sr = None
            logger.debug(f"Background choice changed from '{current_choice}' to '{new_choice}', cleared related audio state.")

        st.session_state.wizard_background_choice = new_choice
        st.session_state.wizard_background_choice_label = selected_label

    def clear_background_upload_state(self):
        pass

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
            elif audio_data.ndim == 2:
                looped_audio = np.tile(audio_data, (num_repeats, 1))
            else:
                logger.error(f"Unsupported audio dimensions for looping: {audio_data.ndim}")
                raise ValueError("Audio data must be 1D or 2D for looping.")
            return looped_audio[:target_length]
        else:  # current_length > target_length
            return audio_data[:target_length]

    # --- Preview Mix Generation ---
    def _generate_preview_mix(self, duration_seconds: int) -> AudioTuple:
        """Generates a short preview mix based on current Step 4 settings."""
        logger.info(f"Starting preview mix generation ({duration_seconds}s).")
        try:
            # Get Audio Data and SR
            affirmation_audio: Optional[AudioData] = st.session_state.get("wizard_affirmation_audio")
            affirmation_sr: Optional[int] = st.session_state.get("wizard_affirmation_sr")
            background_audio: Optional[AudioData] = st.session_state.get("wizard_background_audio")
            background_sr: Optional[int] = st.session_state.get("wizard_background_sr")
            frequency_audio: Optional[AudioData] = st.session_state.get("wizard_frequency_audio")
            frequency_sr: Optional[int] = st.session_state.get("wizard_frequency_sr")

            # Get Volume & Speed Settings from Step 4 State
            affirmation_volume: float = st.session_state.get("wizard_affirmation_volume", 1.0)
            background_volume: float = st.session_state.get("wizard_background_volume", 0.7)
            frequency_volume: float = st.session_state.get("wizard_frequency_volume", 0.2)
            apply_quick_speed: bool = st.session_state.get("wizard_apply_quick_settings", True)
            affirmation_speed: float = QUICK_SUBLIMINAL_PRESET_SPEED if apply_quick_speed else 1.0

            # Validate Affirmation Audio
            if affirmation_audio is None or affirmation_sr is None or affirmation_sr != GLOBAL_SR:
                logger.error("Preview failed: Missing or invalid affirmation audio/SR.")
                raise ValueError("Affirmation audio is missing or has incorrect sample rate.")
            if not isinstance(affirmation_audio, np.ndarray) or affirmation_audio.size == 0:
                raise ValueError("Affirmation audio data is invalid or empty.")

            # Determine Preview Length
            target_length_samples = int(duration_seconds * GLOBAL_SR)
            affirmation_length_samples = affirmation_audio.shape[0]
            preview_length_samples = min(target_length_samples, affirmation_length_samples)

            if preview_length_samples <= 0:
                raise ValueError("Calculated preview length is zero or negative.")

            logger.info(f"Preview target length: {preview_length_samples / GLOBAL_SR:.2f}s.")

            # Prepare Tracks for Preview Mixing
            if affirmation_audio.ndim == 1:
                affirmation_audio = np.stack([affirmation_audio] * 2, axis=-1)
            affirmation_preview = affirmation_audio[:preview_length_samples]
            affirmation_tuple: AudioTuple = (affirmation_preview, GLOBAL_SR)

            background_tuple: AudioTuple = None
            if background_audio is not None and background_sr == GLOBAL_SR and isinstance(background_audio, np.ndarray) and background_audio.size > 0:
                if background_audio.ndim == 1:
                    background_audio = np.stack([background_audio] * 2, axis=-1)
                looped_background = self._loop_audio_to_length(background_audio, preview_length_samples)
                background_tuple = (looped_background, GLOBAL_SR)
                logger.debug("Background preview prepared.")
            else:
                logger.debug(
                    f"Skipping background for preview. Audio exists: {background_audio is not None}, SR match: {background_sr == GLOBAL_SR}, Type valid: {isinstance(background_audio, np.ndarray)}, Size > 0: {background_audio.size > 0 if isinstance(background_audio, np.ndarray) else 'N/A'}"
                )

            frequency_tuple: AudioTuple = None
            if frequency_audio is not None and frequency_sr == GLOBAL_SR and isinstance(frequency_audio, np.ndarray) and frequency_audio.size > 0:
                if frequency_audio.ndim == 1:
                    frequency_audio = np.stack([frequency_audio] * 2, axis=-1)
                looped_frequency = self._loop_audio_to_length(frequency_audio, preview_length_samples)
                frequency_tuple = (looped_frequency, GLOBAL_SR)
                logger.debug("Frequency preview prepared.")
            else:
                logger.debug(
                    f"Skipping frequency for preview. Audio exists: {frequency_audio is not None}, SR match: {frequency_sr == GLOBAL_SR}, Type valid: {isinstance(frequency_audio, np.ndarray)}, Size > 0: {frequency_audio.size > 0 if isinstance(frequency_audio, np.ndarray) else 'N/A'}"
                )

            # --- Mix Preview Tracks ---
            logger.info("Mixing preview tracks...")
            # --- ADDED: Debug logging before mix call ---
            logger.info(f"Preview Mixing - Affirm Vol: {affirmation_volume:.2f}, Speed: {affirmation_speed:.2f}")
            logger.info(f"Preview Mixing - BG Audio Present: {background_tuple is not None}, BG Vol: {background_volume:.2f}")
            logger.info(f"Preview Mixing - Freq Audio Present: {frequency_tuple is not None}, Freq Vol: {frequency_volume:.2f}")
            # --- END ADDED ---
            preview_mix = mix_wizard_tracks(
                affirmation_audio=affirmation_tuple,
                background_audio=background_tuple,
                frequency_audio=frequency_tuple,
                affirmation_speed=affirmation_speed,
                affirmation_volume=affirmation_volume,
                background_volume=background_volume,
                frequency_volume=frequency_volume,
                target_sr=GLOBAL_SR,
            )

            if preview_mix is None or not isinstance(preview_mix, np.ndarray) or preview_mix.size == 0:
                raise ValueError("Preview mixing resulted in empty or invalid audio data.")

            logger.info(f"Preview mixing successful. Length: {len(preview_mix) / GLOBAL_SR:.2f}s.")
            return preview_mix, GLOBAL_SR

        except Exception as e_preview:
            logger.exception("Error generating wizard preview mix.")
            raise e_preview  # Propagate exception

    # --- Core Processing Logic (Full Export) ---
    def _process_and_export(self):
        """Handles the final processing and export logic for the wizard."""
        logger.info("Starting wizard export process.")
        st.session_state.wizard_export_buffer = None
        st.session_state.wizard_export_error = None
        processing_success = False

        try:
            # Get Audio Data and SR
            affirmation_audio_data: Optional[AudioData] = st.session_state.get("wizard_affirmation_audio")
            affirmation_sr: Optional[int] = st.session_state.get("wizard_affirmation_sr")
            background_audio_data: Optional[AudioData] = st.session_state.get("wizard_background_audio")
            background_sr: Optional[int] = st.session_state.get("wizard_background_sr")
            frequency_audio_data: Optional[AudioData] = st.session_state.get("wizard_frequency_audio")
            frequency_sr: Optional[int] = st.session_state.get("wizard_frequency_sr")

            # Get Volume & Speed Settings from Step 4 State
            affirmation_volume_slider: float = st.session_state.get("wizard_affirmation_volume", 1.0)
            background_volume: float = st.session_state.get("wizard_background_volume", 0.7)
            frequency_volume: float = st.session_state.get("wizard_frequency_volume", 0.2)
            apply_quick_settings: bool = st.session_state.get("wizard_apply_quick_settings", True)
            eff_affirm_speed: float = QUICK_SUBLIMINAL_PRESET_SPEED if apply_quick_settings else 1.0
            eff_affirm_volume: float = affirmation_volume_slider

            # Validate Affirmation Audio
            if affirmation_audio_data is None or affirmation_sr is None:
                st.session_state.wizard_export_error = "Affirmation audio is missing. Please go back to Step 1."
                logger.error("Wizard export failed: Missing affirmation audio.")
                return
            if affirmation_sr != GLOBAL_SR:
                st.session_state.wizard_export_error = f"Affirmation audio has incorrect sample rate ({affirmation_sr} Hz). Expected {GLOBAL_SR} Hz."
                logger.error(st.session_state.wizard_export_error)
                return

            export_format: str = st.session_state.get("wizard_export_format", "WAV").lower()

            # Prepare Tracks for Mixing
            target_length_samples = 0
            try:
                if not isinstance(affirmation_audio_data, np.ndarray) or affirmation_audio_data.size == 0:
                    raise ValueError("Affirmation audio data is invalid or empty.")

                target_length_samples = affirmation_audio_data.shape[0]
                if target_length_samples == 0:
                    raise ValueError("Affirmation audio has zero length.")

                logger.info(f"Target mix length: {target_length_samples / GLOBAL_SR:.2f}s based on affirmation length.")

                if affirmation_audio_data.ndim == 1:
                    affirmation_audio_data = np.stack([affirmation_audio_data] * 2, axis=-1)
                elif affirmation_audio_data.ndim != 2:
                    raise ValueError(f"Unsupported affirmation audio dimensions: {affirmation_audio_data.ndim}")

                affirmation_tuple: AudioTuple = (affirmation_audio_data, GLOBAL_SR)

                # Prepare Background Audio
                background_tuple: AudioTuple = None
                if background_audio_data is not None and background_sr is not None:
                    if background_sr != GLOBAL_SR:
                        logger.warning(f"Skipping background: SR mismatch ({background_sr} vs {GLOBAL_SR}).")
                    elif not isinstance(background_audio_data, np.ndarray) or background_audio_data.size == 0:
                        logger.warning("Skipping background: Audio data is invalid or empty.")
                    else:
                        if background_audio_data.ndim == 1:
                            background_audio_data = np.stack([background_audio_data] * 2, axis=-1)
                        elif background_audio_data.ndim != 2:
                            logger.warning(f"Skipping background: Unsupported audio dimensions: {background_audio_data.ndim}.")
                            background_audio_data = None  # Set to None to prevent further processing

                        if background_audio_data is not None:
                            looped_background = self._loop_audio_to_length(background_audio_data, target_length_samples)
                            background_tuple = (looped_background, GLOBAL_SR)
                            logger.info(f"Background audio prepared (looped/trimmed to {target_length_samples / GLOBAL_SR:.2f}s).")
                else:
                    logger.info("No background audio data provided or SR missing.")

                # Prepare Frequency Audio
                frequency_tuple: AudioTuple = None
                if frequency_audio_data is not None and frequency_sr is not None:
                    if frequency_sr != GLOBAL_SR:
                        logger.warning(f"Skipping frequency: SR mismatch ({frequency_sr} vs {GLOBAL_SR}).")
                    elif not isinstance(frequency_audio_data, np.ndarray) or frequency_audio_data.size == 0:
                        logger.warning("Skipping frequency: Audio data is invalid or empty.")
                    else:
                        if frequency_audio_data.ndim == 1:
                            frequency_audio_data = np.stack([frequency_audio_data] * 2, axis=-1)
                        elif frequency_audio_data.ndim != 2:
                            logger.warning(f"Skipping frequency: Unsupported audio dimensions: {frequency_audio_data.ndim}.")
                            frequency_audio_data = None  # Set to None to prevent further processing

                        if frequency_audio_data is not None:
                            looped_frequency = self._loop_audio_to_length(frequency_audio_data, target_length_samples)
                            frequency_tuple = (looped_frequency, GLOBAL_SR)
                            logger.info(f"Frequency audio prepared (looped/trimmed to {target_length_samples / GLOBAL_SR:.2f}s).")
                else:
                    logger.info("No frequency audio data provided or SR missing.")

            except Exception as e_prep:
                logger.exception("Error preparing tracks for mixing.")
                st.session_state.wizard_export_error = f"Track Preparation Failed: {e_prep}"
                return

            # --- Mix Tracks ---
            logger.info("Wizard mixing tracks...")
            full_mix: Optional[AudioData] = None
            try:
                if "mix_wizard_tracks" not in globals() or not callable(mix_wizard_tracks):
                    raise NameError("Mixing function 'mix_wizard_tracks' is not available.")

                # --- ADDED: Debug logging before mix call ---
                logger.info(f"Final Mixing - Affirm Vol: {eff_affirm_volume:.2f}, Speed: {eff_affirm_speed:.2f}")
                logger.info(f"Final Mixing - BG Audio Present: {background_tuple is not None}, BG Vol: {background_volume:.2f}")
                logger.info(f"Final Mixing - Freq Audio Present: {frequency_tuple is not None}, Freq Vol: {frequency_volume:.2f}")
                # --- END ADDED ---
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
                return

            # --- Export to selected format ---
            logger.info(f"Exporting final mix as {export_format.upper()}...")
            export_buffer = None
            try:
                if "save_audio_to_bytesio" not in globals() or not callable(save_audio_to_bytesio):
                    raise NameError("Saving function 'save_audio_to_bytesio' is not available.")

                if export_format == "wav":
                    export_buffer = save_audio_to_bytesio(full_mix, GLOBAL_SR)
                    if export_buffer and export_buffer.getbuffer().nbytes > 0:
                        logger.info("Wizard WAV mix buffer generated successfully.")
                        processing_success = True
                    else:
                        logger.error("save_audio_to_bytesio returned empty buffer for WAV.")
                        st.session_state.wizard_export_error = "Failed to save WAV buffer (empty)."

                elif export_format == "mp3":
                    if not PYDUB_AVAILABLE:
                        st.session_state.wizard_export_error = "MP3 export requires 'pydub' and 'ffmpeg'."
                        logger.error("MP3 export requested but pydub is not available.")
                    else:
                        logger.info("Converting full mix to MP3 using pydub...")
                        full_mix_clipped = np.clip(full_mix, -1.0, 1.0)
                        audio_int16 = (full_mix_clipped * 32767).astype(np.int16)

                        channels = audio_int16.shape[1] if audio_int16.ndim == 2 else 1
                        if channels != 2:
                            logger.warning(f"Mix has {channels} channels before MP3 export. Ensuring stereo.")
                            if audio_int16.ndim == 1:
                                audio_int16 = np.stack([audio_int16] * 2, axis=-1)
                            elif audio_int16.shape[1] == 1:
                                audio_int16 = np.concatenate([audio_int16, audio_int16], axis=1)
                            channels = 2

                        segment = AudioSegment(
                            data=audio_int16.tobytes(),
                            sample_width=audio_int16.dtype.itemsize,
                            frame_rate=GLOBAL_SR,
                            channels=channels,
                        )
                        mp3_buffer = BytesIO()
                        segment.export(mp3_buffer, format="mp3", bitrate="192k")
                        mp3_buffer.seek(0)

                        if mp3_buffer.getbuffer().nbytes > 0:
                            logger.info("Wizard MP3 mix buffer generated successfully.")
                            export_buffer = mp3_buffer
                            processing_success = True
                        else:
                            logger.error("MP3 export resulted in empty buffer.")
                            st.session_state.wizard_export_error = "MP3 export failed (empty buffer)."
                else:
                    st.session_state.wizard_export_error = f"Unsupported export format requested: '{export_format}'."
                    logger.error(f"Unsupported export format: {export_format}")

            except NameError as e_func:
                logger.exception(f"A required function ({e_func}) was not found.")
                st.session_state.wizard_export_error = f"Processing function error: {e_func}. Check application setup."
            except Exception as e_export:
                logger.exception(f"Error during wizard audio export ({export_format}).")
                st.session_state.wizard_export_error = f"Export Failed ({export_format.upper()}): {e_export}"

            if processing_success and export_buffer:
                st.session_state.wizard_export_buffer = export_buffer
            elif not st.session_state.get("wizard_export_error"):
                st.session_state.wizard_export_error = "Export process completed but no valid output buffer was generated."
                logger.error("Export finished without errors but buffer is missing.")

        except Exception as e_main:
            logger.exception("Unhandled error during wizard export process.")
            st.session_state.wizard_export_error = f"An unexpected error occurred: {e_main}"

        finally:
            st.session_state.wizard_processing_active = False
            logger.info("Reset wizard_processing_active flag to False.")
            # No st.rerun() here; step_4_export handles rerunning

    # --- Main Rendering Method ---
    def render_wizard(self):
        """Renders the main wizard UI and steps."""
        st.title("✨ MindMorph Quick Create Wizard")

        initialize_wizard_state()  # Ensure state exists

        if not hasattr(self, "tts_generator") or self.tts_generator is None:
            st.error("TTS Engine failed to initialize. Cannot proceed.")
            if st.button("Go Home"):
                self._reset_wizard_state()
            return

        step = st.session_state.get("wizard_step", 1)
        steps_display = ["Affirmations", "Background", "Frequency", "Mix & Export"]
        progress_step = max(1, min(step, len(steps_display)))

        try:
            st.progress(
                (progress_step) / len(steps_display),
                text=f"Step {progress_step}: {steps_display[progress_step - 1]}",
            )
        except IndexError:
            st.progress(0.0)
            logger.error(f"Progress bar index out of range: step={step}, progress_step={progress_step}")

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
                st.error(f"Invalid wizard step encountered: {step}. Resetting to Step 1.")
                logger.error(f"Invalid step detected: {step}. Resetting wizard.")
                st.session_state.wizard_step = 1
                st.rerun()
        except Exception as e_render:
            logger.exception(f"Error rendering wizard step {step}: {e_render}")
            st.error(f"An error occurred while rendering Step {step}. Please try again or reset the wizard.")
            if st.button("Reset Wizard"):
                self._reset_wizard_state()
