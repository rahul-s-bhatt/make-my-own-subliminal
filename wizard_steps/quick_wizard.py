# quick_wizard.py
# ==========================================
# Quick Create Wizard Orchestrator for MindMorph
# ==========================================

import logging
import math  # Needed for ceiling division for looping
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st

# Import necessary components from other modules
from audio_utils.audio_io import save_audio_to_bytesio  # Keep using this

# Import mix_wizard_tracks from audio_utils.audio_mixers
# Assuming mix_wizard_tracks expects audio data as numpy arrays and SR as int
from audio_utils.audio_mixers import mix_wizard_tracks

# Type hint for AudioData
try:
    # Assuming AudioData is defined as np.ndarray or similar
    from audio_utils.audio_effects_pipeline import AudioData
except ImportError:
    AudioData = np.ndarray  # Fallback

# Import wizard state management
from config import GLOBAL_SR, QUICK_SUBLIMINAL_PRESET_SPEED, QUICK_SUBLIMINAL_PRESET_VOLUME
from tts_generator import TTSGenerator

# Import step rendering functions
from wizard_steps import (
    step_1_affirmations,
    step_2_background,
    step_3_frequency,
    step_4_export,
)
from wizard_steps.wizard_state import initialize_wizard_state, reset_wizard_state

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

    def __init__(self, tts_generator: TTSGenerator):
        """Initializes the QuickWizard."""
        self.tts_generator = tts_generator
        initialize_wizard_state()
        logger.debug("QuickWizard initialized and state ensured.")

    # --- State Synchronization Callbacks ---
    # (Keep existing callbacks: sync_affirmation_text, clear_affirmation_upload_state, etc.)
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

        if new_choice != st.session_state.get("wizard_background_choice"):
            st.session_state.wizard_background_audio = None
            st.session_state.wizard_background_sr = None
            logger.debug(f"Background choice changed to {new_choice}, cleared audio.")

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
        """
        Loops or truncates audio_data to match target_length.

        Args:
            audio_data: The numpy array representing the audio.
            target_length: The desired length in samples.

        Returns:
            The looped or truncated audio data as a numpy array.
        """
        current_length = audio_data.shape[0]
        if current_length == target_length:
            return audio_data
        elif current_length < target_length:
            num_repeats = math.ceil(target_length / current_length)
            # Handle potential mono input before tiling (should be stereo by now, but check)
            if audio_data.ndim == 1:
                logger.warning("Looping mono audio. Will tile as 1D.")
                looped_audio = np.tile(audio_data, num_repeats)
            else:
                looped_audio = np.tile(audio_data, (num_repeats, 1))
            return looped_audio[:target_length]
        else:  # current_length > target_length
            return audio_data[:target_length]

    # --- Core Processing Logic ---
    def _process_and_export(self):
        """Handles the final processing and export logic for the wizard."""
        logger.info("Starting wizard export process.")
        st.session_state.wizard_export_buffer = None
        st.session_state.wizard_export_error = None

        # Get base audio data and sample rates from session state
        affirmation_audio_data: Optional[AudioData] = st.session_state.get("wizard_affirmation_audio")
        affirmation_sr: Optional[int] = st.session_state.get("wizard_affirmation_sr")
        background_audio_data: Optional[AudioData] = st.session_state.get("wizard_background_audio")
        background_sr: Optional[int] = st.session_state.get("wizard_background_sr")
        frequency_audio_data: Optional[AudioData] = st.session_state.get("wizard_frequency_audio")
        frequency_sr: Optional[int] = st.session_state.get("wizard_frequency_sr")

        # Get volume settings
        background_volume: float = st.session_state.get("wizard_background_volume", 0.7)
        frequency_volume: float = st.session_state.get("wizard_frequency_volume", 0.2)

        # --- Validate Affirmation Audio ---
        if affirmation_audio_data is None or affirmation_sr is None:
            st.session_state.wizard_export_error = "Affirmation audio is missing."
            logger.error("Wizard export failed: Missing affirmation audio.")
            return

        # --- Determine Affirmation Speed/Volume based on Toggle ---
        apply_quick_settings: bool = st.session_state.get("wizard_apply_quick_settings", True)
        if apply_quick_settings:
            eff_affirm_speed: float = QUICK_SUBLIMINAL_PRESET_SPEED
            eff_affirm_volume: float = QUICK_SUBLIMINAL_PRESET_VOLUME
            logger.info("Applying Quick Wizard preset settings (Speed/Volume) to affirmations.")
        else:
            eff_affirm_speed = 1.0  # Original speed
            eff_affirm_volume = 1.0  # Original volume (will be masked by background)
            logger.info("Using original speed/volume for affirmations (Quick Settings disabled).")

        export_format: str = st.session_state.get("wizard_export_format", "wav").lower()

        # --- Prepare Tracks for Mixing (Including Looping) ---
        try:
            # Ensure all tracks are at the target sample rate (should be handled by load/gen, but double-check)
            if affirmation_sr != GLOBAL_SR:
                logger.warning(f"Affirmation SR ({affirmation_sr}) differs from GLOBAL_SR ({GLOBAL_SR}). This might cause issues.")
                # Ideally, add resampling here if this occurs
                pass  # Assume GLOBAL_SR for now

            target_length_samples = affirmation_audio_data.shape[0]
            logger.info(f"Target mix length based on affirmations: {target_length_samples / GLOBAL_SR:.2f} seconds")

            # Prepare affirmation tuple (no looping needed for this one)
            affirmation_tuple: AudioTuple = (affirmation_audio_data, GLOBAL_SR)

            # Prepare background tuple (loop if necessary)
            background_tuple: AudioTuple = None
            if background_audio_data is not None and background_sr is not None:
                if background_sr != GLOBAL_SR:
                    logger.warning(f"Background SR ({background_sr}) differs from GLOBAL_SR ({GLOBAL_SR}). Resampling required.")
                    # Add resampling logic if needed
                    pass  # Assume GLOBAL_SR for now
                looped_background = self._loop_audio_to_length(background_audio_data, target_length_samples)
                background_tuple = (looped_background, GLOBAL_SR)
                logger.info(f"Background audio prepared (looped/truncated to target length).")

            # Prepare frequency tuple (loop if necessary)
            frequency_tuple: AudioTuple = None
            if frequency_audio_data is not None and frequency_sr is not None:
                if frequency_sr != GLOBAL_SR:
                    logger.warning(f"Frequency SR ({frequency_sr}) differs from GLOBAL_SR ({GLOBAL_SR}). Resampling required.")
                    # Add resampling logic if needed
                    pass  # Assume GLOBAL_SR for now
                looped_frequency = self._loop_audio_to_length(frequency_audio_data, target_length_samples)
                frequency_tuple = (looped_frequency, GLOBAL_SR)
                logger.info(f"Frequency audio prepared (looped/truncated to target length).")

        except Exception as e_prep:
            logger.exception("Error preparing tracks for mixing.")
            st.session_state.wizard_export_error = f"Track Preparation Failed: {e_prep}"
            return

        # --- Mix Tracks ---
        logger.info("Wizard mixing tracks...")
        spinner_msg = f"Generating final mix ({export_format.upper()})..."
        # NOTE: The spinner is now handled in step_4_export.py around this call
        # with st.spinner(spinner_msg): # Remove spinner from here
        try:
            # Pass effective speed/volume and potentially looped tracks
            full_mix = mix_wizard_tracks(
                affirmation_audio=affirmation_tuple,  # Already checked not None
                background_audio=background_tuple,  # May be None
                frequency_audio=frequency_tuple,  # May be None
                affirmation_speed=eff_affirm_speed,
                affirmation_volume=eff_affirm_volume,
                background_volume=background_volume,
                frequency_volume=frequency_volume,
                target_sr=GLOBAL_SR,
            )

            if full_mix is None or full_mix.size == 0:
                raise ValueError("Mixing process resulted in empty or None audio.")

            mix_duration_s = len(full_mix) / GLOBAL_SR if GLOBAL_SR > 0 else 0
            logger.info(f"Mixing successful. Final mix length: {mix_duration_s:.2f} seconds.")

            # --- Export to selected format ---
            export_buffer = None  # Initialize buffer variable
            if export_format == "wav":
                logger.info("Saving final mix to WAV buffer...")
                export_buffer = save_audio_to_bytesio(full_mix, GLOBAL_SR)  # Using the robust version

                # --- ADDED BUFFER CHECK ---
                if export_buffer and export_buffer.getbuffer().nbytes > 0:
                    buffer_size = export_buffer.getbuffer().nbytes
                    # Calculate expected size (approx): samples * channels * bytes_per_sample (2 for int16)
                    expected_min_size = full_mix.shape[0] * 2 * 2 * 0.9  # Samples * stereo * 2 bytes/sample * 90% tolerance
                    logger.info(f"Generated export buffer size: {buffer_size} bytes.")
                    if buffer_size < expected_min_size:
                        logger.error(f"Export buffer size ({buffer_size}) seems unexpectedly small compared to expected minimum ({expected_min_size})!")
                        # Optionally raise an error or set the buffer to None
                        # raise ValueError("Generated WAV buffer is unexpectedly small.")
                        st.session_state.wizard_export_error = "Generated audio buffer was unexpectedly small. Export failed."
                        export_buffer = None  # Prevent download of potentially corrupt buffer
                    else:
                        st.session_state.wizard_export_buffer = export_buffer
                        logger.info("Wizard WAV mix buffer generated and stored in session state.")
                else:
                    # save_audio_to_bytesio failed or returned empty
                    logger.error("save_audio_to_bytesio failed to return a valid buffer.")
                    st.session_state.wizard_export_error = "Failed to save final audio to buffer."
                    export_buffer = None  # Ensure it's None
            # --- END ADDED BUFFER CHECK ---

            elif export_format == "mp3" and PYDUB_AVAILABLE:
                try:
                    logger.info("Wizard converting full mix to MP3...")
                    full_mix_clipped = np.clip(full_mix, -1.0, 1.0)
                    audio_int16 = (full_mix_clipped * 32767).astype(np.int16)
                    channels = 2  # Assume stereo after mixing/processing
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
                        logger.info("Converted mono mix to stereo for MP3 export.")

                    mp3_buffer = BytesIO()
                    segment.export(mp3_buffer, format="mp3", bitrate="192k")
                    mp3_buffer.seek(0)

                    # --- ADDED BUFFER CHECK ---
                    if mp3_buffer.getbuffer().nbytes > 0:
                        buffer_size = mp3_buffer.getbuffer().nbytes
                        logger.info(f"Generated MP3 export buffer size: {buffer_size} bytes.")
                        # MP3 size is harder to predict, just check > 0
                        if buffer_size < 1000:  # Check it's not trivially small
                            logger.error(f"MP3 Export buffer size ({buffer_size}) seems unexpectedly small!")
                            st.session_state.wizard_export_error = "Generated MP3 audio buffer was unexpectedly small. Export failed."
                            export_buffer = None
                        else:
                            st.session_state.wizard_export_buffer = mp3_buffer
                            logger.info("Wizard MP3 mix buffer generated and stored in session state.")
                            export_buffer = mp3_buffer  # Assign to check later if needed
                    else:
                        logger.error("MP3 export resulted in an empty buffer.")
                        st.session_state.wizard_export_error = "MP3 export failed (empty buffer)."
                        export_buffer = None
                    # --- END ADDED BUFFER CHECK ---

                except Exception as e_mp3:
                    logger.exception("Wizard failed to export mix as MP3.")
                    st.session_state.wizard_export_error = f"MP3 Export Failed: {e_mp3}. Ensure ffmpeg is installed and accessible."
                    export_buffer = None
            elif export_format == "mp3" and not PYDUB_AVAILABLE:
                st.session_state.wizard_export_error = "MP3 export requires 'pydub' and 'ffmpeg'."
                logger.error(st.session_state.wizard_export_error)
                export_buffer = None
            else:
                st.session_state.wizard_export_error = f"Unsupported export format '{export_format}'."
                logger.error(st.session_state.wizard_export_error)
                export_buffer = None

            # Final check if buffer creation failed for any reason
            if export_buffer is None or export_buffer.getbuffer().nbytes == 0:
                logger.error("Export process finished but resulted in no valid export buffer.")
                # Error message should already be set in session state by specific format handling

        except Exception as e:
            logger.exception("Error during wizard mix/save.")
            st.session_state.wizard_export_error = f"Processing Failed: {e}"

    # --- Main Rendering Method ---
    def render_wizard(self):
        """Renders the current step of the wizard by calling the appropriate step function."""
        st.title("✨ MindMorph Quick Create Wizard")
        initialize_wizard_state()  # Ensure state exists
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
            logger.warning(f"Progress bar step index out of range: {progress_step}")

        # Render the appropriate step UI
        if step == 1:
            step_1_affirmations.render_step_1(self)
        elif step == 2:
            step_2_background.render_step_2(self)
        elif step == 3:
            step_3_frequency.render_step_3(self)
        elif step == 4:
            # Wrap the call to render_step_4 in a spinner if generation is happening
            # (Need to manage the button state carefully)
            # This might be better handled within render_step_4 itself
            step_4_export.render_step_4(self)
        else:
            st.error("Invalid wizard step detected. Resetting.")
            logger.error(f"Invalid wizard step: {step}. Resetting.")
            self._reset_wizard_state()  # Reruns
