# wizard_audio_processor.py
# ==========================================
# Handles audio processing for the Quick Create Wizard
# Uses constants from quick_wizard_config.py
# ==========================================

import gc
import logging
import math
import os  # Needed for checking file extension
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import colorednoise
import librosa
import numpy as np
import streamlit as st

# Import necessary components from other modules
from audio_utils.audio_io import save_audio_to_bytesio
from audio_utils.audio_mixers import mix_wizard_tracks

# Import constants from the central config file
from .quick_wizard_config import (
    AFFIRM_APPLY_SPEED_KEY,
    AFFIRMATION_VOLUME_KEY,
    BG_VOLUME_KEY,
    DEFAULT_APPLY_SPEED,
    FREQ_VOLUME_KEY,
    NOISE_TYPES,
)

# Import constants from main config
try:
    from config import GLOBAL_SR
except ImportError:
    logging.error("Failed to import GLOBAL_SR from main config.")
    GLOBAL_SR = 22050  # Fallback

# Type hint for AudioData
try:
    from audio_utils.audio_effects_pipeline import AudioData
except ImportError:
    AudioData = np.ndarray  # type: ignore
    logging.warning("AudioData type hint failed.")

# Optional MP3 export/import dependency check
try:
    # Check if ffmpeg/libav is likely available for pydub to use
    # This is a basic check; a more robust check might involve trying a conversion
    import subprocess

    from pydub import AudioSegment

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        PYDUB_AVAILABLE = True
        logging.info("pydub and ffmpeg seem available.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            subprocess.run(
                ["avconv", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            PYDUB_AVAILABLE = True
            logging.info("pydub and libav (avconv) seem available.")
        except (FileNotFoundError, subprocess.CalledProcessError):
            PYDUB_AVAILABLE = False
            logging.warning(
                "pydub found, but ffmpeg or avconv likely missing. MP3 loading/export might fail."
            )

except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub library not found. MP3 loading/export disabled.")


# Import soundfile explicitly to catch its specific error
try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logging.warning("soundfile library not found. Librosa might fail.")


logger = logging.getLogger(__name__)
AudioTuple = Optional[Tuple[AudioData, int]]


class WizardAudioProcessor:
    """Handles audio processing, uses constants from quick_wizard_config."""

    def __init__(self):
        logger.debug("WizardAudioProcessor initialized.")

    # --- Core Utility Methods ---
    def _ensure_float32(self, audio_data: AudioData) -> AudioData:
        """Ensures audio data is numpy float32."""
        if not isinstance(audio_data, np.ndarray):
            # If it's not a numpy array, attempt conversion or raise error
            try:
                audio_data = np.array(audio_data, dtype=np.float32)
            except Exception as e:
                raise TypeError(
                    f"Input must be convertible to a NumPy array. Error: {e}"
                )

        if audio_data.dtype != np.float32:
            # Convert non-float32 numpy arrays
            return audio_data.astype(np.float32)
        return audio_data

    def _ensure_stereo(self, audio_data: AudioData) -> AudioData:
        """Ensures audio data is stereo (2 channels)."""
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("Input must be a NumPy array for stereo conversion.")

        if audio_data.ndim == 1:
            # Stack mono to stereo
            logger.debug("Converting mono audio to stereo.")
            return np.stack([audio_data] * 2, axis=-1)
        elif audio_data.ndim == 2:
            if audio_data.shape[1] == 1:
                # Duplicate single channel
                logger.debug("Duplicating single channel to create stereo.")
                return np.concatenate([audio_data, audio_data], axis=1)
            elif audio_data.shape[1] == 2:
                # Already stereo
                return audio_data
            else:
                # Unsupported number of channels
                raise ValueError(
                    f"Unsupported audio shape for stereo conversion: {audio_data.shape}. Expected 1 or 2 channels."
                )
        else:
            # Unsupported dimensions
            raise ValueError(
                f"Unsupported audio dimensions for stereo conversion: {audio_data.ndim}. Expected 1 or 2."
            )

    def _apply_speed_change(
        self, audio_data: AudioData, sr: int, speed_factor: float
    ) -> AudioData:
        """Applies time stretching using librosa."""
        if not isinstance(audio_data, np.ndarray):
            logger.warning("Speed change skipped: Input is not a NumPy array.")
            return audio_data  # Return original if not numpy array
        # Ensure input is stereo float32 before processing
        try:
            audio_data = self._ensure_float32(self._ensure_stereo(audio_data))
        except (TypeError, ValueError) as e:
            logger.warning(
                f"Speed change skipped: Could not ensure stereo float32. Error: {e}"
            )
            return audio_data

        if speed_factor <= 0 or speed_factor == 1.0:
            return audio_data  # No change needed

        logger.info(f"Applying speed change: {speed_factor:.2f}x")
        try:
            # Librosa expects shape (channels, samples) for time_stretch
            stretched_audio = librosa.effects.time_stretch(
                audio_data.T, rate=speed_factor
            )
            # Transpose back to (samples, channels)
            return stretched_audio.T
        except Exception as e:
            logger.exception(f"Error during time stretch: {e}")
            return audio_data  # Return original on error

    def _loop_audio_to_length(
        self, audio_data: AudioData, target_length: int
    ) -> AudioData:
        """Loops or truncates audio to a target number of samples."""
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("Input must be a NumPy array for looping.")

        if target_length <= 0:
            logger.warning("Looping skipped: Target length is zero or negative.")
            # Return an empty array of the correct type and dimensions
            dtype = audio_data.dtype
            ndim = audio_data.ndim
            shape = (0, audio_data.shape[1]) if ndim == 2 else (0,)
            return np.array([], dtype=dtype).reshape(shape)

        current_length = audio_data.shape[0]
        if current_length == 0:
            logger.warning("Looping skipped: Input audio has zero length.")
            # Return an empty array matching target dimensions if possible
            dtype = audio_data.dtype
            ndim = audio_data.ndim
            shape = (0, audio_data.shape[1]) if ndim == 2 else (0,)
            return np.array([], dtype=dtype).reshape(shape)

        if current_length == target_length:
            return audio_data  # Already correct length
        elif current_length < target_length:
            # Loop the audio
            num_repeats = math.ceil(target_length / current_length)
            looped_audio = (
                np.tile(audio_data, (num_repeats, 1))  # Tile for 2D arrays
                if audio_data.ndim == 2
                else np.tile(audio_data, num_repeats)  # Tile for 1D arrays
            )
            # Truncate to the exact target length
            return looped_audio[:target_length]
        else:  # current_length > target_length
            # Truncate the audio
            return audio_data[:target_length]

    # --- Dynamic Loading/Generation Methods ---

    def load_uploaded_audio(
        self, uploaded_file: st.runtime.uploaded_file_manager.UploadedFile
    ) -> AudioTuple:
        """
        Loads audio from a Streamlit UploadedFile object.
        Attempts to use pydub for MP3s if available, otherwise uses librosa.
        Returns a tuple (numpy_array, sample_rate) or None on failure.
        """
        if uploaded_file is None:
            return None

        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        logger.info(
            f"Attempting to load uploaded file: '{file_name}' (Extension: {file_ext})"
        )

        audio_data: Optional[AudioData] = None
        sr: Optional[int] = None

        # --- Attempt 1: Use pydub for MP3 if available ---
        if file_ext == ".mp3" and PYDUB_AVAILABLE:
            logger.info(
                "MP3 detected and pydub available. Attempting load with pydub..."
            )
            try:
                # Read the file content into memory for pydub
                file_content = BytesIO(uploaded_file.getvalue())
                segment = AudioSegment.from_file(
                    file_content
                )  # Let pydub detect format within MP3 container
                logger.info(
                    f"pydub loaded '{file_name}'. Frame rate: {segment.frame_rate}, Channels: {segment.channels}"
                )

                # Resample to GLOBAL_SR if necessary
                if segment.frame_rate != GLOBAL_SR:
                    logger.info(
                        f"Resampling from {segment.frame_rate} Hz to {GLOBAL_SR} Hz."
                    )
                    segment = segment.set_frame_rate(GLOBAL_SR)

                # Convert to stereo if necessary
                if segment.channels == 1:
                    logger.info("Converting mono pydub segment to stereo.")
                    segment = segment.set_channels(2)
                elif segment.channels != 2:
                    raise ValueError(
                        f"Unsupported number of channels from pydub: {segment.channels}"
                    )

                # Convert pydub segment to numpy array (float32, normalized)
                # pydub samples are integers, librosa expects floats between -1 and 1
                samples = np.array(segment.get_array_of_samples())
                if segment.sample_width == 1:  # 8-bit
                    audio_data = (samples.astype(np.float32) - 128) / 128.0
                elif segment.sample_width == 2:  # 16-bit
                    audio_data = samples.astype(np.float32) / 32768.0
                elif segment.sample_width == 4:  # 32-bit int? (less common)
                    audio_data = samples.astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(
                        f"Unsupported sample width from pydub: {segment.sample_width}"
                    )

                # Reshape to (samples, channels)
                audio_data = audio_data.reshape((-1, 2))
                sr = GLOBAL_SR
                logger.info(
                    f"Successfully converted pydub segment to NumPy array. Shape: {audio_data.shape}"
                )

            except Exception as e_pydub:
                logger.warning(
                    f"pydub failed to load '{file_name}': {e_pydub}. Falling back to librosa."
                )
                audio_data = None  # Ensure reset before trying librosa
                sr = None
                # Seek back to the beginning of the file for librosa
                uploaded_file.seek(0)

        # --- Attempt 2: Use librosa (for non-MP3 or if pydub failed/unavailable) ---
        if audio_data is None:
            if not SOUNDFILE_AVAILABLE and file_ext != ".wav":
                logger.error(
                    f"Cannot load '{file_name}' with librosa: soundfile library missing or format potentially unsupported without it."
                )
                # Optionally try audioread backend if installed? For now, fail.
                return None

            logger.info(f"Attempting load with librosa for '{file_name}'...")
            try:
                # Librosa can often handle file-like objects directly
                audio_data, sr_native = librosa.load(
                    uploaded_file, sr=GLOBAL_SR, mono=False
                )
                # Librosa loads as (channels, samples) if stereo, need (samples, channels)
                if audio_data.ndim == 2 and audio_data.shape[0] == 2:
                    audio_data = audio_data.T
                sr = GLOBAL_SR  # We requested this sample rate
                logger.info(
                    f"Librosa loaded '{file_name}'. Original SR: {sr_native}, Target SR: {sr}. Shape before processing: {audio_data.shape}"
                )

            except (sf.LibsndfileError, RuntimeError, Exception) as e_librosa:
                # Catch specific soundfile errors and general runtime errors
                logger.error(f"Librosa failed to load '{file_name}': {e_librosa}")
                # Check if it was the specific "Format not recognised" error
                if isinstance(
                    e_librosa, sf.LibsndfileError
                ) and "Format not recognised" in str(e_librosa):
                    logger.error(
                        "This often means libsndfile lacks support for this format (e.g., MP3). Try installing ffmpeg and using pydub."
                    )
                return None  # Failed to load

        # --- Final Processing and Validation ---
        if audio_data is not None and sr is not None:
            try:
                # Ensure stereo float32 for consistency downstream
                processed_audio = self._ensure_float32(self._ensure_stereo(audio_data))
                logger.info(
                    f"Successfully loaded and processed '{file_name}'. Final Shape: {processed_audio.shape}, SR: {sr}"
                )
                return (processed_audio, sr)
            except (TypeError, ValueError) as e_process:
                logger.error(
                    f"Error during final processing (stereo/float32) of '{file_name}': {e_process}"
                )
                return None
        else:
            # Should not happen if loading succeeded, but as a safeguard
            logger.error(
                f"Loading failed for '{file_name}' for unknown reasons after load attempts."
            )
            return None

    def generate_noise_audio(
        self, noise_type: str, duration_seconds: float, sr: int
    ) -> AudioTuple:
        """Generates stereo noise audio."""
        if noise_type not in NOISE_TYPES or duration_seconds <= 0:
            logger.warning(
                f"Invalid parameters for noise generation: type={noise_type}, duration={duration_seconds}"
            )
            return None
        logger.info(f"Generating {noise_type} for {duration_seconds:.2f}s at {sr} Hz.")
        num_samples = int(duration_seconds * sr)
        if num_samples <= 0:
            logger.warning("Calculated zero samples for noise generation.")
            return None

        noise_audio: Optional[AudioData] = None
        try:
            # Determine beta value for colorednoise library
            beta = 0  # White noise default
            if noise_type == "Pink Noise":
                beta = 1
            elif noise_type == "Brown Noise":
                beta = 2

            # Generate mono noise
            mono_noise = colorednoise.powerlaw_psd_gaussian(beta, num_samples)

            # Normalize roughly to avoid clipping but maintain character
            max_val = np.max(np.abs(mono_noise))
            if max_val > 1e-6:  # Avoid division by zero or near-zero
                mono_noise = mono_noise / max_val * 0.8  # Normalize to 0.8 peak
            else:
                logger.warning("Generated noise has near-zero amplitude.")

            # Convert to stereo float32
            noise_audio = np.stack([mono_noise] * 2, axis=-1).astype(np.float32)

            logger.info(f"Generated {noise_type}. Shape: {noise_audio.shape}")
            return (noise_audio, sr)
        except Exception as e:
            logger.exception(f"Failed to generate noise: {e}")
            return None
        finally:
            # Explicitly delete large array if created
            if noise_audio is not None:
                del noise_audio
                gc.collect()

    def generate_frequency_audio(
        self,
        freq_choice: str,
        freq_params: Dict[str, Any],
        duration_seconds: float,
        sr: int,
    ) -> AudioTuple:
        """Generates stereo frequency-based audio (Binaural/Isochronic)."""
        if freq_choice == "None" or duration_seconds <= 0:
            return None  # No frequency requested or zero duration
        logger.info(
            f"Generating {freq_choice} for {duration_seconds:.2f}s at {sr} Hz with params: {freq_params}"
        )
        num_samples = int(duration_seconds * sr)
        if num_samples <= 0:
            return None

        # Time vector
        t = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)
        audio_result: Optional[AudioData] = None

        try:
            base_freq = float(freq_params.get("base_freq", 100.0))  # Default 100Hz

            if freq_choice == "Binaural Beats":
                beat_freq = float(freq_params.get("beat_freq", 5.0))  # Default 5Hz
                if beat_freq <= 0:
                    raise ValueError("Beat frequency must be positive.")
                # Calculate left and right frequencies
                f_left = base_freq - beat_freq / 2.0
                f_right = base_freq + beat_freq / 2.0
                if f_left <= 0:
                    raise ValueError("Resulting left frequency must be positive.")
                # Generate sine waves for each channel
                left_channel = 0.5 * np.sin(2 * np.pi * f_left * t)  # Amplitude 0.5
                right_channel = 0.5 * np.sin(2 * np.pi * f_right * t)
                # Stack into stereo float32 array
                audio_result = np.stack([left_channel, right_channel], axis=-1).astype(
                    np.float32
                )

            elif freq_choice == "Isochronic Tones":
                pulse_freq = float(freq_params.get("pulse_freq", 7.0))  # Default 7Hz
                if pulse_freq <= 0:
                    raise ValueError("Pulse frequency must be positive.")
                # Generate base tone
                tone = np.sin(2 * np.pi * base_freq * t)
                # Generate modulation (square wave based on pulse frequency)
                # This creates pulses by turning the tone on/off
                pulse_period_samples = sr / pulse_freq
                # Simple square wave: on for first half of period, off for second
                modulation = (
                    np.mod(np.arange(num_samples), pulse_period_samples)
                    < (pulse_period_samples / 2)
                ).astype(np.float32)
                # Apply modulation to tone, scale amplitude
                mono_iso_tone = 0.8 * tone * modulation  # Amplitude 0.8
                # Stack into stereo float32 array
                audio_result = np.stack([mono_iso_tone] * 2, axis=-1).astype(np.float32)

            else:
                logger.warning(f"Unknown frequency choice requested: {freq_choice}")
                return None

            if audio_result is not None:
                logger.info(f"Generated {freq_choice}. Shape: {audio_result.shape}")
                return (audio_result, sr)
            else:
                # Should not happen if logic is correct, but safeguard
                logger.error(
                    f"Frequency generation failed unexpectedly for {freq_choice}."
                )
                return None

        except Exception as e:
            logger.exception(f"Failed to generate frequency audio '{freq_choice}': {e}")
            return None
        finally:
            if audio_result is not None:
                del audio_result
                gc.collect()

    # --- Track Preparation & Mixing Methods ---
    def _prepare_audio_track(
        self, audio_tuple: AudioTuple, target_length: int, track_name: str
    ) -> AudioTuple:
        """Prepares a single audio track for mixing (ensures SR, stereo, length)."""
        if audio_tuple is None:
            logger.debug(
                f"Skipping preparation for {track_name}: No audio tuple provided."
            )
            return None
        audio_data, sr = audio_tuple

        # Validate sample rate
        if sr != GLOBAL_SR:
            # This should ideally not happen if loading/generation enforces GLOBAL_SR
            logger.warning(
                f"Skipping {track_name}: Sample rate mismatch ({sr} Hz vs expected {GLOBAL_SR} Hz). Resampling not implemented here."
            )
            return None

        # Validate data
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            logger.warning(f"Skipping {track_name}: Invalid or empty audio data.")
            return None

        try:
            # Ensure stereo float32 and correct length
            processed_audio = self._ensure_float32(self._ensure_stereo(audio_data))
            processed_audio = self._loop_audio_to_length(processed_audio, target_length)

            if processed_audio.size == 0:
                logger.warning(
                    f"Skipping {track_name}: Audio became empty after processing (looping/truncating). Target length: {target_length}"
                )
                return None

            logger.debug(
                f"{track_name} prepared. Target Samples: {target_length}, Actual Samples: {processed_audio.shape[0]}, dtype: {processed_audio.dtype}."
            )
            return (processed_audio, GLOBAL_SR)

        except (TypeError, ValueError) as e:
            logger.warning(
                f"Skipping {track_name} due to error during preparation: {e}"
            )
            return None

    def generate_preview_mix(
        self,
        duration_seconds: int,
        affirmation_audio_tuple: AudioTuple,
        background_audio_tuple: AudioTuple,
        frequency_audio_tuple: AudioTuple,
        wizard_state: Dict[str, Any],
    ) -> AudioTuple:
        """Generates a preview mix of the tracks."""
        logger.info(f"Starting preview mix generation ({duration_seconds}s).")
        preview_mix: Optional[AudioData] = None
        processed_affirmation_audio: Optional[AudioData] = None

        try:
            # 1. Validate and Process Affirmations (Required)
            if (
                affirmation_audio_tuple is None
                or not isinstance(affirmation_audio_tuple[0], np.ndarray)
                or affirmation_audio_tuple[0].size == 0
            ):
                raise ValueError("Invalid or missing affirmation audio for preview.")

            affirmation_audio, affirmation_sr = affirmation_audio_tuple
            if affirmation_sr != GLOBAL_SR:
                raise ValueError(
                    f"Affirmation SR incorrect ({affirmation_sr} Hz). Cannot mix."
                )

            # Ensure stereo float32 and apply speed change if requested
            processed_affirmation_audio = self._ensure_float32(
                self._ensure_stereo(affirmation_audio)
            )
            apply_speed = wizard_state.get(AFFIRM_APPLY_SPEED_KEY, DEFAULT_APPLY_SPEED)
            speed_factor = wizard_state.get("wizard_speed_factor", 1.0)
            if apply_speed:
                processed_affirmation_audio = self._apply_speed_change(
                    processed_affirmation_audio, affirmation_sr, speed_factor
                )

            # 2. Get Volumes
            affirmation_volume: float = wizard_state.get(AFFIRMATION_VOLUME_KEY, 1.0)
            background_volume: float = wizard_state.get(BG_VOLUME_KEY, 0.7)
            frequency_volume: float = wizard_state.get(FREQ_VOLUME_KEY, 0.2)

            # 3. Determine Preview Length in Samples
            # Use the minimum of requested duration and actual affirmation length
            target_preview_samples = int(duration_seconds * GLOBAL_SR)
            actual_affirmation_samples = processed_affirmation_audio.shape[0]
            preview_length_samples = min(
                target_preview_samples, actual_affirmation_samples
            )

            if preview_length_samples <= 0:
                raise ValueError("Calculated preview length is zero or negative.")
            logger.info(
                f"Preview target length: {preview_length_samples / GLOBAL_SR:.2f}s ({preview_length_samples} samples)."
            )

            # 4. Prepare Tracks to Preview Length
            affirmation_preview_tuple = (
                processed_affirmation_audio[
                    :preview_length_samples
                ],  # Truncate affirmations
                GLOBAL_SR,
            )
            background_preview_tuple = self._prepare_audio_track(
                background_audio_tuple, preview_length_samples, "Background (Preview)"
            )
            frequency_preview_tuple = self._prepare_audio_track(
                frequency_audio_tuple, preview_length_samples, "Frequency (Preview)"
            )

            # 5. Mix Tracks using the dedicated mixer function
            logger.info("Mixing preview tracks...")
            preview_mix = mix_wizard_tracks(
                affirmation_audio=affirmation_preview_tuple,
                background_audio=background_preview_tuple,
                frequency_audio=frequency_preview_tuple,
                affirmation_volume=affirmation_volume,
                background_volume=background_volume,
                frequency_volume=frequency_volume,
                target_sr=GLOBAL_SR,  # Ensure mixer uses correct SR
            )

            # 6. Validate Mix Result
            if (
                preview_mix is None
                or not isinstance(preview_mix, np.ndarray)
                or preview_mix.size == 0
            ):
                raise ValueError("Preview mixing failed or resulted in empty audio.")

            logger.info(
                f"Preview mixing successful. Length: {len(preview_mix) / GLOBAL_SR:.2f}s."
            )
            return (self._ensure_float32(preview_mix), GLOBAL_SR)  # Ensure float32

        except Exception as e_preview:
            logger.exception("Error generating preview mix.")
            # Re-raise the exception to be caught by the calling function in quick_wizard.py
            raise e_preview
        finally:
            # Cleanup intermediate arrays
            if processed_affirmation_audio is not None:
                del processed_affirmation_audio
            # Input tuples are handled by the caller (quick_wizard)
            gc.collect()

    def process_and_export(
        self,
        affirmation_audio_tuple: AudioTuple,
        background_audio_tuple: AudioTuple,
        frequency_audio_tuple: AudioTuple,
        wizard_state: Dict[str, Any],
    ) -> Tuple[Optional[BytesIO], Optional[str]]:
        """Processes all tracks and exports the final mix to a BytesIO buffer."""
        logger.info("Starting final audio processing and export.")
        export_buffer: Optional[BytesIO] = None
        error_message: Optional[str] = None
        full_mix: Optional[AudioData] = None
        processed_affirmation_audio: Optional[AudioData] = None

        try:
            # 1. Validate and Process Affirmations (Required)
            if (
                affirmation_audio_tuple is None
                or not isinstance(affirmation_audio_tuple[0], np.ndarray)
                or affirmation_audio_tuple[0].size == 0
            ):
                raise ValueError("Invalid or missing affirmation audio for export.")

            affirmation_audio, affirmation_sr = affirmation_audio_tuple
            if affirmation_sr != GLOBAL_SR:
                raise ValueError(
                    f"Affirmation SR incorrect ({affirmation_sr} Hz). Cannot mix."
                )

            # Ensure stereo float32 and apply speed change
            processed_affirmation_audio = self._ensure_float32(
                self._ensure_stereo(affirmation_audio)
            )
            apply_speed = wizard_state.get(AFFIRM_APPLY_SPEED_KEY, DEFAULT_APPLY_SPEED)
            speed_factor = wizard_state.get("wizard_speed_factor", 1.0)
            if apply_speed:
                processed_affirmation_audio = self._apply_speed_change(
                    processed_affirmation_audio, affirmation_sr, speed_factor
                )

            # 2. Get Settings
            affirmation_volume: float = wizard_state.get(AFFIRMATION_VOLUME_KEY, 1.0)
            background_volume: float = wizard_state.get(BG_VOLUME_KEY, 0.7)
            frequency_volume: float = wizard_state.get(FREQ_VOLUME_KEY, 0.2)
            export_format: str = wizard_state.get("wizard_export_format", "WAV").lower()

            # 3. Determine Target Length (based on processed affirmations)
            target_length_samples = processed_affirmation_audio.shape[0]
            if target_length_samples == 0:
                raise ValueError("Affirmation audio has zero length after processing.")
            logger.info(
                f"Target mix length: {target_length_samples / GLOBAL_SR:.2f}s ({target_length_samples} samples)."
            )

            # 4. Prepare Tracks to Final Length
            affirmation_final_tuple: AudioTuple = (
                processed_affirmation_audio,
                GLOBAL_SR,
            )
            # Don't need affirmation_final_tuple after this, free memory early
            processed_affirmation_audio = None
            gc.collect()

            background_final_tuple = self._prepare_audio_track(
                background_audio_tuple, target_length_samples, "Background (Final)"
            )
            frequency_final_tuple = self._prepare_audio_track(
                frequency_audio_tuple, target_length_samples, "Frequency (Final)"
            )

            # 5. Mix Tracks
            logger.info("Mixing final tracks...")
            full_mix = mix_wizard_tracks(
                affirmation_audio=affirmation_final_tuple,  # Pass the prepared tuple
                background_audio=background_final_tuple,
                frequency_audio=frequency_final_tuple,
                affirmation_volume=affirmation_volume,
                background_volume=background_volume,
                frequency_volume=frequency_volume,
                target_sr=GLOBAL_SR,
            )

            # --- Cleanup intermediate tuples ---
            del affirmation_final_tuple, background_final_tuple, frequency_final_tuple
            gc.collect()

            # Validate mix result
            if (
                full_mix is None
                or not isinstance(full_mix, np.ndarray)
                or full_mix.size == 0
            ):
                raise ValueError("Final mixing failed or resulted in empty audio.")

            full_mix = self._ensure_float32(full_mix)  # Ensure float32 before export
            logger.info(
                f"Final mixing successful. Length: {len(full_mix) / GLOBAL_SR:.2f}s."
            )

            # 6. Export to Buffer
            logger.info(f"Exporting final mix as {export_format.upper()}...")
            if export_format == "wav":
                export_buffer = save_audio_to_bytesio(
                    full_mix, GLOBAL_SR
                )  # WAV default
                if not export_buffer or export_buffer.getbuffer().nbytes == 0:
                    error_message = "Failed to save WAV (empty buffer)."
                    export_buffer = None
                else:
                    logger.info("WAV buffer generated successfully.")
            elif export_format == "mp3":
                if not PYDUB_AVAILABLE:
                    error_message = "MP3 export requires 'pydub' and 'ffmpeg'/'libav'."
                    logger.error(error_message)
                else:
                    try:
                        logger.info(
                            "Converting final mix (float32) to MP3 using pydub..."
                        )
                        # Convert float32 [-1.0, 1.0] to int16 for pydub
                        # Clip first to prevent wrap-around issues with large floats
                        full_mix_clipped = np.clip(full_mix, -1.0, 1.0)
                        audio_int16 = (full_mix_clipped * 32767).astype(np.int16)
                        # Free float array memory
                        del full_mix_clipped, full_mix
                        full_mix = None
                        gc.collect()

                        # Create pydub segment from NumPy array
                        segment = AudioSegment(
                            data=audio_int16.tobytes(),
                            sample_width=audio_int16.dtype.itemsize,  # Should be 2 for int16
                            frame_rate=GLOBAL_SR,
                            channels=2,  # Should be stereo after mixing
                        )
                        del audio_int16
                        gc.collect()  # Free int array memory

                        # Export segment to MP3 buffer
                        mp3_buffer = BytesIO()
                        # Specify bitrate (e.g., '192k') - adjust as needed
                        segment.export(mp3_buffer, format="mp3", bitrate="192k")
                        mp3_buffer.seek(0)  # Rewind buffer for reading
                        del segment
                        gc.collect()  # Free pydub segment memory

                        if mp3_buffer.getbuffer().nbytes > 0:
                            logger.info("MP3 buffer generated successfully.")
                            export_buffer = mp3_buffer
                        else:
                            error_message = (
                                "MP3 export failed (pydub created empty buffer)."
                            )
                            logger.error(error_message)

                    except Exception as e_mp3:
                        logger.exception("Error during pydub MP3 conversion/export.")
                        error_message = f"MP3 Export Failed: {e_mp3}"
                        # Ensure intermediate arrays are deleted on error
                        if "full_mix" in locals() and full_mix is not None:
                            del full_mix
                        if "audio_int16" in locals():
                            del audio_int16
                        gc.collect()
            else:
                error_message = (
                    f"Unsupported export format requested: '{export_format}'."
                )
                logger.error(error_message)

        except Exception as e_export:
            logger.exception("Unhandled error during final processing/export wrapper.")
            error_message = f"Unexpected Processing Error: {e_export}"
            export_buffer = None  # Ensure buffer is None on error
            # Cleanup potentially large arrays if an error occurred mid-process
            if (
                "processed_affirmation_audio" in locals()
                and processed_affirmation_audio is not None
            ):
                del processed_affirmation_audio
            if "full_mix" in locals() and full_mix is not None:
                del full_mix
            gc.collect()

        # --- Final Return ---
        if export_buffer and not error_message:
            logger.info("Audio processing and export completed successfully.")
            return export_buffer, None
        else:
            # If buffer creation failed or an error message was set
            if not error_message:
                error_message = "Export failed (unknown reason)."
            logger.error(f"Export failed: {error_message}")
            # Ensure mix is deleted if it exists but export failed
            if "full_mix" in locals() and full_mix is not None:
                del full_mix
                gc.collect()
            return None, error_message
