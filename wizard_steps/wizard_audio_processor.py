# wizard_audio_processor.py
# ==========================================
# Handles audio processing for the Quick Create Wizard
# Uses constants from quick_wizard_config.py
# ==========================================

import gc
import logging
import math
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
from .quick_wizard_config import DEFAULT_APPLY_SPEED  # Need NOISE_TYPES list here
from .quick_wizard_config import (
    AFFIRM_APPLY_SPEED_KEY,
    AFFIRMATION_VOLUME_KEY,
    BG_VOLUME_KEY,
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
    AudioData = np.ndarray
    logging.warning("AudioData type hint failed.")  # type: ignore

# Optional MP3 export dependency check
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not found.")

logger = logging.getLogger(__name__)
AudioTuple = Optional[Tuple[AudioData, int]]


class WizardAudioProcessor:
    """Handles audio processing, uses constants from quick_wizard_config."""

    def __init__(self):
        logger.debug("WizardAudioProcessor initialized.")

    # --- Core Utility Methods ---
    def _ensure_float32(self, audio_data: AudioData) -> AudioData:
        if audio_data.dtype != np.float32:
            return audio_data.astype(np.float32)
        return audio_data

    def _ensure_stereo(self, audio_data: AudioData) -> AudioData:
        if audio_data.ndim == 1:
            return np.stack([audio_data] * 2, axis=-1)
        elif audio_data.ndim == 2 and audio_data.shape[1] == 1:
            return np.concatenate([audio_data, audio_data], axis=1)
        elif audio_data.ndim == 2 and audio_data.shape[1] == 2:
            return audio_data
        else:
            raise ValueError(f"Unsupported audio shape for stereo: {audio_data.shape}")

    def _apply_speed_change(
        self, audio_data: AudioData, sr: int, speed_factor: float
    ) -> AudioData:
        if (
            not isinstance(audio_data, np.ndarray)
            or audio_data.ndim != 2
            or audio_data.shape[1] != 2
        ):
            return audio_data
        if speed_factor <= 0 or speed_factor == 1.0:
            return audio_data
        logger.info(
            f"[_apply_speed_change] - Applying speed change: {speed_factor:.2f}x"
        )
        try:
            audio_float32 = self._ensure_float32(audio_data)
            stretched_audio = librosa.effects.time_stretch(
                audio_float32.T, rate=speed_factor
            ).T
            return self._ensure_float32(stretched_audio)
        except Exception as e:
            logger.exception(f"Error time stretch: {e}")
            return audio_data

    def _loop_audio_to_length(
        self, audio_data: AudioData, target_length: int
    ) -> AudioData:
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("NumPy array needed.")
        audio_data = self._ensure_float32(audio_data)
        if target_length <= 0:
            return np.array([], dtype=np.float32)
        current_length = audio_data.shape[0]
        if current_length == 0:
            return np.array([], dtype=np.float32)
        if current_length == target_length:
            return audio_data
        elif current_length < target_length:
            num_repeats = math.ceil(target_length / current_length)
            looped_audio = (
                np.tile(audio_data, (num_repeats, 1))
                if audio_data.ndim == 2
                else np.tile(audio_data, num_repeats)
            )
            return self._ensure_float32(looped_audio[:target_length])
        else:
            return audio_data[:target_length]

    # --- Dynamic Loading/Generation Methods ---
    def load_uploaded_audio(
        self, uploaded_file: st.runtime.uploaded_file_manager.UploadedFile
    ) -> AudioTuple:
        if uploaded_file is None:
            return None
        logger.info(f"Loading uploaded file: {uploaded_file.name}")
        audio_data: Optional[AudioData] = None
        try:
            audio_data, sr = librosa.load(uploaded_file, sr=GLOBAL_SR, mono=False)
            if audio_data.ndim == 2 and audio_data.shape[0] == 2:
                audio_data = audio_data.T
            elif audio_data.ndim == 1:
                audio_data = np.stack([audio_data] * 2, axis=-1)
            audio_data = self._ensure_float32(self._ensure_stereo(audio_data))
            logger.info(f"Loaded '{uploaded_file.name}'. Shape: {audio_data.shape}")
            return (audio_data, GLOBAL_SR)
        except Exception as e:
            logger.exception(f"Failed load: {e}")
            return None
        finally:
            if audio_data is not None:
                del audio_data
                gc.collect()

    def generate_noise_audio(
        self, noise_type: str, duration_seconds: float, sr: int
    ) -> AudioTuple:
        # Use NOISE_TYPES list from config
        if noise_type not in NOISE_TYPES or duration_seconds <= 0:
            return None
        logger.info(f"Generating {noise_type} for {duration_seconds:.2f}s at {sr} Hz.")
        num_samples = int(duration_seconds * sr)
        if num_samples <= 0:
            return None
        noise_audio: Optional[AudioData] = None
        try:
            beta = 0
            if noise_type == "Pink Noise":
                beta = 1
            elif noise_type == "Brown Noise":
                beta = 2
            mono_noise = colorednoise.powerlaw_psd_gaussian(beta, num_samples)
            noise_audio = np.stack([mono_noise] * 2, axis=-1).astype(np.float32)
            max_val = np.max(np.abs(noise_audio))
            if max_val > 0:
                noise_audio = noise_audio / max_val * 0.8
            noise_audio = self._ensure_float32(noise_audio)
            logger.info(f"Generated {noise_type}. Shape: {noise_audio.shape}")
            return (noise_audio, sr)
        except Exception as e:
            logger.exception(f"Failed generate noise: {e}")
            return None
        finally:
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
        if freq_choice == "None" or duration_seconds <= 0:
            return None
        logger.info(f"Generating {freq_choice} for {duration_seconds:.2f}s at {sr} Hz.")
        num_samples = int(duration_seconds * sr)
        if num_samples <= 0:
            return None
        t = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)
        audio_result: Optional[AudioData] = None
        try:
            base_freq = float(freq_params.get("base_freq", 100.0))
            if freq_choice == "Binaural Beats":
                beat_freq = float(freq_params.get("beat_freq", 5.0))
                if beat_freq <= 0:
                    raise ValueError("Beat frequency must be positive.")
                f_left = base_freq - beat_freq / 2.0
                f_right = base_freq + beat_freq / 2.0
                if f_left <= 0:
                    raise ValueError("Resulting left frequency must be positive.")
                left_channel = 0.5 * np.sin(2 * np.pi * f_left * t)
                right_channel = 0.5 * np.sin(2 * np.pi * f_right * t)
                audio_result = np.stack([left_channel, right_channel], axis=-1).astype(
                    np.float32
                )
            elif freq_choice == "Isochronic Tones":
                pulse_freq = float(freq_params.get("pulse_freq", 7.0))
                if pulse_freq <= 0:
                    raise ValueError("Pulse frequency must be positive.")
                tone = np.sin(2 * np.pi * base_freq * t)
                modulation = ((t * pulse_freq) % 1.0 < 0.5).astype(np.float32)
                mono_iso_tone = 0.8 * tone * modulation
                audio_result = np.stack([mono_iso_tone] * 2, axis=-1).astype(np.float32)
            else:
                logger.warning(f"Unknown frequency choice: {freq_choice}")
                return None
            if audio_result is not None:
                audio_result = self._ensure_float32(audio_result)
                logger.info(f"Generated {freq_choice}. Shape: {audio_result.shape}")
                return (audio_result, sr)
            else:
                return None
        except Exception as e:
            logger.exception(f"Failed generate freq: {e}")
            return None
        finally:
            if audio_result is not None:
                del audio_result
                gc.collect()

    # --- Track Preparation & Mixing Methods ---
    def _prepare_audio_track(
        self, audio_tuple: AudioTuple, target_length: int, track_name: str
    ) -> AudioTuple:
        if audio_tuple is None:
            logger.debug(f"Skipping {track_name}: No tuple.")
            return None
        audio_data, sr = audio_tuple
        if sr != GLOBAL_SR:
            logger.warning(f"Skipping {track_name}: SR mismatch ({sr} vs {GLOBAL_SR}).")
            return None
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            logger.warning(f"Skipping {track_name}: Invalid data.")
            return None
        audio_data = self._ensure_float32(audio_data)
        try:
            processed_audio = self._ensure_stereo(audio_data)
        except ValueError as e:
            logger.warning(f"Skipping {track_name}: Stereo error: {e}")
            return None
        processed_audio = self._loop_audio_to_length(processed_audio, target_length)
        if processed_audio.size == 0:
            logger.warning(f"Skipping {track_name}: Empty after loop.")
            return None
        logger.debug(
            f"{track_name} prepared. Len: {processed_audio.shape[0]}, dtype: {processed_audio.dtype}."
        )
        return (processed_audio, GLOBAL_SR)

    def generate_preview_mix(
        self,
        duration_seconds: int,
        affirmation_audio_tuple: AudioTuple,
        background_audio_tuple: AudioTuple,
        frequency_audio_tuple: AudioTuple,
        wizard_state: Dict[str, Any],
    ) -> AudioTuple:
        logger.info(f"Starting preview mix generation ({duration_seconds}s).")
        preview_mix = None
        processed_affirmation_audio: Optional[AudioData] = None
        try:
            # 1. Process Affirmations
            if (
                affirmation_audio_tuple is None
                or not isinstance(affirmation_audio_tuple[0], np.ndarray)
                or affirmation_audio_tuple[0].size == 0
            ):
                raise ValueError("Invalid affirmation audio.")
            affirmation_audio, affirmation_sr = affirmation_audio_tuple
            if affirmation_sr != GLOBAL_SR:
                raise ValueError(f"Affirmation SR incorrect ({affirmation_sr} Hz).")
            affirmation_audio = self._ensure_float32(
                self._ensure_stereo(affirmation_audio)
            )
            # Use constant key from config
            apply_speed = wizard_state.get(AFFIRM_APPLY_SPEED_KEY, DEFAULT_APPLY_SPEED)
            speed_factor = wizard_state.get(
                "wizard_speed_factor", 1.0
            )  # Keep getting factor from state
            logger.info(
                f"DEBUG PREVIEW - apply_speed from wizard_state: {apply_speed} (Type: {type(apply_speed)})"
            )
            if apply_speed is True:
                affirmation_audio = self._apply_speed_change(
                    affirmation_audio, affirmation_sr, speed_factor
                )
            elif apply_speed is None:
                logger.warning(
                    "DEBUG PREVIEW - 'wizard_apply_speed_change' key not found!"
                )
            processed_affirmation_audio = affirmation_audio

            # 2. Get Volumes using constant keys
            affirmation_volume: float = wizard_state.get(AFFIRMATION_VOLUME_KEY, 1.0)
            background_volume: float = wizard_state.get(BG_VOLUME_KEY, 0.7)
            frequency_volume: float = wizard_state.get(FREQ_VOLUME_KEY, 0.2)

            # 3. Determine Preview Length
            target_length_samples = int(duration_seconds * GLOBAL_SR)
            preview_length_samples = min(
                target_length_samples, processed_affirmation_audio.shape[0]
            )
            if preview_length_samples <= 0:
                raise ValueError("Preview length is zero.")
            logger.info(
                f"Preview target length: {preview_length_samples / GLOBAL_SR:.2f}s."
            )

            # 4. Prepare Tracks
            affirmation_preview_tuple = (
                processed_affirmation_audio[:preview_length_samples],
                GLOBAL_SR,
            )
            background_preview_tuple = self._prepare_audio_track(
                background_audio_tuple, preview_length_samples, "Background (Preview)"
            )
            frequency_preview_tuple = self._prepare_audio_track(
                frequency_audio_tuple, preview_length_samples, "Frequency (Preview)"
            )

            # 5. Mix Tracks
            logger.info("Mixing preview tracks...")
            preview_mix = mix_wizard_tracks(
                affirmation_audio=affirmation_preview_tuple,
                background_audio=background_preview_tuple,
                frequency_audio=frequency_preview_tuple,
                affirmation_volume=affirmation_volume,
                background_volume=background_volume,
                frequency_volume=frequency_volume,
                target_sr=GLOBAL_SR,
            )

            if preview_mix is None:
                raise ValueError("Preview mixing failed (mixer returned None).")
            if not isinstance(preview_mix, np.ndarray) or preview_mix.size == 0:
                raise ValueError("Preview mixing resulted in empty audio.")
            preview_mix = self._ensure_float32(preview_mix)
            logger.info(
                f"Preview mixing successful. Length: {len(preview_mix) / GLOBAL_SR:.2f}s."
            )
            return preview_mix, GLOBAL_SR
        except Exception as e_preview:
            logger.exception("Error generating preview mix.")
            raise e_preview
        finally:
            if processed_affirmation_audio is not None:
                del processed_affirmation_audio
                gc.collect()

    def process_and_export(
        self,
        affirmation_audio_tuple: AudioTuple,
        background_audio_tuple: AudioTuple,
        frequency_audio_tuple: AudioTuple,
        wizard_state: Dict[str, Any],
    ) -> Tuple[Optional[BytesIO], Optional[str]]:
        logger.info("Starting final audio processing and export.")
        export_buffer: Optional[BytesIO] = None
        error_message: Optional[str] = None
        full_mix: Optional[AudioData] = None
        processed_affirmation_audio: Optional[AudioData] = None
        try:
            # 1. Process Affirmations
            if (
                affirmation_audio_tuple is None
                or not isinstance(affirmation_audio_tuple[0], np.ndarray)
                or affirmation_audio_tuple[0].size == 0
            ):
                raise ValueError("Invalid affirmation audio.")
            affirmation_audio, affirmation_sr = affirmation_audio_tuple
            if affirmation_sr != GLOBAL_SR:
                raise ValueError(f"Affirmation SR incorrect ({affirmation_sr} Hz).")
            affirmation_audio = self._ensure_float32(
                self._ensure_stereo(affirmation_audio)
            )
            # Use constant key from config
            apply_speed = wizard_state.get(AFFIRM_APPLY_SPEED_KEY, DEFAULT_APPLY_SPEED)
            speed_factor = wizard_state.get("wizard_speed_factor", 1.0)
            logger.info(
                f"DEBUG EXPORT - apply_speed from wizard_state: {apply_speed} (Type: {type(apply_speed)})"
            )
            if apply_speed is True:
                affirmation_audio = self._apply_speed_change(
                    affirmation_audio, affirmation_sr, speed_factor
                )
            elif apply_speed is None:
                logger.warning(
                    "DEBUG EXPORT - 'wizard_apply_speed_change' key not found!"
                )
            processed_affirmation_audio = affirmation_audio

            # 2. Get Settings using constant keys
            affirmation_volume: float = wizard_state.get(AFFIRMATION_VOLUME_KEY, 1.0)
            background_volume: float = wizard_state.get(BG_VOLUME_KEY, 0.7)
            frequency_volume: float = wizard_state.get(FREQ_VOLUME_KEY, 0.2)
            export_format: str = wizard_state.get(
                "wizard_export_format", "WAV"
            ).lower()  # Keep using string key from state

            # 3. Determine Target Length
            target_length_samples = processed_affirmation_audio.shape[0]
            if target_length_samples == 0:
                raise ValueError("Affirmation audio zero length after processing.")
            logger.info(f"Target mix length: {target_length_samples / GLOBAL_SR:.2f}s.")

            # 4. Prepare Tracks
            affirmation_final_tuple: AudioTuple = (
                processed_affirmation_audio,
                GLOBAL_SR,
            )
            background_final_tuple = self._prepare_audio_track(
                background_audio_tuple, target_length_samples, "Background (Final)"
            )
            frequency_final_tuple = self._prepare_audio_track(
                frequency_audio_tuple, target_length_samples, "Frequency (Final)"
            )

            # 5. Mix Tracks
            logger.info("Mixing final tracks...")
            full_mix = mix_wizard_tracks(
                affirmation_audio=affirmation_final_tuple,
                background_audio=background_final_tuple,
                frequency_audio=frequency_final_tuple,
                affirmation_volume=affirmation_volume,
                background_volume=background_volume,
                frequency_volume=frequency_volume,
                target_sr=GLOBAL_SR,
            )

            # --- Cleanup intermediate ---
            del affirmation_final_tuple
            processed_affirmation_audio = None
            del background_final_tuple, frequency_final_tuple
            gc.collect()
            # ---

            if full_mix is None:
                raise ValueError("Final mixing failed (mixer returned None).")
            if not isinstance(full_mix, np.ndarray) or full_mix.size == 0:
                raise ValueError("Final mixing resulted in empty audio.")
            full_mix = self._ensure_float32(full_mix)
            logger.info(
                f"Final mixing successful. Length: {len(full_mix) / GLOBAL_SR:.2f}s."
            )

            # 6. Export
            logger.info(f"Exporting final mix as {export_format.upper()}...")
            # [ ... Same WAV/MP3 export logic ... ]
            if export_format == "wav":
                export_buffer = save_audio_to_bytesio(full_mix, GLOBAL_SR)
                if not export_buffer or export_buffer.getbuffer().nbytes == 0:
                    error_message = "Failed to save WAV (empty buffer)."
                    export_buffer = None
                else:
                    logger.info("WAV buffer generated successfully.")
            elif export_format == "mp3":
                if not PYDUB_AVAILABLE:
                    error_message = "MP3 export requires 'pydub' and 'ffmpeg'."
                else:
                    try:
                        logger.info("Converting final mix (float32) to MP3...")
                        full_mix_clipped = np.clip(full_mix, -1.0, 1.0)
                        audio_int16 = (full_mix_clipped * 32767).astype(np.int16)
                        del full_mix_clipped, full_mix
                        full_mix = None
                        gc.collect()
                        channels = audio_int16.shape[1] if audio_int16.ndim == 2 else 1
                        if channels != 2:
                            logger.warning(
                                f"Forcing stereo for MP3 (was {channels} channels)."
                            )
                            if audio_int16.ndim == 1:
                                audio_int16 = np.stack([audio_int16] * 2, axis=-1)
                            elif audio_int16.shape[1] == 1:
                                audio_int16 = np.concatenate(
                                    [audio_int16, audio_int16], axis=1
                                )
                            channels = 2
                        if channels == 2:
                            segment = AudioSegment(
                                data=audio_int16.tobytes(),
                                sample_width=audio_int16.dtype.itemsize,
                                frame_rate=GLOBAL_SR,
                                channels=channels,
                            )
                            del audio_int16
                            gc.collect()
                            mp3_buffer = BytesIO()
                            segment.export(mp3_buffer, format="mp3", bitrate="192k")
                            mp3_buffer.seek(0)
                            del segment
                            gc.collect()
                            if mp3_buffer.getbuffer().nbytes > 0:
                                logger.info("MP3 buffer generated.")
                                export_buffer = mp3_buffer
                            else:
                                error_message = "MP3 export failed (empty buffer)."
                        else:
                            error_message = (
                                f"MP3 export failed (could not ensure stereo)."
                            )
                            del audio_int16
                    except Exception as e_mp3:
                        logger.exception("Error during MP3 conversion/export.")
                        error_message = f"MP3 Export Failed: {e_mp3}"
                        if "full_mix" in locals() and full_mix is not None:
                            del full_mix
                        if "audio_int16" in locals():
                            del audio_int16
                            gc.collect()
            else:
                error_message = f"Unsupported export format: '{export_format}'."

        except Exception as e_export:
            logger.exception("Unhandled error during final processing/export.")
            error_message = f"Unexpected Processing Error: {e_export}"
            export_buffer = None
            if (
                "processed_affirmation_audio" in locals()
                and processed_affirmation_audio is not None
            ):
                del processed_affirmation_audio
            if "full_mix" in locals() and full_mix is not None:
                del full_mix
                gc.collect()

        # Final check
        if export_buffer and not error_message:
            logger.info("Audio processing and export completed successfully.")
            return export_buffer, None
        else:
            if not error_message:
                error_message = "Export failed (unknown reason)."
            logger.error(f"Export failed: {error_message}")
            if "full_mix" in locals() and full_mix is not None:
                del full_mix
                gc.collect()
            return None, error_message
