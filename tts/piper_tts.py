# tts/piper_tts.py
# ==========================================
# Piper TTS Generator Implementation
# Removed @st.cache_data from generate() to reduce memory usage from cache.
# ==========================================

import gc  # Garbage collector
import logging
import os
import textwrap
import time
import wave  # For reading/writing WAV from buffer
from io import BytesIO
from typing import Tuple

import librosa  # For resampling
import numpy as np
import streamlit as st  # Import Streamlit for caching decorators

# Import Piper TTS library
from piper.voice import PiperVoice

# Import from local modules
from config import (
    GLOBAL_SR,
    PIPER_VOICE_CONFIG_PATH,
    PIPER_VOICE_MODEL_PATH,
    TTS_CHUNK_SIZE,
)
from tts.base_tts import AudioData, BaseTTSGenerator, SampleRate

logger = logging.getLogger(__name__)


# --- Caching Function for Model Loading ---
# This remains cached as loading the model is expensive but happens once.
@st.cache_resource(show_spinner="Loading TTS Model...")
def _load_piper_voice(model_path: str, config_path: str) -> PiperVoice:
    """
    Loads the PiperVoice model and caches it.
    """
    logger.info(f"Attempting to load Piper voice model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Piper model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Piper config file not found: {config_path}")

    try:
        start_time = time.time()
        voice = PiperVoice.load(model_path, config_path=config_path)
        end_time = time.time()
        logger.info(
            f"Piper voice model loaded successfully in {end_time - start_time:.2f}s."
        )
        return voice
    except Exception as e:
        logger.exception("Failed to load Piper voice model.")
        raise RuntimeError(f"Failed to load Piper voice model: {e}") from e


class PiperTTSGenerator(BaseTTSGenerator):
    """
    TTS Generator using the Piper TTS engine (via piper-tts library).
    Handles chunking, synthesis, resampling, and incremental concatenation.
    Uses Streamlit caching ONLY for model loading. Generation is NOT cached.
    """

    def __init__(
        self,
        model_path: str = PIPER_VOICE_MODEL_PATH,
        config_path: str = PIPER_VOICE_CONFIG_PATH,
        chunk_size: int = TTS_CHUNK_SIZE,
        target_sr: int = GLOBAL_SR,
    ):
        """
        Initializes the PiperTTSGenerator, loading the voice model via a cached function.
        """
        logger.info(f"Initializing PiperTTSGenerator...")
        logger.info(f"Model Path: {model_path}")
        logger.info(f"Config Path: {config_path}")

        self.chunk_size = chunk_size
        self.target_sr = target_sr
        self.voice = _load_piper_voice(model_path, config_path)  # Uses cached function

        self.model_native_sr = None
        if (
            hasattr(self.voice, "config")
            and self.voice.config
            and hasattr(self.voice.config, "sample_rate")
        ):
            self.model_native_sr = self.voice.config.sample_rate
            logger.info(f"Piper model native sample rate: {self.model_native_sr} Hz")
        else:
            self.model_native_sr = 22050  # Fallback
            logger.warning(
                f"Could not automatically determine Piper model's native sample rate. Assuming {self.model_native_sr} Hz."
            )
        # Check if resampling will be needed
        if self.model_native_sr != self.target_sr:
            logger.info(
                f"Resampling from {self.model_native_sr}Hz to {self.target_sr}Hz will be required."
            )

    # Internal helper (not cached)
    def _synthesize_chunk(self, text_chunk: str) -> bytes:
        """Synthesizes a single text chunk into WAV bytes."""
        if not self.voice:
            raise RuntimeError("Piper voice model is not loaded.")
        audio_buffer = BytesIO()
        try:
            # Synthesize directly into the buffer
            with wave.open(audio_buffer, "wb") as wav_file:
                self.voice.synthesize(text_chunk, wav_file)
            audio_bytes = audio_buffer.getvalue()
            return audio_bytes if audio_bytes else b""
        except Exception as e:
            logger.exception(
                f"Piper synthesis/wave handling failed for chunk: '{text_chunk[:50]}...'"
            )
            # Don't raise here, allow process to continue if possible, return empty bytes
            return b""
        finally:
            audio_buffer.close()  # Ensure buffer is closed

    # Internal helper (not cached)
    def _process_wav_bytes(self, wav_bytes: bytes) -> Tuple[AudioData, SampleRate]:
        """Processes raw WAV bytes: reads, converts to float32, ensures stereo."""
        if not wav_bytes:
            # Return empty array and native SR if input bytes are empty
            return np.zeros((0, 2), dtype=np.float32), self.model_native_sr

        audio_int: Optional[np.ndarray] = None
        audio_float: Optional[np.ndarray] = None
        audio_stereo: Optional[np.ndarray] = None

        try:
            with BytesIO(wav_bytes) as bio:
                with wave.open(bio, "rb") as wf:
                    n_channels, sampwidth, framerate, n_frames = wf.getparams()[:4]
                    if n_frames == 0:  # Handle empty WAV file case
                        logger.warning("Processing empty WAV chunk.")
                        return np.zeros((0, 2), dtype=np.float32), framerate
                    audio_frames = wf.readframes(n_frames)

            logger.debug(
                f"Processing WAV: Channels={n_channels}, Rate={framerate}, Width={sampwidth}, Frames={n_frames}"
            )
            # Use self here as normal
            if framerate != self.model_native_sr:
                logger.warning(
                    f"WAV header SR ({framerate}) differs from expected model SR ({self.model_native_sr}). Using header rate."
                )
                # If the header rate is reliable, maybe update the native_sr for this chunk? Risky. Stick to model SR.

            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            if sampwidth not in dtype_map:
                raise ValueError(f"Unsupported sample width: {sampwidth}")
            np_dtype = dtype_map[sampwidth]

            audio_int = np.frombuffer(audio_frames, dtype=np_dtype)
            if n_channels > 1:
                # Ensure reshape happens correctly even if n_frames is small
                audio_int = audio_int.reshape(-1, n_channels)

            # Convert to float32 directly
            if np_dtype == np.int8:
                audio_float = audio_int.astype(np.float32) / 128.0
            elif np_dtype == np.int16:
                audio_float = audio_int.astype(np.float32) / 32768.0
            elif np_dtype == np.int32:
                audio_float = audio_int.astype(np.float32) / 2147483648.0
            else:
                # This case should not be reached due to sampwidth check
                raise ValueError(
                    f"Cannot convert unsupported dtype {np_dtype} to float."
                )

            # Ensure stereo float32
            if audio_float.ndim == 1:
                # Stack creates a new array (N, 2)
                audio_stereo = np.stack([audio_float, audio_float], axis=-1)
            elif audio_float.shape[1] == 1:
                # Concatenate creates a new array (N, 2)
                audio_stereo = np.concatenate([audio_float, audio_float], axis=1)
            elif audio_float.shape[1] == 2:
                # Already stereo, ensure it's float32 (should be from conversion)
                audio_stereo = (
                    audio_float
                    if audio_float.dtype == np.float32
                    else audio_float.astype(np.float32)
                )
            else:
                logger.warning(
                    f"Piper output had {audio_float.shape[1]} channels. Using first two."
                )
                # Slicing creates a view, ensure float32
                audio_stereo = audio_float[:, :2].astype(np.float32)

            # Final check for float32 (redundant but safe)
            if audio_stereo.dtype != np.float32:
                audio_stereo = audio_stereo.astype(np.float32)

            return audio_stereo, framerate  # Return the actual framerate from header

        except Exception as e:
            logger.exception("Failed to process WAV bytes.")
            # Return empty array on error
            return np.zeros((0, 2), dtype=np.float32), self.model_native_sr
        finally:
            # Clean up intermediate arrays
            del audio_int, audio_float, audio_stereo
            # gc.collect() # Might be too frequent here, rely on Python's GC

    # --- Generation function - NO LONGER CACHED ---
    def generate(self, text: str) -> Tuple[AudioData, SampleRate]:
        """
        Generates audio from text using Piper TTS.
        This operation is NOT cached to save memory.
        """
        logger.info(
            f"[PiperTTS] Starting NON-CACHED TTS generation for text length: {len(text)} characters."
        )
        if not text or not text.strip():
            logger.warning(
                "[PiperTTS] Empty or whitespace-only text provided. Returning empty audio."
            )
            return np.zeros((0, 2), dtype=np.float32), self.target_sr

        # Use a list to collect chunks, then concatenate once at the end
        # This can be more memory-efficient than repeated concatenation
        processed_chunks = []
        total_frames = 0
        synthesis_start_time = time.time()

        try:
            chunks = textwrap.wrap(
                text,
                self.chunk_size,
                break_long_words=True,
                replace_whitespace=False,  # Keep whitespace for Piper
                drop_whitespace=False,  # Keep whitespace for Piper
            )
            num_chunks = len(chunks)
            logger.info(f"[PiperTTS] Split text into {num_chunks} chunks.")

            for i, chunk_text in enumerate(chunks):
                chunk_index = i + 1
                # Piper might handle empty strings okay, but skip just in case
                if not chunk_text.strip():
                    logger.debug(f"Skipping empty chunk {chunk_index}/{num_chunks}")
                    continue

                logger.info(
                    f"[PiperTTS] Synthesizing chunk {chunk_index}/{num_chunks}..."
                )
                chunk_start_time = time.time()

                wav_bytes = self._synthesize_chunk(chunk_text)
                if not wav_bytes:
                    logger.warning(
                        f"Synthesize chunk {chunk_index} returned empty bytes."
                    )
                    continue  # Skip processing if synthesis failed

                # Process bytes into float32 stereo numpy array
                chunk_audio_native, native_sr = self._process_wav_bytes(wav_bytes)
                del wav_bytes  # Free memory from raw bytes
                gc.collect()

                if chunk_audio_native.size == 0:
                    logger.warning(
                        f"Processing WAV bytes for chunk {chunk_index} resulted in empty audio."
                    )
                    continue  # Skip if processing failed

                # Use the actual native SR reported by the WAV header for resampling checks
                actual_native_sr = native_sr
                if actual_native_sr != self.model_native_sr:
                    logger.warning(
                        f"Chunk {chunk_index} actual native SR {actual_native_sr} differs from expected model SR {self.model_native_sr}."
                    )

                # --- Resample if necessary ---
                chunk_audio_processed: Optional[np.ndarray] = None
                if actual_native_sr != self.target_sr:
                    logger.info(
                        f"[PiperTTS] Resampling chunk {chunk_index} from {actual_native_sr} Hz to {self.target_sr} Hz."
                    )
                    try:
                        # Ensure input to resample is float32 (should be from _process_wav_bytes)
                        chunk_audio_native_float = chunk_audio_native.astype(np.float32)
                        # librosa expects (channels, samples) or (samples,) for mono
                        # Our data is (samples, channels=2)
                        resampled_audio = librosa.resample(
                            chunk_audio_native_float.T,  # Transpose to (channels, samples)
                            orig_sr=actual_native_sr,
                            target_sr=self.target_sr,
                        ).T  # Transpose back to (samples, channels)
                        chunk_audio_processed = resampled_audio.astype(
                            np.float32
                        )  # Ensure float32 output
                        del chunk_audio_native  # Free memory of original sample rate audio
                        gc.collect()
                    except Exception as e_resample:
                        logger.exception(
                            f"Failed to resample chunk {chunk_index}. Skipping chunk."
                        )
                        del chunk_audio_native  # Clean up even if resampling failed
                        gc.collect()
                        continue  # Skip this chunk if resampling fails
                else:
                    # No resampling needed, ensure it's float32
                    chunk_audio_processed = chunk_audio_native.astype(np.float32)
                # --- End Resampling ---

                if chunk_audio_processed is not None and chunk_audio_processed.size > 0:
                    processed_chunks.append(chunk_audio_processed)
                    total_frames += chunk_audio_processed.shape[0]
                else:
                    logger.warning(
                        f"Chunk {chunk_index} resulted in empty audio after processing/resampling."
                    )

                chunk_end_time = time.time()
                logger.info(
                    f"[PiperTTS] Processed chunk {chunk_index} in {chunk_end_time - chunk_start_time:.2f}s."
                )
                # Explicitly delete loop variable reference (minor effect)
                del chunk_audio_processed
                gc.collect()  # Collect garbage more frequently during loop

            synthesis_end_time = time.time()
            logger.info(
                f"[PiperTTS] Finished processing all {num_chunks} chunks in {synthesis_end_time - synthesis_start_time:.2f}s."
            )

            # --- Concatenate all processed chunks at the end ---
            if not processed_chunks:
                logger.error(
                    "[PiperTTS] TTS process resulted in no valid audio chunks."
                )
                return np.zeros((0, 2), dtype=np.float32), self.target_sr

            logger.info(
                f"Concatenating {len(processed_chunks)} processed audio chunks..."
            )
            final_audio = np.concatenate(processed_chunks, axis=0).astype(np.float32)
            del processed_chunks  # Free memory from the list of arrays
            gc.collect()
            # ---

            if final_audio.size == 0:
                # This case should ideally be caught by the check on processed_chunks list
                logger.error(
                    "[PiperTTS] TTS process resulted in empty final audio after concatenation."
                )
                return np.zeros((0, 2), dtype=np.float32), self.target_sr

            logger.info(
                f"[PiperTTS] TTS generation complete. Final audio shape: {final_audio.shape}, SR: {self.target_sr}Hz"
            )
            return final_audio, self.target_sr

        except Exception as e:
            logger.exception(
                "[PiperTTS] An error occurred during the TTS generation process."
            )
            # Clean up list if error occurred mid-loop
            del processed_chunks
            gc.collect()
            # Return empty array on error
            return np.zeros((0, 2), dtype=np.float32), self.target_sr
