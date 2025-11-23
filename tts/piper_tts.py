# tts/piper_tts.py
# ==========================================
# Piper TTS Generator Implementation
# MODIFIED: Added max_duration_s parameter to generate() for efficient snippet creation.
# ==========================================

import gc
import logging
import os
import textwrap
import time
import wave
from io import BytesIO

# --- MODIFIED: Added Optional ---
from typing import Optional, Tuple

import librosa
import numpy as np
import streamlit as st
from piper.voice import PiperVoice

from config import (
    GLOBAL_SR,
    PIPER_VOICE_CONFIG_PATH,
    PIPER_VOICE_MODEL_PATH,
    TTS_CHUNK_SIZE,
)
from tts.base_tts import AudioData, BaseTTSGenerator, SampleRate

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading TTS Model...")
def _load_piper_voice(model_path: str, config_path: str) -> PiperVoice:
    """Loads the PiperVoice model and caches it."""
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
    TTS Generator using the Piper TTS engine.
    Handles chunking, synthesis, resampling, and optional duration limiting.
    """

    def __init__(
        self,
        model_path: str = PIPER_VOICE_MODEL_PATH,
        config_path: str = PIPER_VOICE_CONFIG_PATH,
        chunk_size: int = TTS_CHUNK_SIZE,
        target_sr: int = GLOBAL_SR,
    ):
        """Initializes the PiperTTSGenerator."""
        logger.info(f"Initializing PiperTTSGenerator...")
        self.chunk_size = chunk_size
        self.target_sr = target_sr
        self.voice = _load_piper_voice(model_path, config_path)
        self.model_native_sr = getattr(
            getattr(self.voice, "config", None), "sample_rate", 22050
        )
        logger.info(f"Piper model native sample rate: {self.model_native_sr} Hz")
        if self.model_native_sr != self.target_sr:
            logger.info(
                f"Resampling from {self.model_native_sr}Hz to {self.target_sr}Hz will be required."
            )

# --- MODIFIED: Cached Synthesis Function ---
@st.cache_data(show_spinner=False)
def _cached_piper_synthesize(
    _voice: PiperVoice, text_chunk: str, target_sr: int, model_native_sr: int
) -> Tuple[AudioData, SampleRate]:
    """
    Synthesizes a single text chunk using the provided voice model.
    Cached based on text chunk, target SR, and model native SR.
    Note: _voice is unhashable, so we prefix with _ to exclude from cache hashing.
          Streamlit will rely on the other arguments for cache invalidation.
          This assumes the voice model instance doesn't change for the same text/sr.
    """
    if not _voice:
        return np.zeros((0, 2), dtype=np.float32), target_sr

    audio_buffer = BytesIO()
    try:
        with wave.open(audio_buffer, "wb") as wav_file:
            _voice.synthesize(text_chunk, wav_file)
        wav_bytes = audio_buffer.getvalue()
    except Exception as e:
        logger.exception(f"Piper synthesis failed for chunk: '{text_chunk[:50]}...'")
        return np.zeros((0, 2), dtype=np.float32), target_sr
    finally:
        audio_buffer.close()

    if not wav_bytes:
        return np.zeros((0, 2), dtype=np.float32), target_sr

    # Process WAV bytes (logic extracted from original _process_wav_bytes)
    try:
        with BytesIO(wav_bytes) as bio, wave.open(bio, "rb") as wf:
            n_channels, sampwidth, framerate, n_frames = wf.getparams()[:4]
            if n_frames == 0:
                return np.zeros((0, 2), dtype=np.float32), framerate
            audio_frames = wf.readframes(n_frames)

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        if sampwidth not in dtype_map:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        np_dtype = dtype_map[sampwidth]
        audio_int = np.frombuffer(audio_frames, dtype=np_dtype)
        if n_channels > 1:
            audio_int = audio_int.reshape(-1, n_channels)

        # Convert to float32
        norm_factor = {
            np.int8: 128.0,
            np.int16: 32768.0,
            np.int32: 2147483648.0,
        }.get(np_dtype, 1.0)
        audio_float = audio_int.astype(np.float32) / norm_factor

        # Ensure stereo float32
        if audio_float.ndim == 1:
            audio_stereo = np.stack([audio_float, audio_float], axis=-1)
        elif audio_float.shape[1] == 1:
            audio_stereo = np.concatenate([audio_float, audio_float], axis=1)
        elif audio_float.shape[1] == 2:
            audio_stereo = audio_float
        else:  # More than 2 channels
            audio_stereo = audio_float[:, :2]

        chunk_audio_native = audio_stereo.astype(np.float32)
        actual_native_sr = framerate

    except Exception as e:
        logger.exception("Failed to process WAV bytes in cached function.")
        return np.zeros((0, 2), dtype=np.float32), target_sr

    # Resample if necessary
    if actual_native_sr != target_sr:
        try:
            chunk_audio_native_float = chunk_audio_native.astype(np.float32)
            resampled_audio = librosa.resample(
                chunk_audio_native_float.T,
                orig_sr=actual_native_sr,
                target_sr=target_sr,
            ).T
            return resampled_audio.astype(np.float32), target_sr
        except Exception as e_resample:
            logger.exception("Failed to resample cached chunk.")
            return np.zeros((0, 2), dtype=np.float32), target_sr
    else:
        return chunk_audio_native, actual_native_sr


class PiperTTSGenerator(BaseTTSGenerator):
    """
    TTS Generator using the Piper TTS engine.
    Handles chunking, synthesis, resampling, and optional duration limiting.
    """

    def __init__(
        self,
        model_path: str = PIPER_VOICE_MODEL_PATH,
        config_path: str = PIPER_VOICE_CONFIG_PATH,
        chunk_size: int = TTS_CHUNK_SIZE,
        target_sr: int = GLOBAL_SR,
    ):
        """Initializes the PiperTTSGenerator."""
        logger.info(f"Initializing PiperTTSGenerator...")
        self.chunk_size = chunk_size
        self.target_sr = target_sr
        self.voice = _load_piper_voice(model_path, config_path)
        self.model_native_sr = getattr(
            getattr(self.voice, "config", None), "sample_rate", 22050
        )
        logger.info(f"Piper model native sample rate: {self.model_native_sr} Hz")
        if self.model_native_sr != self.target_sr:
            logger.info(
                f"Resampling from {self.model_native_sr}Hz to {self.target_sr}Hz will be required."
            )

    # _synthesize_chunk and _process_wav_bytes are replaced by _cached_piper_synthesize
    # We can remove them or keep them if needed for non-cached fallback, but for now we use the cached one.

    # --- MODIFIED: Added max_duration_s parameter ---
    def generate(
        self, text: str, max_duration_s: Optional[float] = None
    ) -> Tuple[AudioData, SampleRate]:
        """
        Generates audio from text using Piper TTS.

        Args:
            text: The text to synthesize.
            max_duration_s: If provided, stop generation once audio exceeds this duration (in seconds).

        Returns:
            A tuple containing the audio data (NumPy array) and the sample rate.
        """
        logger.info(
            f"[PiperTTS] Starting TTS generation (Max duration: {max_duration_s}s) for text length: {len(text)}."
        )
        if not text or not text.strip():
            logger.warning("[PiperTTS] Empty text provided. Returning empty audio.")
            return np.zeros((0, 2), dtype=np.float32), self.target_sr

        processed_chunks = []
        total_processed_duration_s = 0.0
        synthesis_start_time = time.time()

        try:
            chunks = textwrap.wrap(
                text,
                self.chunk_size,
                break_long_words=True,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            num_chunks = len(chunks)
            logger.info(f"[PiperTTS] Split text into {num_chunks} chunks.")

            for i, chunk_text in enumerate(chunks):
                chunk_index = i + 1
                if not chunk_text.strip():
                    continue

                logger.info(
                    f"[PiperTTS] Synthesizing chunk {chunk_index}/{num_chunks}..."
                )
                chunk_start_time = time.time()

                # --- USE CACHED FUNCTION ---
                chunk_audio_processed, chunk_sr = _cached_piper_synthesize(
                    self.voice, chunk_text, self.target_sr, self.model_native_sr
                )
                # ---------------------------

                if chunk_audio_processed is not None and chunk_audio_processed.size > 0:
                    processed_chunks.append(chunk_audio_processed)
                    chunk_duration_s = (
                        chunk_audio_processed.shape[0] / chunk_sr
                    )
                    total_processed_duration_s += chunk_duration_s
                    logger.debug(
                        f"Chunk {chunk_index} duration: {chunk_duration_s:.2f}s. Total duration: {total_processed_duration_s:.2f}s"
                    )

                    # --- Check duration limit ---
                    if (
                        max_duration_s is not None
                        and total_processed_duration_s >= max_duration_s
                    ):
                        logger.info(
                            f"[PiperTTS] Reached max duration ({max_duration_s:.2f}s) at chunk {chunk_index}. Stopping generation."
                        )
                        break  # Stop processing more chunks

                else:
                    logger.warning(
                        f"Chunk {chunk_index} resulted in empty audio after processing/resampling."
                    )

                chunk_end_time = time.time()
                logger.info(
                    f"[PiperTTS] Processed chunk {chunk_index} in {chunk_end_time - chunk_start_time:.2f}s."
                )
                # gc.collect() # Optional

            synthesis_end_time = time.time()
            logger.info(
                f"[PiperTTS] Finished processing chunks in {synthesis_end_time - synthesis_start_time:.2f}s."
            )

            if not processed_chunks:
                logger.error(
                    "[PiperTTS] TTS process resulted in no valid audio chunks."
                )
                return np.zeros((0, 2), dtype=np.float32), self.target_sr

            logger.info(
                f"Concatenating {len(processed_chunks)} processed audio chunks..."
            )
            final_audio = np.concatenate(processed_chunks, axis=0).astype(np.float32)
            del processed_chunks
            gc.collect()

            # --- Trim if duration limit was specified and slightly exceeded ---
            if max_duration_s is not None and final_audio.size > 0:
                max_samples = int(max_duration_s * self.target_sr)
                if final_audio.shape[0] > max_samples:
                    logger.info(
                        f"Trimming final audio from {final_audio.shape[0]} samples to {max_samples} samples to meet max duration."
                    )
                    final_audio = final_audio[:max_samples, :]

            if final_audio.size == 0:
                logger.error("[PiperTTS] TTS process resulted in empty final audio.")
                return np.zeros((0, 2), dtype=np.float32), self.target_sr

            logger.info(
                f"[PiperTTS] TTS generation complete. Final audio shape: {final_audio.shape}, SR: {self.target_sr}Hz"
            )
            return final_audio, self.target_sr

        except Exception as e:
            logger.exception(
                "[PiperTTS] An error occurred during the TTS generation process."
            )
            del processed_chunks  # Ensure cleanup
            gc.collect()
            return np.zeros((0, 2), dtype=np.float32), self.target_sr
