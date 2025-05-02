# tts/piper_tts.py
# ==========================================
# Piper TTS Generator Implementation (with Caching Fix - Underscore Prefix)
# ==========================================

import logging
import os
import textwrap
import time
import wave  # For reading/writing WAV from buffer
from io import BytesIO
from typing import Tuple

import librosa  # For resampling
import numpy as np
import soundfile as sf
import streamlit as st  # Import Streamlit for caching decorators

# Import Piper TTS library
from piper.voice import PiperVoice

# Import from local modules
from config import GLOBAL_SR, PIPER_VOICE_CONFIG_PATH, PIPER_VOICE_MODEL_PATH, TTS_CHUNK_SIZE
from tts.base_tts import AudioData, BaseTTSGenerator, SampleRate

logger = logging.getLogger(__name__)


# --- Caching Function for Model Loading ---
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
        logger.info(f"Piper voice model loaded successfully in {end_time - start_time:.2f}s.")
        return voice
    except Exception as e:
        logger.exception("Failed to load Piper voice model.")
        raise RuntimeError(f"Failed to load Piper voice model: {e}") from e


class PiperTTSGenerator(BaseTTSGenerator):
    """
    TTS Generator using the Piper TTS engine (via piper-tts library).
    Handles chunking, synthesis, resampling, and incremental concatenation.
    Uses Streamlit caching for model loading and generation results.
    """

    def __init__(self, model_path: str = PIPER_VOICE_MODEL_PATH, config_path: str = PIPER_VOICE_CONFIG_PATH, chunk_size: int = TTS_CHUNK_SIZE, target_sr: int = GLOBAL_SR):
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
        if hasattr(self.voice, "config") and self.voice.config and hasattr(self.voice.config, "sample_rate"):
            self.model_native_sr = self.voice.config.sample_rate
            logger.info(f"Piper model native sample rate: {self.model_native_sr} Hz")
        else:
            self.model_native_sr = 22050  # Fallback
            logger.warning(f"Could not automatically determine Piper model's native sample rate. Assuming {self.model_native_sr} Hz.")

    # Internal helper (not cached directly)
    def _synthesize_chunk(self, text_chunk: str) -> bytes:
        """Synthesizes a single text chunk into WAV bytes."""
        # Use self here as normal, the underscore is only needed in the cached method signature
        if not self.voice:
            raise RuntimeError("Piper voice model is not loaded.")
        audio_buffer = BytesIO()
        try:
            with wave.open(audio_buffer, "wb") as wav_file:
                self.voice.synthesize(text_chunk, wav_file)
            audio_bytes = audio_buffer.getvalue()
            return audio_bytes if audio_bytes else b""
        except Exception as e:
            logger.exception(f"Piper synthesis/wave handling failed for chunk: '{text_chunk[:50]}...'")
            raise RuntimeError(f"Piper synthesis failed: {e}") from e
        finally:
            audio_buffer.close()

    # Internal helper (not cached directly)
    def _process_wav_bytes(self, wav_bytes: bytes) -> Tuple[AudioData, SampleRate]:
        """Processes raw WAV bytes: reads, converts to float32, ensures stereo."""
        if not wav_bytes:
            raise ValueError("Received empty WAV bytes for processing.")
        try:
            with BytesIO(wav_bytes) as bio:
                with wave.open(bio, "rb") as wf:
                    n_channels, sampwidth, framerate, n_frames = wf.getparams()[:4]
                    audio_frames = wf.readframes(n_frames)

            logger.debug(f"Processing WAV: Channels={n_channels}, Rate={framerate}, Width={sampwidth}, Frames={n_frames}")
            # Use self here as normal
            if framerate != self.model_native_sr:
                logger.warning(f"WAV header SR ({framerate}) differs from expected model SR ({self.model_native_sr}). Using header rate.")

            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            if sampwidth not in dtype_map:
                raise ValueError(f"Unsupported sample width: {sampwidth}")
            np_dtype = dtype_map[sampwidth]
            audio_int = np.frombuffer(audio_frames, dtype=np_dtype)
            if n_channels > 1:
                audio_int = audio_int.reshape(-1, n_channels) if n_frames > 0 else np.array([], dtype=np_dtype).reshape(0, n_channels)

            # Convert to float
            if np_dtype == np.int8:
                audio_float = audio_int.astype(np.float32) / 128.0
            elif np_dtype == np.int16:
                audio_float = audio_int.astype(np.float32) / 32768.0
            elif np_dtype == np.int32:
                audio_float = audio_int.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Cannot convert unsupported dtype {np_dtype} to float.")

            # Ensure stereo
            if audio_float.ndim == 1:
                audio_stereo = np.stack([audio_float, audio_float], axis=-1)
            elif audio_float.shape[1] == 1:
                audio_stereo = np.concatenate([audio_float, audio_float], axis=1)
            elif audio_float.shape[1] == 2:
                audio_stereo = audio_float
            else:
                logger.warning(f"Piper output had {audio_float.shape[1]} channels. Using first two.")
                audio_stereo = audio_float[:, :2]

            return audio_stereo.astype(np.float32), framerate
        except Exception as e:
            logger.exception("Failed to process WAV bytes.")
            raise ValueError(f"Error processing synthesized WAV data: {e}") from e

    # --- Cache the main generation function ---
    # --- MODIFIED DECORATOR: Removed hash_funcs ---
    # --- MODIFIED SIGNATURE: Renamed self to _self ---
    @st.cache_data(show_spinner="Generating TTS audio...")
    def generate(_self, text: str) -> Tuple[AudioData, SampleRate]:
        """
        Generates audio from text using Piper TTS.
        Results are cached based ONLY on the input text (ignores the instance via _self).
        """
        # --- END MODIFICATIONS ---

        # Inside the method, we still refer to the instance using _self
        logger.info(f"[PiperTTS] Starting TTS generation for text length: {len(text)} characters.")
        if not text or not text.strip():
            logger.warning("[PiperTTS] Empty or whitespace-only text provided. Returning empty audio.")
            return np.zeros((0, 2), dtype=np.float32), _self.target_sr  # Use _self

        final_audio = np.array([], dtype=np.float32).reshape(0, 2)
        try:
            # Use _self to access instance attributes/methods
            chunks = textwrap.wrap(text, _self.chunk_size, break_long_words=True, replace_whitespace=False, drop_whitespace=True)
            num_chunks = len(chunks)
            logger.info(f"[PiperTTS] Split text into {num_chunks} chunks.")

            synthesis_start_time = time.time()
            for i, chunk_text in enumerate(chunks):
                chunk_index = i + 1
                if not chunk_text.strip():
                    continue

                logger.info(f"[PiperTTS] Synthesizing chunk {chunk_index}/{num_chunks}...")
                chunk_start_time = time.time()
                # Call internal methods using _self
                wav_bytes = _self._synthesize_chunk(chunk_text)
                if not wav_bytes:
                    continue

                chunk_audio_native, native_sr = _self._process_wav_bytes(wav_bytes)
                # Access instance attributes using _self
                if native_sr != _self.model_native_sr:
                    logger.warning(f"Chunk {chunk_index} native SR {native_sr} differs from model SR {_self.model_native_sr}.")

                # Resample if necessary
                if native_sr != _self.target_sr:
                    logger.info(f"[PiperTTS] Resampling chunk {chunk_index} from {native_sr} Hz to {_self.target_sr} Hz.")
                    if chunk_audio_native.size > 0:
                        try:
                            chunk_audio_native_float = chunk_audio_native.astype(np.float32)
                            resampled_audio = librosa.resample(chunk_audio_native_float.T, orig_sr=native_sr, target_sr=_self.target_sr).T
                            chunk_audio_processed = resampled_audio
                        except Exception as e_resample:
                            logger.exception(f"Failed to resample chunk {chunk_index}.")
                            raise RuntimeError(f"Resampling failed: {e_resample}") from e_resample
                    else:
                        chunk_audio_processed = chunk_audio_native  # empty
                else:
                    chunk_audio_processed = chunk_audio_native

                # Concatenate
                final_audio = np.concatenate((final_audio, chunk_audio_processed), axis=0)
                chunk_end_time = time.time()
                logger.info(f"[PiperTTS] Processed chunk {chunk_index} in {chunk_end_time - chunk_start_time:.2f}s.")

            synthesis_end_time = time.time()
            logger.info(f"[PiperTTS] Finished processing all {num_chunks} chunks in {synthesis_end_time - synthesis_start_time:.2f}s.")

            if final_audio.size == 0:
                logger.error("[PiperTTS] TTS process resulted in empty final audio.")
                return np.zeros((0, 2), dtype=np.float32), _self.target_sr  # Use _self

            logger.info(f"[PiperTTS] TTS generation complete. Final audio shape: {final_audio.shape}, SR: {_self.target_sr}Hz")  # Use _self
            return final_audio.astype(np.float32), _self.target_sr  # Use _self

        except Exception as e:
            logger.exception("[PiperTTS] An error occurred during the TTS generation process.")
            return np.zeros((0, 2), dtype=np.float32), _self.target_sr  # Use _self
