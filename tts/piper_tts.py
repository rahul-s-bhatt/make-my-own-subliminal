# tts/piper_tts.py
# ==========================================
# Piper TTS Generator Implementation
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
import soundfile as sf  # Potentially needed if Piper outputs format soundfile can't read directly

# Import Piper TTS library (ensure it's in requirements.txt)
from piper.voice import PiperVoice

# Import from local modules
from config import GLOBAL_SR, PIPER_VOICE_CONFIG_PATH, PIPER_VOICE_MODEL_PATH, TTS_CHUNK_SIZE  # Assumed paths in config.py
from tts.base_tts import AudioData, BaseTTSGenerator, SampleRate

logger = logging.getLogger(__name__)


class PiperTTSGenerator(BaseTTSGenerator):
    """
    TTS Generator using the Piper TTS engine (via piper-tts library).
    Handles chunking, synthesis, resampling, and incremental concatenation.
    """

    def __init__(self, model_path: str = PIPER_VOICE_MODEL_PATH, config_path: str = PIPER_VOICE_CONFIG_PATH, chunk_size: int = TTS_CHUNK_SIZE, target_sr: int = GLOBAL_SR):
        """
        Initializes the PiperTTSGenerator.

        Args:
            model_path: Path to the Piper .onnx voice model file.
            config_path: Path to the Piper .onnx.json config file.
            chunk_size: Maximum characters per synthesis chunk.
            target_sr: The desired final sample rate (e.g., GLOBAL_SR).

        Raises:
            FileNotFoundError: If model or config paths are invalid.
            RuntimeError: If the PiperVoice fails to load.
        """
        logger.info(f"Initializing PiperTTSGenerator...")
        logger.info(f"Model Path: {model_path}")
        logger.info(f"Config Path: {config_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper model file not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Piper config file not found: {config_path}")

        self.chunk_size = chunk_size
        self.target_sr = target_sr
        self.voice = None
        self.model_native_sr = None

        try:
            logger.debug("Loading Piper voice model...")
            start_time = time.time()
            self.voice = PiperVoice.load(model_path, config_path=config_path)
            end_time = time.time()
            logger.info(f"Piper voice model loaded successfully in {end_time - start_time:.2f}s.")

            if hasattr(self.voice, "config") and self.voice.config and hasattr(self.voice.config, "sample_rate"):
                self.model_native_sr = self.voice.config.sample_rate
                logger.info(f"Piper model native sample rate: {self.model_native_sr} Hz")
            else:
                self.model_native_sr = 22050  # Fallback
                logger.warning(f"Could not automatically determine Piper model's native sample rate. Assuming {self.model_native_sr} Hz.")

        except Exception as e:
            logger.exception("Failed to load Piper voice model.")
            raise RuntimeError(f"Failed to initialize PiperTTSGenerator: {e}") from e

    def _synthesize_chunk(self, text_chunk: str) -> bytes:
        """
        Synthesizes a single text chunk using the loaded Piper model into WAV bytes.

        Args:
            text_chunk: The text to synthesize.

        Returns:
            Raw WAV audio data as bytes.

        Raises:
            RuntimeError: If synthesis fails.
        """
        if not self.voice:
            raise RuntimeError("Piper voice model is not loaded.")

        # --- MODIFICATION: Use wave.open with BytesIO ---
        audio_buffer = BytesIO()
        try:
            # Open a wave writer backed by the BytesIO buffer
            with wave.open(audio_buffer, "wb") as wav_file:
                # Piper's synthesize method expects an object with wave writer methods
                # It will set channels, framerate, sampwidth, and write frames internally
                logger.debug(f"Calling Piper synthesize for chunk: '{text_chunk[:50]}...'")
                self.voice.synthesize(text_chunk, wav_file)
                logger.debug("Piper synthesize call complete.")

            # Get the complete WAV data (including headers) from the buffer
            audio_bytes = audio_buffer.getvalue()
            if not audio_bytes:
                logger.warning(f"Piper synthesis resulted in empty audio bytes for chunk: '{text_chunk[:50]}...'")
                return b""  # Return empty bytes
            logger.debug(f"Synthesized chunk size: {len(audio_bytes)} bytes.")
            return audio_bytes
        except Exception as e:
            # Catch errors during wave operations or synthesis
            logger.exception(f"Piper synthesis or wave handling failed for chunk: '{text_chunk[:50]}...'")
            raise RuntimeError(f"Piper synthesis failed: {e}") from e
        finally:
            audio_buffer.close()  # Ensure buffer is closed
        # --- END MODIFICATION ---

    def _process_wav_bytes(self, wav_bytes: bytes) -> Tuple[AudioData, SampleRate]:
        """
        Processes raw WAV bytes: reads, converts to float32, ensures stereo.

        Args:
            wav_bytes: Raw WAV audio data (including headers).

        Returns:
            Tuple of (AudioData (float32, stereo), SampleRate (native model rate)).

        Raises:
            ValueError: If WAV data is invalid or empty.
        """
        if not wav_bytes:
            raise ValueError("Received empty WAV bytes for processing.")

        try:
            # Use wave module to read properties and data from bytes
            with BytesIO(wav_bytes) as bio:
                with wave.open(bio, "rb") as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    audio_frames = wf.readframes(n_frames)

                    logger.debug(f"Processing WAV: Channels={n_channels}, Rate={framerate}, Width={sampwidth}, Frames={n_frames}")

                    if framerate != self.model_native_sr:
                        logger.warning(f"WAV header sample rate ({framerate}) differs from expected model native SR ({self.model_native_sr}). Using header rate.")

                    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
                    if sampwidth not in dtype_map:
                        raise ValueError(f"Unsupported sample width in WAV data: {sampwidth}")
                    np_dtype = dtype_map[sampwidth]

                    audio_int = np.frombuffer(audio_frames, dtype=np_dtype)

                    if n_channels > 1:
                        # Ensure reshaping is correct even if n_frames is 0
                        if n_frames > 0:
                            audio_int = audio_int.reshape(-1, n_channels)
                        else:  # Handle empty audio case
                            audio_int = np.array([], dtype=np_dtype).reshape(0, n_channels)

            # Convert integer audio to float32 [-1.0, 1.0]
            if np_dtype == np.int8:
                audio_float = audio_int.astype(np.float32) / 128.0
            elif np_dtype == np.int16:
                audio_float = audio_int.astype(np.float32) / 32768.0
            elif np_dtype == np.int32:
                audio_float = audio_int.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Cannot convert unsupported dtype {np_dtype} to float.")

            # Ensure stereo output (samples, 2)
            if audio_float.ndim == 1:  # If mono
                audio_stereo = np.stack([audio_float, audio_float], axis=-1)
            elif audio_float.shape[1] == 1:  # If (samples, 1)
                audio_stereo = np.concatenate([audio_float, audio_float], axis=1)
            elif audio_float.shape[1] == 2:  # Already stereo
                audio_stereo = audio_float
            else:  # More than 2 channels? Take first two.
                logger.warning(f"Piper output had {audio_float.shape[1]} channels. Using first two.")
                audio_stereo = audio_float[:, :2]

            return audio_stereo.astype(np.float32), framerate  # Return native SR

        except Exception as e:
            logger.exception("Failed to process WAV bytes.")
            raise ValueError(f"Error processing synthesized WAV data: {e}") from e

    def generate(self, text: str) -> Tuple[AudioData, SampleRate]:
        """
        Generates audio from text using Piper TTS.
        """
        logger.info(f"[PiperTTS] Starting TTS generation for text length: {len(text)} characters.")
        if not text or not text.strip():
            logger.warning("[PiperTTS] Empty or whitespace-only text provided. Aborting.")
            raise ValueError("Input text for TTS cannot be empty.")

        final_audio = np.array([], dtype=np.float32).reshape(0, 2)  # Ensure 2D stereo

        try:
            logger.debug(f"[PiperTTS] Wrapping text into chunks of max size: {self.chunk_size}")
            chunks = textwrap.wrap(
                text,
                self.chunk_size,
                break_long_words=True,
                replace_whitespace=False,
                drop_whitespace=True,
            )
            num_chunks = len(chunks)
            logger.info(f"[PiperTTS] Split text into {num_chunks} chunks.")

            synthesis_start_time = time.time()
            for i, chunk_text in enumerate(chunks):
                chunk_index = i + 1
                if not chunk_text.strip():
                    logger.debug(f"[PiperTTS] Skipping empty chunk {chunk_index}/{num_chunks}")
                    continue

                logger.info(f"[PiperTTS] Synthesizing chunk {chunk_index}/{num_chunks}...")
                chunk_start_time = time.time()

                # 1. Synthesize chunk to raw WAV bytes
                wav_bytes = self._synthesize_chunk(chunk_text)
                if not wav_bytes:
                    logger.warning(f"[PiperTTS] Skipping empty audio result for chunk {chunk_index}.")
                    continue

                # 2. Process bytes to get float audio and native SR
                chunk_audio_native, native_sr = self._process_wav_bytes(wav_bytes)
                if native_sr != self.model_native_sr:
                    logger.warning(f"Chunk {chunk_index} native SR {native_sr} differs from expected model SR {self.model_native_sr}.")

                # 3. Resample if necessary
                if native_sr != self.target_sr:
                    logger.warning(f"[PiperTTS] Resampling chunk {chunk_index} from {native_sr} Hz to {self.target_sr} Hz.")
                    if chunk_audio_native.size > 0:
                        try:
                            resampled_audio = librosa.resample(chunk_audio_native.T, orig_sr=native_sr, target_sr=self.target_sr).T
                            chunk_audio_processed = resampled_audio
                        except Exception as e_resample:
                            logger.exception(f"Failed to resample chunk {chunk_index}.")
                            raise RuntimeError(f"Resampling failed for chunk {chunk_index}: {e_resample}") from e_resample
                    else:
                        logger.warning(f"[PiperTTS] Chunk {chunk_index} is empty, cannot resample.")
                        chunk_audio_processed = chunk_audio_native
                else:
                    chunk_audio_processed = chunk_audio_native

                # 4. Concatenate incrementally
                logger.debug(f"[PiperTTS] Concatenating chunk {chunk_index}. Current total samples: {final_audio.shape[0]}, Chunk samples: {chunk_audio_processed.shape[0]}")
                final_audio = np.concatenate((final_audio, chunk_audio_processed), axis=0)
                logger.debug(f"[PiperTTS] Concatenation complete for chunk {chunk_index}. New total samples: {final_audio.shape[0]}")

                chunk_end_time = time.time()
                logger.info(f"[PiperTTS] Processed chunk {chunk_index} in {chunk_end_time - chunk_start_time:.2f}s.")

            synthesis_end_time = time.time()
            logger.info(f"[PiperTTS] Finished processing all {num_chunks} chunks in {synthesis_end_time - synthesis_start_time:.2f}s.")

            if final_audio.size == 0:
                logger.error("[PiperTTS] TTS process resulted in empty final audio.")
                raise RuntimeError("TTS generation produced no audio data.")

            logger.info(f"[PiperTTS] TTS generation complete. Final audio shape: {final_audio.shape}, SR: {self.target_sr}Hz")
            return final_audio.astype(np.float32), self.target_sr

        except Exception as e:
            logger.exception("[PiperTTS] An error occurred during the TTS generation process.")
            raise RuntimeError(f"Piper TTS Generation Failed: {e}") from e
