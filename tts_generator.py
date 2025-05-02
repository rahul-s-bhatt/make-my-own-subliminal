# tts_generator.py
# ==========================================
# Text-to-Speech Generation for MindMorph
# ==========================================

import logging
import os
import tempfile
import textwrap
import time
from typing import List, Optional, Tuple  # Added List

import librosa
import numpy as np
import pyttsx3
import soundfile as sf

# Removed streamlit import
# Import constants from config
from config import GLOBAL_SR, TTS_CHUNK_SIZE

# Define type hints used within this module
AudioData = np.ndarray
SampleRate = int

# Get a logger for this module
logger = logging.getLogger(__name__)


class TTSGenerator:
    """
    Handles Text-to-Speech generation using pyttsx3.

    Includes chunking for long texts, temporary file management,
    and resampling to the global sample rate.

    Provides separate methods suffixed with '_quick_wizard'
    that avoid direct Streamlit UI calls and use exceptions for errors,
    suitable for background processing or integration where UI feedback
    is handled externally. Includes memory optimization for concatenation.
    """

    def __init__(self, chunk_size: int = TTS_CHUNK_SIZE):
        """
        Initializes the TTSGenerator.

        Args:
            chunk_size: Maximum number of characters per TTS chunk.
        """
        self.engine = None
        self.rate = 200
        self.volume = 1.0
        self.chunk_size = chunk_size
        logger.debug(f"TTSGenerator initialized with chunk_size={chunk_size}, rate={self.rate}, volume={self.volume}")

    def _init_engine(self):
        """Initializes the pyttsx3 engine instance if not already done."""
        if self.engine is None:
            try:
                logger.debug("Initializing pyttsx3 engine...")
                self.engine = pyttsx3.init()
                if self.engine is None:
                    raise RuntimeError("pyttsx3.init() returned None")

                self.engine.setProperty("rate", self.rate)
                self.engine.setProperty("volume", self.volume)
                # Note: Setting sample rate directly is not reliably supported by pyttsx3 API
                # The actual output rate depends on the backend (e.g., espeak, SAPI5)
                logger.debug("pyttsx3 engine initialized successfully.")

            except Exception as e:
                logger.exception("Failed to initialize pyttsx3 engine.")
                self.engine = None
                raise RuntimeError("TTS Engine initialization failed.") from e

    # --- Original Methods (Potentially with Streamlit calls - Keep for compatibility if needed) ---
    # (Original _synthesize_chunk, _load_and_resample_chunk, generate methods remain unchanged)
    def _synthesize_chunk(self, chunk_text: str, chunk_index: int, total_chunks: int) -> Optional[str]:
        """
        Synthesizes a single chunk of text to a temporary WAV file.
        (Original version - may contain st calls, kept for compatibility)
        """
        temp_chunk_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk{chunk_index}.wav", mode="wb") as tmp:
                temp_chunk_path = tmp.name
            logger.debug(f"Synthesizing chunk {chunk_index}/{total_chunks} to temporary file: {temp_chunk_path}...")

            if self.engine is None:
                logger.error("TTS engine not initialized before synthesizing chunk.")
                return None

            self.engine.save_to_file(chunk_text, temp_chunk_path)
            self.engine.runAndWait()

            if not os.path.exists(temp_chunk_path) or os.path.getsize(temp_chunk_path) == 0:
                logger.error(f"TTS engine failed to create a non-empty file for chunk {chunk_index}: {temp_chunk_path}")
                if os.path.exists(temp_chunk_path):
                    os.remove(temp_chunk_path)
                return None

            logger.debug(f"Chunk {chunk_index} synthesized successfully.")
            return temp_chunk_path

        except Exception as e_chunk:
            logger.exception(f"Failed to synthesize TTS chunk {chunk_index}.")
            if temp_chunk_path and os.path.exists(temp_chunk_path):
                try:
                    os.remove(temp_chunk_path)
                except OSError:
                    logger.warning(f"Could not clean up failed chunk file: {temp_chunk_path}")
            return None

    def _load_and_resample_chunk(self, file_path: str, target_sr: int) -> Tuple[Optional[AudioData], Optional[SampleRate]]:
        """
        Loads an audio chunk file, ensures stereo, and resamples.
        (Original version - may contain st calls, kept for compatibility)
        """
        try:
            logger.debug(f"Loading chunk file: {file_path}")
            audio_data, sr = sf.read(file_path, dtype="float32", always_2d=True)

            if audio_data.shape[1] == 1:
                logger.debug(f"Chunk {file_path} is mono. Duplicating channel.")
                audio_data = np.concatenate([audio_data, audio_data], axis=1)
            elif audio_data.shape[1] != 2:
                logger.warning(f"Chunk {file_path} has unexpected channel count: {audio_data.shape[1]}. Attempting to use first two.")
                audio_data = audio_data[:, :2]

            if sr != target_sr:
                logger.warning(f"Sample rate mismatch in chunk {file_path}! Expected {target_sr}, got {sr}. Resampling...")
                if audio_data.size > 0:
                    audio_data_resampled = librosa.resample(audio_data.T, orig_sr=sr, target_sr=target_sr)
                    audio_data = audio_data_resampled.T
                    sr = target_sr  # Update sr after resampling
                else:
                    sr = target_sr
                    logger.warning("Chunk audio data is empty, cannot resample.")

            logger.debug(f"Loaded chunk {file_path}. Shape: {audio_data.shape}, SR: {sr}")
            return audio_data.astype(np.float32), sr

        except Exception as e_load:
            logger.exception(f"Failed to load or resample audio chunk: {file_path}")
            return None, None

    def generate(self, text: str) -> Tuple[Optional[AudioData], Optional[SampleRate]]:
        """
        Generates audio data from the input text using TTS.
        (Original version - may contain st calls, kept for compatibility)

        Handles chunking, combines chunks, and resamples. May use Streamlit UI elements.
        """
        logger.info(f"[Original Generate] Starting TTS for text length: {len(text)} chars.")
        if not text or not text.strip():
            logger.warning("[Original Generate] Empty text provided. Aborting.")
            return None, None

        temp_chunk_files: List[str] = []
        audio_chunks: List[AudioData] = []
        final_sr: Optional[SampleRate] = None

        try:
            self._init_engine()
            chunks = textwrap.wrap(text, self.chunk_size, break_long_words=True, replace_whitespace=False, drop_whitespace=True)
            num_chunks = len(chunks)
            logger.info(f"[Original Generate] Split into {num_chunks} chunks.")

            for i, chunk_text in enumerate(chunks):
                chunk_index = i + 1
                if not chunk_text.strip():
                    continue
                temp_file = self._synthesize_chunk(chunk_text, chunk_index, num_chunks)  # Calls original synthesize
                if temp_file:
                    temp_chunk_files.append(temp_file)
                else:
                    logger.error(f"[Original Generate] Synthesis failed for chunk {chunk_index}. Aborting.")
                    raise RuntimeError(f"TTS synthesis failed for chunk {chunk_index}")

            logger.info(f"[Original Generate] Finished synthesizing {num_chunks} chunks.")

            logger.info(f"[Original Generate] Loading/combining {len(temp_chunk_files)} chunks.")
            for i, file_path in enumerate(temp_chunk_files):
                audio_data, chunk_sr = self._load_and_resample_chunk(file_path, GLOBAL_SR)  # Calls original load
                if audio_data is not None and chunk_sr is not None:
                    audio_chunks.append(audio_data)
                    if final_sr is None:
                        final_sr = chunk_sr
                else:
                    logger.error(f"[Original Generate] Failed to load chunk: {file_path}. Aborting.")
                    raise RuntimeError(f"Failed to load audio chunk {file_path}")

            if not audio_chunks:
                logger.error("[Original Generate] No valid audio chunks loaded.")
                return None, None

            logger.info(f"[Original Generate] Concatenating {len(audio_chunks)} chunks...")
            final_audio = np.concatenate(audio_chunks, axis=0)

            if final_sr is None:
                logger.error("[Original Generate] Could not determine final sample rate.")
                return None, None

            if final_sr != GLOBAL_SR:
                logger.warning(f"[Original Generate] Final SR ({final_sr}) != Target SR ({GLOBAL_SR}).")

            logger.info(f"[Original Generate] TTS complete. Shape: {final_audio.shape}, SR: {final_sr}Hz")
            return final_audio.astype(np.float32), final_sr

        except Exception as e:
            logger.exception("[Original Generate] Error during TTS generation.")
            return None, None

        finally:
            if temp_chunk_files:
                logger.debug(f"[Original Generate] Cleaning up {len(temp_chunk_files)} temp files...")
                cleaned_count = 0
                for file_path in temp_chunk_files:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                        except OSError as e_os:
                            logger.warning(f"[Original Generate] Could not delete temp file {file_path}: {e_os}")
                logger.debug(f"[Original Generate] Cleaned up {cleaned_count} files.")

    # --- Quick Wizard Specific Methods (No Streamlit UI Calls, Raise Exceptions) ---

    def _synthesize_chunk_quick_wizard(self, chunk_text: str, chunk_index: int, total_chunks: int) -> str:
        """
        Synthesizes a single chunk of text to a temporary WAV file for Quick Wizard.
        Raises RuntimeError on failure. Does NOT use st.* calls.

        Returns:
            The file path to the temporary WAV file.
        """
        temp_chunk_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk{chunk_index}.wav", mode="wb") as tmp:
                temp_chunk_path = tmp.name

            logger.debug(f"[Quick Wizard] Synthesizing chunk {chunk_index}/{total_chunks} to temp file: {temp_chunk_path}...")

            if self.engine is None:
                logger.error("[Quick Wizard] TTS engine not initialized before synthesizing chunk.")
                raise RuntimeError("TTS engine not initialized.")

            self.engine.save_to_file(chunk_text, temp_chunk_path)
            self.engine.runAndWait()

            if not os.path.exists(temp_chunk_path) or os.path.getsize(temp_chunk_path) == 0:
                logger.error(f"[Quick Wizard] TTS engine failed to create a non-empty file for chunk {chunk_index}: {temp_chunk_path}")
                if os.path.exists(temp_chunk_path):
                    try:
                        os.remove(temp_chunk_path)
                    except OSError:
                        pass
                raise RuntimeError(f"TTS engine failed for chunk {chunk_index}. Check system TTS setup.")

            logger.debug(f"[Quick Wizard] Chunk {chunk_index} synthesized successfully.")
            return temp_chunk_path

        except Exception as e_chunk:
            logger.exception(f"[Quick Wizard] Failed to synthesize TTS chunk {chunk_index}.")
            if temp_chunk_path and os.path.exists(temp_chunk_path):
                try:
                    os.remove(temp_chunk_path)
                except OSError:
                    logger.warning(f"[Quick Wizard] Could not clean up failed chunk file: {temp_chunk_path}")
            raise RuntimeError(f"Error processing TTS chunk {chunk_index}: {e_chunk}") from e_chunk

    def _load_and_resample_chunk_quick_wizard(self, file_path: str, target_sr: int) -> Tuple[AudioData, SampleRate]:
        """
        Loads an audio chunk file, ensures stereo, and resamples for Quick Wizard.
        Raises RuntimeError or ValueError on failure. Does NOT use st.* calls.

        Returns:
            A tuple containing:
                - The loaded and resampled audio data.
                - The target sample rate.
        """
        try:
            logger.debug(f"[Quick Wizard] Loading chunk file: {file_path}")
            audio_data, sr = sf.read(file_path, dtype="float32", always_2d=True)

            if audio_data.shape[1] == 1:
                logger.debug(f"[Quick Wizard] Chunk {file_path} is mono. Duplicating channel.")
                audio_data = np.concatenate([audio_data, audio_data], axis=1)
            elif audio_data.shape[1] != 2:
                logger.error(f"[Quick Wizard] Chunk {file_path} has unexpected channel count: {audio_data.shape[1]}.")
                raise ValueError(f"Unexpected channel count in synthesized chunk: {file_path}")

            if sr != target_sr:
                logger.warning(f"[Quick Wizard] Sample rate mismatch in chunk {file_path}! Expected {target_sr}, got {sr}. Resampling...")
                if audio_data.size > 0:
                    # Adding extra logging before resampling
                    logger.debug(f"[Quick Wizard] Resampling chunk {file_path} from {sr} to {target_sr}. Shape before: {audio_data.T.shape}")
                    audio_data_resampled = librosa.resample(audio_data.T, orig_sr=sr, target_sr=target_sr)
                    audio_data = audio_data_resampled.T
                    sr = target_sr
                    logger.debug(f"[Quick Wizard] Resampling complete for chunk {file_path}. Shape after: {audio_data.shape}")
                else:
                    sr = target_sr
                    logger.warning("[Quick Wizard] Chunk audio data is empty, cannot resample.")

            logger.debug(f"[Quick Wizard] Loaded chunk {file_path}. Shape: {audio_data.shape}, SR: {sr}")
            if sr != target_sr:
                logger.error(f"[Quick Wizard] Resampling failed to produce target SR for {file_path}. Got {sr}, expected {target_sr}")
                raise RuntimeError(f"Resampling failed for chunk {file_path}")

            return audio_data.astype(np.float32), target_sr

        except Exception as e_load:
            logger.exception(f"[Quick Wizard] Failed to load or resample audio chunk: {file_path}")
            raise RuntimeError(f"Error loading audio chunk {file_path}: {e_load}") from e_load

    def generate_quick_wizard(self, text: str) -> Tuple[AudioData, SampleRate]:
        """
        Generates audio data from the input text using TTS for Quick Wizard.
        Handles chunking, combines chunks, and resamples. Uses incremental
        concatenation to reduce peak memory usage.
        Does NOT use st.* calls. Raises exceptions on failure.

        Args:
            text: The text content to synthesize.

        Returns:
            A tuple containing:
                - The generated audio data as a NumPy array (float32, stereo).
                - The sample rate of the generated audio (matching GLOBAL_SR).

        Raises:
            ValueError: If input text is empty.
            RuntimeError: If TTS engine fails, synthesis fails, or loading/combining fails.
        """
        logger.info(f"[Quick Wizard] Starting TTS generation for text length: {len(text)} characters.")
        if not text or not text.strip():
            logger.warning("[Quick Wizard] Empty or whitespace-only text provided for TTS. Aborting.")
            raise ValueError("Input text for TTS cannot be empty.")

        temp_chunk_files: List[str] = []
        # --- MODIFICATION: Initialize final_audio as empty array ---
        final_audio = np.array([], dtype=np.float32).reshape(0, 2)  # Ensure 2D for stereo
        final_sr: Optional[SampleRate] = None

        try:
            self._init_engine()
            chunks = textwrap.wrap(
                text,
                self.chunk_size,
                break_long_words=True,
                replace_whitespace=False,
                drop_whitespace=True,
            )
            num_chunks = len(chunks)
            logger.info(f"[Quick Wizard] Split text into {num_chunks} chunks.")

            # --- Synthesize each chunk ---
            synthesis_start_time = time.time()
            for i, chunk_text in enumerate(chunks):
                chunk_index = i + 1
                if not chunk_text.strip():
                    logger.debug(f"[Quick Wizard] Skipping empty chunk {chunk_index}/{num_chunks}")
                    continue

                logger.info(f"[Quick Wizard] Synthesizing audio chunk {chunk_index}/{num_chunks}...")
                chunk_start_time = time.time()
                temp_file = self._synthesize_chunk_quick_wizard(chunk_text, chunk_index, num_chunks)
                chunk_end_time = time.time()
                temp_chunk_files.append(temp_file)
                logger.debug(f"[Quick Wizard] Chunk {chunk_index} synthesized in {chunk_end_time - chunk_start_time:.2f}s.")

            synthesis_end_time = time.time()
            logger.info(f"[Quick Wizard] Finished synthesizing all {num_chunks} chunks in {synthesis_end_time - synthesis_start_time:.2f}s.")

            # --- Load, Resample, and Combine Chunks Incrementally ---
            logger.info(f"[Quick Wizard] Loading, resampling, and incrementally combining {len(temp_chunk_files)} chunks.")
            # --- MODIFICATION: Loop and concatenate incrementally ---
            for i, file_path in enumerate(temp_chunk_files):
                logger.info(f"[Quick Wizard] Processing chunk {i + 1}/{len(temp_chunk_files)}: {file_path}")
                # Load and resample the current chunk
                audio_data, chunk_sr = self._load_and_resample_chunk_quick_wizard(file_path, GLOBAL_SR)
                # audio_data is guaranteed to be non-None and have chunk_sr == GLOBAL_SR if no exception
                if final_sr is None:
                    final_sr = chunk_sr  # Set the final SR based on the first successful chunk

                # Append the current chunk's data to the final_audio array
                logger.debug(f"[Quick Wizard] Concatenating chunk {i + 1}. Current total samples: {final_audio.shape[0]}, Chunk samples: {audio_data.shape[0]}")
                final_audio = np.concatenate((final_audio, audio_data), axis=0)
                logger.debug(f"[Quick Wizard] Concatenation complete for chunk {i + 1}. New total samples: {final_audio.shape[0]}")
                # --- End of Incremental Concatenation ---

            if final_audio.size == 0:
                # This should not happen if synthesis succeeded but good to check
                logger.error("[Quick Wizard] No valid audio data was combined.")
                raise RuntimeError("Failed to process synthesized audio chunks.")

            if final_sr is None or final_sr != GLOBAL_SR:
                logger.error(f"[Quick Wizard] Could not determine correct final sample rate ({final_sr}) after combining chunks.")
                raise RuntimeError("Failed to determine sample rate for TTS audio.")

            logger.info(f"[Quick Wizard] TTS generation complete. Final audio shape: {final_audio.shape}, SR: {final_sr}Hz")
            return final_audio.astype(np.float32), final_sr

        except Exception as e:
            logger.exception("[Quick Wizard] An error occurred during the TTS generation process.")
            raise RuntimeError(f"TTS Generation Failed: {e}") from e

        finally:
            # --- Cleanup: Always attempt to delete temporary chunk files ---
            if temp_chunk_files:
                logger.debug(f"[Quick Wizard] Cleaning up {len(temp_chunk_files)} temporary TTS chunk files...")
                cleaned_count = 0
                for file_path in temp_chunk_files:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                        except OSError as e_os:
                            logger.warning(f"[Quick Wizard] Could not delete temporary TTS chunk file {file_path}: {e_os}")
                logger.debug(f"[Quick Wizard] Cleaned up {cleaned_count} temporary files.")
