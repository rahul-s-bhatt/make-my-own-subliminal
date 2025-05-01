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

# Removed streamlit import - No direct UI calls allowed in this refactored version for wizard
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
    is handled externally.
    """

    def __init__(self, chunk_size: int = TTS_CHUNK_SIZE):
        """
        Initializes the TTSGenerator.

        Args:
            chunk_size: Maximum number of characters per TTS chunk.
        """
        self.engine = None
        # Default properties (can be customized if needed)
        self.rate = 200  # Words per minute
        self.volume = 1.0  # Volume (0.0 to 1.0)
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
                logger.debug("pyttsx3 engine initialized successfully.")

            except Exception as e:
                logger.exception("Failed to initialize pyttsx3 engine.")
                self.engine = None
                raise RuntimeError("TTS Engine initialization failed.") from e

    # --- Original Methods (Potentially with Streamlit calls - Keep for compatibility if needed) ---

    def _synthesize_chunk(self, chunk_text: str, chunk_index: int, total_chunks: int) -> Optional[str]:
        """
        Synthesizes a single chunk of text to a temporary WAV file.
        (Original version - may contain st calls, kept for compatibility)
        """
        # This is the original implementation. If it contains st calls, they remain here.
        # For the wizard, we use _synthesize_chunk_quick_wizard
        temp_chunk_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk{chunk_index}.wav", mode="wb") as tmp:
                temp_chunk_path = tmp.name
            logger.debug(f"Synthesizing chunk {chunk_index}/{total_chunks} to temporary file: {temp_chunk_path}...")

            if self.engine is None:
                logger.error("TTS engine not initialized before synthesizing chunk.")
                # Original might have st.error here
                return None

            self.engine.save_to_file(chunk_text, temp_chunk_path)
            self.engine.runAndWait()

            if not os.path.exists(temp_chunk_path) or os.path.getsize(temp_chunk_path) == 0:
                logger.error(f"TTS engine failed to create a non-empty file for chunk {chunk_index}: {temp_chunk_path}")
                # Original might have st.error here
                if os.path.exists(temp_chunk_path):
                    os.remove(temp_chunk_path)
                return None

            logger.debug(f"Chunk {chunk_index} synthesized successfully.")
            return temp_chunk_path

        except Exception as e_chunk:
            logger.exception(f"Failed to synthesize TTS chunk {chunk_index}.")
            # Original might have st.error here
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
        # This is the original implementation. If it contains st calls, they remain here.
        # For the wizard, we use _load_and_resample_chunk_quick_wizard
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
                else:
                    sr = target_sr
                    logger.warning("Chunk audio data is empty, cannot resample.")

            logger.debug(f"Loaded chunk {file_path}. Shape: {audio_data.shape}, SR: {target_sr}")
            return audio_data.astype(np.float32), target_sr

        except Exception as e_load:
            logger.exception(f"Failed to load or resample audio chunk: {file_path}")
            # Original might have st.error here
            return None, None

    def generate(self, text: str) -> Tuple[Optional[AudioData], Optional[SampleRate]]:
        """
        Generates audio data from the input text using TTS.
        (Original version - may contain st calls, kept for compatibility)

        Handles chunking, combines chunks, and resamples. May use Streamlit UI elements.
        """
        # This is the original implementation. If it contains st calls, they remain here.
        # For the wizard, we use generate_quick_wizard
        logger.info(f"[Original Generate] Starting TTS for text length: {len(text)} chars.")
        if not text or not text.strip():
            logger.warning("[Original Generate] Empty text provided. Aborting.")
            return None, None

        temp_chunk_files: List[str] = []
        audio_chunks: List[AudioData] = []
        final_sr: Optional[SampleRate] = None

        # Original might have st placeholders here

        try:
            self._init_engine()
            chunks = textwrap.wrap(text, self.chunk_size, break_long_words=True, replace_whitespace=False, drop_whitespace=True)
            num_chunks = len(chunks)
            logger.info(f"[Original Generate] Split into {num_chunks} chunks.")

            # Original might have st.spinner here
            for i, chunk_text in enumerate(chunks):
                chunk_index = i + 1
                if not chunk_text.strip():
                    continue
                # Original might have st progress/text updates here
                temp_file = self._synthesize_chunk(chunk_text, chunk_index, num_chunks)  # Calls original synthesize
                if temp_file:
                    temp_chunk_files.append(temp_file)
                else:
                    logger.error(f"[Original Generate] Synthesis failed for chunk {chunk_index}. Aborting.")
                    raise RuntimeError(f"TTS synthesis failed for chunk {chunk_index}")

            logger.info(f"[Original Generate] Finished synthesizing {num_chunks} chunks.")
            # Original might clear st placeholders here

            logger.info(f"[Original Generate] Loading/combining {len(temp_chunk_files)} chunks.")
            for i, file_path in enumerate(temp_chunk_files):
                # Original might have st progress/text updates here
                audio_data, chunk_sr = self._load_and_resample_chunk(file_path, GLOBAL_SR)  # Calls original load
                if audio_data is not None and chunk_sr is not None:
                    audio_chunks.append(audio_data)
                    if final_sr is None:
                        final_sr = chunk_sr
                else:
                    logger.error(f"[Original Generate] Failed to load chunk: {file_path}. Aborting.")
                    raise RuntimeError(f"Failed to load audio chunk {file_path}")

            # Original might clear st placeholders here

            if not audio_chunks:
                logger.error("[Original Generate] No valid audio chunks loaded.")
                # Original might have st.error here
                return None, None

            logger.info(f"[Original Generate] Concatenating {len(audio_chunks)} chunks...")
            final_audio = np.concatenate(audio_chunks, axis=0)

            if final_sr is None:
                logger.error("[Original Generate] Could not determine final sample rate.")
                # Original might have st.error here
                return None, None

            if final_sr != GLOBAL_SR:
                logger.warning(f"[Original Generate] Final SR ({final_sr}) != Target SR ({GLOBAL_SR}).")

            logger.info(f"[Original Generate] TTS complete. Shape: {final_audio.shape}, SR: {final_sr}Hz")
            return final_audio.astype(np.float32), final_sr

        except Exception as e:
            # Original might clear st placeholders here
            logger.exception("[Original Generate] Error during TTS generation.")
            # Original might have st.error here
            return None, None

        finally:
            if temp_chunk_files:
                logger.debug(f"[Original Generate] Cleaning up {len(temp_chunk_files)} temp files...")
                # Cleanup logic... (same as below)
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
            # Create a temporary file path using a context manager for safety
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk{chunk_index}.wav", mode="wb") as tmp:
                temp_chunk_path = tmp.name

            logger.debug(f"[Quick Wizard] Synthesizing chunk {chunk_index}/{total_chunks} to temp file: {temp_chunk_path}...")

            # Engine initialization should happen before calling this, checked in generate_quick_wizard
            if self.engine is None:
                # This case should ideally be prevented by the calling function (_init_engine check)
                logger.error("[Quick Wizard] TTS engine not initialized before synthesizing chunk.")
                raise RuntimeError("TTS engine not initialized.")

            # Synthesize the chunk to the file
            self.engine.save_to_file(chunk_text, temp_chunk_path)
            self.engine.runAndWait()  # Blocks until synthesis is complete

            # Verify that the file was created and is not empty
            if not os.path.exists(temp_chunk_path) or os.path.getsize(temp_chunk_path) == 0:
                logger.error(f"[Quick Wizard] TTS engine failed to create a non-empty file for chunk {chunk_index}: {temp_chunk_path}")
                if os.path.exists(temp_chunk_path):  # Clean up empty file if it exists
                    try:
                        os.remove(temp_chunk_path)
                    except OSError:
                        pass  # Ignore cleanup error if main task failed
                raise RuntimeError(f"TTS engine failed for chunk {chunk_index}. Check system TTS setup.")

            logger.debug(f"[Quick Wizard] Chunk {chunk_index} synthesized successfully.")
            return temp_chunk_path  # Return path on success

        except Exception as e_chunk:
            logger.exception(f"[Quick Wizard] Failed to synthesize TTS chunk {chunk_index}.")
            # Clean up partial file if it exists
            if temp_chunk_path and os.path.exists(temp_chunk_path):
                try:
                    os.remove(temp_chunk_path)
                except OSError:
                    logger.warning(f"[Quick Wizard] Could not clean up failed chunk file: {temp_chunk_path}")
            # Re-raise the exception for the caller to handle
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
            # Load audio data, ensuring it's always 2D (for consistency)
            audio_data, sr = sf.read(file_path, dtype="float32", always_2d=True)

            # Ensure stereo
            if audio_data.shape[1] == 1:
                logger.debug(f"[Quick Wizard] Chunk {file_path} is mono. Duplicating channel.")
                audio_data = np.concatenate([audio_data, audio_data], axis=1)
            elif audio_data.shape[1] != 2:
                # This case might indicate a problem with the synthesized file
                logger.error(f"[Quick Wizard] Chunk {file_path} has unexpected channel count: {audio_data.shape[1]}.")
                raise ValueError(f"Unexpected channel count in synthesized chunk: {file_path}")

            # Resample if necessary
            if sr != target_sr:
                logger.warning(f"[Quick Wizard] Sample rate mismatch in chunk {file_path}! Expected {target_sr}, got {sr}. Resampling...")
                if audio_data.size > 0:
                    # Librosa expects (channels, samples) for resample
                    audio_data_resampled = librosa.resample(audio_data.T, orig_sr=sr, target_sr=target_sr)
                    audio_data = audio_data_resampled.T  # Transpose back
                    sr = target_sr  # Update sample rate info after resampling
                else:
                    sr = target_sr  # If empty, just update sr info
                    logger.warning("[Quick Wizard] Chunk audio data is empty, cannot resample.")

            logger.debug(f"[Quick Wizard] Loaded chunk {file_path}. Shape: {audio_data.shape}, SR: {sr}")
            # Ensure the returned sample rate matches the target after potential resampling
            if sr != target_sr:
                logger.error(f"[Quick Wizard] Resampling failed to produce target SR for {file_path}. Got {sr}, expected {target_sr}")
                raise RuntimeError(f"Resampling failed for chunk {file_path}")

            return audio_data.astype(np.float32), target_sr

        except Exception as e_load:
            logger.exception(f"[Quick Wizard] Failed to load or resample audio chunk: {file_path}")
            # Re-raise exception for the caller
            raise RuntimeError(f"Error loading audio chunk {file_path}: {e_load}") from e_load

    def generate_quick_wizard(self, text: str) -> Tuple[AudioData, SampleRate]:
        """
        Generates audio data from the input text using TTS for Quick Wizard.
        Handles chunking, combines chunks, and resamples.
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
        audio_chunks: List[AudioData] = []
        final_sr: Optional[SampleRate] = None

        try:
            # Initialize the TTS engine (raises RuntimeError on failure)
            self._init_engine()

            # Split text into manageable chunks
            logger.debug(f"[Quick Wizard] Wrapping text into chunks of max size: {self.chunk_size}")
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

                # Log progress without UI calls
                logger.info(f"[Quick Wizard] Synthesizing audio chunk {chunk_index}/{num_chunks}...")
                chunk_start_time = time.time()
                # Call the wizard-specific synthesize method (raises exception on failure)
                temp_file = self._synthesize_chunk_quick_wizard(chunk_text, chunk_index, num_chunks)
                chunk_end_time = time.time()
                # temp_file will always be a valid path if no exception was raised
                temp_chunk_files.append(temp_file)
                logger.debug(f"[Quick Wizard] Chunk {chunk_index} synthesized in {chunk_end_time - chunk_start_time:.2f}s.")

            synthesis_end_time = time.time()
            logger.info(f"[Quick Wizard] Finished synthesizing all {num_chunks} chunks in {synthesis_end_time - synthesis_start_time:.2f}s.")

            # --- Load, Resample, and Combine Chunks ---
            logger.info(f"[Quick Wizard] Loading and combining {len(temp_chunk_files)} synthesized audio chunks.")
            for i, file_path in enumerate(temp_chunk_files):
                logger.info(f"[Quick Wizard] Loading chunk {i + 1}/{len(temp_chunk_files)}...")
                # Call the wizard-specific load method (raises exception on failure)
                audio_data, chunk_sr = self._load_and_resample_chunk_quick_wizard(file_path, GLOBAL_SR)
                # If no exception, data is valid
                audio_chunks.append(audio_data)
                if final_sr is None:
                    final_sr = chunk_sr  # Should always be GLOBAL_SR if no exception

            if not audio_chunks:
                # This should not happen if synthesis succeeded but good to check
                logger.error("[Quick Wizard] No valid audio chunks were loaded after synthesis.")
                raise RuntimeError("Failed to process synthesized audio chunks.")

            # Concatenate all loaded audio chunks
            logger.info(f"[Quick Wizard] Concatenating {len(audio_chunks)} audio chunks...")
            final_audio = np.concatenate(audio_chunks, axis=0)

            if final_sr is None or final_sr != GLOBAL_SR:
                # This indicates a logic error if reached without exceptions
                logger.error(f"[Quick Wizard] Could not determine correct final sample rate ({final_sr}) after combining chunks.")
                raise RuntimeError("Failed to determine sample rate for TTS audio.")

            logger.info(f"[Quick Wizard] TTS generation complete. Final audio shape: {final_audio.shape}, SR: {final_sr}Hz")
            return final_audio.astype(np.float32), final_sr  # Ensure float32 output

        except Exception as e:
            # Catch any exceptions during the process and re-raise as RuntimeError
            logger.exception("[Quick Wizard] An error occurred during the TTS generation process.")
            # The original exception 'e' is chained for context
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
