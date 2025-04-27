# tts_generator.py
# ==========================================
# Text-to-Speech Generation for MindMorph
# ==========================================

import logging
import os
import tempfile
import textwrap
import time
from io import BytesIO
from typing import Optional, Tuple

import librosa
import numpy as np
import pyttsx3
import soundfile as sf
import streamlit as st  # TODO: Remove direct Streamlit UI calls

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
                    # This can happen in some environments
                    raise RuntimeError("pyttsx3.init() returned None")

                # Set engine properties
                self.engine.setProperty("rate", self.rate)
                self.engine.setProperty("volume", self.volume)
                logger.debug("pyttsx3 engine initialized successfully.")

            except Exception as e:
                logger.exception("Failed to initialize pyttsx3 engine.")
                self.engine = None  # Ensure engine is None on failure
                # Re-raise a more specific error for the caller
                raise RuntimeError("TTS Engine initialization failed.") from e

    def _synthesize_chunk(self, chunk_text: str, chunk_index: int, total_chunks: int) -> Optional[str]:
        """
        Synthesizes a single chunk of text to a temporary WAV file.

        Args:
            chunk_text: The text content of the chunk.
            chunk_index: The index of the current chunk (1-based).
            total_chunks: The total number of chunks.

        Returns:
            The file path to the temporary WAV file, or None if synthesis fails.
        """
        temp_chunk_path = None
        try:
            # Create a temporary file path
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk{chunk_index}.wav", mode="wb") as tmp:
                temp_chunk_path = tmp.name
            logger.debug(f"Synthesizing chunk {chunk_index}/{total_chunks} to temporary file: {temp_chunk_path}...")

            # Ensure engine is initialized
            if self.engine is None:
                logger.error("TTS engine not initialized before synthesizing chunk.")
                return None

            # Synthesize the chunk to the file
            self.engine.save_to_file(chunk_text, temp_chunk_path)
            self.engine.runAndWait()  # Blocks until synthesis is complete for this chunk

            # Verify that the file was created and is not empty
            if not os.path.exists(temp_chunk_path) or os.path.getsize(temp_chunk_path) == 0:
                logger.error(f"TTS engine failed to create a non-empty file for chunk {chunk_index}: {temp_chunk_path}")
                # TODO: Remove direct Streamlit call. Return status/raise exception.
                st.error(f"TTS engine failed for chunk {chunk_index}. Check system TTS setup.")
                if os.path.exists(temp_chunk_path):  # Clean up empty file if it exists
                    os.remove(temp_chunk_path)
                return None

            logger.debug(f"Chunk {chunk_index} synthesized successfully.")
            return temp_chunk_path

        except Exception as e_chunk:
            logger.exception(f"Failed to synthesize TTS chunk {chunk_index}.")
            # TODO: Remove direct Streamlit call. Return status/raise exception.
            st.error(f"Error processing TTS chunk {chunk_index}: {e_chunk}")
            # Clean up partial file if it exists
            if temp_chunk_path and os.path.exists(temp_chunk_path):
                try:
                    os.remove(temp_chunk_path)
                except OSError:
                    logger.warning(f"Could not clean up failed chunk file: {temp_chunk_path}")
            return None

    def _load_and_resample_chunk(self, file_path: str, target_sr: int) -> Tuple[Optional[AudioData], Optional[SampleRate]]:
        """Loads an audio chunk file, ensures stereo, and resamples."""
        try:
            logger.debug(f"Loading chunk file: {file_path}")
            # Load audio data, ensuring it's always 2D (for consistency)
            audio_data, sr = sf.read(file_path, dtype="float32", always_2d=True)

            # Ensure stereo
            if audio_data.shape[1] == 1:
                logger.debug(f"Chunk {file_path} is mono. Duplicating channel.")
                audio_data = np.concatenate([audio_data, audio_data], axis=1)
            elif audio_data.shape[1] != 2:
                logger.warning(f"Chunk {file_path} has unexpected channel count: {audio_data.shape[1]}. Attempting to use first two.")
                audio_data = audio_data[:, :2]  # Take first two channels if more exist

            # Resample if necessary
            if sr != target_sr:
                logger.warning(f"Sample rate mismatch in chunk {file_path}! Expected {target_sr}, got {sr}. Resampling...")
                if audio_data.size > 0:
                    # Librosa expects (channels, samples) for resample
                    audio_data_resampled = librosa.resample(audio_data.T, orig_sr=sr, target_sr=target_sr)
                    audio_data = audio_data_resampled.T  # Transpose back
                else:
                    sr = target_sr  # If empty, just update sr info
                    logger.warning("Chunk audio data is empty, cannot resample.")

            logger.debug(f"Loaded chunk {file_path}. Shape: {audio_data.shape}, SR: {target_sr}")
            return audio_data.astype(np.float32), target_sr

        except Exception as e_load:
            logger.exception(f"Failed to load or resample audio chunk: {file_path}")
            # TODO: Remove direct Streamlit call. Return status/raise exception.
            st.error(f"Error loading audio chunk: {e_load}")
            return None, None

    def generate(self, text: str) -> Tuple[Optional[AudioData], Optional[SampleRate]]:
        """
        Generates audio data from the input text using TTS.

        Handles chunking for long text, combines chunks, and resamples.

        Args:
            text: The text content to synthesize.

        Returns:
            A tuple containing:
                - The generated audio data as a NumPy array (float32, stereo), or None on failure.
                - The sample rate of the generated audio (should match GLOBAL_SR), or None on failure.
        """
        logger.info(f"Starting TTS generation for text length: {len(text)} characters.")
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided for TTS. Aborting.")
            return None, None

        temp_chunk_files: List[str] = []
        audio_chunks: List[AudioData] = []
        final_sr: Optional[SampleRate] = None

        # Placeholders for Streamlit UI feedback (TODO: Refactor to remove direct UI calls)
        progress_placeholder = st.empty()
        status_text = st.empty()  # Placeholder for chunk status

        try:
            # Initialize the TTS engine (raises RuntimeError on failure)
            self._init_engine()

            # Split text into manageable chunks
            logger.debug(f"Wrapping text into chunks of max size: {self.chunk_size}")
            # Use textwrap to handle splitting nicely
            chunks = textwrap.wrap(
                text,
                self.chunk_size,
                break_long_words=True,  # Break words if they exceed chunk size
                replace_whitespace=False,  # Keep existing newlines etc.
                drop_whitespace=True,  # Remove leading/trailing whitespace from chunks
            )
            num_chunks = len(chunks)
            logger.info(f"Split text into {num_chunks} chunks.")

            # --- Synthesize each chunk ---
            # TODO: Remove st.spinner - handle progress reporting via callbacks or return values
            with st.spinner(f"Synthesizing {num_chunks} audio chunks..."):
                synthesis_start_time = time.time()
                for i, chunk_text in enumerate(chunks):
                    chunk_index = i + 1
                    if not chunk_text.strip():
                        logger.debug(f"Skipping empty chunk {chunk_index}/{num_chunks}")
                        continue

                    # TODO: Update progress reporting without direct st calls
                    status_text.text(f"Synthesizing audio chunk {chunk_index}/{num_chunks}...")
                    progress_placeholder.progress(i / num_chunks)

                    chunk_start_time = time.time()
                    temp_file = self._synthesize_chunk(chunk_text, chunk_index, num_chunks)
                    chunk_end_time = time.time()

                    if temp_file:
                        temp_chunk_files.append(temp_file)
                        logger.debug(f"Chunk {chunk_index} synthesized in {chunk_end_time - chunk_start_time:.2f}s.")
                    else:
                        # Synthesis failed for this chunk, stop processing
                        logger.error(f"Synthesis failed for chunk {chunk_index}. Aborting TTS generation.")
                        # No need to return here, finally block will clean up
                        raise RuntimeError(f"TTS synthesis failed for chunk {chunk_index}")

                synthesis_end_time = time.time()
                logger.info(f"Finished synthesizing all {num_chunks} chunks in {synthesis_end_time - synthesis_start_time:.2f}s.")
                progress_placeholder.progress(1.0)
                status_text.text("Combining audio chunks...")

            # --- Load, Resample, and Combine Chunks ---
            logger.info(f"Loading and combining {len(temp_chunk_files)} synthesized audio chunks.")
            for i, file_path in enumerate(temp_chunk_files):
                # TODO: Update progress reporting without direct st calls
                status_text.text(f"Loading chunk {i + 1}/{len(temp_chunk_files)}...")
                audio_data, chunk_sr = self._load_and_resample_chunk(file_path, GLOBAL_SR)

                if audio_data is not None and chunk_sr is not None:
                    audio_chunks.append(audio_data)
                    if final_sr is None:
                        final_sr = chunk_sr  # Set the final SR based on the first successful chunk
                else:
                    logger.error(f"Failed to load or process chunk file: {file_path}. Aborting.")
                    # No need to return, finally block cleans up
                    raise RuntimeError(f"Failed to load audio chunk {file_path}")

            status_text.empty()
            progress_placeholder.empty()  # Clear Streamlit placeholders

            if not audio_chunks:
                logger.error("No valid audio chunks were loaded after synthesis.")
                # TODO: Remove direct Streamlit call. Raise exception.
                st.error("Failed to process synthesized audio chunks.")
                return None, None

            # Concatenate all loaded audio chunks
            logger.info(f"Concatenating {len(audio_chunks)} audio chunks...")
            final_audio = np.concatenate(audio_chunks, axis=0)

            if final_sr is None:
                # This case should ideally not be reached if chunks were loaded
                logger.error("Could not determine the final sample rate after combining chunks.")
                # TODO: Remove direct Streamlit call. Raise exception.
                st.error("Failed to determine sample rate for TTS audio.")
                return None, None

            # Final check on sample rate (should match GLOBAL_SR due to resampling)
            if final_sr != GLOBAL_SR:
                logger.warning(f"Final audio SR ({final_sr}) unexpectedly differs from target ({GLOBAL_SR}). This indicates a potential issue in resampling.")
                # Attempt resampling again as a fallback? Or just log the warning.
                # For now, just log it. The audio might still be usable.

            logger.info(f"TTS generation complete. Final audio shape: {final_audio.shape}, SR: {final_sr}Hz")
            return final_audio.astype(np.float32), final_sr  # Ensure float32 output

        except Exception as e:
            # Catch any exceptions during the process (e.g., engine init, synthesis, loading)
            status_text.empty()
            progress_placeholder.empty()  # Clear Streamlit placeholders
            logger.exception("An error occurred during the TTS generation process.")
            # TODO: Remove direct Streamlit call. Raise exception.
            st.error(f"TTS Generation Failed: {e}")
            return None, None

        finally:
            # --- Cleanup: Always attempt to delete temporary chunk files ---
            if temp_chunk_files:
                logger.debug(f"Cleaning up {len(temp_chunk_files)} temporary TTS chunk files...")
                cleaned_count = 0
                for file_path in temp_chunk_files:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                        except OSError as e_os:
                            logger.warning(f"Could not delete temporary TTS chunk file {file_path}: {e_os}")
                logger.debug(f"Cleaned up {cleaned_count} temporary files.")

            # Ensure engine resources are released if it was initialized
            # Note: pyttsx3 doesn't have an explicit close/shutdown method documented for standard use.
            # Relying on garbage collection might be intended, but good practice would be to have one.
            # If issues arise, investigate engine lifecycle management further.
            # For now, we assume the engine handles its resources appropriately.
            # self.engine = None # Optionally reset engine instance after use?
