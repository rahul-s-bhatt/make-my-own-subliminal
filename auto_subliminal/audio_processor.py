# auto_subliminal/audio_processor.py
# Handles audio processing tasks for the auto subliminal generator,
# including loading background, mixing, and preview generation.

import logging
import math
import os

import librosa  # For resampling, if not handled by load_audio for all cases
import numpy as np

# Import from existing audio_utils
from audio_utils.audio_io import load_audio
from audio_utils.audio_state_definitions import AudioData, SampleRate

# Import from global config
from config import GLOBAL_SR

logger = logging.getLogger(__name__)


class AutoSubliminalAudioProcessor:
    """
    Processes and mixes audio for the auto subliminal generation feature.
    """

    def __init__(self, output_sr: int = GLOBAL_SR):
        """
        Initializes the audio processor.

        Args:
            output_sr (int): The target sample rate for all processed audio.
        """
        self.output_sr = output_sr
        logger.info(f"AutoSubliminalAudioProcessor initialized with output SR: {self.output_sr} Hz.")

    def _resample_if_needed(self, audio_data: AudioData, current_sr: SampleRate) -> AudioData:
        """Resamples audio_data to self.output_sr if necessary."""
        if current_sr == self.output_sr:
            return audio_data
        if audio_data is None or audio_data.size == 0:
            return audio_data  # Return empty if input is empty

        logger.debug(f"Resampling audio from {current_sr} Hz to {self.output_sr} Hz.")
        try:
            # Librosa expects (channels, samples) or (samples,) for mono
            # Input audio_data is expected as (samples, channels) or (samples,)
            if audio_data.ndim == 1:  # Mono
                resampled_audio = librosa.resample(y=audio_data, orig_sr=current_sr, target_sr=self.output_sr)
            elif audio_data.ndim == 2 and audio_data.shape[1] > 0:  # Stereo or more, use first channel if mono-like (N,1)
                # Transpose to (channels, samples) for librosa
                resampled_audio_transposed = librosa.resample(y=audio_data.T, orig_sr=current_sr, target_sr=self.output_sr)
                resampled_audio = resampled_audio_transposed.T  # Transpose back
            else:
                logger.warning(f"Unsupported audio shape for resampling: {audio_data.shape}. Returning original.")
                return audio_data
            return resampled_audio.astype(np.float32)
        except Exception as e:
            logger.error(f"Error during resampling: {e}", exc_info=True)
            return audio_data  # Return original on error

    def _ensure_stereo(self, audio_data: AudioData) -> AudioData:
        """Ensures audio data is stereo (samples, 2)."""
        if audio_data is None or audio_data.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if audio_data.ndim == 1:  # Mono
            return np.stack([audio_data, audio_data], axis=-1).astype(np.float32)
        elif audio_data.ndim == 2 and audio_data.shape[1] == 1:  # (N, 1)
            return np.concatenate([audio_data, audio_data], axis=1).astype(np.float32)
        elif audio_data.ndim == 2 and audio_data.shape[1] == 2:  # Already stereo
            return audio_data.astype(np.float32)
        else:
            logger.warning(f"Unexpected audio shape {audio_data.shape}. Attempting to use first two channels or duplicate first.")
            if audio_data.ndim == 2 and audio_data.shape[1] > 2:
                return audio_data[:, :2].astype(np.float32)
            # Fallback for other odd shapes - this might not be ideal
            return np.stack([audio_data.flatten(), audio_data.flatten()], axis=-1).astype(np.float32)

    def mix_subliminal(
        self, affirmation_audio_data: AudioData, affirmation_sr: SampleRate, background_audio_path: str | None, affirmation_volume_db_offset: float
    ) -> tuple[AudioData | None, SampleRate | None]:
        """
        Mixes affirmation audio with background audio.

        Args:
            affirmation_audio_data (AudioData): NumPy array of the affirmation audio.
            affirmation_sr (SampleRate): Sample rate of the affirmation audio.
            background_audio_path (str | None): Path to the background audio file.
            affirmation_volume_db_offset (float): Volume adjustment for affirmations (e.g., -20 dB).

        Returns:
            tuple[AudioData | None, SampleRate | None]: Mixed audio data and its sample rate, or (None, None) on failure.
        """
        if affirmation_audio_data is None or affirmation_audio_data.size == 0:
            logger.error("Affirmation audio is empty. Cannot mix.")
            return None, None

        # 1. Process Affirmation Audio
        processed_affirmation = self._resample_if_needed(affirmation_audio_data, affirmation_sr)
        processed_affirmation = self._ensure_stereo(processed_affirmation)

        # Apply volume adjustment (convert dB offset to linear scale)
        # Positive dB offset means louder, negative means quieter.
        # For subliminals, affirmations are typically quieter.
        affirmation_linear_gain = librosa.db_to_amplitude(affirmation_volume_db_offset)
        processed_affirmation *= affirmation_linear_gain

        target_mix_len_samples = processed_affirmation.shape[0]
        if target_mix_len_samples == 0:
            logger.error("Processed affirmation audio has zero length after processing.")
            return None, None

        logger.debug(f"Processed affirmation: {target_mix_len_samples} samples at {self.output_sr} Hz, gain: {affirmation_linear_gain:.4f}.")

        # 2. Process Background Audio
        processed_background = np.zeros((target_mix_len_samples, 2), dtype=np.float32)  # Default silent background

        if background_audio_path:
            logger.debug(f"Loading background audio from: {background_audio_path}")
            # load_audio from audio_io.py handles resampling to target_sr and ensures stereo
            bg_audio_data, bg_sr = load_audio(background_audio_path, target_sr=self.output_sr)

            if bg_audio_data is not None and bg_audio_data.size > 0 and bg_sr == self.output_sr:
                bg_audio_data = self._ensure_stereo(bg_audio_data)  # Ensure stereo again after load_audio
                bg_len = bg_audio_data.shape[0]

                if bg_len == 0:
                    logger.warning("Loaded background audio is empty.")
                elif bg_len < target_mix_len_samples:
                    logger.info(f"Looping background audio (len: {bg_len}) to match affirmation len ({target_mix_len_samples}).")
                    num_repeats = math.ceil(target_mix_len_samples / bg_len)
                    looped_bg = np.tile(bg_audio_data, (num_repeats, 1))
                    processed_background = looped_bg[:target_mix_len_samples, :]
                elif bg_len > target_mix_len_samples:
                    logger.info(f"Truncating background audio (len: {bg_len}) to match affirmation len ({target_mix_len_samples}).")
                    processed_background = bg_audio_data[:target_mix_len_samples, :]
                else:  # bg_len == target_mix_len_samples
                    processed_background = bg_audio_data

                # Background volume is typically 1.0 (no change) unless specified otherwise
                # For auto-subliminal, background is often at full volume or slightly reduced.
                # Let's assume background volume is 1.0 for now.
            else:
                logger.warning(f"Failed to load or process background audio from '{background_audio_path}'. Using silent background.")
        else:
            logger.info("No background audio path provided. Using silent background.")

        # 3. Mix (Summation)
        # Ensure both are stereo and have the same length
        if processed_affirmation.shape[0] != processed_background.shape[0] or processed_affirmation.shape[1] != 2 or processed_background.shape[1] != 2:
            logger.error(f"Shape mismatch before mixing. Affirmation: {processed_affirmation.shape}, Background: {processed_background.shape}. Cannot mix.")
            # Fallback: return only affirmations if background processing failed critically
            return processed_affirmation, self.output_sr

        mixed_audio = processed_affirmation + processed_background

        # 4. Clipping
        mixed_audio = np.clip(mixed_audio, -1.0, 1.0)
        logger.info(f"Mixing complete. Final audio shape: {mixed_audio.shape} at {self.output_sr} Hz.")

        return mixed_audio.astype(np.float32), self.output_sr

    def generate_preview(self, source_audio_data: AudioData, source_sr: SampleRate, duration_s: int) -> tuple[AudioData | None, SampleRate | None]:
        """
        Generates a preview by truncating the source audio data.
        """
        if source_audio_data is None or source_audio_data.size == 0:
            logger.warning("Cannot generate preview from empty source audio.")
            return None, None
        if source_sr <= 0:
            logger.warning(f"Invalid source sample rate for preview: {source_sr}")
            return None, None

        target_samples = int(duration_s * source_sr)
        if target_samples <= 0:
            logger.warning(f"Preview duration ({duration_s}s) results in zero samples.")
            return np.zeros((0, source_audio_data.shape[1] if source_audio_data.ndim == 2 else 0), dtype=np.float32), source_sr

        if source_audio_data.ndim == 1:  # Mono
            preview_audio_data = source_audio_data[: min(source_audio_data.shape[0], target_samples)]
        elif source_audio_data.ndim == 2:  # Stereo or more
            preview_audio_data = source_audio_data[: min(source_audio_data.shape[0], target_samples), :]
        else:
            logger.error(f"Unsupported audio shape for preview: {source_audio_data.shape}")
            return None, None

        logger.info(f"Generated preview audio of shape {preview_audio_data.shape} for {duration_s}s at {source_sr}Hz.")
        return preview_audio_data.astype(np.float32), source_sr


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create a dummy affirmation audio
    sr_affirm = 22050
    duration_affirm = 15  # seconds
    affirm_samples = int(sr_affirm * duration_affirm)
    dummy_affirmation_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration_affirm, affirm_samples, endpoint=False), dtype=np.float32)
    dummy_affirmation_audio = np.stack([dummy_affirmation_audio, dummy_affirmation_audio], axis=-1)  # Stereo

    # Create a dummy background sound file for testing load_audio
    test_bg_dir = "temp_test_bg_sounds"
    if not os.path.exists(test_bg_dir):
        os.makedirs(test_bg_dir)

    sr_bg = 44100
    duration_bg = 5  # seconds
    bg_samples = int(sr_bg * duration_bg)
    dummy_bg_audio_data = np.cos(2 * np.pi * 220 * np.linspace(0, duration_bg, bg_samples, endpoint=False), dtype=np.float32) * 0.5
    dummy_bg_audio_data_stereo = np.stack([dummy_bg_audio_data, dummy_bg_audio_data], axis=-1)

    # Save dummy background to a WAV file using audio_io's save_audio_to_bytesio logic (adapted)
    # For testing, we'll use soundfile directly here.
    import soundfile as sf

    dummy_bg_path = os.path.join(test_bg_dir, "dummy_background.wav")
    sf.write(dummy_bg_path, (dummy_bg_audio_data_stereo * 32767).astype(np.int16), sr_bg)
    print(f"Created dummy background file: {dummy_bg_path}")

    processor = AutoSubliminalAudioProcessor(output_sr=GLOBAL_SR)  # GLOBAL_SR from main config

    print("\n--- Testing mix_subliminal ---")
    mixed_audio, mixed_sr = processor.mix_subliminal(
        affirmation_audio_data=dummy_affirmation_audio, affirmation_sr=sr_affirm, background_audio_path=dummy_bg_path, affirmation_volume_db_offset=-20.0
    )

    if mixed_audio is not None and mixed_sr is not None:
        print(f"Mixed audio generated: shape={mixed_audio.shape}, sr={mixed_sr}")
        expected_len_samples = int(duration_affirm * GLOBAL_SR)  # Affirmation duration at target SR
        print(f"Expected length approx: {expected_len_samples} samples. Actual: {mixed_audio.shape[0]}")
        assert mixed_sr == GLOBAL_SR, "Mixed audio SR should match processor's output_sr"
        # Basic check for length (can vary slightly due to resampling precision)
        # assert abs(mixed_audio.shape[0] - expected_len_samples) < GLOBAL_SR * 0.1, "Mixed audio length is too different from expected."

        print("\n--- Testing generate_preview ---")
        preview_duration = 5  # seconds
        preview_audio, preview_sr = processor.generate_preview(mixed_audio, mixed_sr, preview_duration)
        if preview_audio is not None and preview_sr is not None:
            print(f"Preview audio generated: shape={preview_audio.shape}, sr={preview_sr}")
            expected_preview_samples = int(preview_duration * mixed_sr)
            assert preview_audio.shape[0] == expected_preview_samples, "Preview length mismatch"
            assert preview_sr == mixed_sr, "Preview SR mismatch"
        else:
            print("Preview generation failed.")
    else:
        print("Mixing failed.")

    # Clean up dummy file and dir
    if os.path.exists(dummy_bg_path):
        os.remove(dummy_bg_path)
    if os.path.exists(test_bg_dir):
        os.rmdir(test_bg_dir)
    print("\nCleaned up dummy files.")
