# auto_subliminal/output_handler.py
# Handles saving the generated affirmations text and audio files for the auto-subliminal feature.

import datetime
import logging
import os
import re  # For sanitizing topic for filename

# Import from feature-specific config
from .config import AUTO_SUB_OUTPUT_SUBDIR

# Base output directory can be defined in main config.py or passed during instantiation.
# For this example, we'll use a default that assumes it's created at the project root.
DEFAULT_APP_BASE_OUTPUT_DIRECTORY = "mindmorph_generations"  # Example global output dir

logger = logging.getLogger(__name__)


class OutputHandler:
    """
    Manages saving of generated subliminal files for the auto feature.
    Files will be saved under <app_base_output_directory>/<AUTO_SUB_OUTPUT_SUBDIR>/
    """

    def __init__(self, app_base_output_directory: str = DEFAULT_APP_BASE_OUTPUT_DIRECTORY):
        """
        Initializes the OutputHandler.

        Args:
            app_base_output_directory (str): The main application directory where all outputs are stored.
                                           The auto-generated files will be in a subdirectory of this.
        """
        self.feature_output_directory = os.path.join(app_base_output_directory, AUTO_SUB_OUTPUT_SUBDIR)
        self._ensure_output_directory_exists()

    def _ensure_output_directory_exists(self):
        """Creates the feature-specific output directory if it doesn't exist."""
        try:
            os.makedirs(self.feature_output_directory, exist_ok=True)
            logger.info(f"Auto subliminal output directory ensured at: '{self.feature_output_directory}'")
        except OSError as e:
            logger.error(f"Could not create auto subliminal output directory '{self.feature_output_directory}': {e}", exc_info=True)
            # Depending on desired behavior, could raise exception or allow operations to fail saving.

    def _generate_filename_prefix(self, topic: str) -> str:
        """
        Generates a safe filename prefix based on the topic and current timestamp.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize topic: remove special characters, replace spaces, limit length
        safe_topic = topic.lower()
        safe_topic = re.sub(r"\s+", "_", safe_topic)  # Replace spaces with underscores
        safe_topic = re.sub(r"[^\w_.-]", "", safe_topic)  # Remove non-alphanumeric (except _, ., -)

        # Prevent overly long filenames from very long topics
        max_topic_len_in_filename = 40
        safe_topic = safe_topic[:max_topic_len_in_filename].strip("_.-")  # Trim leading/trailing problematic chars

        if not safe_topic:  # If topic was all special characters
            safe_topic = "untitled"

        return f"{safe_topic}_{timestamp}"

    def save_affirmations_text(self, topic: str, affirmations: list[str]) -> str | None:
        """
        Saves the list of affirmations to a text file in the feature's output directory.

        Args:
            topic (str): The topic of the affirmations.
            affirmations (list[str]): The list of affirmation strings.

        Returns:
            str | None: The full path to the saved text file, or None on failure.
        """
        if not affirmations:
            logger.warning("No affirmations provided to save for auto-subliminal feature.")
            return None

        filename_prefix = self._generate_filename_prefix(topic)
        # Ensure directory exists before attempting to write
        if not os.path.exists(self.feature_output_directory):
            logger.error(f"Output directory '{self.feature_output_directory}' does not exist. Cannot save text file.")
            return None
        filepath = os.path.join(self.feature_output_directory, f"{filename_prefix}_affirmations.txt")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Automatically Generated Affirmations for Topic: {topic}\n")
                f.write("=" * (len("Automatically Generated Affirmations for Topic: ") + len(topic)) + "\n\n")
                for i, aff in enumerate(affirmations):
                    f.write(f"{i + 1}. {aff}\n")
            logger.info(f"Auto-generated affirmations text saved to: '{filepath}'")
            return filepath
        except IOError as e:
            logger.error(f"IOError saving auto-generated affirmations text to '{filepath}': {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error saving auto-generated affirmations text to '{filepath}': {e}", exc_info=True)
            return None

    def save_audio_file(self, topic: str, audio_data: bytes, filename_suffix: str, audio_format: str) -> str | None:
        """
        Saves audio data (bytes) to a file in the feature's output directory.

        Args:
            topic (str): The topic, used for naming.
            audio_data (bytes): The raw audio data to save.
            filename_suffix (str): Suffix for the filename (e.g., "final_subliminal", "raw_affirmations").
            audio_format (str): The desired audio format/extension (e.g., "mp3", "wav").

        Returns:
            str | None: The full path to the saved audio file, or None on failure.
        """
        if not audio_data:
            logger.warning(f"No audio data provided for auto-gen topic '{topic}' with suffix '{filename_suffix}'.")
            return None
        if not audio_format:
            logger.error("Audio format must be specified for saving audio file.")
            return None

        filename_prefix = self._generate_filename_prefix(topic)
        # Ensure directory exists
        if not os.path.exists(self.feature_output_directory):
            logger.error(f"Output directory '{self.feature_output_directory}' does not exist. Cannot save audio file.")
            return None
        filepath = os.path.join(self.feature_output_directory, f"{filename_prefix}_{filename_suffix}.{audio_format.lower()}")

        try:
            with open(filepath, "wb") as f:
                f.write(audio_data)
            logger.info(f"Auto-generated audio file saved: '{filepath}'")
            return filepath
        except IOError as e:
            logger.error(f"IOError saving auto-generated audio file to '{filepath}': {e}", exc_info=True)
            return None
        except Exception as e:  # Catch other potential errors
            logger.error(f"An unexpected error occurred while saving auto-generated audio to '{filepath}': {e}", exc_info=True)
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Test with a specific base directory for this test run
    test_app_base_dir = "test_mindmorph_outputs"
    handler = OutputHandler(app_base_output_directory=test_app_base_dir)
    print(f"Test output will be in subdirectory under: '{test_app_base_dir}' (specifically: '{handler.feature_output_directory}')")

    sample_topic = "Deep Sleep & Relaxation!"  # Topic with special char
    sample_affirmations = ["I fall asleep easily and peacefully.", "My sleep is deep and restorative.", "I wake up feeling refreshed and energized."]

    txt_path = handler.save_affirmations_text(sample_topic, sample_affirmations)
    if txt_path:
        print(f"Affirmations text saved to: {txt_path}")
        # Example: Check content
        # with open(txt_path, "r", encoding="utf-8") as f_read:
        #     print("--- File Content ---")
        #     print(f_read.read())
        #     print("--- End Content ---")

    dummy_audio_bytes = b"\xca\xfe\xba\xbe" * 1024  # Some dummy bytes

    raw_audio_path = handler.save_audio_file(
        sample_topic,
        dummy_audio_bytes,
        "raw_affirmations_audio",
        "wav",  # Using format from feature config (conceptually)
    )
    if raw_audio_path:
        print(f"Dummy raw audio saved to: {raw_audio_path}")

    final_audio_path = handler.save_audio_file(
        sample_topic,
        dummy_audio_bytes,
        "final_subliminal_audio",
        "mp3",  # Using format from feature config (conceptually)
    )
    if final_audio_path:
        print(f"Dummy final audio saved to: {final_audio_path}")

    # To clean up the test directory:
    # import shutil
    # if os.path.exists(test_app_base_dir):
    #     shutil.rmtree(test_app_base_dir)
    #     print(f"\nCleaned up test directory: '{test_app_base_dir}'")
