# auto_subliminal/background_sound_manager.py
# Manages selection and retrieval of background sounds.

import logging
import os
import random

# from pydub import AudioSegment # Example: if you want to get duration here

# Import from main application config (e.g., if there's a global assets path)
# from config import ASSETS_DIR as GLOBAL_ASSETS_DIR (if defined and needed)

logger = logging.getLogger(__name__)

# Default path for background sounds.
# This assumes your script is run from the project root, or this path is adjusted.
# A more robust way is to define ASSETS_DIR in your main config.py and import it.
# For this module, we'll construct it assuming a standard project structure.
try:
    # Try to determine project root based on this file's location
    # auto_subliminal -> parent (project_root)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DEFAULT_SOUNDS_DIRECTORY = os.path.join(PROJECT_ROOT, "assets", "background_sounds")
except NameError:  # __file__ is not defined (e.g. in some interactive environments)
    PROJECT_ROOT = os.getcwd()
    DEFAULT_SOUNDS_DIRECTORY = os.path.join(PROJECT_ROOT, "assets", "background_sounds")


class BackgroundSoundManager:
    """
    Selects background sounds for the subliminal audio.
    Provides path, name, and potentially duration of sounds.
    """

    def __init__(self, sounds_directory: str = DEFAULT_SOUNDS_DIRECTORY):
        """
        Initializes the BackgroundSoundManager.

        Args:
            sounds_directory (str): Path to the directory containing background sound files.
                                    Defaults to '<project_root>/assets/background_sounds'.
        """
        self.sounds_directory = sounds_directory
        self.available_sounds = self._load_available_sounds()  # List of dicts

        if not self.available_sounds:
            logger.warning(f"No background sounds found or loaded from directory: '{self.sounds_directory}'. Background sound selection will be limited or disabled.")
        else:
            logger.info(f"Loaded {len(self.available_sounds)} background sounds from '{self.sounds_directory}'.")

    def _load_available_sounds(self) -> list[dict]:
        """
        Scans the sounds directory and loads a list of available sound file paths and names.
        Future: Could also load duration here using a library like pydub or librosa.
        """
        if not os.path.isdir(self.sounds_directory):
            logger.error(f"Sounds directory not found: '{self.sounds_directory}'")
            return []

        supported_formats = (".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a")  # Common audio formats
        sounds_info = []
        try:
            for filename in os.listdir(self.sounds_directory):
                if filename.lower().endswith(supported_formats) and not filename.startswith("."):
                    full_path = os.path.join(self.sounds_directory, filename)
                    duration_s = None  # Placeholder for actual duration loading
                    # Example: Using pydub to get duration (requires pydub and ffmpeg/ffprobe)
                    # try:
                    #     audio = AudioSegment.from_file(full_path)
                    #     duration_s = len(audio) / 1000.0 # pydub duration is in ms
                    # except Exception as e:
                    #     logger.warning(f"Could not get duration for '{filename}': {e}")
                    sounds_info.append({"name": filename, "path": full_path, "duration_s": duration_s})
        except OSError as e:
            logger.error(f"Error accessing sounds directory '{self.sounds_directory}': {e}", exc_info=True)
            return []
        except Exception as e:  # Catch any other unexpected errors during listing/processing
            logger.error(f"Unexpected error loading sounds from '{self.sounds_directory}': {e}", exc_info=True)
            return []

        if not sounds_info:
            logger.warning(f"No supported audio files found in '{self.sounds_directory}'.")
        return sounds_info

    def select_sound_info(self, topic: str = None, preferred_sound_name: str = None) -> dict | None:  # type: ignore
        """
        Selects a background sound and returns its info (path, name, duration).

        Args:
            topic (str, optional): The topic of the subliminal. (Currently not used for selection logic).
            preferred_sound_name (str, optional): A specific sound filename (e.g., "rain.mp3").

        Returns:
            dict | None: A dictionary with {"name": str, "path": str, "duration_s": float | None}
                         for the selected sound, or None if no sound is available/selected.
        """
        if not self.available_sounds:
            logger.warning("No background sounds available for selection.")
            return None

        if preferred_sound_name:
            for sound_info in self.available_sounds:
                if preferred_sound_name.lower() == sound_info["name"].lower():
                    logger.info(f"Using preferred background sound: {sound_info['name']}")
                    return sound_info
            logger.warning(f"Preferred sound '{preferred_sound_name}' not found in available sounds. Selecting randomly.")

        # Current logic: Random selection.
        # Future enhancements:
        # - Select based on topic (e.g., "ocean sounds" for "relaxation")
        # - Allow user to specify a category (e.g., "nature", "noise", "ambient")
        try:
            selected_sound = random.choice(self.available_sounds)
            logger.info(f"Randomly selected background sound: {selected_sound['name']} (Topic: '{topic}')")
            return selected_sound
        except IndexError:  # Should not happen if self.available_sounds is checked, but as a safeguard
            logger.error("Cannot select sound, available_sounds list is unexpectedly empty after check.")
            return None

    def get_available_sounds_list(self) -> list[dict]:
        """Returns a list of available sounds with their names, paths, and durations."""
        return self.available_sounds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # For testing, ensure the DEFAULT_SOUNDS_DIRECTORY or a test one exists and has files
    print(f"Attempting to use sound directory: {DEFAULT_SOUNDS_DIRECTORY}")
    if not os.path.exists(DEFAULT_SOUNDS_DIRECTORY):
        os.makedirs(DEFAULT_SOUNDS_DIRECTORY)
        logger.info(f"Created dummy directory for testing: {DEFAULT_SOUNDS_DIRECTORY}")
        # Create some dummy files for testing
        with open(os.path.join(DEFAULT_SOUNDS_DIRECTORY, "test_rain.mp3"), "w") as f:
            f.write("dummy mp3")
        with open(os.path.join(DEFAULT_SOUNDS_DIRECTORY, "test_forest.wav"), "w") as f:
            f.write("dummy wav")

    manager = BackgroundSoundManager()  # Uses default directory

    print("\nAvailable sounds in manager:")
    available = manager.get_available_sounds_list()
    if available:
        for s_info in available:
            print(f"  - Name: {s_info['name']}, Path: {s_info['path']}, Duration: {s_info['duration_s']}s")
    else:
        print("  No sounds loaded by manager.")

    selected_info = manager.select_sound_info(topic="general relaxation")
    if selected_info:
        print(f"\nSelected sound for 'general relaxation': {selected_info['name']} (Path: {selected_info['path']})")
    else:
        print("\nNo sound selected for 'general relaxation'.")

    selected_pref_info = manager.select_sound_info(topic="focused study", preferred_sound_name="test_rain.mp3")
    if selected_pref_info:
        print(f"\nSelected sound with preference 'test_rain.mp3': {selected_pref_info['name']}")
    else:
        print("\nPreferred sound 'test_rain.mp3' not found or no sounds available.")

    # Clean up dummy files (optional)
    # if os.path.exists(os.path.join(DEFAULT_SOUNDS_DIRECTORY, "test_rain.mp3")):
    #     os.remove(os.path.join(DEFAULT_SOUNDS_DIRECTORY, "test_rain.mp3"))
    # if os.path.exists(os.path.join(DEFAULT_SOUNDS_DIRECTORY, "test_forest.wav")):
    #     os.remove(os.path.join(DEFAULT_SOUNDS_DIRECTORY, "test_forest.wav"))
    # if os.path.exists(DEFAULT_SOUNDS_DIRECTORY) and not os.listdir(DEFAULT_SOUNDS_DIRECTORY):
    #     os.rmdir(DEFAULT_SOUNDS_DIRECTORY)
