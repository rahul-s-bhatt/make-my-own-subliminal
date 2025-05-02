# config.py
# ==========================================
# Configuration and Constants for MindMorph
# ==========================================

import os
from pathlib import Path
from typing import Any, Dict

# --- Core Settings ---
GLOBAL_SR: int = 22050  # Global Sample Rate in Hz
PROJECT_FILE_VERSION: str = "1.0"  # For future compatibility checks of saved projects

# --- External Services / Integrations ---
GA_MEASUREMENT_ID: str = "G-B5LWHH5H7N"  # Replace with your actual ID if needed
GOOGLE_FORM_URL: str = "https://forms.gle/eXGtvAzEoEZCHpK69"  # Replace with your actual URL if needed
PATERON_URL: str = "https://patreon.com/MindMorphing?utm_medium=unknown&utm_source=join_link&utm_campaign=creatorshare_creator&utm_content=copyLink"


# --- Audio Processing ---
ULTRASONIC_TARGET_FREQ: int = 17500  # Target frequency in Hz for "silent" shift
ULTRASONIC_FILTER_CUTOFF: int = 17000  # Low-pass filter slightly below target
ULTRASONIC_FILTER_ORDER: int = 8  # Steep filter order

# --- UI / Preview ---
PREVIEW_DURATION_S: int = 30  # Default duration for track previews in editor (seconds)
MIX_PREVIEW_DURATION_S: int = 10  # Duration for the master mix preview (seconds)
MIX_PREVIEW_PROCESSING_BUFFER_S: int = 5  # Extra seconds to process for preview to handle speed changes
TRACK_SNIPPET_DURATION_S: int = 30  # Duration (in seconds) of audio loaded into memory for editing/preview (Uploads/TTS)
GENERATOR_SNIPPET_DURATION_S: int = 10  # Duration (in seconds) for generated noise/frequency snippets

# --- TTS ---
TTS_CHUNK_SIZE: int = 1500  # Max characters per chunk for TTS generation

# --- Resource Limits ---
MAX_AUDIO_DURATION_S: int = 300  # 5 minutes max for uploaded/generated audio (UI Hint)
MAX_AFFIRMATION_CHARS: int = 5000  # Max characters for TTS input
MAX_TRACK_LIMIT: int = 5  # Maximum number of tracks allowed in the Advanced Editor
MAX_UPLOAD_SIZE_MB: int = 10  # Maximum size for uploaded files in Megabytes
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Calculated size in bytes

# --- Presets ---
# <<< ADDED: Constants for the Quick Subliminal Preset >>>
QUICK_SUBLIMINAL_PRESET_SPEED: float = 2.0
QUICK_SUBLIMINAL_PRESET_VOLUME: float = 0.05

# --- PIPER VOICES ---
# PIPER_VOICE_MODEL_PATH = r"assets//voices//female//kristin//medium//en_US-kristin-medium.onnx"  # e.g., en_US-lessac-medium.onnx
# PIPER_VOICE_CONFIG_PATH = r"assets//voices//female//kristin//medium//en_US-kristin-medium.onnx.json"  # e.g., en_US-lessac-medium.onnx.json
# # Assuming your config.py is at the project root or you can determine the root
# # BASE_DIR = Path(__file__).resolve().parent # If config.py is at root
# # Or define it relative to your execution context
BASE_DIR = Path(".").resolve()  # Path relative to where the script is run

ASSETS_DIR = BASE_DIR / "assets"  # type: ignore
VOICES_DIR = ASSETS_DIR / "voices" / "female" / "kristin" / "medium"  # type: ignore

PIPER_VOICE_MODEL_PATH = str(VOICES_DIR / "en_US-kristin-medium.onnx")
PIPER_VOICE_CONFIG_PATH = str(VOICES_DIR / "en_US-kristin-medium.onnx.json")


# --- Track Types ---
# Define constants for track types for consistency
TRACK_TYPE_AFFIRMATION: str = "🗣️ Affirmation"
TRACK_TYPE_BACKGROUND: str = "🎵 Background/Mask"
TRACK_TYPE_FREQUENCY: str = "🧠 Frequency"
TRACK_TYPE_VOICE: str = "🎤 Voice Recording"
TRACK_TYPE_OTHER: str = "⚪ Other"

# List of all available track types
TRACK_TYPES: list[str] = [
    TRACK_TYPE_AFFIRMATION,
    TRACK_TYPE_BACKGROUND,
    TRACK_TYPE_FREQUENCY,
    TRACK_TYPE_VOICE,
    TRACK_TYPE_OTHER,
]


# --- Default Track Parameters ---
def get_default_track_params() -> Dict[str, Any]:
    """Returns a dictionary with default parameters for an audio track."""
    return {
        "sr": GLOBAL_SR,
        "name": "New Track",
        "track_type": TRACK_TYPE_OTHER,
        "volume": 1.0,
        "mute": False,
        "solo": False,
        "speed_factor": 1.0,
        "pitch_shift": 0,
        "pan": 0.0,
        "filter_type": "off",
        "filter_cutoff": 8000.0,
        "loop_to_fit": False,
        "reverse_audio": False,
        "ultrasonic_shift": False,
        "preview_temp_file_path": None,
        "preview_settings_hash": None,
        "update_counter": 0,
    }


# --- Logging ---
LOG_FILE: str = "editor_oop.log"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s"
LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: int = 3

# --- Asset Paths ---
ASSETS_DIR: str = "assets"
FAVICON_PATH: str = os.path.join(ASSETS_DIR, "favico.png")
LOGO_PATH: str = os.path.join(ASSETS_DIR, "logo.png")
