# config.py
# ==========================================
# Configuration and Constants for MindMorph
# ==========================================

import os
from typing import Any, Dict

import numpy as np

# --- Core Settings ---
GLOBAL_SR: int = 44100  # Global Sample Rate in Hz
PROJECT_FILE_VERSION: str = "1.0"  # For future compatibility checks of saved projects

# --- Audio Processing ---
ULTRASONIC_TARGET_FREQ: int = 17500  # Target frequency in Hz for "silent" shift
ULTRASONIC_FILTER_CUTOFF: int = 17000  # Low-pass filter slightly below target
ULTRASONIC_FILTER_ORDER: int = 8  # Steep filter order

# --- UI / Preview ---
PREVIEW_DURATION_S: int = 60  # Default duration for track previews in editor (seconds)
MIX_PREVIEW_DURATION_S: int = 10  # Duration for the master mix preview (seconds)
MIX_PREVIEW_PROCESSING_BUFFER_S: int = 5  # Extra seconds to process for preview to handle speed changes

# --- TTS ---
TTS_CHUNK_SIZE: int = 1500  # Max characters per chunk for TTS generation

# --- Resource Limits ---
MAX_AUDIO_DURATION_S: int = 300  # 5 minutes max for uploaded/generated audio
MAX_AFFIRMATION_CHARS: int = 5000  # Max characters for TTS input

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
# Used when creating a new track or loading projects to ensure all keys exist
def get_default_track_params() -> Dict[str, Any]:
    """Returns a dictionary with default parameters for an audio track."""
    return {
        "original_audio": np.zeros((0, 2), dtype=np.float32),  # Placeholder for audio data
        "sr": GLOBAL_SR,  # Sample rate
        "name": "New Track",  # Default track name
        "track_type": TRACK_TYPE_OTHER,  # Default track category
        "volume": 1.0,  # Volume multiplier (0.0 to 2.0)
        "mute": False,  # Muted state in the mix
        "solo": False,  # Soloed state in the mix
        "speed_factor": 1.0,  # Playback speed multiplier
        "pitch_shift": 0,  # Pitch shift in semitones
        "pan": 0.0,  # Stereo panning (-1.0 Left to 1.0 Right)
        "filter_type": "off",  # Filter type ('off', 'lowpass', 'highpass')
        "filter_cutoff": 8000.0,  # Filter cutoff frequency in Hz
        "loop_to_fit": False,  # Loop track during final mix
        "reverse_audio": False,  # Reverse audio playback
        "ultrasonic_shift": False,  # Apply ultrasonic shift effect
        "preview_temp_file_path": None,  # Path to the temporary preview audio file
        "preview_settings_hash": None,  # Hash of settings used for the current preview
        "update_counter": 0,  # Counter to help refresh UI elements like audix
        # --- Source/Generation Information ---
        "source_type": "unknown",  # How the track was created ('upload', 'tts', 'noise', 'binaural', etc.)
        "original_filename": None,  # Original filename if source_type is 'upload'
        "tts_text": None,  # Text used if source_type is 'tts'
        # --- Generation Parameters (for reconstruction on load) ---
        "gen_duration": None,  # Duration used for generated tracks
        "gen_freq_left": None,  # Left frequency for binaural
        "gen_freq_right": None,  # Right frequency for binaural
        "gen_freq": None,  # Frequency for Solfeggio
        "gen_carrier_freq": None,  # Carrier frequency for Isochronic
        "gen_pulse_freq": None,  # Pulse frequency for Isochronic
        "gen_noise_type": None,  # Type of noise generated
        "gen_volume": None,  # Volume used during generation
    }


# --- Logging ---
LOG_FILE: str = "editor_oop.log"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s"
LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: int = 3

# --- Asset Paths ---
# Define paths for assets like icons and logos
ASSETS_DIR: str = "assets"
FAVICON_PATH: str = os.path.join(ASSETS_DIR, "favico.png")
LOGO_PATH: str = os.path.join(ASSETS_DIR, "logo.png")
