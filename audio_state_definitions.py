# audio_state_definitions.py
# ==========================================
# Type Definitions for MindMorph Application State
# ==========================================

from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union  # Use Union for SourceInfo

import numpy as np

# --- Basic Types ---
TrackID = str
AudioData = np.ndarray  # Assuming AudioData is just a numpy array
TrackType = str  # Define TrackType as a string alias
SampleRate = int

# --- Source Information Structures ---
# Using TypedDict for better structure and type checking


class SourceInfoUpload(TypedDict):
    type: str  # Literal['upload'] # Use Literal in Python 3.8+ for more safety
    temp_file_path: Optional[str]  # Path to the temporarily stored full uploaded file (Optional before re-upload)
    original_filename: str


class SourceInfoTTS(TypedDict):
    type: str  # Literal['tts']
    text: str
    # Add any specific TTS params if needed, e.g., voice, speed (if not track params)
    # tts_params: Dict[str, Any]


class SourceInfoNoise(TypedDict):
    type: str  # Literal['noise']
    noise_type: str
    target_duration_s: Optional[float]  # Hint for regeneration length
    # Volume is a track param, not source info


class SourceInfoFrequency(TypedDict):
    type: str  # Literal['frequency']
    freq_type: str  # Literal['binaural', 'isochronic', 'solfeggio']
    target_duration_s: Optional[float]  # Hint for regeneration length
    # Specific params needed for generation
    f_left: Optional[float]
    f_right: Optional[float]
    freq: Optional[float]
    carrier: Optional[float]
    pulse: Optional[float]


# Union type for source_info dictionary
SourceInfo = Union[SourceInfoUpload, SourceInfoTTS, SourceInfoNoise, SourceInfoFrequency]


# --- Track Data Structure ---


class TrackDataDict(TypedDict):
    # Core data
    audio_snippet: Optional[AudioData]  # Stores the audio snippet (e.g., 30s)
    source_info: Optional[SourceInfo]  # Info to reload/regen full audio
    sr: int
    # Metadata & Controls
    name: str
    track_type: TrackType  # Use the defined TrackType alias
    volume: float
    mute: bool
    solo: bool
    speed_factor: float
    pitch_shift: int  # Semitones
    pan: float  # -1.0 (L) to 1.0 (R)
    filter_type: str
    filter_cutoff: float
    loop_to_fit: bool
    reverse_audio: bool
    ultrasonic_shift: bool
    # Preview state
    preview_temp_file_path: Optional[str]
    preview_settings_hash: Optional[str]
    update_counter: int
    # Optional track ID within the dict itself if needed, though usually the dict key
    # track_id: Optional[TrackID]
