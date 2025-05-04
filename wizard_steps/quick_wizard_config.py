# wizard_steps/quick_wizard_config.py
# ==========================================
# Central configuration for the Quick Create Wizard
# Contains session state keys, default values, and static lists.
# ==========================================

# --- Session State Keys ---

# General
WIZARD_STEP_KEY = "wizard_step"
WIZARD_PROCESSING_ACTIVE_KEY = "wizard_processing_active"  # For final export
WIZARD_PREVIEW_ACTIVE_KEY = "wizard_preview_active"  # For preview generation <--- NEW KEY

# Step 1: Affirmations
AFFIRMATION_TEXT_KEY = "wizard_affirmation_text"
AFFIRMATION_SOURCE_KEY = "wizard_affirmation_source"  # Though currently hardcoded to 'text'
AFFIRM_APPLY_SPEED_KEY = "wizard_apply_speed_change"  # Boolean flag for speed change
AFFIRMATION_VOLUME_KEY = "wizard_affirmation_volume"

# Step 2: Background
BG_CHOICE_KEY = "wizard_background_choice"  # 'none', 'upload', 'noise'
BG_CHOICE_LABEL_KEY = "wizard_background_choice_label"  # User-facing label
BG_UPLOADED_FILE_KEY = "wizard_background_uploaded_file"  # Stores UploadedFile object
BG_NOISE_TYPE_KEY = "wizard_background_noise_type"  # String name of noise
BG_VOLUME_KEY = "wizard_background_volume"

# Step 3: Frequency
FREQ_CHOICE_KEY = "wizard_frequency_choice"  # 'None', 'Binaural Beats', 'Isochronic Tones'
FREQ_PARAMS_KEY = "wizard_frequency_params"  # Dict storing parameters
FREQ_VOLUME_KEY = "wizard_frequency_volume"

# Step 4: Export / Preview Results
OUTPUT_FILENAME_KEY = "wizard_output_filename"
EXPORT_FORMAT_KEY = "wizard_export_format"
EXPORT_BUFFER_KEY = "wizard_export_buffer"
EXPORT_ERROR_KEY = "wizard_export_error"
PREVIEW_BUFFER_KEY = "wizard_preview_buffer"
PREVIEW_ERROR_KEY = "wizard_preview_error"

# Legacy Audio Keys (might still be needed for cleanup logic)
LEGACY_AFFIRM_AUDIO_KEY = "wizard_affirmation_audio"
LEGACY_AFFIRM_SR_KEY = "wizard_affirmation_sr"
LEGACY_BG_AUDIO_KEY = "wizard_background_audio"
LEGACY_BG_SR_KEY = "wizard_background_sr"
LEGACY_FREQ_AUDIO_KEY = "wizard_frequency_audio"
LEGACY_FREQ_SR_KEY = "wizard_frequency_sr"


# --- Default Values ---
DEFAULT_STEP = 1
DEFAULT_PROCESSING_ACTIVE = False  # Default for export processing
DEFAULT_PREVIEW_ACTIVE = False  # Default for preview processing <--- NEW DEFAULT

DEFAULT_AFFIRMATION_TEXT = ""
DEFAULT_AFFIRMATION_SOURCE = "text"
DEFAULT_APPLY_SPEED = False
DEFAULT_AFFIRMATION_VOLUME = 1.0

DEFAULT_BG_CHOICE = "none"
DEFAULT_BG_CHOICE_LABEL = "None"
DEFAULT_BG_UPLOADED_FILE = None
DEFAULT_NOISE_TYPE = "Brown Noise"  # Choose your preferred default
DEFAULT_BG_VOLUME = 0.7

DEFAULT_FREQ_CHOICE = "None"
DEFAULT_FREQ_PARAMS = {}  # Initialize empty, specific defaults handled in Step 3 UI
DEFAULT_FREQ_VOLUME = 0.2

DEFAULT_OUTPUT_FILENAME = "mindmorph_quick_mix"
DEFAULT_EXPORT_FORMAT = "WAV"

# --- Static Lists / Options ---
NOISE_TYPES = ["White Noise", "Pink Noise", "Brown Noise"]
FREQUENCY_TYPES = ["None", "Binaural Beats", "Isochronic Tones"]
EXPORT_FORMATS = ["WAV", "MP3"]  # MP3 availability checked elsewhere

# Frequency Parameter Defaults (used in Step 3 UI)
DEFAULT_BASE_FREQ = 100.0
DEFAULT_BEAT_FREQ = 5.0  # For Binaural
DEFAULT_PULSE_FREQ = 7.0  # For Isochronic
