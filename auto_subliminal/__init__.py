# auto_subliminal/__init__.py
# This file makes Python treat the `auto_subliminal` directory as a package.

# You can make key classes or functions available for easier import if desired, e.g.:
from . import config as auto_sub_config  # Allow access to feature config
from .affirmation_generator import AffirmationGenerator
from .background_sound_manager import BackgroundSoundManager
from .generator import AutoSubliminalGenerator
from .output_handler import OutputHandler

# This helps in organizing imports when using this module elsewhere:
# from auto_subliminal import AutoSubliminalGenerator, auto_sub_config
