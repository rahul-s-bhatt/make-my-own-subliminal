# tts/base_tts.py
# ==========================================
# Abstract Base Class for TTS Generators
# ==========================================

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

# Define type hints used within this module
AudioData = np.ndarray
SampleRate = int

logger = logging.getLogger(__name__)


class BaseTTSGenerator(ABC):
    """
    Abstract base class defining the interface for TTS generators.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the TTS engine and any necessary resources.
        Specific implementations will define required kwargs.
        """
        pass

    @abstractmethod
    def generate(self, text: str) -> Tuple[AudioData, SampleRate]:
        """
        Generates audio data from the input text using the specific TTS engine.

        Args:
            text: The text content to synthesize.

        Returns:
            A tuple containing:
                - The generated audio data as a NumPy array (float32, stereo).
                - The sample rate of the generated audio (should match GLOBAL_SR
                  after potential resampling by the implementation).

        Raises:
            ValueError: If input text is empty or invalid.
            RuntimeError: If any critical error occurs during synthesis.
        """
        pass

    # Optional: Add methods for listing voices, setting properties etc. if needed
    # def list_voices(self) -> List[str]:
    #     raise NotImplementedError
    #
    # def set_voice(self, voice_id: str):
    #     raise NotImplementedError
