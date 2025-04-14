import os
import tempfile
from abc import ABC, abstractmethod

import streamlit as st
from pydub import AudioSegment
from pydub.generators import Sine


class ModeWrapper:
    def __init__(self):
        self.modes = [
            MusicMode(),
            WhisperLayerMode(),
            EmbedTonesMode(),
            SolfeggioMode(),
            IsochronicMode(),
            MorphicFieldMode()
        ]

    def initialize_all(self):
        for mode in self.modes:
            mode.initialize()

    def apply_all(self, voice: AudioSegment):
        for mode in self.modes:
            voice = mode.modify_voice(voice)
        return voice


class VoiceMode(ABC):
    def __init__(self):
        self.voice = None

    @abstractmethod
    def initialize(self):
        """Setup streamlit UI components"""
        pass

    @abstractmethod
    def modify_voice(self, voice: AudioSegment) -> AudioSegment:
        """Modify voice"""


class MusicMode(VoiceMode):
    def __init__(self):
        super().__init__()
        self.volume_mix = 0

    def initialize(self):
        self.music_file = st.file_uploader(
            "🎼 Upload Background Music (mp3 or wav):", type=["mp3", "wav"])
        self.volume_mix = st.slider(
            "🔊 Background Music Volume (relative to voice):", 0, 100, 30)

    def modify_voice(self, voice):
        if self.music_file:
            music_ext = os.path.splitext(self.music_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=music_ext) as music_tmp:
                music_tmp.write(self.music_file.read())
                music_path = music_tmp.name

            music = AudioSegment.from_file(music_path)
            music = music - (100 - self.volume_mix)
            voice = music.overlay(voice, loop=True)
        return voice


class WhisperLayerMode(VoiceMode):
    def __init__(self):
        super().__init__()
        self.enabled = False

    def initialize(self):
        self.enabled = st.checkbox("👻 Add Whisper Layer")

    def modify_voice(self, voice: AudioSegment):
        if self.enabled:
            whisper = voice - 15
            whisper = whisper.set_frame_rate(16000).low_pass_filter(2000)
            voice = voice.overlay(whisper, position=500)
        return voice


class EmbedTonesMode(VoiceMode):
    def __init__(self):
        super().__init__()
        self.enabled = False

    def initialize(self):
        self.enabled = st.checkbox("🧘 Embed Theta Binaural (4.5Hz)")

    def modify_voice(self, voice):
        if self.enabled:
            tone_left = Sine(200).to_audio_segment(
                duration=len(voice), volume=-25).pan(-1)
            tone_right = Sine(204.5).to_audio_segment(
                duration=len(voice), volume=-25).pan(1)
            tone_combined = tone_left.overlay(tone_right)
            voice = tone_combined.overlay(voice)
        return voice


class SolfeggioMode(VoiceMode):
    def __init__(self):
        super().__init__()
        self.enabled = False

    def initialize(self):
        self.solfeggio_freq = []
        self.solfeggio_options = {
            None: "None",
            174: "174 Hz – Pain Relief & Security",
            285: "285 Hz – Tissue Healing",
            396: "396 Hz – Liberating Fear & Guilt",
            417: "417 Hz – Undoing Situations",
            528: "528 Hz – DNA Repair & Transformation",
            639: "639 Hz – Connection & Relationships",
            741: "741 Hz – Awakening Intuition",
            852: "852 Hz – Returning to Spiritual Order",
            963: "963 Hz – Pineal Gland Activation & Oneness"
        }
        self.solfeggio_label = st.selectbox(
            "🎶 Add Solfeggio Frequency (Optional)", list(self.solfeggio_options.values()))
        self.solfeggio_freq = [freq for freq, label in self.solfeggio_options.items(
        ) if label == self.solfeggio_label][0]
        self.enabled = self.solfeggio_freq != None

    def modify_voice(self, voice):
        if self.enabled:
            solfeggio = Sine(self.solfeggio_freq).to_audio_segment(
                duration=len(voice), volume=-20)
            voice = voice.overlay(solfeggio)
        return voice


class IsochronicMode(VoiceMode):
    def __init__(self):
        super().__init__()
        self.enabled = False

    def initialize(self):
        self.enabled = st.checkbox(
            "🌀 Add Isochronic Tones (7.83Hz - Earth/Healing Base)")

    def modify_voice(self, voice):
        if self.enabled:
            pulse = Sine(150).to_audio_segment(
                duration=64, volume=-30).fade_in(5).fade_out(5)
            pattern = pulse * (len(voice) // len(pulse))
            voice = pattern.overlay(voice)
        return voice


class MorphicFieldMode(VoiceMode):
    def __init__(self):
        super().__init__()
        self.enabled = False

    def initialize(self):
        self.enabled = st.checkbox("🌌 Morphic Field Loop Mode")

    def modify_voice(self, voice):
        if self.enabled:
            morphic = voice.low_pass_filter(
                4000).pan(-1).overlay(voice.high_pass_filter(5000).pan(1), gain_during_overlay=-5)
            voice = morphic.overlay(voice, gain_during_overlay=-3)
        return voice
