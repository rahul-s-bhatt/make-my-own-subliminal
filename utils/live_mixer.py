import os
import tempfile

import streamlit as st
from pydub import AudioSegment

from utils.audio_engine import AudioEngine


class LiveMixer:
    """
    A plug-and-play live audio mixer for Streamlit.

    Usage:
        mixer = LiveMixer(voice_path, bg_path)
        mixer.render()
    Ensure voice_path, bg_path point to temporary WAV files containing each layer.
    """

    def __init__(self, audio_engine: AudioEngine, affirmations_text_input: str):
        self.voice_path = audio_engine.generate_tts_to_tempfile(affirmations_text_input)
        self.music_file = st.file_uploader("🎼 Upload Background Music (mp3 or wav):", type=["mp3", "wav"])
        self.volume_mix = st.slider("🔊 Background Music Volume (relative to voice):", 0, 100, 30)
        # self.freq_path = freq_path
        # self.bg_path = bg_path
        # default slider values
        self.default_voice_vol = 100
        # self.default_freq_vol = 100
        self.default_bg_vol = 30

    def render(self):
        st.subheader("🔀 Live Mix Adjustment")
        # Slider controls
        voice_vol = st.slider("🔊 Affirmations Volume", 0, 100, self.default_voice_vol, key="mix_voice")
        # freq_vol = st.slider("🔔 Frequency Volume", 0, 100, self.default_freq_vol, key="mix_freq")
        # bg_vol = st.slider("🎚 Background Volume", 0, 100, self.default_bg_vol, key="mix_bg")

        # Load layers and apply volume adjustments
        voice = AudioSegment.from_file(self.voice_path) + (voice_vol - 100)
        # freq = AudioSegment.from_file(self.freq_path) + (freq_vol - 100)
        if self.music_file:
            music_ext = os.path.splitext(self.music_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=music_ext) as music_tmp:
                music_tmp.write(self.music_file.read())
                music_path = music_tmp.name

            music = AudioSegment.from_file(music_path)
            music = music - (100 - self.volume_mix)
            voice = music.overlay(voice, loop=True)

        # Overlay layers
        # mixed = bg.overlay(freq).overlay(voice)
        mixed = voice.overlay(voice)

        # Export to temp file and playback
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        mixed.export(tmp.name, format="wav")
        st.audio(tmp.name)
        # Clean up: remove old temp files at end of session if needed
        os.remove(tmp.name)
