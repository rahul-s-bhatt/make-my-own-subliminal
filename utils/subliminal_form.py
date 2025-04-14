import streamlit as st

from modes.basic_mode import ModeWrapper
from utils.audio_engine import AudioEngine


class SubliminalForm:
    def __init__(self):
        self.is_submitted = False
        self.generated_audio = False
        self.affirmations_text_input = ""
        self.playback_speed = ["1x", "2x", "3x"]
        self.audio_play_speed_str = "1x"
        self.audio_play_speed = 1.0
        self.affirmations_loop_count = 1
        self.mode_wrapper = ModeWrapper()
        self.audio_engine = AudioEngine()

    def create_form(self):
        with st.form("subliminal_form"):
            st.subheader("🔤 Affirmation Settings")
            self.affirmations_text_input = st.text_area(
                "📝 Enter Your Affirmations:", height=300)
            self.speed = st.selectbox(
                "🚀 Choose Speech Speed:", self.playback_speed, index=0)
            self.affirmations_loop_count = st.slider(
                "🔁 Repeat Affirmation Track", 1, 20, 1)
            st.subheader("🎵 Audio Settings")
            self.mode_wrapper.initialize_all()
            self.is_submitted = st.form_submit_button("🎧 Generate Subliminal")

    def on_submit(self):
        if self.is_submitted:
            if not self.affirmations_text_input.strip():
                st.warning("Please enter some affirmations first.")
            else:
                with st.spinner("Generating audio..."):
                    self.audio_play_speed = {"1x": 1.0, "2x": 2.0, "3x": 3.0}[
                        self.speed]  # type: ignore
                    self.audio_engine.engine_configuration(
                        self.affirmations_text_input, self.audio_play_speed)
                    self.audio_engine.audio_post_processing(
                        self.mode_wrapper, self.affirmations_loop_count)
