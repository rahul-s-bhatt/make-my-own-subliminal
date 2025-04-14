from io import BytesIO

import pyttsx3
import streamlit as st
from pydub import AudioSegment

from modes.basic_mode import ModeWrapper


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

    def create_form(self):
        with st.form("subliminal_form"):
            st.subheader("🔤 Affirmation Settings")
            self.affirmations_text_input = st.text_area("📝 Enter Your Affirmations:", height=300)
            self.speed = st.selectbox("🚀 Choose Speech Speed:", self.playback_speed, index=0)
            self.affirmations_loop_count = st.slider("🔁 Repeat Affirmation Track", 1, 20, 1)
            st.subheader("🎵 Audio Settings")            
            self.mode_wrapper.initialize_all()
            self.is_submitted = st.form_submit_button("🎧 Generate Subliminal")

    def on_submit(self):
        if self.is_submitted:
            if not self.affirmations_text_input.strip():
                st.warning("Please enter some affirmations first.")
            else:
                with st.spinner("Generating audio..."):
                    self.audio_play_speed = {"1x": 1.0, "2x": 2.0, "3x": 3.0}[self.speed] # type: ignore
                    engine = pyttsx3.init()
                    engine.setProperty('rate', int(200 * self.audio_play_speed))

                    voices = engine.getProperty('voices')
                    engine.setProperty('voice', voices[0].id)

                    # Use BytesIO to store the audio in memory
                    audio_buffer = BytesIO()
                    engine.save_to_file(self.affirmations_text_input, audio_buffer)
                    engine.runAndWait()
                    audio_buffer.seek(0)  # Go to the beginning of the buffer

                    voice = AudioSegment.from_wav(audio_buffer) * self.affirmations_loop_count
                    voice = self.mode_wrapper.apply_all(voice)

                    # Export the looped audio to the in-memory buffer
                    output_buffer = BytesIO()
                    voice.export(output_buffer, format="wav")
                    output_buffer.seek(0)

                    file_name = "subliminal_audio.wav"
                    st.success("✅ Subliminal Generated!")
                    st.audio(output_buffer.getvalue(), format="audio/wav")
                    st.download_button("📥 Download WAV",data=output_buffer.getvalue(),file_name=file_name,mime="audio/wav",)