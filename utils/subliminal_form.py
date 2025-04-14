import os
import shutil
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
                    if voices:
                        engine.setProperty('voice', voices[0].id)

                    os.makedirs("output", exist_ok=True)
                    voice_path = os.path.join("output", "voice.wav")
                    engine.save_to_file(self.affirmations_text_input, voice_path)
                    engine.runAndWait()

                    try:                    
                        voice = AudioSegment.from_file(voice_path, format="wav") * self.affirmations_loop_count
                        voice = self.mode_wrapper.apply_all(voice)
                        
                        output_file_name = st.text_input("💾 Output File Name:", value="subliminal.wav")
                        output_path = os.path.join("output", output_file_name)
                        voice.export(output_path, format="wav")

                        st.audio(output_path, format="audio/wav")

                        st.success("✅ Subliminal Generated!")
                        with open(output_path, "rb") as f:
                            st.download_button("📥 Download WAV",data=f,file_name=output_file_name,mime="audio/wav",)
                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
                    finally:
                        if os.path.exists("output"):
                            try:
                                shutil.rmtree("output")  # Remove the entire directory tree
                                # print("✅ Temporary 'output' folder deleted.")
                            except OSError as e:
                                st.error(f"Error deleting temporary 'output' folder: {e}")