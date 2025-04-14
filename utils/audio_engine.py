import os
import shutil
from io import BytesIO

import pyttsx3
import streamlit as st
from pydub import AudioSegment

from modes.basic_mode import ModeWrapper


class AudioEngine:
    def __init__(self):
        self.audio_file_path = ""

    def engine_configuration(self, affirmations_text_input: str, audio_play_speed: float):
        engine = pyttsx3.init()
        engine.setProperty('rate', int(200 * audio_play_speed))

        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)

        os.makedirs("output", exist_ok=True)
        self.audio_file_path = os.path.join("output", "voice.wav")
        engine.save_to_file(affirmations_text_input, self.audio_file_path)
        engine.runAndWait()

    def audio_post_processing(self, mode_wrapper: ModeWrapper, affirmations_loop_count: int):
        try:
            voice = AudioSegment.from_file(
                self.audio_file_path, format="wav") * affirmations_loop_count
            voice = mode_wrapper.apply_all(voice)

            output_file_name = st.text_input(
                "💾 Output File Name:", value="subliminal.wav")
            output_path = os.path.join("output", output_file_name)
            voice.export(output_path, format="wav")

            st.audio(output_path, format="audio/wav")

            st.success("✅ Subliminal Generated!")
            with open(output_path, "rb") as f:
                st.download_button("📥 Download WAV", data=f,
                                   file_name=output_file_name, mime="audio/wav",)
        except Exception as e:
            st.error(f"Error processing audio: {e}")
        finally:
            if os.path.exists("output"):
                try:
                    shutil.rmtree("output")  # Remove the entire directory tree
                    # print("✅ Temporary 'output' folder deleted.")
                except OSError as e:
                    st.error(f"Error deleting temporary 'output' folder: {e}")
