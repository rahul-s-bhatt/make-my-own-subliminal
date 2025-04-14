import os
import shutil
import tempfile
from io import BytesIO

import pyttsx3
import streamlit as st
from pydub import AudioSegment

from modes.basic_mode import ModeWrapper


class AudioEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.tts_temp_path = None

    def engine_configuration(self, audio_play_speed: float):
        self.engine.setProperty('rate', int(200 * audio_play_speed))
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id)

    def generate_tts_to_tempfile(self, affirmations_text_input: str):
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        self.tts_temp_path = temp_wav.name
        temp_wav.close()
        self.engine.save_to_file(affirmations_text_input, self.tts_temp_path)
        self.engine.runAndWait()

    def process_and_output_audio(self, mode_wrapper: ModeWrapper, affirmations_text_input: str, loop_count: int, output_file_name: str):
        try:
            # Generate TTS to a temp .wav file
            self.generate_tts_to_tempfile(affirmations_text_input)

            # Load and loop the voice
            voice = AudioSegment.from_file(
                self.tts_temp_path, format="wav") * loop_count
            voice = mode_wrapper.apply_all(voice)

            # Export to disk for large affirmations
            output_path = os.path.join(
                tempfile.gettempdir(), output_file_name)
            voice.export(output_path, format="wav")
            st.audio(output_path, format="audio/wav")
            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Download WAV", data=f, file_name=output_file_name, mime="audio/wav")

            st.success("✅ Subliminal Generated!")

        except Exception as e:
            st.error(f"Error processing audio: {e}")

        finally:
            # Cleanup temp file
            if self.tts_temp_path and os.path.exists(self.tts_temp_path):
                os.remove(self.tts_temp_path)
