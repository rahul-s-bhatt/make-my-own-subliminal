import streamlit as st

from modes.basic_mode import ModeWrapper
from utils.audio_engine import AudioEngine

MAX_AFFIRMATIONS = 10000


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
        tab1, tab2, tab3 = st.tabs(
            ["🔤 Affirmations", "🎵 Audio", "🧬 Advanced Modes"])
        with st.form("subliminal_form"):

            with tab1:
                st.subheader("🔤 Affirmation Settings")
                self.affirmations_text_input = st.text_area(
                    "📝 Enter Your Affirmations:", height=300)
            with tab2:
                st.subheader("🎵 Audio Settings")
                self.speed = st.selectbox(
                    "🚀 Choose Speech Speed:", self.playback_speed, index=0)
                self.affirmations_loop_count = st.slider(
                    "🔁 Repeat Affirmation Track", 1, 20, 1)
            with tab3:
                self.mode_wrapper.initialize_all()

            # File name input
            self.output_file_name = st.text_input(
                "💾 Output File Name:", value="subliminal.wav")
            self.is_submitted = st.form_submit_button("🎧 Generate Subliminal")

    def on_submit(self):
        # with st.sidebar:
        #     if self.is_submitted:
        #         num_lines = len(
        #             self.affirmations_text_input.strip().splitlines())
        #         estimated_duration = num_lines * 2
        #         estimated_size_mb = round(
        #             (estimated_duration * 44100 * 2 * 16 / 8) / (1024**2), 1)

        #         st.info(
        #             f"🔢 Estimated duration: {estimated_duration//60} min, size: {estimated_size_mb} MB")

        #         if estimated_size_mb > 200:
        #             st.warning(
        #                 "⚠️ This may take a while and result in a large file. Consider shortening affirmations.")
        if self.is_submitted:
            if not self.affirmations_text_input.strip():
                st.warning("Please enter some affirmations first.")
            else:
                affirmations_lines_count = len(
                    self.affirmations_text_input.strip().splitlines())
                if affirmations_lines_count > MAX_AFFIRMATIONS:
                    st.warning(
                        f"Please limit your affirmations to {MAX_AFFIRMATIONS} lines.")
                    return

                with st.spinner("Generating audio..."):
                    self.audio_play_speed = {"1x": 1.0, "2x": 2.0, "3x": 3.0}[
                        self.speed]  # type: ignore
                    self.audio_engine.engine_configuration(
                        self.audio_play_speed)
                    self.audio_engine.generate_tts_to_tempfile(
                        self.affirmations_text_input)
                    self.audio_engine.process_and_output_audio(
                        self.mode_wrapper, self.affirmations_text_input, self.affirmations_loop_count, self.output_file_name)
