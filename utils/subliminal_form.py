import streamlit as st

from modes.basic_mode import ModeWrapper
from utils.audio_engine import AudioEngine
from utils.form_util import FormUtil
from utils.live_mixer import LiveMixer


class SubliminalForm:
    def __init__(self, mode_wrapper: ModeWrapper, audio_engine: AudioEngine, form_util: FormUtil):
        self.affirmations_text_input = ""
        self.affirmations_loop_count = 1
        self.is_submitted = False
        self.preview_path = None
        self.output_file_name = "subliminal.wav"
        self.speechSpeed = "1x"
        self.playback_speed = ["1x", "2x", "3x"]

        # Object
        self.mode_wrapper = mode_wrapper
        self.audio_engine = audio_engine
        self.form_util = form_util

        # Mix adjustment defaults
        self.mix_affirmation_speed = 1.0
        self.mix_affirmation_volume = 100
        self.mix_frequency_volume = 100
        self.mix_background_volume = 30

    def create_form(self):
        tab1, tab2, tab3 = st.tabs(
            [
                "🔤 Affirmations",
                "🧬 Advanced Modes",
                "🎚 Pro Audio Editor & 🔀 Mix Adjustment",
            ]
        )

        with st.form("subliminal_form"):
            # Tab 1: Affirmations
            with tab1:
                st.subheader("✍️ Affirmation Settings")
                self.affirmations_text_input = st.text_area("📝 Enter Your Affirmations:", height=300)
                self.speechSpeed = st.selectbox("🚀 Choose Speech Speed:", self.playback_speed, index=0)

                self.audio_play_speed = {"1x": 1.0, "2x": 2.0, "3x": 3.0}[self.speechSpeed]  # type: ignore

                self.affirmations_loop_count = st.slider("🔁 Repeat Affirmation Track", 1, 20, 1)

            # Tab 2: Advanced Modes (other modes also handled by ModeWrapper)
            with tab2:
                st.subheader("⚙️ Advanced Customization")
                # self.mode_wrapper.initialize_all()

            # Tab 3: Pro Audio Editor (Preview & Trim) & Mix Adjustment
            with tab3:
                st.subheader("🎚 Pro Audio Editor (Beta)")
                liveMixer = LiveMixer(self.audio_engine, self.affirmations_text_input)
                liveMixer.render()

            # Final generate button
            # self.output_file_name = st.text_input("💾 Output File Name:", value=self.output_file_name)
            # self.is_submitted = st.form_submit_button("🎧 Generate Subliminal")

    def on_submit(self):
        if self.is_submitted:
            # Get text from instance variable
            self.affirmations_text_input = self.affirmations_text_input.strip()

            # --- 1. Basic Validation ---
            self.form_util._pre_processing_validation(self.affirmations_text_input)

            # --- 2. Calculate and Display Estimate ---
            self.form_util._calculate_display_estimates(
                self.affirmations_text_input,
                self.affirmations_loop_count,
                self.audio_play_speed,
            )
            # --- 3. Proceed with Generation ---
            with st.spinner("Generating audio... Please wait."):
                self.audio_engine.engine_configuration(self.audio_play_speed)
                self.audio_engine.generate_tts_to_tempfile(self.affirmations_text_input)
                self.audio_engine.process_and_output_audio(
                    self.mode_wrapper,
                    self.affirmations_text_input,
                    self.affirmations_loop_count,
                    self.output_file_name,
                )
