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
        if self.is_submitted:
            # Get text from instance variable
            self.affirmations_text_input = self.affirmations_text_input.strip()

            # --- 1. Basic Validation ---
            self.pre_processing_validation()

            # --- 2. Calculate and Display Estimate ---
            self.calculate_display_estimates()

            # --- 3. Proceed with Generation ---
            with st.spinner("Generating audio... Please wait."):
                self.audio_engine.engine_configuration(
                    self.audio_play_speed)
                self.audio_engine.generate_tts_to_tempfile(
                    self.affirmations_text_input)
                self.audio_engine.process_and_output_audio(
                    self.mode_wrapper, self.affirmations_text_input, self.affirmations_loop_count, self.output_file_name)

    def pre_processing_validation(self):
        if not self.affirmations_text_input:
            st.warning("Please enter some affirmations first.")
            return  # Stop processing

        lines = self.affirmations_text_input.splitlines()
        self.affirmations_lines_count = len(lines)
        if self.affirmations_lines_count > MAX_AFFIRMATIONS:
            st.error(
                f"⚠️ Input Limit Exceeded! You entered {self.affirmations_lines_count} lines. "
                f"Please limit your affirmations to {MAX_AFFIRMATIONS} lines."
            )
            return  # Stop processing

    def calculate_display_estimates(self):
        self.audio_play_speed = {"1x": 1.0, "2x": 2.0, "3x": 3.0}[
            self.speed]  # type: ignore

        duration_sec, size_mb, num_lines = self.estimate_generation(
            self.affirmations_text_input,
            self.audio_play_speed,
            self.affirmations_loop_count  # Get loop count from instance variable
        )

        estimate_message = ""
        if duration_sec > 0:
            est_minutes = int(duration_sec // 60)
            est_seconds = int(duration_sec % 60)
            duration_str = f"{est_minutes} min {est_seconds} sec" if est_minutes > 0 else f"{duration_sec:.1f} sec"

            estimate_message = (
                f"⚙️ **Estimated Generation:**\n"
                f"- Duration: ~{duration_str} ({num_lines} lines x {self.affirmations_loop_count} loops @ {self.speed} speed)\n"
                f"- Size: ~{size_mb} MB (uncompressed WAV)"
            )
            st.info(estimate_message)  # Display the estimate

            # Optional: Add a warning for very large estimates
            if size_mb > 200 or duration_sec > 600:  # e.g., > 200MB or > 10 minutes
                st.warning(
                    "⚠️ This may take a significant amount of time and result in a large file.")

    def estimate_generation(self, affirmations_text, speed_multiplier, loop_count):
        """Calculates estimated duration (sec) and size (MB) for WAV."""
        lines = affirmations_text.strip().splitlines()
        num_lines = len(lines)
        if num_lines == 0:
            return 0, 0, 0  # duration_sec, size_mb, num_lines

        # --- Estimation Parameters (ADJUST THESE BASED ON YOUR TTS/AUDIO) ---
        # Average time (in seconds) to speak one line at normal speed
        avg_sec_per_line_at_1x = 2.0
        sample_rate = 44100           # Sample rate of the output WAV
        num_channels = 2              # 1 for mono, 2 for stereo output WAV
        # Bit depth of the output WAV (usually 16)
        bits_per_sample = 16
        # --- End Parameters ---

        # Calculate duration
        duration_per_line_adjusted = avg_sec_per_line_at_1x / speed_multiplier
        total_duration_sec = duration_per_line_adjusted * num_lines * loop_count

        # Calculate size (uncompressed WAV)
        bytes_per_second = sample_rate * num_channels * (bits_per_sample / 8)
        estimated_size_bytes = total_duration_sec * bytes_per_second
        estimated_size_mb = round(
            estimated_size_bytes / (1024**2), 1) if estimated_size_bytes > 0 else 0

        return total_duration_sec, estimated_size_mb, num_lines
