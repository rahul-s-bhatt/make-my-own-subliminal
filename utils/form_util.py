import streamlit as st

MAX_AFFIRMATIONS = 10000


class FormUtil:
    def _pre_processing_validation(self, affirmations_text_input: str):
        if not affirmations_text_input:
            st.warning("Please enter some affirmations first.")
            return  # Stop processing

        lines = affirmations_text_input.splitlines()
        affirmations_lines_count = len(lines)
        if affirmations_lines_count > MAX_AFFIRMATIONS:
            st.error(f"⚠️ Input Limit Exceeded! You entered {affirmations_lines_count} lines. Please limit your affirmations to {MAX_AFFIRMATIONS} lines.")
            return  # Stop processing

    def _calculate_display_estimates(self, affirmations_text_input: str, affirmations_loop_count: int, audio_play_speed: float):
        duration_sec, size_mb, num_lines = self._estimate_generation(affirmations_text_input, audio_play_speed, affirmations_loop_count)

        estimate_message = ""
        if duration_sec > 0:
            est_minutes = int(duration_sec // 60)
            est_seconds = int(duration_sec % 60)
            duration_str = f"{est_minutes} min {est_seconds} sec" if est_minutes > 0 else f"{duration_sec:.1f} sec"

            estimate_message = (
                f"⚙️ **Estimated Generation:**\n"
                f"- Duration: ~{duration_str} ({num_lines} lines x {affirmations_loop_count} loops @ {audio_play_speed} speed)\n"
                f"- Size: ~{size_mb} MB (uncompressed WAV)"
            )
            st.info(estimate_message)  # Display the estimate

            # Optional: Add a warning for very large estimates
            if size_mb > 200 or duration_sec > 600:  # e.g., > 200MB or > 10 minutes
                st.warning("⚠️ This may take a significant amount of time and result in a large file.")

    def _estimate_generation(self, affirmations_text: str, speed_multiplier, loop_count: int):
        """Calculates estimated duration (sec) and size (MB) for WAV."""
        lines = affirmations_text.strip().splitlines()
        num_lines = len(lines)
        if num_lines == 0:
            return 0, 0, 0  # duration_sec, size_mb, num_lines

        # --- Estimation Parameters (ADJUST THESE BASED ON YOUR TTS/AUDIO) ---
        # Average time (in seconds) to speak one line at normal speed
        avg_sec_per_line_at_1x = 2.0
        sample_rate = 44100  # Sample rate of the output WAV
        num_channels = 2  # 1 for mono, 2 for stereo output WAV
        # Bit depth of the output WAV (usually 16)
        bits_per_sample = 16
        # --- End Parameters ---

        # Calculate duration
        duration_per_line_adjusted = avg_sec_per_line_at_1x / speed_multiplier
        total_duration_sec = duration_per_line_adjusted * num_lines * loop_count

        # Calculate size (uncompressed WAV)
        bytes_per_second = sample_rate * num_channels * (bits_per_sample / 8)
        estimated_size_bytes = total_duration_sec * bytes_per_second
        estimated_size_mb = round(estimated_size_bytes / (1024**2), 1) if estimated_size_bytes > 0 else 0

        return total_duration_sec, estimated_size_mb, num_lines
