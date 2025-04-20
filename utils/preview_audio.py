import os
import tempfile

import streamlit as st
from pydub import AudioSegment
from streamlit_advanced_audio import WaveSurferOptions, audix

from modes.basic_mode import ModeWrapper
from utils.audio_engine import AudioEngine


class PreviewAudio:
    def _preview_audio(self, audio_engine: AudioEngine, mode_wrapper: ModeWrapper, audio_play_speed: float, affirmations_text_input: str, affirmations_loop_count: int):
        if st.button("🔄 Generate Audio Preview"):
            audio_engine.engine_configuration(audio_play_speed)
            audio_engine.generate_tts_to_tempfile(affirmations_text_input)
            voice = AudioSegment.from_file(audio_engine.tts_temp_path, format="wav") * affirmations_loop_count
            voice = mode_wrapper.apply_all(voice)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            preview_path = tmp.name
            tmp.close()
            voice.export(preview_path, format="wav")

        if preview_path:
            options = WaveSurferOptions(
                wave_color="#2B88D9",
                progress_color="#b91d47",
                height=100,
                bar_width=2,
                bar_gap=1,
            )
            result = audix(preview_path, wavesurfer_options=options)
            if result and result.get("selectedRegion"):
                start_t = result["selectedRegion"]["start"]
                end_t = result["selectedRegion"]["end"]
                st.success(f"Selected: {round(start_t, 2)}s to {round(end_t, 2)}s")
                if st.button("✂️ Export Trimmed Audio"):
                    audio = AudioSegment.from_file(preview_path)
                    trimmed = audio[int(start_t * 1000) : int(end_t * 1000)]
                    export_path = os.path.join(
                        tempfile.gettempdir(),
                        "trimmed_subliminal.wav",
                    )
                    trimmed.export(export_path, format="wav")
                    st.audio(export_path)
                    with open(export_path, "rb") as f:
                        st.download_button(
                            "📥 Download Trimmed WAV",
                            data=f,
                            file_name="trimmed_subliminal.wav",
                            mime="audio/wav",
                        )

        st.subheader("🔀 Mix Adjustment")
        mix_affirmation_speed = st.slider(
            "🎵 Affirmations Speed",
            0.5,
            3.0,
            mix_affirmation_speed,
            step=0.1,
        )
        mix_affirmation_volume = st.slider("🔊 Affirmations Volume", 0, 100, mix_affirmation_volume)
        mix_frequency_volume = st.slider("🔔 Frequency Volume", 0, 100, mix_frequency_volume)
        mix_background_volume = st.slider("🎚 Background Volume", 0, 100, mix_background_volume)
