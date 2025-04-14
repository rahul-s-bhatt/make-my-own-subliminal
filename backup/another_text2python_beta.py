import json
import os
import tempfile

import pyttsx3
import streamlit as st
from pydub import AudioSegment
from pydub.generators import Sine

st.set_page_config(page_title="Subliminal Generator", layout="centered")
st.title("🧠 Subliminal Audio Generator")

st.markdown("""
Transform affirmations into **subliminal audio fields** with high-speed speech, optional background music, whisper layering, Solfeggio frequencies, and more.
""")

pro_mode = st.toggle("✨ Enable Pro Mode")

with st.form("subliminal_form"):
    st.subheader("🔤 Affirmation Settings")
    text_input = st.text_area("📝 Enter Your Affirmations:", height=300)
    speed = st.selectbox("🚀 Choose Speech Speed:", ["1x", "2x", "3x"])
    loop_count = st.slider("🔁 Repeat Affirmation Track", 1, 20, 1)
    file_name = st.text_input("💾 Output File Name:", value="subliminal.wav")

    st.subheader("🎵 Audio Settings")
    music_file = st.file_uploader("🎼 Upload Background Music (mp3 or wav):", type=["mp3", "wav"])
    volume_mix = st.slider("🔊 Background Music Volume (relative to voice):", 0, 100, 30)
    whisper_layer = st.checkbox("👻 Add Whisper Layer")
    embed_tones = st.checkbox("🧘 Embed Theta Binaural (4.5Hz)")

    if pro_mode:
        st.subheader("🧬 Pro Mode: Frequency & Field Customization")
        solfeggio_options = {
            None: "None",
            174: "174 Hz – Pain Relief & Security",
            285: "285 Hz – Tissue Healing",
            396: "396 Hz – Liberating Fear & Guilt",
            417: "417 Hz – Undoing Situations",
            528: "528 Hz – DNA Repair & Transformation",
            639: "639 Hz – Connection & Relationships",
            741: "741 Hz – Awakening Intuition",
            852: "852 Hz – Returning to Spiritual Order",
            963: "963 Hz – Pineal Gland Activation & Oneness"
        }
        solfeggio_label = st.selectbox("🎶 Add Solfeggio Frequency (Optional)", list(solfeggio_options.values()))
        solfeggio_freq = [freq for freq, label in solfeggio_options.items() if label == solfeggio_label][0]
        isochronic = st.checkbox("🌀 Add Isochronic Tones (7.83Hz - Earth/Healing Base)")
        morphic_field_mode = st.checkbox("🌌 Morphic Field Loop Mode")

    submitted = st.form_submit_button("🎧 Generate Subliminal")

    if submitted:
        if not text_input.strip():
            st.warning("Please enter some affirmations first.")
        else:
            with st.spinner("Generating audio..."):
                rate_multiplier = {"1x": 1.0, "2x": 2.0, "3x": 3.0}[speed]
                engine = pyttsx3.init()
                engine.setProperty('rate', int(200 * rate_multiplier))

                voices = engine.getProperty('voices')
                if voices:
                    engine.setProperty('voice', voices[0].id)

                os.makedirs("output", exist_ok=True)
                voice_path = os.path.join("output", "voice.wav")
                engine.save_to_file(text_input, voice_path)
                engine.runAndWait()

                voice = AudioSegment.from_wav(voice_path) * loop_count

                if whisper_layer:
                    whisper = voice - 15
                    whisper = whisper.set_frame_rate(16000).low_pass_filter(2000)
                    voice = voice.overlay(whisper, position=500)

                if embed_tones:
                    tone_left = Sine(200).to_audio_segment(duration=len(voice), volume=-25).pan(-1)
                    tone_right = Sine(204.5).to_audio_segment(duration=len(voice), volume=-25).pan(1)
                    tone_combined = tone_left.overlay(tone_right)
                    voice = tone_combined.overlay(voice)

                if pro_mode:
                    if solfeggio_freq:
                        solfeggio = Sine(solfeggio_freq).to_audio_segment(duration=len(voice), volume=-20)
                        voice = voice.overlay(solfeggio)

                    if isochronic:
                        pulse = Sine(150).to_audio_segment(duration=64, volume=-30).fade_in(5).fade_out(5)
                        pattern = pulse * (len(voice) // len(pulse))
                        voice = pattern.overlay(voice)

                    if morphic_field_mode:
                        morphic = voice.low_pass_filter(4000).pan(-1).overlay(voice.high_pass_filter(5000).pan(1), gain_during_overlay=-5)
                        voice = morphic.overlay(voice, gain_during_overlay=-3)

                if music_file:
                    music_ext = os.path.splitext(music_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=music_ext) as music_tmp:
                        music_tmp.write(music_file.read())
                        music_path = music_tmp.name

                    music = AudioSegment.from_file(music_path)
                    music = music - (100 - volume_mix)
                    combined = music.overlay(voice, loop=True)
                else:
                    combined = voice

                output_path = os.path.join("output", file_name)
                combined.export(output_path, format="wav")

                session_data = {
                    "affirmations": text_input[:100] + ("..." if len(text_input) > 100 else ""),
                    "speed": speed,
                    "loop_count": loop_count,
                    "volume": volume_mix,
                    "whisper_layer": whisper_layer,
                    "embed_tones": embed_tones,
                    "solfeggio": solfeggio_freq,
                    "isochronic": isochronic if pro_mode else False,
                    "morphic_field_mode": morphic_field_mode if pro_mode else False,
                    "output_file": file_name
                }
                with open(os.path.join("output", "session_config.txt"), "w") as cfg:
                    cfg.write(json.dumps(session_data, indent=4))

                st.success("✅ Subliminal Generated!")
                st.audio(output_path)
                with open(output_path, "rb") as f:
                    st.download_button("📥 Download WAV", data=f, file_name=file_name, mime="audio/wav")
                st.download_button("🧾 Download Session Config", data=json.dumps(session_data, indent=4), file_name="session_config.txt", mime="text/plain")
