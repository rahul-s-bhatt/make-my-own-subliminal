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

# 1. Declare a flag before the form
generated_audio = False
output_path = ""
session_data = {}
file_name = "subliminal.wav"  # default
pro_mode = st.toggle("✨ Enable Pro Mode")
solfeggio_freq = []

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
                rate_multiplier = {"1x": 1.0, "2x": 2.0, "3x": 3.0}[speed] # type: ignore
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
                generated_audio = True

if generated_audio: 
    st.success("✅ Subliminal Generated!")
    st.audio(output_path)
    with open(output_path, "rb") as f:
        st.download_button("📥 Download WAV", data=f, file_name=file_name, mime="audio/wav")
