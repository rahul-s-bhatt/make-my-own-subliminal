import os
import tempfile
from io import BytesIO

import pyttsx3
import streamlit as st
from pydub import AudioSegment
from pydub.generators import Sine
from streamlit_advanced_audio import WaveSurferOptions, audix

st.set_page_config(page_title="🎧 Pro Audio Editor", layout="wide")
st.markdown("## 🎚 Pro Audio Editor (Beta)")

# Session state to hold layers
if "audio_layers" not in st.session_state:
    st.session_state.audio_layers = {}
    print("Initialized audio_layers in session state.")

# Affirmation text input
affirmation_text = st.text_area("💬 Enter Affirmations (One per line):", height=300)

# Generate TTS from affirmations
if affirmation_text:
    print("Generating base audio from affirmations...")
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    temp_wav = os.path.join(tempfile.gettempdir(), "affirmation_base.wav")

    engine.save_to_file(affirmation_text, temp_wav)
    engine.runAndWait()

    base_audio = AudioSegment.from_wav(temp_wav)
    print(f"Loaded base audio segment from affirmations: duration={len(base_audio)}ms")
    st.session_state.audio_layers["base"] = base_audio

    options = WaveSurferOptions(wave_color="#2B88D9", progress_color="#b91d47", height=100, bar_width=2, bar_gap=1)
    audix(temp_wav, wavesurfer_options=options)
    print("Waveform displayed with Audix.")

    st.markdown("---")

    # Add Background Music
    bg_music = st.file_uploader("🎼 Upload Background Music:", type=["wav", "mp3"], key="bg_music")
    if bg_music:
        print(f"Background music uploaded: {bg_music.name}")
        bg_path = os.path.join(tempfile.gettempdir(), bg_music.name)
        with open(bg_path, "wb") as f:
            f.write(bg_music.read())
            print(f"Background music written to temp path: {bg_path}")
        bg_audio = AudioSegment.from_file(bg_path)

        volume = st.slider("🔉 Background Music Volume (dB)", -30, 10, -10, key="bg_volume")
        print(f"Background music volume set to: {volume} dB")
        st.session_state.audio_layers["bg_music"] = bg_audio + volume

    # Add Binaural Beats
    if st.checkbox("🧠 Add Binaural Beats"):
        freq = st.slider("🎵 Frequency (Hz)", 100, 1000, 432)
        print(f"Binaural frequency selected: {freq} Hz")
        duration = len(base_audio)
        sine = Sine(freq).to_audio_segment(duration=duration)
        print(f"Generated sine wave of duration: {duration}ms")
        volume = st.slider("🔉 Binaural Beats Volume (dB)", -30, 10, -15, key="bb_volume")
        print(f"Binaural beats volume set to: {volume} dB")
        st.session_state.audio_layers["binaural"] = sine + volume

    # Preview and Export
    if st.button("🔁 Preview Final Mix"):
        print("Preview mix button clicked.")
        final = st.session_state.audio_layers["base"]
        for name, layer in st.session_state.audio_layers.items():
            if name != "base":
                print(f"Overlaying layer: {name}")
                final = final.overlay(layer)

        preview_path = os.path.join(tempfile.gettempdir(), "final_mix.wav")
        final.export(preview_path, format="wav")
        print(f"Final mix exported to: {preview_path}")
        st.audio(preview_path)

    if st.button("📥 Download Final Mix"):
        print("Download mix button clicked.")
        final = st.session_state.audio_layers["base"]
        for name, layer in st.session_state.audio_layers.items():
            if name != "base":
                print(f"Overlaying layer: {name}")
                final = final.overlay(layer)

        download_path = os.path.join(tempfile.gettempdir(), "final_output.wav")
        final.export(download_path, format="wav")
        print(f"Final output exported to: {download_path}")
        with open(download_path, "rb") as f:
            st.download_button("📤 Download WAV", data=f, file_name="final_output.wav", mime="audio/wav")
