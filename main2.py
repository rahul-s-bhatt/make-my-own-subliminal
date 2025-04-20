import os
import tempfile
from typing import Dict, Optional

import pyttsx3
import streamlit as st
from pydub import AudioSegment
from pydub.generators import Sine
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_advanced_audio import WaveSurferOptions, audix

# Initialize Streamlit app
st.set_page_config(page_title="🎧 Pro Audio Editor", layout="wide")
st.markdown("## 🎚 Pro Audio Editor (Beta)")

# Session state to hold audio layers: Dict[layer_name, AudioSegment]
if "audio_layers" not in st.session_state:
    st.session_state.audio_layers: Dict[str, AudioSegment] = {}  # type: ignore
    print("Initialized audio_layers in session state.")

# Affirmation text input
affirmation_text: str = st.text_area("💬 Enter Affirmations (One per line):", height=200)

# Generate TTS from affirmations into base audio layer
if affirmation_text:
    print("Generating base audio from affirmations...")
    engine: pyttsx3.Engine = pyttsx3.init()
    engine.setProperty("rate", 150)

    temp_base: str = os.path.join(tempfile.gettempdir(), "affirmation_base.wav")
    engine.save_to_file(affirmation_text, temp_base)
    engine.runAndWait()

    base_audio: AudioSegment = AudioSegment.from_wav(temp_base)
    print(f"Loaded base audio segment: duration={len(base_audio)}ms")
    st.session_state.audio_layers["base"] = base_audio

    # Default waveform styling options
    default_opts: WaveSurferOptions = WaveSurferOptions(wave_color="#2B88D9", progress_color="#b91d47", height=80, bar_width=2, bar_gap=1)
    st.markdown("### Base Layer")
    audix(temp_base, wavesurfer_options=default_opts)
    print("Displayed base waveform with Audix.")

    st.markdown("---")

    # Add extra layers
    with st.expander("📥 Upload Additional Audio Layers"):
        # Background music
        bg_music: Optional[UploadedFile] = st.file_uploader("Background Music (wav/mp3)", type=["wav", "mp3"], key="bg_music")
        if bg_music:
            print(f"Background music uploaded: {bg_music.name}")
            path: str = os.path.join(tempfile.gettempdir(), bg_music.name)
            with open(path, "wb") as f:
                f.write(bg_music.read())
            bg: AudioSegment = AudioSegment.from_file(path)
            st.session_state.audio_layers["bg_music"] = bg
            st.success("Background music loaded.")

        # Binaural Beats
        if st.checkbox("Add Binaural Beats", key="add_bb"):
            freq: int = st.slider("Beat Frequency (Hz)", 100, 1000, 432)
            print(f"Selected binaural beat frequency: {freq} Hz")
            duration: int = len(base_audio)
            bb: AudioSegment = Sine(freq).to_audio_segment(duration=duration)
            st.session_state.audio_layers["binaural"] = bb
            st.success(f"Binaural beats @{freq}Hz added.")

    # Track controls: pitch, speed, and real-time effects
    st.markdown("## 🎛️ Track Controls")
    for name, audio in st.session_state.audio_layers.items():
        st.markdown(f"#### Track: {name}")

        # Export current audio to temporary WAV for visualization
        tmp_path: str = os.path.join(tempfile.gettempdir(), f"{name}.wav")
        audio.export(tmp_path, format="wav")
        print(f"Exported audio layer '{name}' to temporary file: {tmp_path}")
        audix(tmp_path, wavesurfer_options=default_opts)

        # Pitch shift (semitones) and speed factor (multiplier)
        pitch: int = st.slider(f"Pitch Shift (semitones) for {name}", -12, 12, 0, key=f"{name}_pitch")
        speed: float = st.slider(f"Speed Factor for {name}", 0.5, 2.0, 1.0, key=f"{name}_speed")

        if st.button(f"Apply to {name}", key=f"apply_{name}"):
            print(f"Applying pitch={pitch}st, speed={speed}x to {name}")
            factor: float = speed * (2 ** (pitch / 12))
            mod: AudioSegment = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * factor)}).set_frame_rate(audio.frame_rate)
            st.session_state.audio_layers[name] = mod
            print(f"Updated '{name}' layer in session state.")
            st.success(f"Updated {name}: pitch={pitch}st, speed={speed}x")

        st.markdown("---")

    # Preview & Export combined mix
    if st.button("🔁 Preview Final Mix"):
        print("Previewing final mix...")
        mix: Optional[AudioSegment] = None
        for layer_name, layer in st.session_state.audio_layers.items():
            mix = layer if mix is None else mix.overlay(layer)
            print(f"Overlayed layer: {layer_name}")
        out: str = os.path.join(tempfile.gettempdir(), "preview_mix.wav")
        mix.export(out, format="wav")  # type: ignore
        print(f"Final preview exported to: {out}")
        st.audio(out)

    if st.button("📥 Download Final Mix"):
        print("Preparing download for final mix...")
        mix: Optional[AudioSegment] = None
        for layer_name, layer in st.session_state.audio_layers.items():
            mix = layer if mix is None else mix.overlay(layer)
            print(f"Overlayed layer: {layer_name}")
        dl: str = os.path.join(tempfile.gettempdir(), "final_output.wav")
        mix.export(dl, format="wav")  # type: ignore
        print(f"Final mix exported to: {dl}")
        with open(dl, "rb") as f:
            st.download_button("Download WAV", data=f, file_name="final_output.wav", mime="audio/wav")
