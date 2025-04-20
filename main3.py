import uuid
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_advanced_audio import audix

# Data Types
AudioData = np.ndarray  # Audio data as NumPy array (float32)
SampleRate = int  # Sample rate in Hz (e.g., 44100)
TrackID = str  # Unique identifier for each track

# Session State Initialization
if "tracks" not in st.session_state:
    st.session_state.tracks = {}  # Dict[TrackID, dict] storing audio data and metadata
if "play_position" not in st.session_state:
    st.session_state.play_position = 0.0  # Current playback position in seconds


# Utility Functions
def load_audio(file: UploadedFile) -> tuple[AudioData, SampleRate]:
    """Load audio file into NumPy array."""
    audio, sr = librosa.load(file, sr=None, mono=False)
    return audio.T, sr  # type: ignore # Transpose to match (samples, channels)


def save_audio(audio: AudioData, sr: SampleRate, filename: str) -> BytesIO:
    """Save NumPy audio array to WAV file in memory."""
    buffer = BytesIO()
    sf.write(buffer, audio, sr, format="wav")
    buffer.seek(0)
    return buffer


def generate_binaural_beats(duration: float, freq_left: float, freq_right: float, sr: SampleRate) -> AudioData:
    """Generate stereo binaural beats."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = np.sin(2 * np.pi * freq_left * t)
    right = np.sin(2 * np.pi * freq_right * t)
    return np.stack([left, right], axis=1)


def apply_solfeggio_frequency(audio: AudioData, freq: float, sr: SampleRate) -> AudioData:
    """Modulate audio with a solfeggio frequency."""
    t = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)
    modulation = np.sin(2 * np.pi * freq * t)
    if audio.ndim == 2:  # Stereo
        modulation = np.stack([modulation, modulation], axis=1)
    return audio * modulation


# Main Streamlit App
st.title("Subliminal Audio Editor")

# Sidebar for Track Management
with st.sidebar:
    st.subheader("Track Management")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])
    if uploaded_file:
        audio, sr = load_audio(uploaded_file)
        track_id = str(uuid.uuid4())
        st.session_state.tracks[track_id] = {"audio": audio, "sr": sr, "name": uploaded_file.name, "mute": False, "solo": False, "volume": 1.0}
        st.success(f"Loaded {uploaded_file.name}")

# Main Interface
for track_id, track in st.session_state.tracks.items():
    with st.expander(f"Track: {track['name']} ({track_id[:8]})", expanded=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            # Waveform Visualization using a dictionary for options
            options = {"height": 100, "normalize": True, "waveColor": "violet", "progressColor": "purple"}
            audix(audio=track["audio"].T, sample_rate=track["sr"], options=options, key=f"audix_{track_id}")  # type: ignore

        with col2:
            track["mute"] = st.checkbox("Mute", value=track["mute"], key=f"mute_{track_id}")
            track["solo"] = st.checkbox("Solo", value=track["solo"], key=f"solo_{track_id}")
            track["volume"] = st.slider("Volume", 0.0, 2.0, track["volume"], key=f"vol_{track_id}")

        # Playback Controls
        st.subheader("Playback")
        play_button = st.button("Play", key=f"play_{track_id}")
        stop_button = st.button("Stop", key=f"stop_{track_id}")
        if play_button:
            st.session_state.play_position = 0.0  # Reset to start
            st.audio(save_audio(track["audio"] * track["volume"], track["sr"], "temp.wav"), format="audio/wav")
        if stop_button:
            st.session_state.play_position = 0.0

        # Effects
        st.subheader("Effects")
        effect = st.selectbox("Add Effect", ["None", "Binaural Beats", "Solfeggio Frequency", "Morphic Field"], key=f"effect_{track_id}")

        if effect == "Binaural Beats":
            freq_left = st.slider("Left Freq (Hz)", 100, 1000, 440, key=f"left_{track_id}")
            freq_right = st.slider("Right Freq (Hz)", 100, 1000, 444, key=f"right_{track_id}")
            if st.button("Apply Binaural", key=f"apply_bb_{track_id}"):
                duration = len(track["audio"]) / track["sr"]
                binaural = generate_binaural_beats(duration, freq_left, freq_right, track["sr"])
                track["audio"] = binaural  # Replace audio with binaural beats
                st.success("Binaural beats applied!")

        elif effect == "Solfeggio Frequency":
            solfeggio_freqs = [174, 285, 396, 417, 528, 639, 741, 852, 963]
            freq = st.selectbox("Solfeggio Freq (Hz)", solfeggio_freqs, key=f"solf_{track_id}")
            if st.button("Apply Solfeggio", key=f"apply_sf_{track_id}"):
                track["audio"] = apply_solfeggio_frequency(track["audio"], freq, track["sr"])  # type: ignore
                st.success("Solfeggio frequency applied!")

        elif effect == "Morphic Field":
            intensity = st.slider("Intensity", 0.1, 1.0, 0.5, key=f"morph_{track_id}")
            if st.button("Apply Morphic", key=f"apply_mf_{track_id}"):
                # Simplified morphic field as noise overlay
                noise = np.random.normal(0, intensity, track["audio"].shape)
                track["audio"] = np.clip(track["audio"] + noise, -1.0, 1.0)
                st.success("Morphic field applied!")

        # Save Audio
        if st.button("Save Track", key=f"save_{track_id}"):
            buffer = save_audio(track["audio"] * track["volume"], track["sr"], f"{track['name']}_processed.wav")
            st.download_button(label="Download", data=buffer, file_name=f"{track['name']}_processed.wav", mime="audio/wav", key=f"download_{track_id}")

# Instructions
st.markdown("""
### How to Use:
1. Upload an audio file via the sidebar.
2. Adjust track settings (mute, solo, volume).
3. Apply effects like binaural beats, solfeggio frequencies, or morphic fields.
4. Play the track or save the processed audio.
""")
