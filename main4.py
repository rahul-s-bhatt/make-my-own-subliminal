import tempfile
import uuid
from io import BytesIO

import librosa
import numpy as np
import pyttsx3
import soundfile as sf
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_advanced_audio import WaveSurferOptions, audix

# Constants
GLOBAL_SR = 44100  # Fixed sample rate for all tracks

# Data Types
AudioData = np.ndarray  # Audio data as NumPy array (float32, shape: (samples, 2))
SampleRate = int  # Sample rate in Hz (44100)
TrackID = str  # Unique identifier for each track

# Session State Initialization
if "tracks" not in st.session_state:
    st.session_state.tracks = {}  # Dict[TrackID, dict] storing track data
if "play_position" not in st.session_state:
    st.session_state.play_position = 0.0  # Current playback position in seconds


# Utility Functions
def load_audio(file: UploadedFile, target_sr: SampleRate = GLOBAL_SR) -> tuple[AudioData, SampleRate]:
    """Load audio file into NumPy array and resample to target_sr."""
    audio, sr = librosa.load(file, sr=None, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)  # Convert mono to stereo
    else:
        audio = audio.T  # Transpose to (samples, channels)
    if sr != target_sr:
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr).T
    return audio.astype(np.float32), target_sr


def save_audio(audio: AudioData, sr: SampleRate, filename: str) -> BytesIO:
    """Save NumPy audio array to WAV file in memory."""
    buffer = BytesIO()
    sf.write(buffer, audio, sr, format="wav")
    buffer.seek(0)
    return buffer


def save_audio_to_temp(audio: AudioData, sr: SampleRate) -> str:
    """Save NumPy audio array to a temporary WAV file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, sr, format="wav")
        return tmp.name


def generate_binaural_beats(duration: float, freq_left: float, freq_right: float, sr: SampleRate, volume: float) -> AudioData:
    """Generate stereo binaural beats."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = volume * np.sin(2 * np.pi * freq_left * t)
    right = volume * np.sin(2 * np.pi * freq_right * t)
    return np.stack([left, right], axis=1).astype(np.float32)


def generate_solfeggio_frequency(duration: float, freq: float, sr: SampleRate, volume: float) -> AudioData:
    """Generate a stereo sine wave at the solfeggio frequency."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = volume * np.sin(2 * np.pi * freq * t)
    return np.stack([sine_wave, sine_wave], axis=1).astype(np.float32)


def apply_speed_change(audio: AudioData, sr: SampleRate, speed_factor: float) -> AudioData:
    """Apply speed change effect by resampling."""
    if speed_factor == 1.0:
        return audio
    # Resample to change speed and pitch
    new_sr = sr / speed_factor
    audio_resampled = librosa.resample(audio.T, orig_sr=sr, target_sr=new_sr).T
    return audio_resampled.astype(np.float32)


def mix_tracks(tracks: dict, preview: bool = False) -> AudioData:
    """Mix all tracks based on their settings."""
    if not tracks:
        return np.zeros((0, 2), dtype=np.float32)

    # Determine if any track is soloed
    solo_tracks = [t for t in tracks.values() if t["solo"]]
    active_tracks = solo_tracks if solo_tracks else tracks.values()

    # Find max length
    max_len = max(len(t["processed_audio"]) for t in active_tracks if not t["mute"])
    if preview:
        max_len = min(max_len, int(GLOBAL_SR * 10))  # 10 seconds for preview

    mix = np.zeros((max_len, 2), dtype=np.float32)
    for t in active_tracks:
        if t["mute"]:
            continue
        audio = t["processed_audio"]
        if len(audio) < max_len:
            audio = np.pad(audio, ((0, max_len - len(audio)), (0, 0)), mode="constant")
        elif len(audio) > max_len:
            audio = audio[:max_len]
        mix += audio * t["volume"]

    mix = np.clip(mix, -1.0, 1.0)
    return mix


# Main Streamlit App
st.title("🎧 Pro Subliminal Audio Editor")

# Sidebar for Track Management
with st.sidebar:
    st.subheader("Track Management")

    # OLD CODE BLOCK (causes duplicates) Upload Audio Files
    # uploaded_files = st.file_uploader("Upload Audio Files", type=["wav", "mp3"], accept_multiple_files=True, key="upload_files")
    # for file in uploaded_files:
    #     audio, sr = load_audio(file)
    #     track_id = str(uuid.uuid4())
    #     st.session_state.tracks[track_id] = {
    #         "original_audio": audio,
    #         "processed_audio": audio.copy(),
    #         "sr": sr,
    #         "name": file.name,
    #         "volume": 1.0,
    #         "mute": False,
    #         "solo": False,
    #         "speed_factor": 1.0,
    #     }
    #     st.success(f"Loaded {file.name}")

    # --- Corrected File Upload Logic ---
    uploaded_files = st.file_uploader("Upload Audio Files", type=["wav", "mp3"], accept_multiple_files=True, key="upload_files")

    # Get names of tracks already loaded
    loaded_track_names = [t["name"] for t in st.session_state.tracks.values()]

    if uploaded_files is not None:
        for file in uploaded_files:
            # Check if a track with this name is already loaded
            if file.name not in loaded_track_names:
                try:
                    with st.spinner(f"Loading {file.name}..."):
                        audio, sr = load_audio(file)
                    track_id = str(uuid.uuid4())
                    st.session_state.tracks[track_id] = {
                        "original_audio": audio,
                        "processed_audio": audio.copy(),  # Start with a copy
                        "sr": sr,
                        "name": file.name,
                        "volume": 1.0,
                        "mute": False,
                        "solo": False,
                        "speed_factor": 1.0,
                        "pitch_shift": 0,  # New: Pitch shift in semitones
                        "pan": 0.0,  # New: Panning (-1 Left, 0 Center, 1 Right)
                        "filter_type": "off",  # New: 'off', 'lowpass', 'highpass'
                        "filter_cutoff": 1000.0,  # New: Cutoff frequency in Hz
                    }
                    st.success(f"Loaded {file.name}")
                    # Update the list of loaded names immediately to handle multiple uploads at once
                    loaded_track_names.append(file.name)
                except Exception as e:
                    st.error(f"Failed to load {file.name}: {e}")
            # else:
            #     # Optional: Notify user if file is already loaded
            #     # st.info(f"Track '{file.name}' is already loaded.")

    # Generate Affirmations from Text
    st.subheader("Generate Affirmations")
    affirmation_text = st.text_area("Enter Affirmations (One per line)", height=100, key="affirmation_text")
    if st.button("Generate TTS", key="generate_tts"):
        if affirmation_text:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            buffer = BytesIO()
            engine.save_to_file(affirmation_text, buffer)
            engine.runAndWait()
            buffer.seek(0)
            audio, sr = load_audio(buffer)
            track_id = str(uuid.uuid4())
            st.session_state.tracks[track_id] = {
                "original_audio": audio,
                "processed_audio": audio.copy(),
                "sr": sr,
                "name": "Affirmations",
                "volume": 1.0,
                "mute": False,
                "solo": False,
                "speed_factor": 1.0,
            }
            st.success("Affirmations generated!")

    # Generate Binaural Beats Track
    st.subheader("Generate Binaural Beats")
    bb_duration = st.number_input("Duration (seconds)", min_value=1, value=60, key=f"bb_duration_{uuid.uuid4()}")
    bb_freq_left = st.number_input("Left Frequency (Hz)", min_value=100, max_value=1000, value=440, key=f"bb_freq_left_{uuid.uuid4()}")
    bb_freq_right = st.number_input("Right Frequency (Hz)", min_value=100, max_value=1000, value=444, key=f"bb_freq_right_{uuid.uuid4()}")
    bb_volume = st.slider("Volume", 0.0, 1.0, 0.5, key=f"bb_volume_{uuid.uuid4()}")
    if st.button("Generate Binaural Beats", key="generate_bb"):
        audio = generate_binaural_beats(bb_duration, bb_freq_left, bb_freq_right, GLOBAL_SR, bb_volume)
        track_id = str(uuid.uuid4())
        st.session_state.tracks[track_id] = {
            "original_audio": audio,
            "processed_audio": audio.copy(),
            "sr": GLOBAL_SR,
            "name": "Binaural Beats",
            "volume": 1.0,
            "mute": False,
            "solo": False,
            "speed_factor": 1.0,
        }
        st.success("Binaural Beats track generated!")

    # Generate Solfeggio Frequency Track
    st.subheader("Generate Solfeggio Frequency")
    solfeggio_freqs = [174, 285, 396, 417, 528, 639, 741, 852, 963]
    solf_freq = st.selectbox("Solfeggio Frequency (Hz)", solfeggio_freqs, key=f"solf_freq_{uuid.uuid4()}")
    solf_duration = st.number_input("Duration (seconds)", min_value=1, value=60, key=f"solf_duration_{uuid.uuid4()}")
    solf_volume = st.slider("Volume", 0.0, 1.0, 0.5, key=f"solf_volume_{uuid.uuid4()}")
    if st.button("Generate Solfeggio Track", key="generate_solf"):
        audio = generate_solfeggio_frequency(solf_duration, solf_freq, GLOBAL_SR, solf_volume)
        track_id = str(uuid.uuid4())
        st.session_state.tracks[track_id] = {
            "original_audio": audio,
            "processed_audio": audio.copy(),
            "sr": GLOBAL_SR,
            "name": f"Solfeggio {solf_freq}Hz",
            "volume": 1.0,
            "mute": False,
            "solo": False,
            "speed_factor": 1.0,
        }
        st.success(f"Solfeggio {solf_freq}Hz track generated!")

# Main Interface for Tracks
for track_id in list(st.session_state.tracks.keys()):
    track = st.session_state.tracks[track_id]
    with st.expander(f"Track: {track['name']} ({track_id[:8]})", expanded=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            # Waveform Visualization

            ### Old CODE
            # options = {"height": 100, "normalize": True, "waveColor": "violet", "progressColor": "purple"}
            # temp_wav_path = save_audio_to_temp(track["processed_audio"], track["sr"])
            # audix(data=temp_wav_path, sample_rate=track["sr"], options=options, key=f"audix_{track_id}")
            # Clean up temporary file (optional, Streamlit handles this with delete=False)
            # os.remove(temp_wav_path)  # Uncomment if you want to manually clean up

            ### NEW code
            # Define WaveSurfer options using the correct parameter
            ws_options = WaveSurferOptions(
                height=100,
                normalize=True,
                wave_color="violet",
                progress_color="purple",
                # You can add other valid WaveSurferOptions here if needed
            )
            temp_wav_path = save_audio_to_temp(track["processed_audio"], track["sr"])
            # Call audix with the correct parameter name
            audix_result = audix(
                data=temp_wav_path,
                sample_rate=track["sr"],
                wavesurfer_options=ws_options,  # Use wavesurfer_options
                key=f"audix_{track_id}",
            )
            # Consider handling the audix_result if needed (e.g., playback position)
            # Clean up temporary file - Important! Since you use delete=False
            # You might want to remove the file after audix has used it,
            # although Streamlit might handle temp file cleanup eventually.
            # import os # Add import os at the top
            # try:
            #     os.remove(temp_wav_path)
            # except OSError as e:
            #     st.warning(f"Could not remove temp file {temp_wav_path}: {e}")

        with col2:
            track["name"] = st.text_input("Name", value=track["name"], key=f"name_{track_id}")
            track["volume"] = st.slider("Volume", 0.0, 2.0, track["volume"], key=f"vol_{track_id}")
            track["mute"] = st.checkbox("Mute", value=track["mute"], key=f"mute_{track_id}")
            track["solo"] = st.checkbox("Solo", value=track["solo"], key=f"solo_{track_id}")
            if st.button("Delete Track", key=f"delete_{track_id}"):
                del st.session_state.tracks[track_id]
                st.rerun()

        # Effects
        st.subheader("Effects")
        speed_factor = st.slider("Speed Factor", 0.5, 2.0, track["speed_factor"], key=f"speed_{track_id}")
        if st.button("Apply Speed Change", key=f"apply_speed_{track_id}"):
            track["speed_factor"] = speed_factor
            track["processed_audio"] = apply_speed_change(track["original_audio"], track["sr"], speed_factor)
            st.success(f"Speed changed to {speed_factor}x")

# Master Controls
st.subheader("Master Controls")
if st.button("Preview Mix (10s)", key="preview_mix"):
    with st.spinner("Generating preview..."):
        mix = mix_tracks(st.session_state.tracks, preview=True)
        if mix.size > 0:
            st.audio(save_audio(mix, GLOBAL_SR, "preview.wav"), format="audio/wav")
        else:
            st.warning("No tracks to mix.")

if st.button("Export Full Mix", key="export_mix"):
    with st.spinner("Generating full mix..."):
        mix = mix_tracks(st.session_state.tracks)
        if mix.size > 0:
            buffer = save_audio(mix, GLOBAL_SR, "full_mix.wav")
            st.download_button(label="Download Full Mix", data=buffer, file_name="full_mix.wav", mime="audio/wav", key="download_full_mix")
        else:
            st.warning("No tracks to mix.")

# Instructions
st.markdown("""
### How to Use:
1. **Upload Audio Files**: Use the sidebar to upload WAV or MP3 files.
2. **Generate Affirmations**: Enter text in the sidebar and generate TTS audio.
3. **Generate Special Tracks**: Create binaural beats or solfeggio frequency tracks with custom parameters.
4. **Adjust Track Settings**: In each track expander, modify the name, volume, mute, solo, and apply speed change effects.
5. **Preview and Export**: Use the master controls to preview a 10-second mix or export the full mix as a WAV file.

### Notes:
- All tracks are resampled to 44.1 kHz and converted to stereo for consistency.
- Speed change affects both speed and pitch (e.g., speeding up raises pitch).
- Solo takes precedence over mute; if any track is soloed, only soloed tracks are mixed.
""")
