# make-my-own-subliminal (MindMorph - Pro Subliminal Audio Editor)

A Streamlit-based Python application for creating and editing subliminal audio tracks with advanced audio processing and text-to-speech features.

## Features

- Create custom subliminal audio tracks
- Advanced audio processing (mixing, effects, background sounds)
- Text-to-speech (Piper TTS, ONNX models)
- Modular UI with wizard-based workflow
- Docker support for easy deployment

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/make-my-own-subliminal.git
cd make-my-own-subliminal
```

### 2. Install Python Dependencies

It is recommended to use Python 3.11.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download Voice Models

Place your Piper TTS voice models (`.onnx`, `.json`) in the `assets/voices/` directory as required.

### 4. Run the Application

```bash
streamlit run main.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

---

## Docker Usage

Build and run the app in a container:

```bash
docker build -t mindmorph .
docker run -p 8501:8501 mindmorph
```

---

## Project Structure

- [`main.py`](main.py:1): Application entry point (Streamlit router)
- [`auto_subliminal/`](auto_subliminal/): Auto-create subliminal workflow UI and logic
- [`audio_utils/`](audio_utils/): Audio processing utilities
- [`assets/`](assets/): Voice models, background sounds, and images

---

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to your fork and open a Pull Request

---

## License

See [`LICENSE`](LICENSE:1) for details.
