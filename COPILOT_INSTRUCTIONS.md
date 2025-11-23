# Copilot Instructions for MindMorph (make-my-own-subliminal)

This document provides context, rules, and coding conventions for AI assistants working on the MindMorph project.

## Project Overview
**MindMorph** is a Streamlit-based Python application for creating and editing subliminal audio tracks. It features advanced audio processing, text-to-speech integration (Piper TTS), and a modular wizard-based UI.

## Technology Stack
- **Language**: Python 3.11
- **Framework**: Streamlit
- **Audio Processing**: `librosa`, `soundfile`, `pydub`, `scipy`
- **Text-to-Speech**: Piper TTS (via `onnxruntime`)
- **Containerization**: Docker

## Critical Constraints & Rules

### 1. Dependency Constraints
- **NumPy**: Must be `< 2.0` to avoid ABI incompatibility with libraries like `scipy` and `numba`.
- **Streamlit**: Use `streamlit-advanced-audio` for audio components.

### 2. State Management
- **Centralized State**: Use `app_state.py` or `config.py` for maintaining application state and global configurations.
- **Feature Configs**: Create `<feature>_config.py` for feature-specific configurations if needed.
- **Memory Efficiency**: The application must be extremely memory efficient. Clear large objects from memory (session state) immediately after export/download and return to the home page.

### 3. Audio Processing Logic
- **Looping**: If the background noise/music is shorter than the affirmation track, it **must** be looped until the end of the track.
- **Affirmation Expansion**: When a user selects a topic for "Auto Subliminal", you **must** use `affirmation_expander.py` to generate/expand the affirmations.
- **Preview**: Before exporting the final result, always generate a **10-second preview** audio.

### 4. Project Structure
- `main.py`: Application entry point and router.
- `auto_subliminal/`: Logic and UI for the auto-creation workflow.
- `audio_utils/`: Core audio processing functions.
- `assets/`: Stores voice models, background sounds, and images.

## Coding Style
- **Type Hinting**: Use Python type hints for all function signatures.
- **Docstrings**: Include docstrings for all major functions and classes.
- **Error Handling**: Gracefully handle audio processing errors and display user-friendly messages via `st.error`.
