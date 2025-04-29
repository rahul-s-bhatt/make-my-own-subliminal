# wizard_steps/step_1_affirmations.py
# ==========================================
# UI Rendering for Wizard Step 1: Affirmations
# ==========================================

import logging

import streamlit as st

# Import necessary components from other modules
from audio_io import load_audio, save_audio_to_bytesio
from config import GLOBAL_SR, MAX_AFFIRMATION_CHARS, MAX_AUDIO_DURATION_S
from tts_generator import TTSGenerator  # Type hint

# Constants from the original wizard file
WIZARD_MAX_UPLOAD_SIZE_MB = 15
WIZARD_MAX_UPLOAD_SIZE_BYTES = WIZARD_MAX_UPLOAD_SIZE_MB * 1024 * 1024

logger = logging.getLogger(__name__)


def render_step_1(wizard):
    """
    Renders the UI for Step 1: Affirmations.

    Args:
        wizard: The instance of the main QuickWizard class.
    """
    st.subheader("Step 1: Add Your Affirmations")
    st.write("This is the core message of your subliminal. Type your affirmations or upload an audio file.")

    # Determine the index for the radio button based on current state
    source_options = ["Type Text", "Upload Audio File"]
    current_source = None
    if st.session_state.wizard_affirmation_source == "text":
        current_source = "Type Text"
    elif st.session_state.wizard_affirmation_source == "upload":
        current_source = "Upload Audio File"

    try:
        source_index = source_options.index(current_source) if current_source else 0
    except ValueError:
        source_index = 0  # Default to 'Type Text' if state is inconsistent

    affirmation_source = st.radio("Affirmation Source:", source_options, index=source_index, key="wizard_affirm_source_radio", horizontal=True, label_visibility="collapsed")

    # --- Logic for Text Input ---
    if affirmation_source == "Type Text":
        st.caption(f"Enter your affirmations below (max {MAX_AFFIRMATION_CHARS} characters).")
        text_input = st.text_area(
            "Affirmations",
            value=st.session_state.wizard_affirmation_text,
            height=200,
            key="wizard_affirm_text_area",
            label_visibility="collapsed",
            max_chars=MAX_AFFIRMATION_CHARS,
            on_change=wizard.sync_affirmation_text,  # Use callback to update state
        )
        # st.session_state.wizard_affirmation_text = text_input # Update state directly (alternative)
        st.caption(f"{len(st.session_state.wizard_affirmation_text)} / {MAX_AFFIRMATION_CHARS} characters")

        if st.button("Generate Affirmation Audio", key="wizard_generate_tts_button"):
            current_text = st.session_state.wizard_affirmation_text
            if not current_text or not current_text.strip():
                st.warning("Please enter some affirmation text.")
            else:
                with st.spinner("Generating audio from text..."):
                    try:
                        # Use the TTS generator passed via the wizard instance
                        audio, sr = wizard.tts_generator.generate(current_text)
                        if audio is not None and sr is not None:
                            st.session_state.wizard_affirmation_audio = audio
                            st.session_state.wizard_affirmation_sr = sr
                            st.session_state.wizard_affirmation_source = "text"
                            st.success("Affirmation audio generated!")
                            st.rerun()  # Rerun to show preview/next button
                        else:
                            st.error("Failed to generate audio from text.")
                    except Exception as e:
                        logger.exception("Error generating TTS in wizard step 1")
                        st.error(f"Error generating audio: {e}")

    # --- Logic for File Upload ---
    elif affirmation_source == "Upload Audio File":
        st.caption(f"Upload your pre-recorded affirmations (WAV or MP3, max {WIZARD_MAX_UPLOAD_SIZE_MB}MB).")
        uploaded_file = st.file_uploader(
            "Upload Affirmation Audio",
            type=["wav", "mp3"],
            key="wizard_affirm_file_uploader",
            label_visibility="collapsed",
            # Clear previous audio if a new file is uploaded or removed
            on_change=wizard.clear_affirmation_upload_state,
        )
        if uploaded_file is not None:
            # --- File Size Validation ---
            if uploaded_file.size > WIZARD_MAX_UPLOAD_SIZE_BYTES:
                st.error(f"❌ File '{uploaded_file.name}' ({uploaded_file.size / (1024 * 1024):.1f} MB) exceeds the {WIZARD_MAX_UPLOAD_SIZE_MB} MB limit.")
                # Clear the invalid upload from state if needed
                st.session_state.wizard_affirmation_audio = None
                st.session_state.wizard_affirmation_sr = None
                st.session_state.wizard_affirmation_source = None
                st.session_state.wizard_affirmation_text = ""  # Clear text reference too
            else:
                # Only load if audio is not already loaded from this file
                # (Prevents reloading on every rerun after successful upload)
                # We use the text field storing the filename as a check
                expected_text = f"Uploaded: {uploaded_file.name}"
                if st.session_state.get("wizard_affirmation_text") != expected_text:
                    with st.spinner(f"Loading '{uploaded_file.name}'..."):
                        try:
                            # Load audio (ensure target_sr is used correctly)
                            audio, sr = load_audio(uploaded_file, target_sr=GLOBAL_SR)
                            if audio is not None and sr is not None and audio.size > 0:
                                duration_seconds = len(audio) / sr if sr > 0 else 0
                                if duration_seconds > MAX_AUDIO_DURATION_S:
                                    st.error(f"❌ File '{uploaded_file.name}' too long ({duration_seconds:.1f}s). Max is {MAX_AUDIO_DURATION_S // 60} min.")
                                    st.session_state.wizard_affirmation_audio = None  # Clear invalid
                                    st.session_state.wizard_affirmation_sr = None
                                    st.session_state.wizard_affirmation_source = None
                                    st.session_state.wizard_affirmation_text = ""
                                else:
                                    st.session_state.wizard_affirmation_audio = audio
                                    st.session_state.wizard_affirmation_sr = sr
                                    # Store filename as text reference AND set source
                                    st.session_state.wizard_affirmation_text = expected_text
                                    st.session_state.wizard_affirmation_source = "upload"
                                    st.success(f"Loaded '{uploaded_file.name}' successfully!")
                                    st.rerun()  # Rerun to show preview/next button
                            elif audio is not None:
                                st.warning(f"File '{uploaded_file.name}' appears to be empty or invalid.")
                                st.session_state.wizard_affirmation_audio = None  # Clear invalid
                                st.session_state.wizard_affirmation_sr = None
                                st.session_state.wizard_affirmation_source = None
                                st.session_state.wizard_affirmation_text = ""
                            else:
                                st.error(f"Failed to load audio from '{uploaded_file.name}'.")
                                st.session_state.wizard_affirmation_audio = None  # Clear invalid
                                st.session_state.wizard_affirmation_sr = None
                                st.session_state.wizard_affirmation_source = None
                                st.session_state.wizard_affirmation_text = ""
                        except Exception as e:
                            logger.exception(f"Error loading affirmation audio in wizard: {uploaded_file.name}")
                            st.error(f"Error loading file: {e}")
                            st.session_state.wizard_affirmation_audio = None  # Clear on error
                            st.session_state.wizard_affirmation_sr = None
                            st.session_state.wizard_affirmation_source = None
                            st.session_state.wizard_affirmation_text = ""

    # --- Display Preview and Navigation ---
    if st.session_state.get("wizard_affirmation_audio") is not None:
        st.divider()
        st.caption("Affirmation Audio Preview (Original):")
        try:
            # Ensure audio data and sr are valid before attempting to save/play
            audio_data = st.session_state.wizard_affirmation_audio
            sample_rate = st.session_state.wizard_affirmation_sr
            if audio_data is not None and sample_rate is not None:
                audio_bytes = save_audio_to_bytesio(audio_data, sample_rate)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                else:
                    st.warning("Could not generate affirmation preview.")
            else:
                st.warning("Affirmation audio data or sample rate missing for preview.")

        except Exception as e:
            logger.error(f"Error creating wizard affirmation preview bytes: {e}")
            st.warning("Could not play affirmation preview due to an error.")

        st.button("Next: Add Background Sound", on_click=wizard._go_to_step, args=(2,), type="primary", key="wizard_step1_next")
    elif affirmation_source == "Upload Audio File" and not uploaded_file:
        # If upload is selected but no file is present yet, show a disabled Next button or instruction
        st.info("Upload an audio file to proceed.")
        st.button("Next: Add Background Sound", disabled=True, key="wizard_step1_next_disabled")
    elif affirmation_source == "Type Text" and not st.session_state.get("wizard_affirmation_text", "").strip():
        st.info("Enter text and click 'Generate Affirmation Audio' to proceed.")
        st.button("Next: Add Background Sound", disabled=True, key="wizard_step1_next_disabled")
