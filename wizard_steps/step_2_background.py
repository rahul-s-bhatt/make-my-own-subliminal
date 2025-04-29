# wizard_steps/step_2_background.py
# ==========================================
# UI Rendering for Wizard Step 2: Background Sound
# ==========================================

import logging

import streamlit as st

# Import necessary components
from audio_generators import generate_noise
from audio_io import load_audio, save_audio_to_bytesio
from config import GLOBAL_SR, MAX_AUDIO_DURATION_S

# Constants
WIZARD_MAX_UPLOAD_SIZE_MB = 15
WIZARD_MAX_UPLOAD_SIZE_BYTES = WIZARD_MAX_UPLOAD_SIZE_MB * 1024 * 1024
WIZARD_NOISE_OPTIONS = ["None", "White Noise", "Pink Noise", "Brown Noise"]  # Keep local or import from config/wizard_config
WIZARD_NOISE_PREVIEW_DURATION_S = 10  # Generate only 10 seconds for preview/step

logger = logging.getLogger(__name__)


def render_step_2(wizard):
    """
    Renders the UI for Step 2: Background Sound.

    Args:
        wizard: The instance of the main QuickWizard class.
    """
    st.subheader("Step 2: Add Background Sound (Optional)")
    st.write("Add music or ambient noise to mask the affirmations.")

    choice_options = ["Upload Music/Sound", "Generate Noise", "None (Skip)"]
    try:
        current_choice_label = st.session_state.get("wizard_background_choice_label", "None (Skip)")
        choice_index = choice_options.index(current_choice_label)
    except ValueError:
        choice_index = 2  # Default to "None (Skip)"

    # Let the callback handle state changes, including clearing audio if choice changes
    choice = st.radio(
        "Background Sound Source:",
        choice_options,
        key="wizard_bg_choice_radio",
        horizontal=True,
        index=choice_index,
        on_change=wizard.sync_background_choice,  # Callback updates state and clears audio
        args=(choice_options,),
    )

    # --- Logic for Upload ---
    if st.session_state.wizard_background_choice == "upload":
        st.caption(f"Upload background audio (WAV or MP3, max {WIZARD_MAX_UPLOAD_SIZE_MB}MB).")
        uploaded_file = st.file_uploader(
            "Upload Background Audio",
            type=["wav", "mp3"],
            key="wizard_bg_file_uploader",
            label_visibility="collapsed",
            on_change=wizard.clear_background_upload_state,  # Clear if file changes/removed
        )
        if uploaded_file is not None:
            if uploaded_file.size > WIZARD_MAX_UPLOAD_SIZE_BYTES:
                st.error(f"❌ File '{uploaded_file.name}' ({uploaded_file.size / (1024 * 1024):.1f} MB) exceeds the {WIZARD_MAX_UPLOAD_SIZE_MB} MB limit.")
                # Ensure state is cleared by the callback or explicitly here if needed
                if st.session_state.get("wizard_background_audio") is not None:
                    st.session_state.wizard_background_audio = None
                    st.session_state.wizard_background_sr = None
            else:
                # Load only if audio is not already loaded from this file
                if st.session_state.get("wizard_background_audio") is None:
                    with st.spinner(f"Loading '{uploaded_file.name}'..."):
                        try:
                            audio, sr = load_audio(uploaded_file, target_sr=GLOBAL_SR)
                            if audio is not None and sr is not None and audio.size > 0:
                                duration_seconds = len(audio) / sr if sr > 0 else 0
                                if duration_seconds > MAX_AUDIO_DURATION_S * 2:
                                    st.warning(f"⚠️ File '{uploaded_file.name}' is quite long ({duration_seconds:.1f}s). It will be used but may increase processing time.")
                                st.session_state.wizard_background_audio = audio
                                st.session_state.wizard_background_sr = sr
                                st.success(f"Loaded '{uploaded_file.name}'!")
                                # Rerun might still be needed here after successful load to show slider correctly
                                # Let's test without it first, rely on implicit rerun. If slider doesn't show, add back.
                                # st.rerun()
                            elif audio is not None:
                                st.warning(f"File '{uploaded_file.name}' appears to be empty or invalid.")
                            else:
                                st.error(f"Failed to load audio from '{uploaded_file.name}'.")
                        except Exception as e:
                            logger.exception(f"Error loading background audio in wizard: {uploaded_file.name}")
                            st.error(f"Error loading file: {e}")

        # Show volume slider only if audio is loaded
        if st.session_state.get("wizard_background_audio") is not None:
            st.session_state.wizard_background_volume = st.slider("Background Volume", 0.0, 1.0, st.session_state.wizard_background_volume, 0.05, key="wizard_bg_upload_vol")

    # --- Logic for Noise Generation ---
    elif st.session_state.wizard_background_choice == "noise":
        noise_options_display = WIZARD_NOISE_OPTIONS[1:]
        try:
            current_noise_index = noise_options_display.index(st.session_state.wizard_background_noise_type)
        except ValueError:
            current_noise_index = 0

        noise_type = st.selectbox("Noise Type:", noise_options_display, key="wizard_bg_noise_type_select", index=current_noise_index)
        # Update state if changed - NO rerun needed here, just set state
        if noise_type != st.session_state.wizard_background_noise_type:
            st.session_state.wizard_background_noise_type = noise_type
            st.session_state.wizard_background_audio = None  # Force regeneration on next run
            st.session_state.wizard_background_sr = None
            # <<< REMOVED st.rerun() >>>

        current_bg_volume = st.session_state.wizard_background_volume
        new_bg_volume = st.slider("Noise Volume", 0.0, 1.0, current_bg_volume, 0.05, key="wizard_bg_noise_vol")
        # Update state if volume changed - NO rerun needed here
        if new_bg_volume != current_bg_volume:
            st.session_state.wizard_background_volume = new_bg_volume
            # Only force regeneration if volume change should affect the sample *significantly*
            # For simplicity, let's regenerate if volume changes. Clear audio state.
            st.session_state.wizard_background_audio = None
            st.session_state.wizard_background_sr = None
            # <<< REMOVED st.rerun() >>>

        # Generate noise if not already generated for the current type/volume
        if st.session_state.get("wizard_background_audio") is None:
            st.info(f"Generating a {WIZARD_NOISE_PREVIEW_DURATION_S}-second sample of {noise_type}. This will be looped during final export.")
            with st.spinner(f"Generating {noise_type} sample..."):
                volume = st.session_state.wizard_background_volume  # Use current volume
                audio = generate_noise(noise_type, WIZARD_NOISE_PREVIEW_DURATION_S, GLOBAL_SR, volume)
                if audio is not None:
                    st.session_state.wizard_background_audio = audio
                    st.session_state.wizard_background_sr = GLOBAL_SR
                    # Rerun IS needed here to show the UI update after generation finishes
                    st.rerun()
                else:
                    st.error(f"Failed to generate {noise_type}.")

    # --- Logic for None ---
    elif st.session_state.wizard_background_choice == "none":
        st.caption("No background sound will be added.")
        pass

    # --- Navigation Buttons ---
    st.divider()
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        st.button("Back: Affirmations", on_click=wizard._go_to_step, args=(1,), key="wizard_step2_back")
    with col_nav2:
        st.button("Next: Add Frequency", on_click=wizard._go_to_step, args=(3,), type="primary", key="wizard_step2_next")
