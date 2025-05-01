# wizard_steps/step_2_background.py
# ==========================================
# UI Rendering for Wizard Step 2: Background Sound
# ==========================================

import logging
import os  # Import os for path operations if needed
import tempfile  # Import tempfile

import streamlit as st

# Import necessary components
from audio_utils.audio_generators import generate_noise
from audio_utils.audio_io import load_audio
from config import MAX_AUDIO_DURATION_S  # Import upload limits
from config import GLOBAL_SR, MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB

# Constants
WIZARD_NOISE_OPTIONS = [
    "None",
    "White Noise",
    "Pink Noise",
    "Brown Noise",
]  # Keep local or import from config/wizard_config
WIZARD_NOISE_PREVIEW_DURATION_S = 10  # Generate only 10 seconds for preview/step

logger = logging.getLogger(__name__)


def _process_background_choice(wizard):
    """
    Helper function to load/generate background audio based on choice.
    This ensures audio is ready before moving to the next step if needed,
    or handles the 'None' case.
    """
    choice = st.session_state.get("wizard_background_choice")
    audio_loaded = st.session_state.get("wizard_background_audio") is not None

    if choice == "upload" and not audio_loaded:
        # This case should ideally be handled by the file uploader logic below,
        # but we can add a warning if they try to proceed without a successful upload.
        logger.warning(
            "Proceeding from Step 2 with 'Upload' selected but no audio loaded."
        )
        # Optionally show a warning: st.warning("Please ensure background audio is loaded before proceeding.")
        pass  # Allow proceeding, maybe user intends to skip background
    elif choice == "noise" and not audio_loaded:
        # Generate noise if it wasn't generated during interaction
        noise_type = st.session_state.wizard_background_noise_type
        volume = st.session_state.wizard_background_volume
        logger.info(f"Generating background noise '{noise_type}' before proceeding.")
        with st.spinner(f"Generating {noise_type} sample..."):
            audio = generate_noise(
                noise_type, WIZARD_NOISE_PREVIEW_DURATION_S, GLOBAL_SR, volume
            )
            if audio is not None:
                st.session_state.wizard_background_audio = audio
                st.session_state.wizard_background_sr = GLOBAL_SR
            else:
                st.error(f"Failed to generate {noise_type}.")
                # Should we prevent proceeding? For now, allow, but log error.
                logger.error(
                    f"Failed to generate background noise {noise_type} before Step 3."
                )
    elif choice == "none":
        # Ensure audio state is cleared if 'None' is chosen
        if audio_loaded:
            st.session_state.wizard_background_audio = None
            st.session_state.wizard_background_sr = None
            logger.info("Cleared background audio state as 'None' was selected.")


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
        current_choice_label = st.session_state.get(
            "wizard_background_choice_label", "None (Skip)"
        )
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
        st.caption(f"Upload background audio (WAV or MP3, max {MAX_UPLOAD_SIZE_MB}MB).")
        uploaded_file = st.file_uploader(
            "Upload Background Audio",
            type=["wav", "mp3"],
            key="wizard_bg_file_uploader",
            label_visibility="collapsed",
            # Removed on_change here, sync_background_choice handles clearing if radio changes
        )
        if uploaded_file is not None:
            # Check if this specific file has already been processed and loaded
            # We need a way to track the loaded file's identity if the widget persists
            # For simplicity, let's assume if audio exists, it's from the current uploader state
            # A more robust way would involve storing filename in state.
            audio_already_loaded = (
                st.session_state.get("wizard_background_audio") is not None
            )

            if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
                st.error(
                    f"❌ File '{uploaded_file.name}' ({uploaded_file.size / (1024 * 1024):.1f} MB) exceeds the {MAX_UPLOAD_SIZE_MB} MB limit."
                )
                if audio_already_loaded:  # Clear if a previous valid file was loaded
                    st.session_state.wizard_background_audio = None
                    st.session_state.wizard_background_sr = None
            elif not audio_already_loaded:  # Only load if not already loaded
                with st.spinner(f"Processing '{uploaded_file.name}'..."):
                    try:
                        # Save temporarily to load with librosa
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                        ) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            temp_file_path = tmp.name

                        audio_data, sr = load_audio(
                            temp_file_path, target_sr=GLOBAL_SR
                        )  # Resample to global SR

                        # Clean up temp file
                        try:
                            os.remove(temp_file_path)
                        except OSError:
                            logger.warning(
                                f"Could not remove temp audio file: {temp_file_path}"
                            )

                        if (
                            audio_data is not None
                            and sr is not None
                            and audio_data.size > 0
                        ):
                            duration_seconds = len(audio_data) / sr if sr > 0 else 0
                            if (
                                duration_seconds > MAX_AUDIO_DURATION_S * 2
                            ):  # Example check
                                st.warning(
                                    f"⚠️ File '{uploaded_file.name}' is quite long ({duration_seconds:.1f}s). It will be used but may increase processing time."
                                )
                            st.session_state.wizard_background_audio = audio_data
                            st.session_state.wizard_background_sr = sr
                            st.success(f"Loaded '{uploaded_file.name}'!")
                            st.rerun()  # Rerun needed to show slider after load
                        elif audio_data is not None:
                            st.warning(
                                f"File '{uploaded_file.name}' appears to be empty or invalid."
                            )
                        else:
                            st.error(
                                f"Failed to load audio from '{uploaded_file.name}'."
                            )
                    except Exception as e:
                        logger.exception(
                            f"Error loading background audio in wizard: {uploaded_file.name}"
                        )
                        st.error(f"Error loading file: {e}")
                        # Clean up temp file in case of error during processing
                        if "temp_file_path" in locals() and os.path.exists(
                            temp_file_path
                        ):
                            try:
                                os.remove(temp_file_path)
                            except OSError:
                                pass

        # Show volume slider only if audio is loaded
        if st.session_state.get("wizard_background_audio") is not None:
            st.session_state.wizard_background_volume = st.slider(
                "Background Volume",
                0.0,
                1.0,
                st.session_state.wizard_background_volume,
                0.05,
                key="wizard_bg_upload_vol",
            )

    # --- Logic for Noise Generation ---
    elif st.session_state.wizard_background_choice == "noise":
        noise_options_display = WIZARD_NOISE_OPTIONS[1:]  # Exclude "None"
        try:
            current_noise_index = noise_options_display.index(
                st.session_state.wizard_background_noise_type
            )
        except ValueError:
            current_noise_index = 0

        noise_type = st.selectbox(
            "Noise Type:",
            noise_options_display,
            key="wizard_bg_noise_type_select",
            index=current_noise_index,
        )

        # Update state if changed - NO rerun needed here, just set state
        if noise_type != st.session_state.wizard_background_noise_type:
            st.session_state.wizard_background_noise_type = noise_type
            st.session_state.wizard_background_audio = (
                None  # Force regeneration on next run
            )
            st.session_state.wizard_background_sr = None

        current_bg_volume = st.session_state.wizard_background_volume
        new_bg_volume = st.slider(
            "Noise Volume", 0.0, 1.0, current_bg_volume, 0.05, key="wizard_bg_noise_vol"
        )

        # Update state if volume changed
        if new_bg_volume != current_bg_volume:
            st.session_state.wizard_background_volume = new_bg_volume
            # Force regeneration if volume changes
            st.session_state.wizard_background_audio = None
            st.session_state.wizard_background_sr = None
            st.rerun()  # Rerun needed to trigger regeneration display

        # Generate noise if not already generated for the current type/volume
        if st.session_state.get("wizard_background_audio") is None:
            st.info(
                f"Generating a {WIZARD_NOISE_PREVIEW_DURATION_S}-second sample of {noise_type}. This will be looped during final export."
            )
            with st.spinner(f"Generating {noise_type} sample..."):
                volume = st.session_state.wizard_background_volume  # Use current volume
                audio = generate_noise(
                    noise_type, WIZARD_NOISE_PREVIEW_DURATION_S, GLOBAL_SR, volume
                )
                if audio is not None:
                    st.session_state.wizard_background_audio = audio
                    st.session_state.wizard_background_sr = GLOBAL_SR
                    st.rerun()  # Rerun IS needed here to show the UI update
                else:
                    st.error(f"Failed to generate {noise_type}.")

    # --- Logic for None ---
    elif st.session_state.wizard_background_choice == "none":
        st.caption("No background sound will be added.")
        # Ensure audio state is cleared if 'None' is chosen
        if st.session_state.get("wizard_background_audio") is not None:
            st.session_state.wizard_background_audio = None
            st.session_state.wizard_background_sr = None
            logger.info("Cleared background audio state as 'None' was selected.")

    # --- Navigation Buttons ---
    st.divider()
    # Use 3 columns for Home, Back, Next
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])  # Adjust ratios

    with col_nav_1:
        # Add Back to Home button
        if st.button(
            "🏠 Back to Home",
            key="wizard_step2_home",
            use_container_width=True,
            help="Exit Wizard and return to main menu.",
        ):
            wizard._reset_wizard_state()
            # st.rerun() is handled by reset_wizard_state indirectly

    with col_nav_2:
        if st.button(
            "⬅️ Back: Affirmations", key="wizard_step2_back", use_container_width=True
        ):
            wizard._go_to_step(1)

    with col_nav_3:
        # Validation for 'Next' - always enabled from this step onwards? Or validate selection?
        # For now, assume always enabled if they reached step 2.
        if st.button(
            "Next: Frequency ➡️",
            key="wizard_step2_next",
            type="primary",
            use_container_width=True,
        ):
            # Process background choice before moving on
            _process_background_choice(wizard)  # Call helper to load/generate if needed
            wizard._go_to_step(3)
