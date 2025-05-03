# wizard_steps/step_1_affirmations.py
# ==========================================
# Step 1 UI for Quick Create Wizard: Affirmations
# ==========================================

import logging
import os
import tempfile  # Needed for saving uploaded audio temporarily

import streamlit as st

from affirmation_expander import expand_affirmations

# Import necessary components from other modules
from audio_utils.audio_io import load_audio
from config import (
    GLOBAL_SR,
    MAX_AFFIRMATION_CHARS,  # Import limits and SR
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
)
from utils import read_text_file

# NOTE: TTSGenerator is no longer imported directly here.
# The wizard instance passed to render_step_1 will hold the TTS generator.

logger = logging.getLogger(__name__)

# Define keys for widgets in this step to ensure consistency
AFFIRM_SOURCE_RADIO_KEY = "wizard_affirm_source_radio"
AFFIRM_TEXT_AREA_KEY = "wizard_affirm_text_area"
AFFIRM_FILE_UPLOADER_KEY = "wizard_affirm_file_uploader"
AFFIRM_ORIGINAL_TEXT_KEY = "wizard_original_affirmation_text"
AFFIRM_PENDING_UPDATE_KEY = "wizard_affirm_text_pending_update"
AFFIRM_PENDING_TRUNCATED_KEY = "wizard_affirm_truncated_pending"
# --- ADDED: Key for volume slider ---
AFFIRM_VOLUME_SLIDER_KEY = "wizard_affirm_vol_slider"
# --- END ADDED ---


def render_step_1(wizard):  # Pass the wizard instance which has tts_generator
    """Renders Step 1: Affirmations Input."""
    st.subheader("Step 1: Enter Your Affirmations")
    st.markdown("Provide the core affirmations you want to use. You can type them, upload a file, or record audio.")

    # --- Initialize state variables if they don't exist ---
    if AFFIRM_ORIGINAL_TEXT_KEY not in st.session_state:
        st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
    if AFFIRM_PENDING_UPDATE_KEY not in st.session_state:
        st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None
    if AFFIRM_PENDING_TRUNCATED_KEY not in st.session_state:
        st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
    if AFFIRM_TEXT_AREA_KEY not in st.session_state:
        st.session_state[AFFIRM_TEXT_AREA_KEY] = ""  # Ensure main text state exists
    # Ensure affirmation volume state exists (handled by initialize_wizard_state, but good practice)
    if "wizard_affirmation_volume" not in st.session_state:
        st.session_state.wizard_affirmation_volume = 1.0

    # --- Apply pending update at the start of the run ---
    if st.session_state.get(AFFIRM_PENDING_UPDATE_KEY) is not None:
        logger.debug("Applying pending affirmation text update.")
        st.session_state[AFFIRM_TEXT_AREA_KEY] = st.session_state[AFFIRM_PENDING_UPDATE_KEY]
        # Display warning if needed from the pending state
        if st.session_state.get(AFFIRM_PENDING_TRUNCATED_KEY):
            st.warning(
                f"⚠️ Text was automatically shortened to fit the {MAX_AFFIRMATION_CHARS} character limit.",
                icon="✂️",
            )
        # Clear pending state
        st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None
        st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False

    # --- Input Type Selection ---
    col_input_type, col_main_input = st.columns([1, 2])
    with col_input_type:
        st.markdown("**Choose Input Method:**")
        source_options = ["Type/Paste Text", "Upload Audio File", "Upload Text File"]
        current_source_label = "Type/Paste Text"
        current_source = st.session_state.get("wizard_affirmation_source")
        if current_source == "upload_audio":
            current_source_label = "Upload Audio File"
        elif current_source == "upload_text":
            current_source_label = "Upload Text File"

        try:
            default_index = source_options.index(current_source_label)
        except ValueError:
            default_index = 0

        selected_source_label = st.radio(
            "Affirmation Source:",
            options=source_options,
            index=default_index,
            key=AFFIRM_SOURCE_RADIO_KEY,
            label_visibility="collapsed",
        )

        new_source = "text"
        if selected_source_label == "Upload Audio File":
            new_source = "upload_audio"
        elif selected_source_label == "Upload Text File":
            new_source = "upload_text"

        if new_source != st.session_state.get("wizard_affirmation_source"):
            logger.info(f"Wizard Step 1: Affirmation source changed to '{new_source}'")
            st.session_state.wizard_affirmation_source = new_source
            # Clear conflicting data and original text backup
            if new_source != "text":
                st.session_state[AFFIRM_TEXT_AREA_KEY] = ""
                st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None
                st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
            if new_source != "upload_audio":
                st.session_state.wizard_affirmation_audio = None
                st.session_state.wizard_affirmation_sr = None
                # --- ADDED: Reset volume when switching away from audio ---
                st.session_state.wizard_affirmation_volume = 1.0  # Reset to default
                # --- END ADDED ---
            st.rerun()

    # --- Main Input Area ---
    with col_main_input:
        source = st.session_state.get("wizard_affirmation_source", "text")

        if source == "text":
            st.markdown("**Type or Paste Affirmations:** (one per line recommended)")
            affirmation_text_value = st.session_state.get(AFFIRM_TEXT_AREA_KEY, "")

            def clear_original_on_edit():  # Callback
                current_val = st.session_state.get(AFFIRM_TEXT_AREA_KEY)
                original_val = st.session_state.get(AFFIRM_ORIGINAL_TEXT_KEY)
                if original_val is not None and current_val != original_val:
                    st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                    logger.debug("Cleared original affirmation backup due to manual edit.")

            affirmation_text = st.text_area(
                "Affirmations Text Area",
                value=affirmation_text_value,
                key=AFFIRM_TEXT_AREA_KEY,
                height=200,
                max_chars=MAX_AFFIRMATION_CHARS,
                label_visibility="collapsed",
                help="Enter your affirmations here.",
                on_change=clear_original_on_edit,
            )
            st.caption(f"{len(affirmation_text_value)} / {MAX_AFFIRMATION_CHARS} characters")

            # Expansion and Undo Buttons
            button_col_1, button_col_2 = st.columns(2)
            with button_col_1:  # Expand
                expand_disabled = not affirmation_text_value.strip()
                if st.button(
                    "✨ Expand Affirmations",
                    key="wizard_expand_affirmations",
                    disabled=expand_disabled,
                    use_container_width=True,
                    help="Generate variations of your affirmations.",
                ):
                    with st.spinner("Expanding affirmations..."):
                        try:
                            st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = affirmation_text_value
                            expanded_text, truncated = expand_affirmations(
                                base_text=affirmation_text_value,
                                max_chars=MAX_AFFIRMATION_CHARS,
                                multiplier=3,
                            )
                            st.session_state[AFFIRM_PENDING_UPDATE_KEY] = expanded_text
                            st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = truncated
                            logger.info(f"Wizard affirmation expansion complete. Truncated: {truncated}")
                            st.rerun()
                        except Exception as e:
                            logger.error(
                                f"Error during wizard affirmation expansion: {e}",
                                exc_info=True,
                            )
                            st.error(f"Failed to expand affirmations: {e}")
                            st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                            st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None
                            st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
            with button_col_2:  # Undo
                undo_disabled = st.session_state.get(AFFIRM_ORIGINAL_TEXT_KEY) is None
                if st.button(
                    "↩️ Undo Expansion",
                    key="wizard_undo_expansion",
                    disabled=undo_disabled,
                    use_container_width=True,
                    help="Revert to the text before expansion.",
                ):
                    original_text = st.session_state.get(AFFIRM_ORIGINAL_TEXT_KEY)
                    if original_text is not None:
                        st.session_state[AFFIRM_PENDING_UPDATE_KEY] = original_text
                        st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
                        st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                        logger.info("User staged affirmation undo for next run.")
                        st.rerun()
                    else:
                        logger.warning("Undo clicked but no original text found in state.")

        elif source == "upload_audio":
            # Audio Upload UI
            st.markdown("**Upload Affirmation Audio:**")
            uploaded_audio_file = st.file_uploader(
                "Select an audio file (.wav, .mp3)",
                type=["wav", "mp3"],
                key="wizard_affirm_audio_uploader",
                label_visibility="collapsed",
                help=f"Upload your pre-recorded affirmations (Max {MAX_UPLOAD_SIZE_MB}MB).",
            )
            if uploaded_audio_file:
                # Processing logic for uploaded audio file
                if uploaded_audio_file.size > MAX_UPLOAD_SIZE_BYTES:
                    st.error(f"❌ File '{uploaded_audio_file.name}' exceeds {MAX_UPLOAD_SIZE_MB} MB limit.")
                    st.session_state.wizard_affirmation_audio = None
                    st.session_state.wizard_affirmation_sr = None
                    # --- ADDED: Reset volume if upload fails ---
                    st.session_state.wizard_affirmation_volume = 1.0
                    # --- END ADDED ---
                else:
                    with st.spinner(f"Processing '{uploaded_audio_file.name}'..."):
                        temp_file_path = None
                        try:
                            with tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=os.path.splitext(uploaded_audio_file.name)[1],
                            ) as tmp:
                                tmp.write(uploaded_audio_file.getvalue())
                                temp_file_path = tmp.name
                            audio_data, sr = load_audio(temp_file_path, target_sr=GLOBAL_SR)
                            if audio_data is not None and sr is not None and audio_data.size > 0:
                                st.session_state.wizard_affirmation_audio = audio_data
                                st.session_state.wizard_affirmation_sr = sr
                                st.success(f"✅ Loaded audio: '{uploaded_audio_file.name}' ({len(audio_data) / sr:.1f}s)")
                                logger.info(f"Wizard Step 1: Loaded affirmation audio '{uploaded_audio_file.name}'")
                                st.session_state[AFFIRM_TEXT_AREA_KEY] = ""  # Clear text area
                                st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None  # Clear backup
                                st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None
                                st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
                                # --- ADDED: Rerun to show volume slider ---
                                st.rerun()
                                # --- END ADDED ---
                            else:
                                st.error(f"❌ Failed to load audio from '{uploaded_audio_file.name}'.")
                                st.session_state.wizard_affirmation_audio = None
                                st.session_state.wizard_affirmation_sr = None
                                # --- ADDED: Reset volume on failure ---
                                st.session_state.wizard_affirmation_volume = 1.0
                                # --- END ADDED ---
                        except Exception as e:
                            logger.error(
                                f"Error processing audio file '{uploaded_audio_file.name}': {e}",
                                exc_info=True,
                            )
                            st.error(f"Error processing audio file: {e}")
                            st.session_state.wizard_affirmation_audio = None
                            st.session_state.wizard_affirmation_sr = None
                            # --- ADDED: Reset volume on exception ---
                            st.session_state.wizard_affirmation_volume = 1.0
                            # --- END ADDED ---
                        finally:
                            if temp_file_path and os.path.exists(temp_file_path):
                                try:
                                    os.remove(temp_file_path)
                                except OSError:
                                    logger.warning(f"Could not remove temp audio file: {temp_file_path}")

        elif source == "upload_text":
            # Text File Upload UI
            st.markdown("**Upload Affirmation Text File:**")
            uploaded_text_file = st.file_uploader(
                "Select a text file (.txt, .docx)",
                type=["txt", "docx"],
                key=AFFIRM_FILE_UPLOADER_KEY,
                label_visibility="collapsed",
                help="Upload affirmations from a text or Word document.",
            )
            if uploaded_text_file:
                # Processing logic for text file
                try:
                    text_content = read_text_file(uploaded_text_file)
                    if text_content is not None:
                        if len(text_content) > MAX_AFFIRMATION_CHARS:
                            st.error(f"❌ Text in file '{uploaded_text_file.name}' too long ({len(text_content)} chars). Max {MAX_AFFIRMATION_CHARS}.")
                            st.session_state[AFFIRM_TEXT_AREA_KEY] = ""
                        else:
                            st.session_state[AFFIRM_PENDING_UPDATE_KEY] = text_content
                            st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
                            logger.info(f"Wizard Step 1: Staged text from file '{uploaded_text_file.name}' for update.")
                            st.success(f"✅ Loaded text from: '{uploaded_text_file.name}'")
                            st.session_state.wizard_affirmation_audio = None  # Clear audio state
                            st.session_state.wizard_affirmation_sr = None
                            # --- ADDED: Reset volume when loading text ---
                            st.session_state.wizard_affirmation_volume = 1.0
                            # --- END ADDED ---
                            st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                            st.session_state.wizard_affirmation_source = "text"  # Switch back to text view
                            st.rerun()
                    else:
                        st.error(f"Could not read text content from '{uploaded_text_file.name}'. Check file format.")
                        st.session_state[AFFIRM_TEXT_AREA_KEY] = ""
                except Exception as e:
                    logger.error(
                        f"Error reading text file '{uploaded_text_file.name}': {e}",
                        exc_info=True,
                    )
                    st.error(f"Failed to read text file: {e}")
                    st.session_state[AFFIRM_TEXT_AREA_KEY] = ""

    # --- ADDED: Affirmation Volume Slider (Show only when audio is loaded) ---
    if st.session_state.get("wizard_affirmation_audio") is not None:
        st.divider()  # Add visual separation
        st.markdown("**Adjust Affirmation Volume:**")
        current_affirm_volume = st.session_state.get("wizard_affirmation_volume", 1.0)
        new_affirm_volume = st.slider(
            "Affirmation Volume",
            min_value=0.0,
            max_value=1.0,
            value=current_affirm_volume,
            step=0.05,
            key=AFFIRM_VOLUME_SLIDER_KEY,
            label_visibility="collapsed",  # Hide label as we have markdown title
            help="Adjust the volume of the affirmation track. This overrides the default volume if 'Apply Quick Settings' is checked in Step 4.",
        )
        # Update state if the slider value changes
        if new_affirm_volume != current_affirm_volume:
            st.session_state.wizard_affirmation_volume = new_affirm_volume
            logger.debug(f"Affirmation volume updated to: {new_affirm_volume}")
            # No rerun needed, just update state for next step processing
    # --- END ADDED ---

    st.divider()  # Existing divider before navigation

    # --- Navigation ---
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])
    with col_nav_1:  # Home
        if st.button(
            "🏠 Back to Home",
            key="wizard_step1_home",
            use_container_width=True,
            help="Exit Wizard and return to main menu.",
        ):
            wizard._reset_wizard_state()
    with col_nav_2:  # Back
        st.button("⬅️ Back", key="wizard_step1_back", disabled=True, use_container_width=True)
    with col_nav_3:  # Next
        # Validation logic
        next_disabled = True
        current_source = st.session_state.get("wizard_affirmation_source", "text")
        affirm_text_present = st.session_state.get(AFFIRM_TEXT_AREA_KEY, "").strip()
        affirm_audio_present = st.session_state.get("wizard_affirmation_audio") is not None
        if (
            (current_source == "text" and affirm_text_present)
            or (current_source == "upload_audio" and affirm_audio_present)
            or (current_source == "upload_text" and affirm_text_present)
        ):
            next_disabled = False
        elif current_source == "upload_text" and not affirm_text_present:
            st.warning("Upload a valid text file or select another source.")

        if st.button(
            "Next: Background Sound ➡️",
            key="wizard_step1_next",
            type="primary",
            use_container_width=True,
            disabled=next_disabled,
        ):
            # Generate TTS only if source is text and audio doesn't exist yet
            if st.session_state.wizard_affirmation_source == "text" and st.session_state.wizard_affirmation_audio is None:
                text_to_gen = st.session_state.get(AFFIRM_TEXT_AREA_KEY, "")
                if text_to_gen.strip():
                    logger.info("Wizard Step 1: Generating TTS audio using wizard's TTS generator.")
                    with st.spinner("Generating affirmation audio... Please wait."):
                        try:
                            audio_data, sr = wizard.tts_generator.generate(text_to_gen)

                            if audio_data is not None and sr is not None:
                                st.session_state.wizard_affirmation_audio = audio_data
                                st.session_state.wizard_affirmation_sr = sr
                                logger.info(f"Wizard TTS generation successful. SR: {sr}")
                                st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                                st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None
                                st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
                                # --- ADDED: Rerun after TTS to show slider before navigating ---
                                st.rerun()  # Show slider, then user clicks next again
                                # wizard._go_to_step(2) # Original: Navigate immediately
                                # --- END ADDED ---
                            else:
                                logger.error("TTS generation returned None or empty data without raising an exception.")
                                st.error("Audio generation failed. Please check logs or try again.")

                        except Exception as e:
                            logger.error(
                                f"Error generating TTS in wizard step 1: {e}",
                                exc_info=True,
                            )
                            st.error(f"Audio generation failed: {e}")
                else:
                    st.error("Cannot proceed without affirmation text.")

            elif st.session_state.wizard_affirmation_audio is not None:
                # If audio already exists (e.g., from upload or previous TTS run), proceed directly
                st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None
                st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
                wizard._go_to_step(2)  # Reruns
            else:
                st.error("Please provide affirmations (text or audio) before proceeding.")
