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
from audio_io import load_audio
from config import GLOBAL_SR, MAX_AFFIRMATION_CHARS, MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB  # Import limits and SR
from utils import read_text_file

logger = logging.getLogger(__name__)

# Define keys for widgets in this step to ensure consistency
AFFIRM_SOURCE_RADIO_KEY = "wizard_affirm_source_radio"
AFFIRM_TEXT_AREA_KEY = "wizard_affirm_text_area"
AFFIRM_FILE_UPLOADER_KEY = "wizard_affirm_file_uploader"
AFFIRM_ORIGINAL_TEXT_KEY = "wizard_original_affirmation_text"
AFFIRM_PENDING_UPDATE_KEY = "wizard_affirm_text_pending_update"
AFFIRM_PENDING_TRUNCATED_KEY = "wizard_affirm_truncated_pending"


def render_step_1(wizard):  # Pass the wizard instance
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

    # --- Apply pending update at the start of the run ---
    if st.session_state.get(AFFIRM_PENDING_UPDATE_KEY) is not None:
        logger.debug("Applying pending affirmation text update.")
        st.session_state[AFFIRM_TEXT_AREA_KEY] = st.session_state[AFFIRM_PENDING_UPDATE_KEY]
        # Display warning if needed from the pending state
        if st.session_state.get(AFFIRM_PENDING_TRUNCATED_KEY):
            st.warning(f"⚠️ Text was automatically shortened to fit the {MAX_AFFIRMATION_CHARS} character limit.", icon="✂️")
        # Clear pending state
        st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None
        st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False

    # Use columns for layout
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

        selected_source_label = st.radio("Affirmation Source:", options=source_options, index=default_index, key=AFFIRM_SOURCE_RADIO_KEY, label_visibility="collapsed")

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
                st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None  # Clear backup if switching away from text
                st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None  # Clear pending update
                st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
            if new_source != "upload_audio":
                st.session_state.wizard_affirmation_audio = None
                st.session_state.wizard_affirmation_sr = None
            st.rerun()

    with col_main_input:
        source = st.session_state.get("wizard_affirmation_source", "text")

        if source == "text":
            st.markdown("**Type or Paste Affirmations:** (one per line recommended)")

            affirmation_text_value = st.session_state.get(AFFIRM_TEXT_AREA_KEY, "")

            # Callback to clear original text if user manually edits after expansion
            def clear_original_on_edit():
                current_val = st.session_state.get(AFFIRM_TEXT_AREA_KEY)
                original_val = st.session_state.get(AFFIRM_ORIGINAL_TEXT_KEY)
                # Clear backup only if it exists and the text area value is different from the backup
                # (prevents clearing if rerun happens without actual edit)
                if original_val is not None and current_val != original_val:
                    st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                    logger.debug("Cleared original affirmation backup due to manual edit.")

            affirmation_text = st.text_area(
                "Affirmations Text Area",
                value=affirmation_text_value,  # Read from state
                key=AFFIRM_TEXT_AREA_KEY,  # Use key to allow Streamlit to manage widget value
                height=200,
                max_chars=MAX_AFFIRMATION_CHARS,
                label_visibility="collapsed",
                help="Enter your affirmations here.",
                on_change=clear_original_on_edit,  # Clear backup if user types
            )
            st.caption(f"{len(affirmation_text_value)} / {MAX_AFFIRMATION_CHARS} characters")  # Show count based on state value

            # --- Expansion and Undo Buttons ---
            button_col_1, button_col_2 = st.columns(2)
            with button_col_1:
                # Disable expand if text is empty
                expand_disabled = not affirmation_text_value.strip()  # Check state value
                if st.button(
                    "✨ Expand Affirmations", key="wizard_expand_affirmations", disabled=expand_disabled, use_container_width=True, help="Generate variations of your affirmations."
                ):
                    with st.spinner("Expanding affirmations..."):
                        try:
                            # Store original text BEFORE expanding
                            st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = affirmation_text_value

                            expanded_text, truncated = expand_affirmations(
                                base_text=affirmation_text_value,  # Use current value
                                max_chars=MAX_AFFIRMATION_CHARS,
                                multiplier=3,
                            )
                            # Store result in PENDING state
                            st.session_state[AFFIRM_PENDING_UPDATE_KEY] = expanded_text
                            st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = truncated
                            logger.info(f"Wizard affirmation expansion complete. Staged for next run. Truncated: {truncated}")
                            # Rerun to apply the pending update
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Error during wizard affirmation expansion: {e}", exc_info=True)
                            st.error(f"Failed to expand affirmations: {e}")
                            # Clear original text if expansion failed
                            st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                            st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None  # Clear pending on error
                            st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False

            with button_col_2:
                # Show Undo button only if original text is stored
                undo_disabled = st.session_state.get(AFFIRM_ORIGINAL_TEXT_KEY) is None
                if st.button("↩️ Undo Expansion", key="wizard_undo_expansion", disabled=undo_disabled, use_container_width=True, help="Revert to the text before expansion."):
                    original_text = st.session_state.get(AFFIRM_ORIGINAL_TEXT_KEY)
                    if original_text is not None:
                        # Stage the original text as a pending update
                        st.session_state[AFFIRM_PENDING_UPDATE_KEY] = original_text
                        st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False  # Reset truncated status
                        # Clear the stored original text backup immediately
                        st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                        logger.info("User staged affirmation undo for next run.")
                        st.rerun()  # Rerun to apply the undo
                    else:
                        logger.warning("Undo clicked but no original text found in state.")
            # --- End Expansion/Undo Buttons ---

        elif source == "upload_audio":
            st.markdown("**Upload Affirmation Audio:**")
            uploaded_audio_file = st.file_uploader(
                "Select an audio file (.wav, .mp3)",
                type=["wav", "mp3"],
                key="wizard_affirm_audio_uploader",  # Unique key for this uploader
                label_visibility="collapsed",
                help=f"Upload your pre-recorded affirmations (Max {MAX_UPLOAD_SIZE_MB}MB).",
            )
            if uploaded_audio_file:
                if uploaded_audio_file.size > MAX_UPLOAD_SIZE_BYTES:
                    st.error(f"❌ File '{uploaded_audio_file.name}' ({uploaded_audio_file.size / (1024 * 1024):.1f} MB) exceeds the {MAX_UPLOAD_SIZE_MB} MB limit.")
                    st.session_state.wizard_affirmation_audio = None  # Clear state if invalid
                    st.session_state.wizard_affirmation_sr = None
                else:
                    # Process the uploaded audio file immediately for the wizard
                    with st.spinner(f"Processing '{uploaded_audio_file.name}'..."):
                        try:
                            # Save temporarily to load with librosa
                            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio_file.name)[1]) as tmp:
                                tmp.write(uploaded_audio_file.getvalue())
                                temp_file_path = tmp.name

                            audio_data, sr = load_audio(temp_file_path, target_sr=GLOBAL_SR)  # Resample to global SR

                            # Clean up temp file
                            try:
                                os.remove(temp_file_path)
                            except OSError:
                                logger.warning(f"Could not remove temp audio file: {temp_file_path}")

                            if audio_data is not None and sr is not None and audio_data.size > 0:
                                st.session_state.wizard_affirmation_audio = audio_data
                                st.session_state.wizard_affirmation_sr = sr
                                st.success(f"✅ Loaded audio: '{uploaded_audio_file.name}' ({len(audio_data) / sr:.1f}s)")
                                logger.info(f"Wizard Step 1: Loaded affirmation audio '{uploaded_audio_file.name}'")
                                # Clear text state if audio is uploaded
                                st.session_state[AFFIRM_TEXT_AREA_KEY] = ""
                                st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                                st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None  # Clear pending
                                st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
                            else:
                                st.error(f"❌ Failed to load audio from '{uploaded_audio_file.name}'.")
                                st.session_state.wizard_affirmation_audio = None
                                st.session_state.wizard_affirmation_sr = None

                        except Exception as e:
                            logger.error(f"Error processing uploaded audio file '{uploaded_audio_file.name}': {e}", exc_info=True)
                            st.error(f"Error processing audio file: {e}")
                            st.session_state.wizard_affirmation_audio = None
                            st.session_state.wizard_affirmation_sr = None
                            # Clean up temp file in case of error during processing
                            if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                                try:
                                    os.remove(temp_file_path)
                                except OSError:
                                    pass

        elif source == "upload_text":
            st.markdown("**Upload Affirmation Text File:**")
            uploaded_text_file = st.file_uploader(
                "Select a text file (.txt, .docx)",
                type=["txt", "docx"],
                key=AFFIRM_FILE_UPLOADER_KEY,
                label_visibility="collapsed",
                help="Upload affirmations from a text or Word document.",
            )
            if uploaded_text_file:
                try:
                    text_content = read_text_file(uploaded_text_file)
                    if text_content is not None:
                        if len(text_content) > MAX_AFFIRMATION_CHARS:
                            st.error(f"❌ Text in file '{uploaded_text_file.name}' is too long ({len(text_content)} chars). Maximum is {MAX_AFFIRMATION_CHARS}.")
                            st.session_state[AFFIRM_TEXT_AREA_KEY] = ""
                        else:
                            # Stage the loaded text as a pending update
                            st.session_state[AFFIRM_PENDING_UPDATE_KEY] = text_content
                            st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False  # Reset truncated status
                            logger.info(f"Wizard Step 1: Staged text from file '{uploaded_text_file.name}' for update.")
                            st.success(f"✅ Loaded text from: '{uploaded_text_file.name}'")
                            # Clear other states
                            st.session_state.wizard_affirmation_audio = None
                            st.session_state.wizard_affirmation_sr = None
                            st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                            # Switch back to text view and rerun to apply pending update
                            st.session_state.wizard_affirmation_source = "text"
                            st.rerun()
                    else:
                        st.session_state[AFFIRM_TEXT_AREA_KEY] = ""
                except Exception as e:
                    logger.error(f"Error reading text file '{uploaded_text_file.name}': {e}", exc_info=True)
                    st.error(f"Failed to read text file: {e}")
                    st.session_state[AFFIRM_TEXT_AREA_KEY] = ""

    st.divider()

    # --- Navigation ---
    # Use 3 columns for Home, Back, Next
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])  # Adjust ratios as needed

    with col_nav_1:
        # Add Back to Home button
        if st.button("🏠 Back to Home", key="wizard_step1_home", use_container_width=True, help="Exit Wizard and return to main menu."):
            wizard._reset_wizard_state()  # Call the reset method from the wizard instance
            # st.rerun() is handled by reset_wizard_state indirectly

    with col_nav_2:
        # Back button is not needed on Step 1
        st.button("⬅️ Back", key="wizard_step1_back", disabled=True, use_container_width=True)

    with col_nav_3:
        # Validate input before allowing 'Next'
        next_disabled = True
        current_source = st.session_state.get("wizard_affirmation_source", "text")
        # Use the state key for text validation
        if current_source == "text" and st.session_state.get(AFFIRM_TEXT_AREA_KEY, "").strip():
            next_disabled = False
        elif current_source == "upload_audio" and st.session_state.get("wizard_affirmation_audio") is not None:
            next_disabled = False
        elif current_source == "upload_text":  # This state shouldn't persist due to auto-switch
            if st.session_state.get(AFFIRM_TEXT_AREA_KEY, "").strip():
                next_disabled = False
            else:
                st.warning("Upload a text file or select another source.")

        if st.button("Next: Background Sound ➡️", key="wizard_step1_next", type="primary", use_container_width=True, disabled=next_disabled):
            # Generate TTS only if source is text and audio doesn't exist
            if st.session_state.wizard_affirmation_source == "text" and st.session_state.wizard_affirmation_audio is None:
                text_to_gen = st.session_state.get(AFFIRM_TEXT_AREA_KEY, "")  # Use state key
                if text_to_gen.strip():
                    logger.info("Wizard Step 1: Generating TTS audio before proceeding to Step 2.")
                    with st.spinner("Generating affirmation audio..."):
                        try:
                            audio_data, sr = wizard.tts_generator.generate(text_to_gen)
                            if audio_data is not None and sr is not None:
                                st.session_state.wizard_affirmation_audio = audio_data
                                st.session_state.wizard_affirmation_sr = sr
                                logger.info("Wizard TTS generation successful.")
                                # Clear original text backup after successful generation/navigation
                                st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                                st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None  # Clear pending
                                st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
                                wizard._go_to_step(2)
                            else:
                                st.error("Failed to generate audio from text. Please try again or check logs.")
                        except Exception as e:
                            logger.error(f"Error generating TTS in wizard step 1: {e}", exc_info=True)
                            st.error(f"Audio generation failed: {e}")
                else:
                    st.error("Cannot proceed without affirmation text.")
            elif st.session_state.wizard_affirmation_audio is not None:
                # Clear original text backup when navigating with existing audio
                st.session_state[AFFIRM_ORIGINAL_TEXT_KEY] = None
                st.session_state[AFFIRM_PENDING_UPDATE_KEY] = None  # Clear pending
                st.session_state[AFFIRM_PENDING_TRUNCATED_KEY] = False
                wizard._go_to_step(2)
            else:
                st.error("Please provide affirmations (text or audio) before proceeding.")
