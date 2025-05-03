# wizard_steps/step_4_export.py
# ==========================================
# UI Rendering for Wizard Step 4: Review and Export
# ==========================================

import logging
import re

import numpy as np
import streamlit as st

# --- MODIFIED: No longer need preset volume, only speed ---
from config import QUICK_SUBLIMINAL_PRESET_SPEED

# --- END MODIFIED ---

# Optional MP3 export dependency check
try:
    # Check based on quick_wizard's PYDUB_AVAILABLE (which should be accurate)
    from quick_wizard import PYDUB_AVAILABLE
except ImportError:
    # Fallback if quick_wizard structure changes or direct run
    try:
        from pydub import AudioSegment

        PYDUB_AVAILABLE = True
    except ImportError:
        PYDUB_AVAILABLE = False


logger = logging.getLogger(__name__)


def render_step_4(wizard):
    """
    Renders the UI for Step 4: Review and Export.

    Args:
        wizard: The instance of the main QuickWizard class.
    """
    st.subheader("Step 4: Review and Export")
    st.write("Your subliminal is ready to be generated!")

    # --- Summary ---
    st.markdown("**Summary:**")
    summary_data = []

    # Affirmations Summary
    affirm_source = st.session_state.get("wizard_affirmation_source")
    affirm_text = st.session_state.get("wizard_affirmation_text", "")
    affirm_audio_exists = st.session_state.get("wizard_affirmation_audio") is not None
    # --- ADDED: Get affirmation volume from state ---
    affirm_volume = st.session_state.get("wizard_affirmation_volume", 1.0)
    # --- END ADDED ---

    affirm_summary_line = "- **Affirmations:** ⚠️ **MISSING** (Go back to Step 1)"  # Default if missing
    if affirm_audio_exists:
        # --- MODIFIED: Include volume from slider in summary ---
        affirm_vol_percent = f"{affirm_volume:.0%}"  # Format as percentage
        if affirm_source == "text":
            # Check if text exists - implies TTS was used or will be used
            if affirm_text.strip():
                affirm_summary_line = f"- **Affirmations:** Text Input ('{affirm_text[:30]}...') (Volume: {affirm_vol_percent})"
            else:
                # Audio exists but no text (shouldn't happen with current logic, but handle defensively)
                affirm_summary_line = f"- **Affirmations:** Audio Ready (Volume: {affirm_vol_percent})"
        elif affirm_source == "upload_audio":
            affirm_summary_line = f"- **Affirmations:** Uploaded Audio File (Volume: {affirm_vol_percent})"
        else:  # Fallback if source is None but audio exists
            affirm_summary_line = f"- **Affirmations:** Audio Ready (Volume: {affirm_vol_percent})"
        # --- END MODIFIED ---

    summary_data.append(affirm_summary_line)

    # Background Summary
    bg_choice = st.session_state.get("wizard_background_choice")
    bg_volume = st.session_state.get("wizard_background_volume", 0)
    bg_audio_exists = st.session_state.get("wizard_background_audio") is not None
    if bg_choice == "upload" and bg_audio_exists:
        summary_data.append(f"- **Background:** Uploaded Audio (Volume: {bg_volume:.0%})")
    elif bg_choice == "noise" and bg_audio_exists:
        noise_type = st.session_state.get("wizard_background_noise_type", "Unknown Noise")
        summary_data.append(f"- **Background:** {noise_type} (Volume: {bg_volume:.0%})")
    else:
        summary_data.append("- **Background:** None")

    # Frequency Summary
    freq_choice = st.session_state.get("wizard_frequency_choice", "None")
    freq_volume = st.session_state.get("wizard_frequency_volume", 0)
    freq_audio_exists = st.session_state.get("wizard_frequency_audio") is not None
    if freq_choice != "None" and freq_audio_exists:
        summary_data.append(f"- **Frequency:** {freq_choice} (Volume: {freq_volume:.0%})")
    else:
        summary_data.append("- **Frequency:** None")

    st.markdown("\n".join(summary_data))

    # --- Quick Settings Toggle (Now only for Speed) ---
    # --- MODIFIED: Update label and help text ---
    apply_speed_setting = st.checkbox(
        f"Apply Quick Subliminal Speed ({QUICK_SUBLIMINAL_PRESET_SPEED}x)",
        value=st.session_state.get("wizard_apply_quick_settings", True),  # Keep state key for now
        key="wizard_apply_quick_settings_checkbox",
        help=f"Check this to automatically speed up the affirmations to {QUICK_SUBLIMINAL_PRESET_SPEED}x. Volume is controlled separately in Step 1.",
    )
    # Update the state variable based on the checkbox
    st.session_state.wizard_apply_quick_settings = apply_speed_setting
    # --- END MODIFIED ---
    st.divider()

    # --- Export Options ---
    st.session_state.wizard_output_filename = st.text_input(
        "Output Filename (no extension):",
        value=st.session_state.wizard_output_filename,
        key="wizard_filename_input",
    )
    export_formats = ["WAV"]
    help_text = "Export in WAV format (lossless, larger file size)."
    if PYDUB_AVAILABLE:
        export_formats.append("MP3")
        help_text = "Choose WAV (lossless, large) or MP3 (compressed, smaller - requires ffmpeg)."
    else:
        help_text += " MP3 export disabled (requires 'pydub' library and 'ffmpeg')."
    try:
        current_format_index = export_formats.index(
            st.session_state.wizard_export_format.upper()  # Ensure comparison is case-insensitive
        )
    except ValueError:
        current_format_index = 0  # Default to WAV
        st.session_state.wizard_export_format = "WAV"  # Ensure state matches default

    selected_format = st.radio(
        "Export Format:",
        export_formats,
        key="wizard_export_format_radio",
        horizontal=True,
        help=help_text,
        index=current_format_index,
    )
    # Update state if radio button changes
    if selected_format != st.session_state.wizard_export_format:
        st.session_state.wizard_export_format = selected_format
        st.rerun()  # Rerun to update button label potentially

    # --- Generate Button ---
    is_processing = st.session_state.get("wizard_processing_active", False)
    # Check if affirmation audio exists *now* (it might have been generated in Step 1)
    affirmations_missing = st.session_state.get("wizard_affirmation_audio") is None
    mp3_unavailable = st.session_state.wizard_export_format == "MP3" and not PYDUB_AVAILABLE

    # Disable if processing, or if affirmations missing, or if MP3 chosen but unavailable
    export_disabled = is_processing or affirmations_missing or mp3_unavailable

    export_tooltip = ""
    if is_processing:
        export_tooltip = "Processing... Please wait."
    elif affirmations_missing:
        export_tooltip = "Affirmation audio is missing. Please go back to Step 1 and ensure audio is generated or uploaded."
    elif mp3_unavailable:
        export_tooltip = "MP3 export requires 'pydub' and 'ffmpeg'. Choose WAV or install dependencies."
    else:
        export_tooltip = "Click to generate the final audio mix."

    generate_button_label = "⏳ Processing..." if is_processing else f"Generate & Prepare Download (. {st.session_state.wizard_export_format.lower()})"

    if st.button(
        generate_button_label,
        key="wizard_generate_button",
        type="primary",
        disabled=export_disabled,
        help=export_tooltip,
        use_container_width=True,
    ):
        # Set the processing flag and rerun to disable button
        st.session_state.wizard_processing_active = True
        logger.info("Set wizard_processing_active flag to True.")
        st.rerun()

    # Perform processing only if flag was just set (triggered by the button click above)
    if st.session_state.get("wizard_processing_active", False):
        # Check if we need to START processing (buffer doesn't exist and no error)
        if st.session_state.get("wizard_export_buffer") is None and st.session_state.get("wizard_export_error") is None:
            logger.info("Processing flag is True, starting export process...")
            with st.spinner("Generating audio mix... This may take a moment."):
                wizard._process_and_export()  # Call the main processing function
            # The finally block in _process_and_export resets the flag.
            # Rerun AFTER processing finishes to show download/error.
            logger.info("Processing finished, triggering rerun to display results.")
            st.rerun()

    # --- Show Download Button or Error ---
    export_buffer = st.session_state.get("wizard_export_buffer")
    export_error = st.session_state.get("wizard_export_error")

    if export_buffer:
        # Sanitize filename
        raw_filename = st.session_state.wizard_output_filename
        sanitized_filename = re.sub(r'[\\/*?:"<>|]', "", raw_filename).strip()
        if not sanitized_filename:
            sanitized_filename = "mindmorph_quick_mix"  # Default filename if empty

        file_ext = st.session_state.wizard_export_format.lower()
        download_filename = f"{sanitized_filename}.{file_ext}"
        mime_type = f"audio/{file_ext}" if file_ext == "wav" else "audio/mpeg"

        st.download_button(
            label=f"⬇️ Download: {download_filename}",
            data=export_buffer,
            file_name=download_filename,
            mime=mime_type,
            key="wizard_download_button",
            use_container_width=True,
            help="Click to download the generated subliminal audio file.",
            on_click=wizard._reset_wizard_state,  # Reset wizard after download starts
        )
        # Clear buffer after showing button to prevent re-showing on unrelated reruns
        # st.session_state.wizard_export_buffer = None # Keep buffer until reset by download click

    elif export_error:
        st.error(f"Export Failed: {export_error}")
        # Clear error after display so it doesn't persist
        st.session_state.wizard_export_error = None
        # Ensure processing flag is false if error occurred and wasn't reset
        if st.session_state.get("wizard_processing_active"):
            st.session_state.wizard_processing_active = False

    # --- Navigation Buttons ---
    st.divider()
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])
    with col_nav_1:  # Home
        if st.button(
            "🏠 Back to Home",
            key="wizard_step4_home",
            use_container_width=True,
            help="Exit Wizard and return to main menu.",
            disabled=is_processing,  # Disable if processing
        ):
            if not is_processing:
                wizard._reset_wizard_state()
    with col_nav_2:  # Back
        if st.button(
            "⬅️ Back: Frequency",
            key="wizard_step4_back",
            use_container_width=True,
            disabled=is_processing,  # Disable if processing
        ):
            if not is_processing:
                # Clear any potential export results before going back
                st.session_state.wizard_export_buffer = None
                st.session_state.wizard_export_error = None
                wizard._go_to_step(3)
    with col_nav_3:  # Finish Placeholder (remains disabled)
        st.button(
            "Finish",
            key="wizard_step4_finish_placeholder",
            disabled=True,
            use_container_width=True,
        )
