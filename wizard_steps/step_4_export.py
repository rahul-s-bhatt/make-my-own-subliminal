# wizard_steps/step_4_export.py
# ==========================================
# UI Rendering for Wizard Step 4: Review and Export
# ==========================================

import logging
import re

import numpy as np
import streamlit as st

from config import QUICK_SUBLIMINAL_PRESET_SPEED, QUICK_SUBLIMINAL_PRESET_VOLUME

# Optional MP3 export dependency check
try:
    # from pydub import AudioSegment # No longer needed here, check happens in quick_wizard
    PYDUB_AVAILABLE = True  # Assume available if library installed
except ImportError:
    PYDUB_AVAILABLE = (
        False  # Should ideally check based on quick_wizard's PYDUB_AVAILABLE
    )

logger = logging.getLogger(__name__)


def render_step_4(wizard):
    """
    Renders the UI for Step 4: Review and Export.

    Args:
        wizard: The instance of the main QuickWizard class.
    """
    st.subheader("Step 4: Review and Export")
    st.write("Your subliminal is ready to be generated!")

    # --- Summary --- (Keep as is)
    st.markdown("**Summary:**")
    summary_data = []
    affirm_source = st.session_state.get("wizard_affirmation_source")
    affirm_text = st.session_state.get("wizard_affirmation_text", "")
    affirm_audio_exists = st.session_state.get("wizard_affirmation_audio") is not None
    if affirm_source == "text":
        summary_data.append(f"- **Affirmations:** Text Input ('{affirm_text[:30]}...')")
    elif affirm_source == "upload_audio" and affirm_audio_exists:
        summary_data.append(f"- **Affirmations:** Uploaded Audio File")
    elif affirm_audio_exists:
        summary_data.append(f"- **Affirmations:** Audio Ready")
    else:
        summary_data.append("- **Affirmations:** ⚠️ **MISSING** (Go back to Step 1)")
    bg_choice = st.session_state.get("wizard_background_choice")
    bg_volume = st.session_state.get("wizard_background_volume", 0)
    if (
        bg_choice == "upload"
        and st.session_state.get("wizard_background_audio") is not None
    ):
        summary_data.append(
            f"- **Background:** Uploaded Audio (Volume: {bg_volume:.0%})"
        )
    elif (
        bg_choice == "noise"
        and st.session_state.get("wizard_background_audio") is not None
    ):
        noise_type = st.session_state.get(
            "wizard_background_noise_type", "Unknown Noise"
        )
        summary_data.append(f"- **Background:** {noise_type} (Volume: {bg_volume:.0%})")
    else:
        summary_data.append("- **Background:** None")
    freq_choice = st.session_state.get("wizard_frequency_choice", "None")
    freq_volume = st.session_state.get("wizard_frequency_volume", 0)
    if (
        freq_choice != "None"
        and st.session_state.get("wizard_frequency_audio") is not None
    ):
        summary_data.append(
            f"- **Frequency:** {freq_choice} (Volume: {freq_volume:.0%})"
        )
    else:
        summary_data.append("- **Frequency:** None")
    st.markdown("\n".join(summary_data))

    # Quick Settings Toggle (Keep as is)
    apply_settings = st.checkbox(
        f"Apply Quick Subliminal Settings (Speed={QUICK_SUBLIMINAL_PRESET_SPEED}x, Volume={QUICK_SUBLIMINAL_PRESET_VOLUME:.0%})",
        value=st.session_state.get("wizard_apply_quick_settings", True),
        key="wizard_apply_quick_settings_checkbox",
        help="Check this to automatically speed up and lower the volume of the affirmations for a typical subliminal effect.",
    )
    st.session_state.wizard_apply_quick_settings = apply_settings
    st.divider()

    # --- Export Options --- (Keep as is)
    st.session_state.wizard_output_filename = st.text_input(
        "Output Filename (no extension):",
        value=st.session_state.wizard_output_filename,
        key="wizard_filename_input",
    )
    export_formats = ["WAV"]
    help_text = "Export in WAV format (lossless, larger file size)."
    # Check PYDUB_AVAILABLE status (ideally passed or checked from quick_wizard instance)
    # For now, assume PYDUB_AVAILABLE reflects the environment status
    if PYDUB_AVAILABLE:
        export_formats.append("MP3")
        help_text = "Choose WAV (lossless, large) or MP3 (compressed, smaller - requires ffmpeg)."
    else:
        help_text += " MP3 export disabled (requires 'pydub' library and 'ffmpeg')."
    try:
        current_format_index = export_formats.index(
            st.session_state.wizard_export_format
        )
    except ValueError:
        current_format_index = 0
    st.session_state.wizard_export_format = st.radio(
        "Export Format:",
        export_formats,
        key="wizard_export_format_radio",
        horizontal=True,
        help=help_text,
        index=current_format_index,
    )

    # --- Generate Button ---
    # --- MODIFIED: Disable button based on processing flag ---
    is_processing = st.session_state.get("wizard_processing_active", False)
    affirmations_missing = st.session_state.get("wizard_affirmation_audio") is None
    mp3_unavailable = (
        st.session_state.wizard_export_format == "MP3" and not PYDUB_AVAILABLE
    )
    # Disable if processing, or if affirmations missing, or if MP3 chosen but unavailable
    export_disabled = is_processing or affirmations_missing or mp3_unavailable

    export_tooltip = ""
    if is_processing:
        export_tooltip = "Processing... Please wait."
    elif affirmations_missing:
        export_tooltip = "Affirmation audio is missing. Please go back to Step 1."
    elif mp3_unavailable:
        export_tooltip = "MP3 export requires 'pydub' and 'ffmpeg'. Choose WAV or install dependencies."
    else:
        export_tooltip = "Click to generate the final audio mix."

    generate_button_label = (
        "⏳ Processing..."
        if is_processing
        else f"Generate & Prepare Download (. {st.session_state.wizard_export_format.lower()})"
    )

    if st.button(
        generate_button_label,
        key="wizard_generate_button",
        type="primary",
        disabled=export_disabled,
        help=export_tooltip,
        use_container_width=True,  # Make button full width
    ):
        # --- Set the processing flag and rerun ---
        st.session_state.wizard_processing_active = True
        logger.info("Set wizard_processing_active flag to True.")
        st.rerun()  # Rerun immediately to disable the button visually

    # --- Perform processing only if flag was just set (avoid re-running on subsequent reruns) ---
    # This logic runs *after* the rerun caused by setting the flag above.
    # The button is now disabled, and we can start the actual processing.
    if st.session_state.get("wizard_processing_active", False):
        # Check if we need to START processing (i.e., buffer doesn't exist yet from this run)
        # This prevents re-processing if the page reruns for other reasons while processing is True
        if (
            st.session_state.get("wizard_export_buffer") is None
            and st.session_state.get("wizard_export_error") is None
        ):
            logger.info("Processing flag is True, starting export process...")
            # Call the processing method on the main wizard instance
            wizard._process_and_export()
            # The finally block in _process_and_export will reset the flag.
            # We need another rerun AFTER processing finishes to show download/error.
            logger.info("Processing finished, triggering rerun to display results.")
            st.rerun()  # Rerun AFTER processing to show download button or error

    # --- Show Download Button or Error --- (Keep as is)
    if st.session_state.get("wizard_export_buffer"):
        sanitized_filename = re.sub(
            r'[\\/*?:"<>|]', "", st.session_state.wizard_output_filename
        ).strip()
        if not sanitized_filename:
            sanitized_filename = "mindmorph_quick_mix"
        file_ext = st.session_state.wizard_export_format.lower()
        download_filename = f"{sanitized_filename}.{file_ext}"
        mime_type = f"audio/{file_ext}" if file_ext == "wav" else "audio/mpeg"
        st.download_button(
            label=f"⬇️ Download: {download_filename}",
            data=st.session_state.wizard_export_buffer,
            file_name=download_filename,
            mime=mime_type,
            key="wizard_download_button",
            use_container_width=True,  # Make button full width
            help="Click to download the generated subliminal audio file.",
            on_click=wizard._reset_wizard_state,  # Reset after download click
        )
    elif st.session_state.get("wizard_export_error"):
        st.error(f"Export Failed: {st.session_state.wizard_export_error}")
        st.session_state.wizard_export_error = None  # Clear error after display

    # --- Navigation Buttons --- (Keep as is)
    st.divider()
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])
    with col_nav_1:  # Home
        if st.button(
            "🏠 Back to Home",
            key="wizard_step4_home",
            use_container_width=True,
            help="Exit Wizard and return to main menu.",
        ):
            wizard._reset_wizard_state()
    with col_nav_2:  # Back
        # Disable back button if processing
        if st.button(
            "⬅️ Back: Frequency",
            key="wizard_step4_back",
            use_container_width=True,
            disabled=is_processing,
        ):
            if not is_processing:  # Double check before navigating
                wizard._go_to_step(3)
    with col_nav_3:  # Finish Placeholder
        st.button(
            "Finish",
            key="wizard_step4_finish_placeholder",
            disabled=True,
            use_container_width=True,
        )
