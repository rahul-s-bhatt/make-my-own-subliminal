# wizard_steps/step_4_export.py
# ==========================================
# UI Rendering for Wizard Step 4: Review and Export
# ==========================================

import logging
import re

import streamlit as st

# Import necessary components/constants
# Optional MP3 export dependency check (can be done in main wizard class too)
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Constants from the original wizard file
WIZARD_AFFIRMATION_SPEED = 10.0  # Or import from config/wizard_config
WIZARD_AFFIRMATION_VOLUME = 0.10  # Or import from config/wizard_config

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
    # Affirmations
    affirm_source = st.session_state.get("wizard_affirmation_source")
    affirm_text = st.session_state.get("wizard_affirmation_text", "")
    if affirm_source == "text":
        summary_data.append(f"- **Affirmations:** Text Input ('{affirm_text[:30]}...')")
    elif affirm_source == "upload":
        # Clean up the "Uploaded: " part for display
        display_name = affirm_text.replace("Uploaded: ", "")
        summary_data.append(f"- **Affirmations:** Uploaded File ('{display_name}')")
    elif st.session_state.get("wizard_affirmation_audio") is not None:
        summary_data.append(f"- **Affirmations:** Audio Loaded (Source unclear - check state)")  # Fallback
    else:
        summary_data.append("- **Affirmations:** ⚠️ **MISSING** (Go back to Step 1)")  # Should not happen if logic is correct

    # Background
    bg_choice = st.session_state.get("wizard_background_choice")
    bg_volume = st.session_state.get("wizard_background_volume", 0)
    if bg_choice == "upload":
        summary_data.append(f"- **Background:** Uploaded Audio (Volume: {bg_volume:.0%})")
    elif bg_choice == "noise":
        noise_type = st.session_state.get("wizard_background_noise_type", "Unknown Noise")
        summary_data.append(f"- **Background:** {noise_type} (Volume: {bg_volume:.0%})")
    else:  # 'none'
        summary_data.append("- **Background:** None")

    # Frequency
    freq_choice = st.session_state.get("wizard_frequency_choice", "None")
    freq_volume = st.session_state.get("wizard_frequency_volume", 0)
    if freq_choice != "None":
        summary_data.append(f"- **Frequency:** {freq_choice} (Volume: {freq_volume:.0%})")
    else:
        summary_data.append("- **Frequency:** None")

    st.markdown("\n".join(summary_data))
    st.caption(f"Affirmations will be automatically sped up ({WIZARD_AFFIRMATION_SPEED}x) and volume lowered ({WIZARD_AFFIRMATION_VOLUME:.0%}).")
    st.divider()

    # --- Export Options ---
    st.session_state.wizard_output_filename = st.text_input("Output Filename (no extension):", value=st.session_state.wizard_output_filename, key="wizard_filename_input")

    export_formats = ["WAV"]
    help_text = "Export in WAV format (lossless, larger file size)."
    if PYDUB_AVAILABLE:
        export_formats.append("MP3")
        help_text = "Choose WAV (lossless, large) or MP3 (compressed, smaller - requires ffmpeg)."
    else:
        help_text += " MP3 export disabled (requires 'pydub' library and 'ffmpeg')."

    # Get current format index
    try:
        current_format_index = export_formats.index(st.session_state.wizard_export_format)
    except ValueError:
        current_format_index = 0  # Default to WAV

    st.session_state.wizard_export_format = st.radio(
        "Export Format:", export_formats, key="wizard_export_format_radio", horizontal=True, help=help_text, index=current_format_index
    )

    # --- Generate Button ---
    # Disable if affirmations are missing or if MP3 is chosen but unavailable
    affirmations_missing = st.session_state.get("wizard_affirmation_audio") is None
    mp3_unavailable = st.session_state.wizard_export_format == "MP3" and not PYDUB_AVAILABLE
    export_disabled = affirmations_missing or mp3_unavailable
    export_tooltip = ""
    if affirmations_missing:
        export_tooltip = "Affirmation audio is missing. Please go back to Step 1."
    elif mp3_unavailable:
        export_tooltip = "MP3 export requires 'pydub' and 'ffmpeg'. Please choose WAV or install dependencies."

    if st.button(
        f"Generate & Prepare Download (. {st.session_state.wizard_export_format.lower()})",
        key="wizard_generate_button",
        type="primary",
        disabled=export_disabled,
        help=export_tooltip or "Click to generate the final audio mix.",  # Show tooltip if disabled
    ):
        # Call the processing method on the main wizard instance
        wizard._process_and_export()
        st.rerun()  # Rerun to show download button or error

    # --- Show Download Button or Error ---
    if st.session_state.get("wizard_export_buffer"):
        # Sanitize filename (ensure wizard instance or this module has access to re)
        sanitized_filename = re.sub(r'[\\/*?:"<>|]', "", st.session_state.wizard_output_filename).strip()
        if not sanitized_filename:
            sanitized_filename = "mindmorph_quick_mix"  # Default filename
        file_ext = st.session_state.wizard_export_format.lower()
        download_filename = f"{sanitized_filename}.{file_ext}"
        mime_type = f"audio/{file_ext}" if file_ext == "wav" else "audio/mpeg"

        st.download_button(
            label=f"⬇️ Download: {download_filename}",
            data=st.session_state.wizard_export_buffer,
            file_name=download_filename,
            mime=mime_type,
            key="wizard_download_button",
            use_container_width=False,  # Don't make it full width
            help="Click to download the generated subliminal audio file.",
            on_click=wizard._reset_wizard_state,  # Reset after download click
        )
        # Optionally clear buffer after rendering download button? No, reset_wizard_state handles it.
    elif st.session_state.get("wizard_export_error"):
        st.error(f"Export Failed: {st.session_state.wizard_export_error}")
        # Clear the error after displaying it so it doesn't persist on rerun without new attempt
        st.session_state.wizard_export_error = None

    # --- Navigation Buttons ---
    st.divider()
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        st.button("Back: Frequency", on_click=wizard._go_to_step, args=(3,), key="wizard_step4_back")
    with col_nav2:
        # Use the main wizard's reset method
        st.button("✨ Start Over", on_click=wizard._reset_wizard_state, key="wizard_step4_reset")
