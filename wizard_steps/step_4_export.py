# wizard_steps/step_4_export.py
# ==========================================
# UI Rendering for Wizard Step 4: Review, Mix, Preview, and Export
# ==========================================

import logging
import re
from io import BytesIO

import numpy as np
import streamlit as st

from audio_utils.audio_io import save_audio_to_bytesio  # Needed for preview
from config import GLOBAL_SR, QUICK_SUBLIMINAL_PRESET_SPEED  # Added GLOBAL_SR

# Optional MP3 export dependency check
try:
    from quick_wizard import PYDUB_AVAILABLE
except ImportError:
    try:
        from pydub import AudioSegment

        PYDUB_AVAILABLE = True
    except ImportError:
        PYDUB_AVAILABLE = False


logger = logging.getLogger(__name__)

# Define preview duration
PREVIEW_DURATION_SECONDS = 30


def render_step_4(wizard):
    """
    Renders the UI for Step 4: Review, Mix, Preview and Export.

    Args:
        wizard: The instance of the main QuickWizard class.
    """
    st.subheader("Step 4: Mix, Preview & Export")
    st.write("Adjust final levels, preview the mix, and generate your subliminal!")

    # --- Review Selections ---
    st.markdown("**Review Selections:**")
    summary_data = []
    affirm_audio_exists = st.session_state.get("wizard_affirmation_audio") is not None
    bg_audio_exists = st.session_state.get("wizard_background_audio") is not None
    freq_audio_exists = st.session_state.get("wizard_frequency_audio") is not None

    # Affirmations Summary
    affirm_source = st.session_state.get("wizard_affirmation_source")
    affirm_text = st.session_state.get("wizard_affirmation_text", "")
    affirm_summary_line = "- **Affirmations:** ⚠️ **MISSING** (Go back to Step 1)"
    if affirm_audio_exists:
        if affirm_source == "text":
            if affirm_text.strip():
                affirm_summary_line = f"- **Affirmations:** Text Input ('{affirm_text[:30]}...')"
            else:
                affirm_summary_line = "- **Affirmations:** Audio Ready"
        elif affirm_source == "upload_audio":
            affirm_summary_line = "- **Affirmations:** Uploaded Audio File"
        else:
            affirm_summary_line = "- **Affirmations:** Audio Ready"
    summary_data.append(affirm_summary_line)

    # Background Summary
    bg_choice = st.session_state.get("wizard_background_choice")
    if bg_choice == "upload" and bg_audio_exists:
        summary_data.append("- **Background:** Uploaded Audio")
    elif bg_choice == "noise" and bg_audio_exists:
        noise_type = st.session_state.get("wizard_background_noise_type", "Unknown Noise")
        summary_data.append(f"- **Background:** {noise_type}")
    else:
        summary_data.append("- **Background:** None")

    # Frequency Summary
    freq_choice = st.session_state.get("wizard_frequency_choice", "None")
    if freq_choice != "None" and freq_audio_exists:
        summary_data.append(f"- **Frequency:** {freq_choice}")
    else:
        summary_data.append("- **Frequency:** None")

    st.markdown("\n".join(summary_data))
    st.divider()

    # --- Mixing Controls ---
    st.markdown("**Mixing Controls:**")
    col_vol1, col_vol2, col_vol3 = st.columns(3)
    with col_vol1:
        st.session_state.wizard_affirmation_volume = st.slider(
            "🗣️ Affirmation Vol.",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.wizard_affirmation_volume,
            step=0.05,
            key="wizard_affirm_vol_slider_step4",
            help="Adjust the volume of the affirmation track.",
        )
    with col_vol2:
        disable_bg_vol = not bg_audio_exists
        st.session_state.wizard_background_volume = st.slider(
            "🎵 Background Vol.",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.wizard_background_volume,
            step=0.05,
            key="wizard_bg_vol_slider_step4",
            help="Adjust the volume of the background track (if added).",
            disabled=disable_bg_vol,
        )
    with col_vol3:
        disable_freq_vol = not freq_audio_exists
        st.session_state.wizard_frequency_volume = st.slider(
            "〰️ Frequency Vol.",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.wizard_frequency_volume,
            step=0.05,
            key="wizard_freq_vol_slider_step4",
            help="Adjust the volume of the frequency track (if added).",
            disabled=disable_freq_vol,
        )

    # --- Speed Setting Toggle ---
    apply_speed_setting = st.checkbox(
        f"Apply Quick Subliminal Speed ({QUICK_SUBLIMINAL_PRESET_SPEED}x)",
        value=st.session_state.get("wizard_apply_quick_settings", True),
        key="wizard_apply_quick_settings_checkbox",
        help=f"Check this to automatically speed up the affirmations to {QUICK_SUBLIMINAL_PRESET_SPEED}x.",
    )
    st.session_state.wizard_apply_quick_settings = apply_speed_setting
    st.divider()

    # --- Preview Section ---
    st.markdown("**Preview Mix:**")
    # --- ADDED: Instruction for preview ---
    st.caption("Adjust controls above, then click 'Generate Preview' to hear the changes.")
    # --- END ADDED ---
    preview_disabled = not affirm_audio_exists  # Can't preview without affirmations
    preview_tooltip = "Generate a short preview of the mix with current settings." if not preview_disabled else "Add affirmations in Step 1 to enable preview."

    if st.button(f"🎧 Generate Preview ({PREVIEW_DURATION_SECONDS}s)", key="wizard_preview_button", disabled=preview_disabled, help=preview_tooltip, use_container_width=True):
        with st.spinner(f"Generating {PREVIEW_DURATION_SECONDS}s preview..."):
            try:
                # Call the preview generation function from the wizard instance
                preview_audio, preview_sr = wizard._generate_preview_mix(PREVIEW_DURATION_SECONDS)
                if preview_audio is not None and preview_sr is not None:
                    # Save preview to buffer for st.audio
                    preview_buffer = save_audio_to_bytesio(preview_audio, preview_sr)
                    st.session_state.wizard_preview_buffer = preview_buffer  # Store in state
                    st.session_state.wizard_preview_error = None
                    logger.info("Preview generated successfully.")
                else:
                    st.session_state.wizard_preview_buffer = None
                    st.session_state.wizard_preview_error = "Preview generation failed (empty audio)."
                    logger.error("Preview generation returned None or empty audio.")
            except Exception as e:
                logger.exception("Error generating preview mix.")
                st.session_state.wizard_preview_buffer = None
                st.session_state.wizard_preview_error = f"Preview Error: {e}"

    # Display the preview audio player or error message
    preview_buffer = st.session_state.get("wizard_preview_buffer")
    preview_error = st.session_state.get("wizard_preview_error")

    if preview_buffer:
        st.audio(preview_buffer, format="audio/wav", start_time=0)
    elif preview_error:
        st.error(preview_error)
        # Clear error after displaying
        st.session_state.wizard_preview_error = None
    st.divider()

    # --- Export Options ---
    st.markdown("**Export Settings:**")
    col_export1, col_export2 = st.columns([2, 1])
    with col_export1:
        st.session_state.wizard_output_filename = st.text_input(
            "Output Filename (no extension):", value=st.session_state.wizard_output_filename, key="wizard_filename_input", label_visibility="collapsed"
        )
    with col_export2:
        export_formats = ["WAV"]
        help_text = "Export in WAV format (lossless, larger file size)."
        if PYDUB_AVAILABLE:
            export_formats.append("MP3")
            help_text = "Choose WAV (lossless, large) or MP3 (compressed, smaller - requires ffmpeg)."
        else:
            help_text += " MP3 export disabled (requires 'pydub' library and 'ffmpeg')."
        try:
            current_format_index = export_formats.index(st.session_state.wizard_export_format.upper())
        except ValueError:
            current_format_index = 0
            st.session_state.wizard_export_format = "WAV"

        selected_format = st.radio(
            "Format:", export_formats, key="wizard_export_format_radio", horizontal=True, help=help_text, index=current_format_index, label_visibility="collapsed"
        )
        if selected_format != st.session_state.wizard_export_format:
            st.session_state.wizard_export_format = selected_format
            st.rerun()

    # --- Generate Button ---
    is_processing = st.session_state.get("wizard_processing_active", False)
    affirmations_missing = not affirm_audio_exists  # Re-check
    mp3_unavailable = st.session_state.wizard_export_format == "MP3" and not PYDUB_AVAILABLE

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

    generate_button_label = "⏳ Processing..." if is_processing else f"Generate & Prepare Download (. {st.session_state.wizard_export_format.lower()})"

    if st.button(
        generate_button_label,
        key="wizard_generate_button",
        type="primary",
        disabled=export_disabled,
        help=export_tooltip,
        use_container_width=True,
    ):
        st.session_state.wizard_processing_active = True
        # Clear previous results before starting new export
        st.session_state.wizard_export_buffer = None
        st.session_state.wizard_export_error = None
        st.session_state.wizard_preview_buffer = None  # Clear preview too
        st.session_state.wizard_preview_error = None
        logger.info("Set wizard_processing_active flag to True.")
        st.rerun()

    # Perform processing only if flag was just set
    if st.session_state.get("wizard_processing_active", False):
        if st.session_state.get("wizard_export_buffer") is None and st.session_state.get("wizard_export_error") is None:
            logger.info("Processing flag is True, starting export process...")
            with st.spinner("Generating final audio mix... This may take a moment."):
                wizard._process_and_export()  # Call the main processing function
            logger.info("Processing finished, triggering rerun to display results.")
            st.rerun()  # Rerun AFTER processing finishes

    # --- Show Download Button or Error ---
    export_buffer = st.session_state.get("wizard_export_buffer")
    export_error = st.session_state.get("wizard_export_error")

    if export_buffer:
        raw_filename = st.session_state.wizard_output_filename
        sanitized_filename = re.sub(r'[\\/*?:"<>|]', "", raw_filename).strip()
        if not sanitized_filename:
            sanitized_filename = "mindmorph_quick_mix"

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

    elif export_error:
        st.error(f"Export Failed: {export_error}")
        st.session_state.wizard_export_error = None  # Clear error after display
        if st.session_state.get("wizard_processing_active"):
            st.session_state.wizard_processing_active = False  # Ensure flag is reset on error

    # --- Navigation Buttons ---
    st.divider()
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])
    with col_nav_1:  # Home
        if st.button(
            "🏠 Back to Home",
            key="wizard_step4_home",
            use_container_width=True,
            help="Exit Wizard and return to main menu.",
            disabled=is_processing,
        ):
            if not is_processing:
                wizard._reset_wizard_state()
    with col_nav_2:  # Back
        if st.button(
            "⬅️ Back: Frequency",
            key="wizard_step4_back",
            use_container_width=True,
            disabled=is_processing,
        ):
            if not is_processing:
                st.session_state.wizard_export_buffer = None
                st.session_state.wizard_export_error = None
                st.session_state.wizard_preview_buffer = None
                st.session_state.wizard_preview_error = None
                wizard._go_to_step(3)
    with col_nav_3:  # Finish Placeholder
        st.button(
            "Finish",
            key="wizard_step4_finish_placeholder",
            disabled=True,
            use_container_width=True,
        )
