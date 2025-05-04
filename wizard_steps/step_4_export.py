# wizard_steps/step_4_export.py
# ==========================================
# UI Rendering for Wizard Step 4: Review, Mix, Preview, and Export
# Uses constants from quick_wizard_config.py
# ==========================================

import logging
import re

import streamlit as st

# Import constants from the central config file
from .quick_wizard_config import EXPORT_FORMATS  # Import list of formats
from .quick_wizard_config import (
    AFFIRM_APPLY_SPEED_KEY,
    AFFIRMATION_TEXT_KEY,
    AFFIRMATION_VOLUME_KEY,
    BG_CHOICE_KEY,
    BG_NOISE_TYPE_KEY,
    BG_UPLOADED_FILE_KEY,
    BG_VOLUME_KEY,
    DEFAULT_APPLY_SPEED,
    EXPORT_BUFFER_KEY,
    EXPORT_ERROR_KEY,
    EXPORT_FORMAT_KEY,
    FREQ_CHOICE_KEY,
    FREQ_PARAMS_KEY,
    FREQ_VOLUME_KEY,
    OUTPUT_FILENAME_KEY,
    PREVIEW_BUFFER_KEY,
    PREVIEW_ERROR_KEY,
    WIZARD_PROCESSING_ACTIVE_KEY,
)

# Import necessary components
try:
    pass

    AUDIO_IO_AVAILABLE = True
except ImportError:
    AUDIO_IO_AVAILABLE = False
    logging.warning("audio_utils.audio_io not found.")

# Optional MP3 export dependency check
try:
    pass

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.info("pydub library not found. MP3 export disabled.")

logger = logging.getLogger(__name__)
PREVIEW_DURATION_SECONDS = 10


def render_step_4(wizard):
    """
    Renders the UI for Step 4: Review Settings, Mix, Preview and Export.

    Args:
        wizard: An instance of the QuickWizard class.
    """
    st.subheader("Step 4: Review Settings, Mix & Export")
    st.write("Review your selections, adjust final volume levels...")

    # --- Review Selections (Based on Settings) ---
    st.markdown("**Review Selections:**")
    summary_data = []

    # 1. Affirmations Summary
    affirm_text = st.session_state.get(AFFIRMATION_TEXT_KEY, "").strip()
    speed_setting = st.session_state.get(AFFIRM_APPLY_SPEED_KEY, DEFAULT_APPLY_SPEED)
    speed_indicator = (
        " (Speed Change Enabled)" if speed_setting else " (Speed Change Disabled)"
    )
    if affirm_text:
        summary_data.append(
            f"- **Affirmations:** Text Input ('{affirm_text[:30].strip()}...'){speed_indicator}"
        )
    else:
        summary_data.append("- **Affirmations:** ⚠️ **MISSING** (Go back to Step 1)")

    # 2. Background Summary
    bg_choice = st.session_state.get(BG_CHOICE_KEY, "none")
    if bg_choice == "upload":
        uploaded_file = st.session_state.get(BG_UPLOADED_FILE_KEY)
        if uploaded_file:
            summary_data.append(
                f"- **Background:** Uploaded File ('{uploaded_file.name}')"
            )
        else:
            summary_data.append(
                "- **Background:** Upload Selected (⚠️ **File Missing?** Go back to Step 2)"
            )
    elif bg_choice == "noise":
        noise_type = st.session_state.get(BG_NOISE_TYPE_KEY, "N/A")
        summary_data.append(f"- **Background:** Generated Noise ('{noise_type}')")
    else:
        summary_data.append("- **Background:** None")

    # 3. Frequency Summary
    freq_choice = st.session_state.get(FREQ_CHOICE_KEY, "None")
    if freq_choice != "None":
        freq_params = st.session_state.get(FREQ_PARAMS_KEY, {})
        params_str = ", ".join(
            f"{k.replace('_freq', '').capitalize()}={v}Hz"
            for k, v in freq_params.items()
            if v is not None
        )
        summary_data.append(f"- **Frequency:** {freq_choice} ({params_str})")
    else:
        summary_data.append("- **Frequency:** None")

    st.markdown("\n".join(summary_data))
    st.divider()

    # --- Mixing Controls ---
    st.markdown("**Mixing Controls:**")
    col_vol1, col_vol2, col_vol3 = st.columns(3)
    bg_choice_for_vol = st.session_state.get(BG_CHOICE_KEY, "none")
    freq_choice_for_vol = st.session_state.get(FREQ_CHOICE_KEY, "None")

    with col_vol1:
        st.session_state[AFFIRMATION_VOLUME_KEY] = st.slider(
            "🗣️ Affirmation Vol.",
            0.0,
            1.0,
            st.session_state.get(AFFIRMATION_VOLUME_KEY, 1.0),
            0.05,
            key="wizard_affirm_vol_slider_step4",  # Widget key can be different
        )
    with col_vol2:
        disable_bg_vol = bg_choice_for_vol == "none"
        st.session_state[BG_VOLUME_KEY] = st.slider(
            "🎵 Background Vol.",
            0.0,
            1.0,
            st.session_state.get(BG_VOLUME_KEY, 0.7),
            0.05,
            key="wizard_bg_vol_slider_step4",
            disabled=disable_bg_vol,
        )
    with col_vol3:
        disable_freq_vol = freq_choice_for_vol == "None"
        st.session_state[FREQ_VOLUME_KEY] = st.slider(
            "〰️ Frequency Vol.",
            0.0,
            1.0,
            st.session_state.get(FREQ_VOLUME_KEY, 0.2),
            0.05,
            key="wizard_freq_vol_slider_step4",
            disabled=disable_freq_vol,
        )
    st.divider()

    # --- Preview Section ---
    st.markdown("**Preview Mix:**")
    st.caption(f"Generate a {PREVIEW_DURATION_SECONDS}-second preview...")
    affirmations_present = bool(affirm_text)
    preview_disabled = not affirmations_present
    preview_tooltip = (
        "Generate preview." if affirmations_present else "Add affirmation text first."
    )
    if st.button(
        f"🎧 Generate Preview ({PREVIEW_DURATION_SECONDS}s)",
        key="wizard_preview_button",
        disabled=preview_disabled,
        help=preview_tooltip,
        use_container_width=True,
    ):
        with st.spinner(f"Generating preview..."):
            wizard.generate_preview(PREVIEW_DURATION_SECONDS)
        st.rerun()

    preview_buffer = st.session_state.get(PREVIEW_BUFFER_KEY)
    preview_error = st.session_state.get(PREVIEW_ERROR_KEY)
    if preview_buffer:
        try:
            st.audio(preview_buffer, format="audio/wav")
        except Exception as e:
            st.error(f"Error displaying preview: {e}")
            logger.error(f"Error display preview: {e}")
            st.session_state.pop(PREVIEW_BUFFER_KEY, None)
    elif preview_error:
        st.error(f"Preview Error: {preview_error}")
        st.session_state.pop(PREVIEW_ERROR_KEY, None)
    st.divider()

    # --- Export Options ---
    st.markdown("**Export Settings:**")
    col_export1, col_export2 = st.columns([2, 1])
    with col_export1:
        st.session_state[OUTPUT_FILENAME_KEY] = st.text_input(
            "Output Filename (no extension):",
            st.session_state.get(OUTPUT_FILENAME_KEY, "mindmorph_mix"),
            key="wizard_filename_input",
        )
    with col_export2:
        export_formats_options = EXPORT_FORMATS.copy()  # Use list from config
        help_text = "Choose WAV or MP3."
        if not PYDUB_AVAILABLE:
            if "MP3" in export_formats_options:
                export_formats_options.remove("MP3")
            help_text = "WAV format only (MP3 requires pydub library)."
        current_format = st.session_state.get(EXPORT_FORMAT_KEY, "WAV")
        if current_format not in export_formats_options:
            current_format = "WAV"  # Default if invalid
        try:
            current_format_index = export_formats_options.index(current_format)
        except ValueError:
            current_format_index = 0
        selected_format = st.radio(
            "Format:",
            export_formats_options,
            index=current_format_index,
            key="wizard_export_format_radio",
            horizontal=True,
            help=help_text,
        )
        if selected_format != current_format:
            st.session_state[EXPORT_FORMAT_KEY] = selected_format
            st.rerun()

    # --- Generate Button ---
    is_processing = st.session_state.get(WIZARD_PROCESSING_ACTIVE_KEY, False)
    export_disabled = is_processing or not affirmations_present
    export_tooltip = ""
    if is_processing:
        export_tooltip = "Processing..."
    elif not affirmations_present:
        export_tooltip = "Add affirmation text first."
    else:
        export_tooltip = "Generate final audio."
    if st.session_state.get(EXPORT_FORMAT_KEY) == "MP3" and not PYDUB_AVAILABLE:
        export_disabled = True
        export_tooltip = "MP3 unavailable."

    generate_button_label = (
        "⏳ Processing..."
        if is_processing
        else f"Generate & Prepare Download (. {st.session_state.get(EXPORT_FORMAT_KEY, 'WAV').lower()})"
    )
    if st.button(
        generate_button_label,
        key="wizard_generate_button",
        type="primary",
        disabled=export_disabled,
        help=export_tooltip,
        use_container_width=True,
    ):
        st.session_state.pop(EXPORT_BUFFER_KEY, None)
        st.session_state.pop(EXPORT_ERROR_KEY, None)
        st.session_state.pop(PREVIEW_BUFFER_KEY, None)
        st.session_state.pop(PREVIEW_ERROR_KEY, None)
        logger.info(
            "Generate button clicked, calling wizard.process_and_export_audio()"
        )
        wizard.process_and_export_audio()
        st.rerun()

    # --- Show Download Button or Error ---
    export_buffer = st.session_state.get(EXPORT_BUFFER_KEY)
    export_error = st.session_state.get(EXPORT_ERROR_KEY)
    if export_buffer:
        raw_filename = st.session_state.get(OUTPUT_FILENAME_KEY, "mindmorph_mix")
        sanitized_filename = (
            re.sub(r'[\\/*?:"<>|]', "", raw_filename).strip() or "mindmorph_mix"
        )
        file_ext = st.session_state.get(EXPORT_FORMAT_KEY, "WAV").lower()
        download_filename = f"{sanitized_filename}.{file_ext}"
        mime_type = f"audio/{file_ext}" if file_ext == "wav" else "audio/mpeg"
        try:
            st.download_button(
                label=f"⬇️ Download: {download_filename}",
                data=export_buffer,
                file_name=download_filename,
                mime=mime_type,
                key="wizard_download_button",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error creating download button: {e}")
            logger.error(f"Error dl button: {e}")
            st.session_state.pop(EXPORT_BUFFER_KEY, None)
    elif export_error:
        st.error(f"Export Failed: {export_error}")
        st.session_state.pop(EXPORT_ERROR_KEY, None)
        if st.session_state.get(WIZARD_PROCESSING_ACTIVE_KEY):
            st.session_state[WIZARD_PROCESSING_ACTIVE_KEY] = False

    # --- Navigation Buttons ---
    st.divider()
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])
    nav_disabled = is_processing
    with col_nav_1:
        if st.button(
            "🏠 Back to Home",
            key="wizard_step4_home",
            use_container_width=True,
            help="Exit wizard.",
            disabled=nav_disabled,
        ):
            wizard._reset_wizard_state()
    with col_nav_2:
        if st.button(
            "⬅️ Back: Frequency",
            key="wizard_step4_back",
            use_container_width=True,
            help="Go back to Step 3.",
            disabled=nav_disabled,
        ):
            wizard._go_to_step(3)
    with col_nav_3:
        st.button(
            "Finish ✨",
            key="wizard_step4_finish_placeholder",
            disabled=True,
            use_container_width=True,
        )
