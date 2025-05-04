# wizard_steps/step_4_export.py
# ==========================================
# UI Rendering for Wizard Step 4: Review, Mix, Preview, and Export
# Uses constants from quick_wizard_config.py
# ==========================================

import logging
import re  # Regular expression module for filename sanitization

import streamlit as st

# Import constants from the central config file
# Ensures consistency and avoids magic strings
from .quick_wizard_config import (
    AFFIRM_APPLY_SPEED_KEY,  # State key for speed change checkbox
    AFFIRMATION_TEXT_KEY,  # State key for affirmation text
    AFFIRMATION_VOLUME_KEY,  # State key for affirmation volume slider
    BG_CHOICE_KEY,  # State key for background choice ('none', 'upload', 'noise')
    BG_NOISE_TYPE_KEY,  # State key for selected noise type
    BG_UPLOADED_FILE_KEY,  # State key for the uploaded background file object
    BG_VOLUME_KEY,  # State key for background volume slider
    DEFAULT_APPLY_SPEED,  # Default value for speed change
    EXPORT_BUFFER_KEY,  # State key for the final exported audio buffer
    EXPORT_ERROR_KEY,  # State key for storing export errors
    EXPORT_FORMAT_KEY,  # State key for selected export format ('WAV', 'MP3')
    EXPORT_FORMATS,  # List like ['WAV', 'MP3']
    FREQ_CHOICE_KEY,  # State key for frequency choice ('None', 'Binaural', ...)
    FREQ_PARAMS_KEY,  # State key for frequency parameters dictionary
    FREQ_VOLUME_KEY,  # State key for frequency volume slider
    OUTPUT_FILENAME_KEY,  # State key for the desired output filename
    PREVIEW_BUFFER_KEY,  # State key for the preview audio buffer
    PREVIEW_ERROR_KEY,  # State key for storing preview errors
    WIZARD_PREVIEW_ACTIVE_KEY,  # State key (boolean) indicating if PREVIEW processing is ongoing <--- IMPORTED
    WIZARD_PROCESSING_ACTIVE_KEY,  # State key (boolean) indicating if EXPORT processing is ongoing
)

# Import necessary components (or assume they exist)
try:
    # Assuming audio_io functionality (like saving) is available elsewhere
    # from audio_utils.audio_io import ...
    pass
    AUDIO_IO_AVAILABLE = True  # Assume available for simplicity in this example
except ImportError:
    AUDIO_IO_AVAILABLE = False
    logging.warning("audio_utils.audio_io not found (or relevant parts).")

# Optional MP3 export dependency check (pydub)
try:
    # Assuming pydub check is handled by the audio processor or main app setup
    # from pydub import AudioSegment
    pass
    PYDUB_AVAILABLE = True  # Assume available for simplicity
except ImportError:
    PYDUB_AVAILABLE = False
    logging.info("pydub library not found. MP3 export disabled.")

# Setup logger for this module
logger = logging.getLogger(__name__)
PREVIEW_DURATION_SECONDS = 10  # Define preview length in seconds


def render_step_4(wizard):
    """
    Renders the UI for Step 4: Review Settings, Mix, Preview and Export.

    Args:
        wizard: An instance of the QuickWizard class, providing access to state
                and processing methods like generate_preview() and _reset_wizard_state().
    """
    st.subheader("Step 4: Review Settings, Mix & Export")
    st.write("Review your selections, adjust final volume levels, generate a preview, and export your audio.")

    # --- Read Processing States ---
    # Read both flags at the start of the render function
    is_previewing = st.session_state.get(WIZARD_PREVIEW_ACTIVE_KEY, False)
    is_exporting = st.session_state.get(WIZARD_PROCESSING_ACTIVE_KEY, False)
    # Determine if *any* processing is happening to disable general navigation
    is_processing = is_previewing or is_exporting

    # --- Review Selections ---
    st.markdown("**Review Selections:**")
    summary_data = []

    # 1. Affirmations Summary
    affirm_text = st.session_state.get(AFFIRMATION_TEXT_KEY, "").strip()
    affirmations_present = bool(affirm_text)  # Needed for disabling buttons
    speed_setting = st.session_state.get(AFFIRM_APPLY_SPEED_KEY, DEFAULT_APPLY_SPEED)
    speed_indicator = " (Speed Change Enabled)" if speed_setting else " (Speed Change Disabled)"
    if affirmations_present:
        summary_data.append(f"- **Affirmations:** Text Input ('{affirm_text[:30].strip()}...'){speed_indicator}")
    else:
        summary_data.append("- **Affirmations:** ⚠️ **MISSING** (Go back to Step 1)")

    # 2. Background Summary
    bg_choice = st.session_state.get(BG_CHOICE_KEY, "none")
    if bg_choice == "upload":
        uploaded_file = st.session_state.get(BG_UPLOADED_FILE_KEY)
        if uploaded_file:
            summary_data.append(f"- **Background:** Uploaded File ('{uploaded_file.name}')")
        else:
            summary_data.append("- **Background:** Upload Selected (⚠️ **File Missing?** Go back to Step 2)")
    elif bg_choice == "noise":
        noise_type = st.session_state.get(BG_NOISE_TYPE_KEY, "N/A")
        summary_data.append(f"- **Background:** Generated Noise ('{noise_type}')")
    else:  # bg_choice == "none"
        summary_data.append("- **Background:** None")

    # 3. Frequency Summary
    freq_choice = st.session_state.get(FREQ_CHOICE_KEY, "None")
    if freq_choice != "None":
        freq_params = st.session_state.get(FREQ_PARAMS_KEY, {})
        params_str = ", ".join(f"{k.replace('_freq', '').capitalize()}={v}Hz" for k, v in freq_params.items() if v is not None)
        summary_data.append(f"- **Frequency:** {freq_choice} ({params_str})")
    else:
        summary_data.append("- **Frequency:** None")

    st.markdown("\n".join(summary_data))
    st.divider()

    # --- Mixing Controls (Volume Sliders) ---
    # Disable sliders if any processing is active
    st.markdown("**Mixing Controls:**")
    col_vol1, col_vol2, col_vol3 = st.columns(3)

    bg_choice_for_vol = st.session_state.get(BG_CHOICE_KEY, "none")
    freq_choice_for_vol = st.session_state.get(FREQ_CHOICE_KEY, "None")

    with col_vol1:
        st.session_state[AFFIRMATION_VOLUME_KEY] = st.slider(
            "🗣️ Affirmation Vol.",
            0.0,
            1.0,
            value=st.session_state.get(AFFIRMATION_VOLUME_KEY, 1.0),
            step=0.05,
            key="wizard_affirm_vol_slider_step4",
            help="Adjust the volume of the affirmation track.",
            disabled=is_processing,  # Disable if previewing or exporting
        )
    with col_vol2:
        disable_bg_vol = bg_choice_for_vol == "none" or is_processing
        st.session_state[BG_VOLUME_KEY] = st.slider(
            "🎵 Background Vol.",
            0.0,
            1.0,
            value=st.session_state.get(BG_VOLUME_KEY, 0.7),
            step=0.05,
            key="wizard_bg_vol_slider_step4",
            disabled=disable_bg_vol,
            help="Adjust the volume of the background track. Disabled if 'None' selected or processing.",
        )
    with col_vol3:
        disable_freq_vol = freq_choice_for_vol == "None" or is_processing
        st.session_state[FREQ_VOLUME_KEY] = st.slider(
            "〰️ Frequency Vol.",
            0.0,
            1.0,
            value=st.session_state.get(FREQ_VOLUME_KEY, 0.2),
            step=0.05,
            key="wizard_freq_vol_slider_step4",
            disabled=disable_freq_vol,
            help="Adjust the volume of the frequency track. Disabled if 'None' selected or processing.",
        )
    st.divider()

    # --- Preview Section ---
    st.markdown("**Preview Mix:**")
    st.caption(f"Generate a {PREVIEW_DURATION_SECONDS}-second preview of the mix with current settings.")

    # Determine Preview Button State
    preview_button_disabled = not affirmations_present or is_processing  # Disabled if no text OR if any processing active
    preview_tooltip = ""
    if is_previewing:
        preview_tooltip = "Preview generation in progress..."
    elif is_exporting:
        preview_tooltip = "Final export in progress, please wait."
    elif not affirmations_present:
        preview_tooltip = "Please add affirmation text in Step 1 to enable preview."
    else:
        preview_tooltip = "Generate a short preview of the final mix."

    preview_button_label = "⏳ Generating Preview..." if is_previewing else f"🎧 Generate Preview ({PREVIEW_DURATION_SECONDS}s)"

    # Preview Button
    if st.button(
        preview_button_label,
        key="wizard_preview_button",
        disabled=preview_button_disabled,  # Use combined disabled state
        help=preview_tooltip,
        use_container_width=True,
    ):
        # --- Preview Button Click Logic ---
        logger.info("Preview button clicked. Setting flags and rerunning.")
        # 1. Clear previous results
        st.session_state.pop(PREVIEW_BUFFER_KEY, None)
        st.session_state.pop(PREVIEW_ERROR_KEY, None)
        # 2. Set the preview active flag
        st.session_state[WIZARD_PREVIEW_ACTIVE_KEY] = True
        # 3. Rerun immediately
        st.rerun()
        # --- End Preview Button Click Logic ---

    # Display Preview Audio Player or Error Message (reads state from previous run)
    preview_buffer = st.session_state.get(PREVIEW_BUFFER_KEY)
    preview_error = st.session_state.get(PREVIEW_ERROR_KEY)

    if preview_buffer:
        try:
            st.audio(preview_buffer, format="audio/wav")
        except Exception as e:
            st.error(f"Error displaying preview audio player: {e}")
            logger.error(f"Error displaying preview buffer: {e}")
            st.session_state.pop(PREVIEW_BUFFER_KEY, None)
    elif preview_error:
        st.error(f"Preview Generation Failed: {preview_error}")
        st.session_state.pop(PREVIEW_ERROR_KEY, None)  # Clear error after display
    st.divider()

    # --- Export Options ---
    st.markdown("**Export Settings:**")
    col_export1, col_export2 = st.columns([2, 1])

    with col_export1:
        # Filename Input (disabled during processing)
        st.session_state[OUTPUT_FILENAME_KEY] = st.text_input(
            "Output Filename (no extension):",
            st.session_state.get(OUTPUT_FILENAME_KEY, "mindmorph_quick_mix"),
            key="wizard_filename_input",
            help="Enter the desired name for your exported file.",
            disabled=is_processing,
        )
    with col_export2:
        # Export Format Selection (disabled during processing)
        export_formats_options = EXPORT_FORMATS.copy()
        help_text = "Choose WAV (lossless) or MP3 (compressed)."
        if not PYDUB_AVAILABLE:
            if "MP3" in export_formats_options:
                export_formats_options.remove("MP3")
            help_text = "WAV format only (MP3 export requires additional libraries)."

        current_format = st.session_state.get(EXPORT_FORMAT_KEY, "WAV")
        if current_format not in export_formats_options:
            current_format = "WAV"
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
            disabled=is_processing,  # Disable during processing
        )
        # Update state only if changed AND not processing
        if selected_format != current_format and not is_processing:
            st.session_state[EXPORT_FORMAT_KEY] = selected_format
            st.rerun()

    # --- Generate Button ---
    # Determine Export Button State
    export_button_disabled = not affirmations_present or is_processing  # Disabled if no text OR if any processing active
    export_tooltip = ""
    if is_exporting:
        export_tooltip = "Final export in progress..."
    elif is_previewing:
        export_tooltip = "Preview generation in progress, please wait."
    elif not affirmations_present:
        export_tooltip = "Please add affirmation text in Step 1 to enable generation."
    else:
        export_tooltip = "Generate the final audio file based on current settings."

    # Add specific disable reason if MP3 selected but unavailable
    if st.session_state.get(EXPORT_FORMAT_KEY) == "MP3" and not PYDUB_AVAILABLE:
        export_button_disabled = True  # Override other conditions
        export_tooltip = "MP3 format is unavailable. Please install required libraries or choose WAV."

    export_button_label = "⏳ Processing Export..." if is_exporting else f"Generate & Prepare Download (. {st.session_state.get(EXPORT_FORMAT_KEY, 'WAV').lower()})"

    # The Generate Button itself
    if st.button(
        export_button_label,
        key="wizard_generate_button",
        type="primary",
        disabled=export_button_disabled,  # Use combined disabled state
        help=export_tooltip,
        use_container_width=True,
    ):
        # --- Generate Button Click Logic ---
        logger.info("Generate button clicked. Setting flags and rerunning.")
        # 1. Clear previous results (export and preview)
        st.session_state.pop(EXPORT_BUFFER_KEY, None)
        st.session_state.pop(EXPORT_ERROR_KEY, None)
        st.session_state.pop(PREVIEW_BUFFER_KEY, None)
        st.session_state.pop(PREVIEW_ERROR_KEY, None)
        # 2. Set the export active flag
        st.session_state[WIZARD_PROCESSING_ACTIVE_KEY] = True
        # 3. Rerun immediately
        st.rerun()
        # --- End Generate Button Click Logic ---

    # --- Show Download Button or Error Message ---
    # This displays results after the export process has run
    export_buffer = st.session_state.get(EXPORT_BUFFER_KEY)
    export_error = st.session_state.get(EXPORT_ERROR_KEY)

    if export_buffer:
        # Sanitize filename
        raw_filename = st.session_state.get(OUTPUT_FILENAME_KEY, "mindmorph_quick_mix")
        sanitized_filename = re.sub(r'[\\/*?:"<>|]', "", raw_filename).strip() or "mindmorph_quick_mix"
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
                help="Click to download the generated audio file.",
            )
        except Exception as e:
            st.error(f"Error creating download button: {e}")
            logger.error(f"Error creating download button widget: {e}")
            st.session_state.pop(EXPORT_BUFFER_KEY, None)

    elif export_error:
        st.error(f"Export Failed: {export_error}")
        st.session_state.pop(EXPORT_ERROR_KEY, None)  # Clear error after display
        # Ensure processing flag is reset if error displayed
        if st.session_state.get(WIZARD_PROCESSING_ACTIVE_KEY):
            st.session_state[WIZARD_PROCESSING_ACTIVE_KEY] = False
            logger.warning("Resetting export processing flag due to error display.")
        if st.session_state.get(WIZARD_PREVIEW_ACTIVE_KEY):  # Also check preview flag
            st.session_state[WIZARD_PREVIEW_ACTIVE_KEY] = False
            logger.warning("Resetting preview processing flag due to export error display.")

    # --- Navigation Buttons ---
    st.divider()
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 2])
    # Disable navigation if any processing is active
    nav_disabled = is_processing

    with col_nav_1:
        if st.button(
            "🏠 Back to Home",
            key="wizard_step4_home",
            use_container_width=True,
            help="Exit the wizard and return to the main application.",
            disabled=nav_disabled,
        ):
            wizard._reset_wizard_state()
            st.rerun()
    with col_nav_2:
        if st.button(
            "⬅️ Back: Frequency",
            key="wizard_step4_back",
            use_container_width=True,
            help="Go back to Step 3 (Frequency selection).",
            disabled=nav_disabled,
        ):
            wizard._go_to_step(3)  # Handles rerun internally
    with col_nav_3:
        st.button(
            "Finish ✨",
            key="wizard_step4_finish_placeholder",
            disabled=True,
            use_container_width=True,
            help="Wizard complete (Action TBD).",
        )
