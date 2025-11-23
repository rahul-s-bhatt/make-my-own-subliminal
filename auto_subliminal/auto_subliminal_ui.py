# auto_subliminal/auto_subliminal_ui.py
# ==========================================
# UI and Workflow Logic for Auto-Create Subliminal Feature
# ==========================================

import logging
import os

import numpy as np  # For st.audio with numpy array
import streamlit as st

# Import from main application config
from config import (
    GLOBAL_SR,
    PIPER_VOICE_CONFIG_PATH,
    PIPER_VOICE_MODEL_PATH,  # For TTS initialization
)

# except ImportError as e:
#     st.error(f"Auto-Subliminal Core Modules Import Error (within auto_subliminal package): {e}. Please check installation/paths.", icon="🚨")
#     # Define dummy classes if import fails
#     class AffirmationGenerator:
#         pass
#     class BackgroundSoundManager:
#         pass
#     class OutputHandler:
#         pass
#     class AutoSubliminalAudioProcessor:
#         pass
#     class AutoSubliminalGeneratorMain:
#         pass
# Import TTS engine (top-level import)
# try:
from tts.piper_tts import PiperTTSGenerator

# --- MODIFIED: Relative imports for modules within the auto_subliminal package ---
# try:
from .affirmation_generator import AffirmationGenerator
from .audio_processor import AutoSubliminalAudioProcessor
from .background_sound_manager import BackgroundSoundManager
from .generator import AutoSubliminalGenerator as AutoSubliminalGeneratorMain
from .output_handler import OutputHandler

# except ImportError as e:
#     st.error(f"PiperTTSGenerator Import Error: {e}. TTS functionality will be affected.", icon="🗣️")

#     class PiperTTSGenerator:
#         pass  # Dummy class


logger = logging.getLogger(__name__)


@st.cache_resource
def get_auto_subliminal_components():
    """Initializes and returns components for AutoSubliminalGenerator."""
    logger.info("Initializing components for Auto-Subliminal Generator (from auto_subliminal/auto_subliminal_ui.py)...")
    try:
        # Path handling for Piper model.
        # __file__ here refers to auto_subliminal/auto_subliminal_ui.py
        # So, os.path.dirname(__file__) is auto_subliminal/
        # os.path.join(..., "..") gives the project root.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        piper_model_path_abs = PIPER_VOICE_MODEL_PATH
        if not os.path.isabs(piper_model_path_abs):
            piper_model_path_abs = os.path.join(project_root, piper_model_path_abs)

        piper_config_path_abs = PIPER_VOICE_CONFIG_PATH
        if not os.path.isabs(piper_config_path_abs):
            piper_config_path_abs = os.path.join(project_root, piper_config_path_abs)

        if not os.path.exists(piper_model_path_abs) or not os.path.exists(piper_config_path_abs):
            st.error(f"Piper TTS model or config not found. Please check paths.\nModel: {piper_model_path_abs}\nConfig: {piper_config_path_abs}", icon="🗣️")
            logger.error(f"Piper TTS model or config not found. Model: '{piper_model_path_abs}', Config: '{piper_config_path_abs}'.")
            return None

        tts_engine = PiperTTSGenerator(model_path=piper_model_path_abs, config_path=piper_config_path_abs, target_sr=GLOBAL_SR)
        affirm_gen = AffirmationGenerator()

        sound_mgr_default_path = os.path.join(project_root, "assets", "background_sounds")
        sound_mgr = BackgroundSoundManager(sounds_directory=sound_mgr_default_path)

        output_hdlr_default_path = os.path.join(project_root, "mindmorph_generations")
        output_hdlr = OutputHandler(app_base_output_directory=output_hdlr_default_path)

        audio_proc = AutoSubliminalAudioProcessor(output_sr=GLOBAL_SR)

        auto_generator = AutoSubliminalGeneratorMain(
            affirmation_gen=affirm_gen,
            sound_manager=sound_mgr,
            output_hdlr=output_hdlr,
            tts_engine=tts_engine,
            audio_processor=audio_proc,  # type: ignore
        )
        logger.info("Auto-Subliminal Generator and components initialized successfully (from auto_subliminal/auto_subliminal_ui.py).")
        return auto_generator
    except ImportError as e:
        st.error(f"Failed to import modules for Auto-Create feature: {e}. Please ensure all dependencies are installed.", icon="🚨")
        logger.error(f"ImportError for Auto-Create feature components: {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"An error occurred initializing the Auto-Create feature components: {e}", icon="🔥")
        logger.error(f"Initialization error for Auto-Create feature components: {e}", exc_info=True)
        return None


def _dummy_send_ga_event(event_name: str, event_params: dict):
    logger.debug(f"Dummy GA Event: {event_name}, {event_params} (real function not available here)")


def render_auto_subliminal_workflow(send_ga_event_func=None):
    """
    Renders the UI and handles logic for the Auto-Create Subliminal workflow.
    """
    actual_send_ga_event = send_ga_event_func if callable(send_ga_event_func) else _dummy_send_ga_event

    st.title("🚀 Auto-Create Subliminal")
    st.caption("✨ Experimental Beta Feature ✨")
    st.markdown("Simply enter a topic, and MindMorph will generate a complete subliminal audio package for you, including affirmations, background sound, and a preview.")
    st.markdown("---")

    if "auto_sub_topic" not in st.session_state:
        st.session_state.auto_sub_topic = ""
    if "auto_sub_results" not in st.session_state:
        st.session_state.auto_sub_results = None
    if "auto_sub_processing" not in st.session_state:
        st.session_state.auto_sub_processing = False
    if "auto_sub_show_results" not in st.session_state:
        st.session_state.auto_sub_show_results = False

    if not st.session_state.auto_sub_show_results:
        topic = st.text_input(
            "Enter your desired topic (e.g., 'deep relaxation', 'increased focus'):",
            value=st.session_state.auto_sub_topic,
            key="auto_sub_topic_input_widget",
            on_change=lambda: setattr(st.session_state, "auto_sub_topic", st.session_state.auto_sub_topic_input_widget),
        )

        generate_button_disabled = not st.session_state.auto_sub_topic.strip() or st.session_state.auto_sub_processing
        if st.button("🔮 Generate Subliminal Package", type="primary", disabled=generate_button_disabled, use_container_width=True):
            if st.session_state.auto_sub_topic.strip():
                st.session_state.auto_sub_processing = True
                st.session_state.auto_sub_results = None
                st.rerun()
            else:
                st.warning("Please enter a topic.")

    if st.session_state.auto_sub_processing and not st.session_state.auto_sub_show_results:
        with st.spinner(f"Generating subliminal for '{st.session_state.auto_sub_topic}'... This may take a moment."):
            auto_generator_instance = get_auto_subliminal_components()
            if auto_generator_instance:
                try:
                    results = auto_generator_instance.generate_package(st.session_state.auto_sub_topic)
                    st.session_state.auto_sub_results = results
                    if results:
                        actual_send_ga_event("auto_subliminal_generated", {"topic": st.session_state.auto_sub_topic, "status": "success"})
                    else:
                        st.error("Sorry, something went wrong during generation. Please try a different topic or check logs.", icon="😔")
                        actual_send_ga_event("auto_subliminal_generated", {"topic": st.session_state.auto_sub_topic, "status": "failure"})
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}", icon="🔥")
                    st.session_state.auto_sub_results = None
                    actual_send_ga_event("auto_subliminal_generated", {"topic": st.session_state.auto_sub_topic, "status": "exception"})
            else:
                st.session_state.auto_sub_results = None

            st.session_state.auto_sub_processing = False
            st.session_state.auto_sub_show_results = True
            st.rerun()

    if st.session_state.auto_sub_show_results:
        results = st.session_state.auto_sub_results
        if results:
            st.subheader(f'Results for: "{st.session_state.auto_sub_topic}"')

            preview_audio_np = results.get("preview_audio_data")
            preview_sr = results.get("preview_audio_sr")
            if preview_audio_np is not None and preview_audio_np.size > 0 and preview_sr:
                st.markdown("#### 🎧 Audio Preview (approx. 10 seconds)")
                st.audio(preview_audio_np, format="audio/wav", sample_rate=preview_sr)
            else:
                st.markdown("#### 🎧 Audio Preview")
                st.warning("Preview audio could not be generated for this topic.", icon="⚠️")

            affirmations_list = results.get("affirmations_list", [])
            if affirmations_list:
                with st.expander("📜 View Generated Affirmations", expanded=False):
                    st.markdown(f"_{len(affirmations_list)} affirmations generated (Truncated: {results.get('affirmations_truncated', False)})_")
                    for i, aff in enumerate(affirmations_list):
                        st.markdown(f"{i + 1}. {aff}")

            st.markdown("---")
            st.markdown("#### 💾 Download Your Files")
            col_dl1, col_dl2, col_dl3 = st.columns(3)

            text_file_path = results.get("affirmations_text_file")
            if text_file_path and os.path.exists(text_file_path):
                with open(text_file_path, "r", encoding="utf-8") as f:
                    text_data = f.read()
                with col_dl1:
                    st.download_button(label="📄 Affirmations (.txt)", data=text_data, file_name=os.path.basename(text_file_path), mime="text/plain", use_container_width=True)

            raw_audio_path = results.get("raw_affirmation_audio_file")
            if raw_audio_path and os.path.exists(raw_audio_path):
                with open(raw_audio_path, "rb") as f:
                    raw_audio_bytes = f.read()
                with col_dl2:
                    st.download_button(
                        label="🎤 Raw Affirmations (.wav)", data=raw_audio_bytes, file_name=os.path.basename(raw_audio_path), mime="audio/wav", use_container_width=True
                    )

            final_audio_path = results.get("final_subliminal_audio_file")
            if final_audio_path and os.path.exists(final_audio_path):
                with open(final_audio_path, "rb") as f:
                    final_audio_bytes = f.read()
                file_ext = os.path.splitext(final_audio_path)[1].lower()
                mime_type = "audio/mpeg" if file_ext == ".mp3" else "audio/wav"
                label_text = "🎧 Final Subliminal" + (f" ({file_ext})" if file_ext else "")
                with col_dl3:
                    st.download_button(
                        label=label_text, data=final_audio_bytes, file_name=os.path.basename(final_audio_path), mime=mime_type, use_container_width=True, type="primary"
                    )

        elif not st.session_state.auto_sub_processing:
            st.error("Could not generate the subliminal package. Please try a different topic or check the application logs.", icon="❌")

        if st.button("✨ Create Another Subliminal", key="auto_sub_create_another", use_container_width=True):
            st.session_state.auto_sub_topic = ""
            st.session_state.auto_sub_results = None
            st.session_state.auto_sub_processing = False
            st.session_state.auto_sub_show_results = False
            st.rerun()

    st.markdown("---")
    if st.button("🏠 Back to Home", key="auto_sub_back_home_from_ui_v2", help="Exit Auto-Create feature.", use_container_width=True):
        keys_to_clear = ["auto_sub_topic", "auto_sub_results", "auto_sub_processing", "auto_sub_show_results"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.selected_workflow = None
        st.rerun()
