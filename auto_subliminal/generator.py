# auto_subliminal/generator.py
# Main orchestrator for the "one-click" subliminal generation feature.

import logging
import os  # For os.path.basename

# Import from main application config
from config import MIX_PREVIEW_DURATION_S  # Using this for consistency, or use AUTO_SUB_PREVIEW_DURATION_S

# Assuming these modules are in the same package (auto_subliminal)
from .affirmation_generator import AffirmationGenerator
from .background_sound_manager import BackgroundSoundManager
from .config import AUTO_SUB_AFFIRMATION_VOLUME_DB_OFFSET, AUTO_SUB_PREVIEW_DURATION_S
from .output_handler import OutputHandler

# Placeholder for actual TTS and Audio Mixing utilities/classes
# These would likely come from other parts of your application (e.g., tts.piper_tts, audio_utils.mixer)
# from tts.base_tts import BaseTTSGenerator # Example: your actual TTS class
# from audio_utils.audio_processing import AudioMixer # Example: your actual Mixer class

logger = logging.getLogger(__name__)


class AutoSubliminalGenerator:
    """
    Generates a complete subliminal audio package from a single topic.
    """

    def __init__(
        self,
        affirmation_gen: AffirmationGenerator,
        sound_manager: BackgroundSoundManager,
        output_hdlr: OutputHandler,
        tts_engine: any,  # Should be an instance of your actual TTS class # type: ignore
        audio_mixer: any,  # Should be an instance of your actual AudioMixer class # type: ignore
    ):
        self.affirmation_generator = affirmation_gen
        self.background_sound_manager = sound_manager
        self.output_handler = output_hdlr
        self.tts_engine = tts_engine
        self.audio_mixer = audio_mixer  # This mixer needs to handle looping of background
        logger.info("AutoSubliminalGenerator initialized.")

    def generate_package(self, topic: str) -> dict | None:
        """
        Generates the full subliminal package for the given topic.

        Args:
            topic (str): The topic for the subliminal.

        Returns:
            dict | None: A dictionary containing:
                         'affirmations_list': list[str],
                         'affirmations_text_file': str | None,
                         'raw_affirmation_audio_file': str | None,
                         'final_subliminal_audio_file': str | None,
                         'preview_audio_data': bytes | None
                         or None if generation fails at a critical step.
        """
        if not topic.strip():
            logger.error("Topic cannot be empty for auto-generation.")
            return None

        logger.info(f"Starting auto subliminal generation for topic: '{topic}'")
        output_files = {}  # For file paths
        generated_data = {}  # For in-memory data like affirmation list and preview bytes

        # 1. Generate and Expand Affirmations
        try:
            affirmations, truncated = self.affirmation_generator.generate_and_expand_affirmations(topic)
            if not affirmations:
                logger.error(f"Failed to generate affirmations for topic: {topic}")
                return None
            generated_data["affirmations_list"] = affirmations
            generated_data["affirmations_truncated"] = truncated
            logger.info(f"Generated and expanded {len(affirmations)} affirmations (truncated: {truncated}).")
        except Exception as e:
            logger.error(f"Exception during affirmation generation/expansion: {e}", exc_info=True)
            return None

        # 2. Save Affirmations Text
        try:
            affirmations_text_path = self.output_handler.save_affirmations_text(topic, affirmations)
            output_files["affirmations_text_file"] = affirmations_text_path
        except Exception as e:
            logger.error(f"Exception during saving affirmations text: {e}", exc_info=True)
            # Non-critical, can proceed

        # 3. Convert Affirmations to Speech (TTS) -> Raw Affirmation Audio Data (bytes)
        raw_affirmation_audio_bytes: bytes | None = None
        try:
            full_affirmation_text = "\n".join(affirmations)  # TTS might prefer one block

            # Actual TTS call:
            # raw_affirmation_audio_bytes = self.tts_engine.synthesize_to_bytes(full_affirmation_text)
            raw_affirmation_audio_bytes = f"TTS audio data for: {topic}".encode("utf-8")  # Placeholder

            if not raw_affirmation_audio_bytes:
                logger.error(f"TTS engine failed to produce audio for topic: {topic}")
                return None  # Critical failure
            generated_data["raw_affirmation_audio_bytes"] = raw_affirmation_audio_bytes
            logger.info("Successfully generated raw affirmation audio using TTS.")
        except Exception as e:
            logger.error(f"Exception during TTS generation: {e}", exc_info=True)
            return None

        # 4. Save Raw Affirmation Audio
        try:
            raw_audio_path = self.output_handler.save_audio_file(
                topic,
                raw_affirmation_audio_bytes,
                "raw_affirmations",
                "wav",  # WAV for raw
            )
            output_files["raw_affirmation_audio_file"] = raw_audio_path
        except Exception as e:
            logger.error(f"Exception during saving raw affirmation audio: {e}", exc_info=True)
            # Potentially non-critical if final mix succeeds

        # 5. Select Background Sound
        background_sound_info = None  # Dict: {"name": str, "path": str, "duration_s": float | None}
        try:
            background_sound_info = self.background_sound_manager.select_sound_info(topic)
            if background_sound_info:
                logger.info(f"Selected background sound: {background_sound_info['name']}")
                generated_data["background_sound_path"] = background_sound_info["path"]
            else:
                logger.warning(f"No background sound selected for topic: {topic}. Final mix will lack background.")
                generated_data["background_sound_path"] = None
        except Exception as e:
            logger.error(f"Exception during background sound selection: {e}", exc_info=True)
            # Potentially non-critical

        # 6. Mix Affirmation Audio with Background Sound -> Final Subliminal Audio Data (bytes)
        final_subliminal_audio_bytes: bytes | None = None
        try:
            # Actual mixing call:
            # The audio_mixer.mix_subliminal method needs to handle:
            # - Loading affirmation audio (from bytes or saved path)
            # - Loading background audio (from path)
            # - Looping background if it's shorter than affirmation audio.
            # - Adjusting affirmation audio volume (e.g., AUTO_SUB_AFFIRMATION_VOLUME_DB_OFFSET).
            # - Outputting mixed audio as bytes.
            #
            # final_subliminal_audio_bytes = self.audio_mixer.mix_subliminal(
            #     affirmation_audio_bytes=raw_affirmation_audio_bytes,
            #     background_audio_path=generated_data.get("background_sound_path"),
            #     affirmation_volume_db=AUTO_SUB_AFFIRMATION_VOLUME_DB_OFFSET,
            #     # output_sr=GLOBAL_SR from config.py, # Ensure consistent SR
            # )

            # Placeholder for mixing:
            bg_path = generated_data.get("background_sound_path")
            if bg_path:
                final_subliminal_audio_bytes = f"Mixed audio: [Affirmations for {topic}] + [BG: {os.path.basename(bg_path)}]".encode("utf-8")
            else:
                final_subliminal_audio_bytes = raw_affirmation_audio_bytes  # No background

            if not final_subliminal_audio_bytes:
                logger.error(f"Audio mixer failed to produce final subliminal audio for topic: {topic}")
                return None  # Critical failure
            generated_data["final_subliminal_audio_bytes"] = final_subliminal_audio_bytes
            logger.info("Successfully mixed final subliminal audio.")
        except Exception as e:
            logger.error(f"Exception during audio mixing: {e}", exc_info=True)
            return None

        # 7. Save Final Subliminal Audio
        try:
            final_audio_path = self.output_handler.save_audio_file(
                topic,
                final_subliminal_audio_bytes,
                "final_subliminal",
                "mp3",  # MP3 for final
            )
            output_files["final_subliminal_audio_file"] = final_audio_path
            if not final_audio_path:  # If saving failed, this is problematic
                logger.error("Critical failure: Failed to save the final subliminal audio file path.")
                # Depending on strictness, might return None here
        except Exception as e:
            logger.error(f"Exception during saving final subliminal audio: {e}", exc_info=True)
            return None  # Saving final audio is critical

        # 8. Generate Preview Audio (short segment of the final mix)
        preview_audio_bytes: bytes | None = None
        try:
            # Actual preview generation call:
            # The audio_mixer.generate_preview method would take the final mixed audio (or ingredients)
            # and return a short segment (e.g., AUTO_SUB_PREVIEW_DURATION_S).
            #
            # preview_audio_bytes = self.audio_mixer.generate_preview(
            #     source_audio_bytes=final_subliminal_audio_bytes, # or provide ingredients again
            #     duration_s=AUTO_SUB_PREVIEW_DURATION_S
            # )

            # Placeholder for preview:
            preview_audio_bytes = f"Preview of: {topic}".encode("utf-8")  # Truncate or take first N bytes of final_subliminal_audio_bytes

            if not preview_audio_bytes:
                logger.warning(f"Failed to generate preview audio for topic: {topic}")
            generated_data["preview_audio_data"] = preview_audio_bytes
            logger.info("Successfully generated preview audio.")
        except Exception as e:
            logger.error(f"Exception during preview audio generation: {e}", exc_info=True)
            # Non-critical, main files are generated

        logger.info(f"Auto subliminal package generation completed for topic: '{topic}'.")

        # Combine file paths and in-memory data for the return dictionary
        return {**output_files, **generated_data}


# Placeholder classes for TTS and Mixer (replace with your actual implementations)
class PlaceholderTTSEngine:
    def synthesize_to_bytes(self, text: str) -> bytes:
        logger.info(f"TTS Placeholder: Synthesizing text - '{text[:50]}...'")
        # In a real scenario, this would return actual audio bytes from a TTS engine
        return f"TTS audio for: {text}".encode("utf-8")


class PlaceholderAudioMixer:
    def mix_subliminal(self, affirmation_audio_bytes: bytes, background_audio_path: str | None, affirmation_volume_db: float) -> bytes:
        affirm_data_str = affirmation_audio_bytes.decode("utf-8", errors="ignore")[:30]
        bg_name = os.path.basename(background_audio_path) if background_audio_path else "NoBgAudio"
        logger.info(f"AudioMixer Placeholder: Mixing '{affirm_data_str}...' with '{bg_name}' (Affirm Vol: {affirmation_volume_db}dB). BG Looping implied.")
        # Real mixer would load background, loop if needed, mix with affirmation_audio_bytes at specified volume.
        return f"Mixed: {affirm_data_str}... + {bg_name}".encode("utf-8")

    def generate_preview(self, source_audio_bytes: bytes, duration_s: int) -> bytes:
        logger.info(f"AudioMixer Placeholder: Generating {duration_s}s preview.")
        # Real preview generator would take the first 'duration_s' of source_audio_bytes
        return source_audio_bytes[: min(len(source_audio_bytes), duration_s * 10000)]  # Rough estimate for bytes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Setup dummy directories and files for testing
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dummy_assets_dir = os.path.join(project_root_dir, "assets")
    dummy_bg_dir = os.path.join(dummy_assets_dir, "background_sounds")
    if not os.path.exists(dummy_bg_dir):
        os.makedirs(dummy_bg_dir)

    dummy_rain_sound = os.path.join(dummy_bg_dir, "rain_test.mp3")
    if not os.path.exists(dummy_rain_sound):
        with open(dummy_rain_sound, "w") as f:
            f.write("dummy rain sound file content")

    # Initialize components with placeholders
    affirm_gen = AffirmationGenerator()
    sound_mgr = BackgroundSoundManager(sounds_directory=dummy_bg_dir)  # Uses the dummy dir
    output_hdlr = OutputHandler(base_output_directory="generated_subliminals_main_test")

    tts_engine_placeholder = PlaceholderTTSEngine()
    audio_mixer_placeholder = PlaceholderAudioMixer()

    auto_generator = AutoSubliminalGenerator(
        affirmation_gen=affirm_gen, sound_manager=sound_mgr, output_hdlr=output_hdlr, tts_engine=tts_engine_placeholder, audio_mixer=audio_mixer_placeholder
    )

    sample_topic = "enhanced creativity"
    result_package = auto_generator.generate_package(sample_topic)

    if result_package:
        print(f"\n--- Generation successful for '{sample_topic}' ---")
        print(f"Affirmations List ({len(result_package.get('affirmations_list', []))} items): First few...")
        for i, aff in enumerate(result_package.get("affirmations_list", [])[:3]):
            print(f"  {i + 1}. {aff}")
        print(f"Affirmations Truncated: {result_package.get('affirmations_truncated')}")
        print(f"Affirmations Text File: {result_package.get('affirmations_text_file')}")
        print(f"Raw Affirmation Audio File: {result_package.get('raw_affirmation_audio_file')}")
        print(f"Background Sound Path Used: {result_package.get('background_sound_path')}")
        print(f"Final Subliminal Audio File: {result_package.get('final_subliminal_audio_file')}")

        preview_data = result_package.get("preview_audio_data")
        if preview_data:
            print(f"Preview Audio Data: {len(preview_data)} bytes (Placeholder: '{preview_data.decode('utf-8', errors='ignore')[:60]}...')")
        else:
            print("Preview Audio Data: Not generated or failed.")
    else:
        print(f"\n--- Generation failed for '{sample_topic}' ---")

    # Optional: Clean up test directories and files
    # import shutil
    # if os.path.exists(output_hdlr.output_directory): shutil.rmtree(output_hdlr.output_directory)
    # if os.path.exists(dummy_rain_sound): os.remove(dummy_rain_sound)
    # if os.path.exists(dummy_bg_dir) and not os.listdir(dummy_bg_dir): os.rmdir(dummy_bg_dir)
    # if os.path.exists(dummy_assets_dir) and not os.listdir(dummy_assets_dir): os.rmdir(dummy_assets_dir)
