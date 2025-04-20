import os
import tempfile
import textwrap

import pyttsx3
from pydub import AudioSegment


def init_engine():
    engine = pyttsx3.init()
    engine.setProperty("rate", 240)  # Speed
    engine.setProperty("volume", 1.0)
    return engine


def speak_to_mp3(text, output_file="output.mp3", chunk_size=3000):
    engine = init_engine()
    chunks = textwrap.wrap(text, chunk_size)

    temp_files = []

    print(f"🧠 Total Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            temp_file_path = tf.name

        print(f"🔊 [{i + 1}/{len(chunks)}] Synthesizing chunk...")
        engine.save_to_file(chunk, temp_file_path)
        engine.runAndWait()  # <-- VERY important to wait after each save
        temp_files.append(temp_file_path)

    # Merge all temp wavs to MP3
    combined = AudioSegment.empty()
    for file_path in temp_files:
        try:
            audio = AudioSegment.from_wav(file_path)
            combined += audio
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
        finally:
            os.remove(file_path)

    combined.export(output_file, format="mp3")
    print(f"✅ MP3 saved: {output_file}")


# === Example ===
if __name__ == "__main__":
    with open("long_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    speak_to_mp3(text, "final_output.mp3")
