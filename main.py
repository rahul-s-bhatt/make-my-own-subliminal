import streamlit as st

from modes.basic_mode import ModeWrapper
from utils.audio_engine import AudioEngine
from utils.form_util import FormUtil
from utils.subliminal_form import SubliminalForm


def main():
    try:
        st.set_page_config(
            page_title="Subliminal Generator", layout="centered")
        st.title("🧠 Subliminal Audio Generator")

        st.markdown("""Transform affirmations into **subliminal audio fields** with high-speed speech, optional background music, whisper layering, Solfeggio frequencies, and more.""")

        sub_form = SubliminalForm(ModeWrapper(), AudioEngine(), FormUtil())
        sub_form.create_form()
        sub_form.on_submit()
    except Exception as e:
        st.error(
            f"An error occurred during the execution of the Streamlit app: {e}")
        st.error("Please check the traceback in your console for more details.")


if __name__ == "__main__":
    main()
