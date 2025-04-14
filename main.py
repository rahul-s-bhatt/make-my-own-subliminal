import streamlit as st

from utils.subliminal_form import SubliminalForm

st.set_page_config(page_title="Subliminal Generator", layout="centered")
st.title("🧠 Subliminal Audio Generator")

st.markdown("""Transform affirmations into **subliminal audio fields** with high-speed speech, optional background music, whisper layering, Solfeggio frequencies, and more.""")

sub_form = SubliminalForm()
sub_form.create_form()
sub_form.on_submit()