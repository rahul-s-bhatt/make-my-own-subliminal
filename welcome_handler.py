# welcome_handler.py
# ==========================================
# Welcome Message Display Logic for MindMorph
# ==========================================

import logging

import streamlit as st

logger = logging.getLogger(__name__)


def display_welcome_message():
    """Displays the initial welcome message and mode explanation."""
    # Check if the message has already been shown/dismissed
    if "welcome_message_shown" not in st.session_state:
        with st.container(border=True):
            st.markdown("### 👋 Welcome to MindMorph!")
            st.markdown("Create custom subliminal audio by layering sounds and applying effects.")
            st.markdown("---")
            st.markdown("#### ✨ Choose Your Experience:")
            st.markdown("Use the **'Select Editor Mode'** option at the top of the main panel:")
            st.markdown("- **Easy Mode:** Simplified interface, perfect for getting started quickly.")
            st.markdown("- **Advanced Mode:** Access all features like detailed frequency generation and audio effects.")
            st.markdown("*(You can switch modes any time!)*")
            st.markdown("---")
            st.markdown("#### Quick Start Workflow:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("##### 1. Add Tracks ➕")
                st.markdown("Use the **sidebar** (👈).")
                st.caption("Upload, TTS, Noise, Freq.")
            with col2:
                st.markdown("##### 2. Edit Tracks 🎚️")
                st.markdown("Adjust **settings** below.")
                st.caption("Click 'Update Preview'!")
            with col3:
                st.markdown("##### 3. Mix & Export 🔊")
                st.markdown("Use **master controls** (bottom).")
                st.caption("Preview or Download")
            st.markdown("---")
            st.markdown("*(Click button below to hide this guide. Find details in Instructions at page bottom.)*")
            # Center the button
            button_cols = st.columns([1, 1.5, 1])  # Adjust ratios as needed
            with button_cols[1]:
                if st.button(
                    "Got it! Let's Start Creating ✨",
                    key="dismiss_welcome_button",  # Unique key
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.welcome_message_shown = True
                    logger.info("Welcome message dismissed by user.")
                    st.rerun()  # Rerun to hide the message immediately
        # Return True if the message was shown (so main loop knows to maybe skip rest of UI)
        return True
    # Return False if the message was already dismissed
    return False
