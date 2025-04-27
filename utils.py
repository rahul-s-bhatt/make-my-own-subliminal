# utils.py
# ==========================================
# General Utility Functions for MindMorph
# ==========================================

import logging
from typing import Optional

import docx  # Required for reading .docx files
import streamlit as st  # TODO: Remove direct Streamlit UI calls
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Get a logger for this module
logger = logging.getLogger(__name__)


def read_text_file(uploaded_file: UploadedFile) -> Optional[str]:
    """
    Reads content from uploaded txt or docx file.

    Handles potential decoding errors for txt and parsing errors for docx.

    Args:
        uploaded_file: The UploadedFile object from Streamlit.

    Returns:
        The text content as a string, or None if reading fails or the
        file type is unsupported.
    """
    if uploaded_file is None:
        logger.error("read_text_file called with None object.")
        return None

    file_name = uploaded_file.name
    file_type = uploaded_file.type
    logger.info(f"Attempting to read file: {file_name}, Type: {file_type}")

    try:
        # --- Read .txt file ---
        if file_name.lower().endswith(".txt") or file_type == "text/plain":
            logger.debug(f"Reading .txt file: {file_name}")
            try:
                # Read bytes and decode with UTF-8, replacing errors
                file_content_bytes = uploaded_file.getvalue()
                return file_content_bytes.decode("utf-8", errors="replace")
            except Exception as e_decode:
                logger.exception(f"Error decoding .txt file {file_name} as UTF-8.")
                # TODO: Remove direct Streamlit call. Raise exception or return status.
                st.error(f"Error decoding text file '{file_name}': {e_decode}. Ensure it's UTF-8 encoded.")
                return None

        # --- Read .docx file ---
        elif file_name.lower().endswith(".docx") or file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            logger.debug(f"Processing .docx file: {file_name}")
            try:
                # Need to operate on the file-like object directly
                uploaded_file.seek(0)  # Ensure reading from the start
                document = docx.Document(uploaded_file)
                logger.debug(f"docx document parsed. Found {len(document.paragraphs)} paragraphs.")

                # Extract text from paragraphs
                para_texts = []
                for i, para in enumerate(document.paragraphs):
                    if para is None:
                        logger.warning(f"Paragraph {i} in '{file_name}' is None. Skipping.")
                        continue
                    # Ensure paragraph text is not None before stripping/appending
                    para_text = getattr(para, "text", "")
                    if para_text and para_text.strip():  # Only add non-empty paragraphs
                        para_texts.append(para_text.strip())

                logger.debug(f"Extracted {len(para_texts)} non-empty paragraphs from '{file_name}'.")
                return "\n".join(para_texts)  # Join paragraphs with newlines

            except Exception as e_docx:
                logger.exception(f"Error processing .docx file: {file_name}")
                # TODO: Remove direct Streamlit call. Raise exception or return status.
                st.error(f"Error reading Word document '{file_name}': {e_docx}")
                return None

        # --- Unsupported file type ---
        else:
            logger.error(f"Unsupported file type for text reading: {file_type} (Filename: {file_name})")
            # TODO: Remove direct Streamlit call. Raise exception or return status.
            st.error(f"Unsupported file type: '{file_type}'. Please upload a .txt or .docx file.")
            return None

    except Exception as e_outer:
        # Catch any other unexpected errors during file handling
        logger.exception(f"An unexpected error occurred while reading file: {file_name}")
        # TODO: Remove direct Streamlit call. Raise exception or return status.
        st.error(f"Error reading file '{file_name}': {e_outer}")
        return None


# Potential future additions:
# - Logging setup function
# - Function to sanitize filenames
# - etc.
