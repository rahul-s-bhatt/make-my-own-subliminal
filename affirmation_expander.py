# affirmation_expander.py
# ==========================================
# Logic for Expanding Affirmations
# ==========================================

import logging
import random
from typing import List, Tuple

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- Expansion Components ---

# <<< REMOVED SCENARIOS LIST >>>
# SCENARIOS = [ ... ]

# Emotions remain as a way to add framing variation
EMOTIONS = [
    "grateful for",
    "joyful about",
    "excited for",
    "proud of",
    "thankful for",
    "happy about",
    "thrilled with",
    "content with",
    "serene about",
    "optimistic about",
]

# Person/Tense remains for variation
PERSON_TENSE = [
    ("I", "am"),
    ("I", "feel"),
    # ("Self", "feels") # Example - could add more perspectives
]

# --- Helper Functions ---


# <<< MODIFIED: Removed 'scenario' parameter >>>
def _format_affirmation(subject: str, verb: str, emotion: str, core: str) -> str:
    """Formats a single expanded affirmation without a specific scenario."""
    # Ensure core ends nicely before appending '.' if needed.
    core_cleaned = core.strip().rstrip(".").rstrip()
    # Format: "I am grateful for [core]."
    return f"{subject.capitalize()} {verb} {emotion} {core_cleaned}."


def _expand_affirmation_single(base: str, multiplier: int) -> List[str]:
    """Generates multiple variations for a single base affirmation."""
    base = base.strip()
    if not base:
        return []

    # Clean the base affirmation itself
    base_cleaned = base.capitalize().rstrip(".").rstrip()
    results = set()  # Use a set to store unique variations for this single base
    # Add the original base affirmation
    results.add(base_cleaned)

    attempts = 0
    # Try a bit harder to get unique variations up to the multiplier limit
    max_attempts = multiplier * 5

    # Generate variations using only Person/Tense and Emotion
    while len(results) < (multiplier + 1) and attempts < max_attempts:
        subject, verb = random.choice(PERSON_TENSE)
        emotion = random.choice(EMOTIONS)
        # <<< MODIFIED: Call updated _format_affirmation >>>
        composed = _format_affirmation(subject, verb, emotion, base_cleaned)
        results.add(composed)  # Add the formatted variation
        attempts += 1

    # Return unique variations including the original
    return list(results)


# --- Main Expansion Function ---


def expand_affirmations(
    base_text: str, max_chars: int, multiplier: int = 3
) -> Tuple[str, bool]:
    """
    Expands each line of the base text into multiple affirmations,
    ensures uniqueness, and truncates if exceeding max_chars.

    Args:
        base_text: A string containing base affirmations, potentially multi-line.
        max_chars: The maximum allowed character count for the final output string.
        multiplier: How many variations to attempt generating per base affirmation.

    Returns:
        A tuple containing:
            - The expanded affirmations as a single string (lines separated by '\n').
            - A boolean indicating if the result was truncated due to max_chars.
    """
    logger.info(
        f"Starting affirmation expansion. Multiplier: {multiplier}, Max Chars: {max_chars}"
    )
    was_truncated = False
    # Split input text into lines and filter out empty ones
    base_lines = [line.strip() for line in base_text.splitlines() if line.strip()]

    if not base_lines:
        logger.warning("No valid base affirmation lines found in input text.")
        return "", False

    all_expanded_unique = set()
    logger.debug(f"Expanding {len(base_lines)} base lines...")
    for base in base_lines:
        # Generate variations for the current base line
        expanded_for_base = _expand_affirmation_single(base, multiplier)
        # Add the generated variations to the overall set (ensures uniqueness)
        all_expanded_unique.update(expanded_for_base)

    logger.debug(f"Generated {len(all_expanded_unique)} unique affirmations initially.")

    # Convert set to list and sort for consistent truncation (optional, but good practice)
    sorted_affirmations = sorted(list(all_expanded_unique))

    # Check character limit and truncate if necessary
    final_affirmations_list = []
    current_char_count = 0
    # Account for newline character between lines when calculating length
    newline_char_count = 1

    for affirmation in sorted_affirmations:
        # Calculate length needed for this affirmation + newline
        prospective_length = len(affirmation) + newline_char_count
        # Check if adding this affirmation exceeds the limit
        if current_char_count + prospective_length <= max_chars:
            final_affirmations_list.append(affirmation)
            current_char_count += prospective_length
        else:
            # Stop adding more affirmations if the next one exceeds the limit
            was_truncated = True
            logger.warning(
                f"Character limit ({max_chars}) reached. Truncating expanded affirmations."
            )
            break  # Exit the loop

    # Adjust final count: the very last item doesn't have a trailing newline
    if final_affirmations_list:
        current_char_count -= newline_char_count

    # Join the selected affirmations into a single string
    final_text = "\n".join(final_affirmations_list)

    logger.info(
        f"Expansion complete. Final affirmations: {len(final_affirmations_list)}, Chars: {len(final_text)}, Truncated: {was_truncated}"
    )
    return final_text, was_truncated


# --- Grammar Correction (Placeholder for Future Enhancement) ---
# def expand_and_correct_affirmations(...)
#   # ... expansion logic ...
#   if correct_grammar and language_tool_is_available:
#       # ... call grammar correction (synchronously for now) ...
#   # ... limit check ...
#   return final_text, was_truncated
