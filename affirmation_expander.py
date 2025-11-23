# affirmation_expander.py
# ==========================================
# Logic for Generating and Expanding Affirmations using Rule-Based Methods
# ==========================================

import logging
import random
import re  # For keyword extraction
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)

# --- Linguistic Components for Affirmation Generation & Expansion ---

# (Existing components)
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
    "confident in my ability to experience",
    "open to receiving",
    "worthy of",
]

PERSON_TENSE_SUBJECT_VERB = [
    ("I", "am"),
    ("I", "feel"),
    ("I", "choose to be"),
    ("I", "embrace being"),
    ("My mind", "is"),
    ("My reality", "is"),
    ("I", "attract"),
    ("I", "create"),
    ("I", "welcome"),
    ("I", "allow"),
    ("I", "easily"),
    ("I", "naturally"),
]

# (New components)
ACTION_VERBS_PREFIX = [
    "achieving",
    "embracing",
    "cultivating",
    "welcoming",
    "manifesting",
    "experiencing",
    "strengthening my ability for",
    "developing",
    "mastering",
    "attracting more",
    "creating a reality of",
    "living a life filled with",
    "easily and effortlessly stepping into",
]

ACTION_VERBS_SUFFIX = [  # Can be appended to the core affirmation if it makes sense
    "in my life",
    "with ease and grace",
    "effortlessly",
    "abundantly",
    "more and more each day",
    "in wonderful ways",
]

TIMEFRAMES_ADVERBS = [
    "now",
    "every day",
    "consistently",
    "more and more",
    "with each passing moment",
    "starting today",
    "in this present moment",
    "continuously",
    "increasingly",
    "from this moment forward",
]

INTENSIFIERS = [
    "deeply",
    "fully",
    "completely",
    "truly",
    "absolutely",
    "perfectly",
    "profoundly",
    "unconditionally",
    "vibrantly",
]

PERSPECTIVES = [
    "I know that",
    "I believe that",
    "I understand that",
    "It is my truth that",
    "My subconscious mind accepts that",
    "I am certain that",
]

# --- Helper Functions ---


def _clean_affirmation_text(text: str) -> str:
    """Cleans and standardizes affirmation text."""
    text = text.strip()
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text.capitalize()


def _format_affirmation_complex(
    base_core: str,
    perspective: str = "",
    subject: str = "I",
    verb: str = "am",
    intensifier: str = "",
    action_prefix: str = "",
    emotion: str = "",
    timeframe: str = "",
    action_suffix: str = "",
) -> str:
    """
    Formats a single affirmation using various components.
    Tries to build a grammatically sensible sentence.
    """
    parts = []

    if perspective:
        parts.append(perspective.capitalize())

    # Subject-Verb start or Emotion-based start
    if emotion and not action_prefix:  # "I am grateful for [core]"
        parts.append(subject.capitalize())
        parts.append(verb)
        if intensifier:
            parts.append(intensifier)
        parts.append(emotion)
        parts.append(base_core)
    elif action_prefix:  # "I am achieving [core]" or "[Perspective] I am achieving [core]"
        if not perspective:  # Add subject if no perspective already did
            parts.append(subject.capitalize())
            parts.append(verb)  # e.g. I am achieving...
        else:  # Perspective might imply subject, or we add it. For "I know that I am achieving..."
            if subject.lower() not in perspective.lower():  # Avoid "I know that I I am..."
                parts.append(subject)
                parts.append(verb)

        if intensifier:
            parts.append(intensifier)
        parts.append(action_prefix)
        parts.append(base_core)
    else:  # Simple "I am [core]"
        parts.append(subject.capitalize())
        parts.append(verb)
        if intensifier:
            parts.append(intensifier)
        parts.append(base_core)

    if timeframe:
        # Decide placement: usually at end, or start if it's a phrase like "Starting today"
        if timeframe.lower().startswith(("starting", "from this moment")):
            parts.insert(0, timeframe.capitalize())
        else:
            parts.append(timeframe)

    if action_suffix:
        parts.append(action_suffix)

    # Join parts and clean up potential double spaces, then capitalize and punctuate.
    constructed_affirmation = " ".join(part for part in parts if part)  # Filter out empty strings
    # Basic cleanup for multiple spaces that might have formed
    constructed_affirmation = re.sub(r"\s+", " ", constructed_affirmation).strip()
    return _clean_affirmation_text(constructed_affirmation)


def _expand_affirmation_single(base_affirmation: str, multiplier: int) -> Set[str]:
    """
    Generates multiple variations for a single base affirmation using enhanced components.
    """
    base_affirmation = base_affirmation.strip().rstrip(".").strip()
    if not base_affirmation:
        return set()

    variations: Set[str] = set()
    variations.add(_clean_affirmation_text(base_affirmation))  # Add the original

    # Determine if the base affirmation is a full sentence or just a core concept
    # This is a heuristic. A more robust way would involve NLP part-of-speech tagging.
    is_full_sentence_heuristic = len(base_affirmation.split()) > 3 or any(
        base_affirmation.lower().startswith(s_v[0].lower() + " " + s_v[1].lower()) for s_v in PERSON_TENSE_SUBJECT_VERB
    )

    for _ in range(multiplier * 5):  # More attempts to get unique variations
        if len(variations) >= multiplier + 1:  # +1 for the original
            break

        core_concept = base_affirmation  # Assume base is the core for now

        # Randomly select components
        chosen_perspective = random.choice([""] + PERSPECTIVES)  # Chance of no perspective
        subj, vb = random.choice(PERSON_TENSE_SUBJECT_VERB)
        chosen_intensifier = random.choice([""] + INTENSIFIERS)
        chosen_action_prefix = ""
        chosen_emotion = ""

        # Decide if we use action_prefix or emotion, try not to use both if it sounds awkward
        if random.random() < 0.6 and ACTION_VERBS_PREFIX:  # 60% chance to use an action prefix
            chosen_action_prefix = random.choice(ACTION_VERBS_PREFIX)
        elif EMOTIONS:  # else, try emotion
            chosen_emotion = random.choice(EMOTIONS)

        chosen_timeframe = random.choice([""] + TIMEFRAMES_ADVERBS)
        chosen_action_suffix = random.choice([""] + ACTION_VERBS_SUFFIX)

        # If the base affirmation seems like a full sentence, we might try to embed it
        # or use it as a core for simpler modifications.
        # If it's just a keyword (e.g., "wealth"), we build around it more heavily.

        # This logic can be quite complex to make it always sound natural.
        # For now, we treat the base_affirmation as the main object of the sentence.

        new_affirmation = _format_affirmation_complex(
            base_core=core_concept,
            perspective=chosen_perspective,
            subject=subj,
            verb=vb,
            intensifier=chosen_intensifier,
            action_prefix=chosen_action_prefix,
            emotion=chosen_emotion,
            timeframe=chosen_timeframe,
            action_suffix=chosen_action_suffix,
        )
        variations.add(new_affirmation)

    return variations


def generate_affirmations_from_topic_keywords(topic: str, num_target_affirmations: int = 10, max_variations_per_template: int = 2) -> List[str]:
    """
    Generates a set of base affirmations from a topic string using keyword extraction
    and predefined templates. This is a rule-based alternative to LLM for initial generation.
    """
    topic = topic.lower().strip()
    if not topic:
        return []

    # Simple keyword extraction (can be improved with NLP libraries for better results)
    # For now, just use the topic as a whole or split it if it's a phrase.
    keywords = [kw.strip() for kw in topic.split() if len(kw.strip()) > 2]
    if not keywords:  # If topic was very short or only small words
        keywords = [topic]

    main_keyword = topic  # Use the full topic as the primary keyword for some templates

    generated: Set[str] = set()

    # Templates that use the main keyword (full topic phrase)
    topic_templates = [
        lambda kw: f"I am embracing {kw}",
        lambda kw: f"My life is filled with {kw}",
        lambda kw: f"{kw} flows to me easily and abundantly",
        lambda kw: f"I am a magnet for {kw}",
        lambda kw: f"I welcome more {kw} into my experience",
        lambda kw: f"Every day, my connection to {kw} strengthens",
        lambda kw: f"I am worthy of experiencing true {kw}",
        lambda kw: f"My reality is one of {kw}",
        lambda kw: f"I choose to focus on {kw}",
        lambda kw: f"I am {random.choice(INTENSIFIERS)} grateful for the {kw} in my life",
    ]

    for template_func in random.sample(topic_templates, min(len(topic_templates), num_target_affirmations)):
        affirmation = template_func(main_keyword)
        generated.add(_clean_affirmation_text(affirmation))
        if len(generated) >= num_target_affirmations:
            break

    # If still need more, try to use individual keywords with more generic templates
    # This part can be expanded significantly
    if len(generated) < num_target_affirmations and len(keywords) > 1:
        for keyword in random.sample(keywords, min(len(keywords), num_target_affirmations - len(generated))):
            # Use the _expand_affirmation_single with the keyword as a base, take a few results
            expanded_from_keyword = list(_expand_affirmation_single(keyword, max_variations_per_template))
            for aff in expanded_from_keyword:
                generated.add(aff)
                if len(generated) >= num_target_affirmations:
                    break
            if len(generated) >= num_target_affirmations:
                break

    # Fallback if very few generated
    if not generated and topic:
        generated.add(_clean_affirmation_text(f"I am open to positive experiences regarding {topic}"))
        generated.add(_clean_affirmation_text(f"{topic} is a positive part of my life"))

    return sorted(list(generated))[:num_target_affirmations]


def expand_affirmations(base_text: str, max_chars: int, multiplier: int = 3) -> Tuple[str, bool]:
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
    logger.info(f"Starting affirmation expansion. Multiplier: {multiplier}, Max Chars: {max_chars}")
    was_truncated = False
    base_lines = [line.strip() for line in base_text.splitlines() if line.strip()]

    if not base_lines:
        logger.warning("No valid base affirmation lines found in input text for expansion.")
        return "", False

    all_expanded_unique: Set[str] = set()
    logger.debug(f"Expanding {len(base_lines)} base lines...")

    for base in base_lines:
        expanded_for_base = _expand_affirmation_single(base, multiplier)
        all_expanded_unique.update(expanded_for_base)

    logger.debug(f"Generated {len(all_expanded_unique)} unique affirmations initially from expansion.")

    sorted_affirmations = sorted(list(all_expanded_unique))
    final_affirmations_list = []
    current_char_count = 0
    newline_char_count = 1  # For the '\n' character

    for affirmation in sorted_affirmations:
        prospective_length = len(affirmation) + newline_char_count
        if current_char_count + prospective_length <= max_chars:
            final_affirmations_list.append(affirmation)
            current_char_count += prospective_length
        else:
            was_truncated = True
            logger.warning(f"Character limit ({max_chars}) reached. Truncating expanded affirmations.")
            break

    if final_affirmations_list:  # Remove the last newline count if list is not empty
        current_char_count -= newline_char_count

    final_text = "\n".join(final_affirmations_list)
    logger.info(f"Expansion complete. Final affirmations: {len(final_affirmations_list)}, Chars: {len(final_text)} (calculated: {current_char_count}), Truncated: {was_truncated}")
    return final_text, was_truncated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    print("\n--- Testing generate_affirmations_from_topic_keywords ---")
    topic1 = "financial freedom"
    topic_generated_affirmations = generate_affirmations_from_topic_keywords(topic1, num_target_affirmations=7)
    print(f"\nAffirmations for topic '{topic1}':")
    for aff in topic_generated_affirmations:
        print(f"- {aff}")

    topic2 = "health"
    topic_generated_affirmations_2 = generate_affirmations_from_topic_keywords(topic2, num_target_affirmations=5)
    print(f"\nAffirmations for topic '{topic2}':")
    for aff in topic_generated_affirmations_2:
        print(f"- {aff}")

    print("\n--- Testing expand_affirmations with pre-defined base text ---")
    sample_base_text = "I am confident.\nI attract success.\nMy mind is clear and focused"
    expanded_output, truncated_flag = expand_affirmations(sample_base_text, max_chars=500, multiplier=2)
    print(f"\nExpanded Output (Truncated: {truncated_flag}):")
    print(expanded_output)
    print(f"Character count: {len(expanded_output)}")

    print("\n--- Testing expand_affirmations with topic-generated affirmations as base ---")
    base_for_expansion = "\n".join(topic_generated_affirmations)  # Use affirmations from topic1
    if base_for_expansion:
        further_expanded_output, further_truncated_flag = expand_affirmations(base_for_expansion, max_chars=1000, multiplier=1)
        print(f"\nFurther Expanded Output (From topic '{topic1}', Truncated: {further_truncated_flag}):")
        print(further_expanded_output)
        print(f"Character count: {len(further_expanded_output)}")
    else:
        print(f"\nSkipping further expansion as no base affirmations were generated for topic '{topic1}'.")

    print("\n--- Testing truncation ---")
    short_limit_output, short_truncated_flag = expand_affirmations(sample_base_text, max_chars=100, multiplier=3)
    print(f"\nShort Limit Expansion (Truncated: {short_truncated_flag}):")
    print(short_limit_output)
    print(f"Character count: {len(short_limit_output)}")
