# auto_subliminal/affirmation_generator.py
# This module is responsible for generating affirmations based on a given topic,
# using the rule-based methods from affirmation_expander.py.

import logging

# Assuming affirmation_expander.py is at the project root or accessible in PYTHONPATH
# If auto_subliminal is a sub-package, adjust the import path accordingly.
# try:
from affirmation_expander import expand_affirmations, generate_affirmations_from_topic_keywords

# except ImportError as e:
#     logging.error(f"Failed to import from 'affirmation_expander'. Ensure it's in PYTHONPATH. Error: {e}")
#     # Define fallback functions if import fails, so the rest of the app can be tested (with warnings)
#     def generate_affirmations_from_topic_keywords(topic: str, num_target_affirmations: int = 5, **kwargs) -> list[str]:
#         logging.warning("Using fallback 'generate_affirmations_from_topic_keywords'.")
#         return [f"Fallback: Positive affirmation about {topic} #{i + 1}" for i in range(num_target_affirmations)]
#     def expand_affirmations(base_text: str, max_chars: int, multiplier: int = 1, **kwargs) -> tuple[str, bool]:
#         logging.warning("Using fallback 'expand_affirmations'.")
#         affirm_list = [line.strip() for line in base_text.splitlines() if line.strip()]
#         final_text = "\n".join(affirm_list)
#         truncated = False
#         if len(final_text) > max_chars:
#             final_text = final_text[:max_chars]
#             truncated = True
#         return final_text, truncated
# Import from main application config for global limits
from config import MAX_AFFIRMATION_CHARS

# Import from feature-specific config for feature-specific parameters
from .config import AUTO_SUB_DEFAULT_INITIAL_AFFIRMATIONS, AUTO_SUB_MAIN_EXPANSION_MULTIPLIER

logger = logging.getLogger(__name__)


class AffirmationGenerator:
    """
    Generates and expands affirmations for a given topic using rule-based methods.
    """

    def __init__(self):
        """
        Initializes the AffirmationGenerator for rule-based generation.
        No external LLM client is needed.
        """
        logger.info("AffirmationGenerator (rule-based) initialized.")

    def generate_and_expand_affirmations(
        self,
        topic: str,
        num_initial_affirmations: int = AUTO_SUB_DEFAULT_INITIAL_AFFIRMATIONS,
        main_expansion_multiplier: int = AUTO_SUB_MAIN_EXPANSION_MULTIPLIER,
        max_total_chars: int = MAX_AFFIRMATION_CHARS,
    ) -> tuple[list[str], bool]:
        """
        Generates initial affirmations from a topic using rule-based methods,
        then expands them further.

        Args:
            topic (str): The topic for which to generate affirmations.
            num_initial_affirmations (int): Target number of affirmations to generate from the topic keywords.
            main_expansion_multiplier (int): Multiplier for the subsequent expansion step.
            max_total_chars (int): Overall character limit for the final expanded text block.

        Returns:
            tuple[list[str], bool]:
                - A list of final, expanded affirmation strings.
                - A boolean indicating if the result was truncated due to max_chars.
        """
        if not topic or not topic.strip():
            logger.warning("Topic is empty. Cannot generate affirmations.")
            return [], False

        # 1. Generate initial set of affirmations from the topic using rule-based keyword generator
        logger.info(f"Generating initial set of {num_initial_affirmations} affirmations for topic: '{topic}' using rule-based generator.")
        try:
            initial_affirmations_list = generate_affirmations_from_topic_keywords(
                topic=topic,
                num_target_affirmations=num_initial_affirmations,
                # max_variations_per_template is an internal param of generate_affirmations_from_topic_keywords
            )
        except Exception as e:
            logger.error(f"Error in 'generate_affirmations_from_topic_keywords': {e}", exc_info=True)
            initial_affirmations_list = []

        if not initial_affirmations_list:
            logger.warning(f"No initial affirmations could be generated for topic: '{topic}' using rules. Creating a generic fallback.")
            # Fallback: create a very simple affirmation if all else fails
            initial_affirmations_list = [f"I am experiencing positive changes regarding {topic.capitalize()}."]
            if len(topic.split()) > 1:  # If topic is a phrase
                initial_affirmations_list.append(f"My life is improving in the area of {topic.capitalize()}.")

        logger.info(f"Generated {len(initial_affirmations_list)} initial affirmations. Now performing main expansion...")
        base_text_for_main_expansion = "\n".join(initial_affirmations_list)

        # 2. Expand these initial affirmations further using the main expand_affirmations function
        #    from affirmation_expander.py
        try:
            expanded_text_str, was_truncated = expand_affirmations(base_text=base_text_for_main_expansion, max_chars=max_total_chars, multiplier=main_expansion_multiplier)
        except Exception as e:
            logger.error(f"Error in 'expand_affirmations': {e}", exc_info=True)
            # Fallback to using initial affirmations if expansion fails
            expanded_text_str = base_text_for_main_expansion
            was_truncated = False
            if len(expanded_text_str) > max_total_chars:
                expanded_text_str = expanded_text_str[:max_total_chars]
                was_truncated = True

        final_affirmations_list = [line.strip() for line in expanded_text_str.splitlines() if line.strip()]

        logger.info(f"Rule-based generation and expansion complete. Total final affirmations: {len(final_affirmations_list)}. Truncated: {was_truncated}")
        return final_affirmations_list, was_truncated


if __name__ == "__main__":
    # Setup basic logging for direct script execution testing
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create an instance of the generator
    affirmation_gen = AffirmationGenerator()

    # Test case 1: Simple topic
    test_topic_1 = "self confidence"
    print(f"\n--- Testing with topic: '{test_topic_1}' ---")
    affirmations1, truncated1 = affirmation_gen.generate_and_expand_affirmations(test_topic_1)
    if affirmations1:
        print(f"Generated {len(affirmations1)} affirmations (Truncated: {truncated1}):")
        for i, aff in enumerate(affirmations1[:5]):  # Print first 5
            print(f"  {i + 1}. {aff}")
        # print("\nFull list for topic 1:\n" + "\n".join(affirmations1))
    else:
        print("No affirmations generated.")

    # Test case 2: More complex topic
    test_topic_2 = "attracting positive relationships and deep connections"
    print(f"\n--- Testing with topic: '{test_topic_2}' ---")
    affirmations2, truncated2 = affirmation_gen.generate_and_expand_affirmations(
        test_topic_2,
        num_initial_affirmations=8,  # Request more initial ones
        main_expansion_multiplier=1,  # Less aggressive final expansion
    )
    if affirmations2:
        print(f"Generated {len(affirmations2)} affirmations (Truncated: {truncated2}):")
        for i, aff in enumerate(affirmations2[:5]):  # Print first 5
            print(f"  {i + 1}. {aff}")
    else:
        print("No affirmations generated.")

    # Test case 3: Short character limit to test truncation
    test_topic_3 = "achieving goals"
    print(f"\n--- Testing with topic: '{test_topic_3}' and short max_chars ---")
    affirmations3, truncated3 = affirmation_gen.generate_and_expand_affirmations(
        test_topic_3,
        max_total_chars=200,  # Very short limit
    )
    if affirmations3:
        print(f"Generated {len(affirmations3)} affirmations (Truncated: {truncated3}):")
        # for aff in affirmations3: print(f"  - {aff}")
        print("Full text (first 250 chars):\n" + "\n".join(affirmations3)[:250] + "...")
        joined_affirmations_long = "\n".join(affirmations3)
        print(f"Total characters in generated string: {len(joined_affirmations_long)}")

    else:
        print("No affirmations generated.")

    # Test case 4: Empty topic
    print(f"\n--- Testing with empty topic ---")
    affirmations4, truncated4 = affirmation_gen.generate_and_expand_affirmations("")
    print(f"Generated {len(affirmations4)} affirmations (Truncated: {truncated4}) for empty topic.")
