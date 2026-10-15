from __future__ import annotations

import sys
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from scoring import (
    detect_refusal,
    expand_gold_aliases,
    fact_class,
    gold_answer_variants,
    normalize_answer,
    score_response,
)


class ScoringTest(unittest.TestCase):
    def test_normalization(self) -> None:
        self.assertEqual(normalize_answer("  The South-Carolina. "), "south carolina")

    def test_all_aliases_are_used(self) -> None:
        result = score_response("Answer: The Palmetto State.", ["South Carolina", "Palmetto State"])
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["matched_alias"], "Palmetto State")

    def test_alias_inside_long_contradiction_is_not_correct(self) -> None:
        result = score_response("Paris, not London", ["Paris"])
        self.assertFalse(result["is_correct"])
        self.assertTrue(result["diagnostic_alias_present"])

    def test_verbose_answer_uses_safe_alias_span(self) -> None:
        result = score_response(
            "Paris is the capital and largest city of France.",
            ["Paris"],
            question="What is the capital of France?",
        )
        self.assertTrue(result["is_correct"])
        self.assertFalse(result["strict_is_correct"])
        self.assertEqual(result["match_type"], "safe_alias_span")

    def test_regular_plural_alias_span_is_correct(self) -> None:
        result = score_response(
            "Kniphofia is commonly known as red-hot pokers or torch lilies.",
            ["Red Hot Poker", "Torch Lily", "Kniphofia"],
            question="What is the common name of the flowering plant Kniphofia?",
        )
        self.assertTrue(result["is_correct"])
        self.assertFalse(result["strict_is_correct"])
        self.assertEqual(result["match_type"], "safe_regular_plural_span")
        self.assertEqual(result["matched_alias"], "Red Hot Poker")

    def test_alias_already_in_question_is_not_promoted(self) -> None:
        result = score_response(
            "The question mentions Paris, but gives no answer.",
            ["Paris", "The Seine"],
            question="Which river flows through Paris?",
        )
        self.assertFalse(result["is_correct"])

    def test_exact_echo_of_question_exposed_alias_is_not_correct(self) -> None:
        result = score_response(
            "Kniphofia",
            ["Kniphofia", "Red Hot Poker"],
            question="What is the common name of the flowering plant Kniphofia?",
        )
        self.assertFalse(result["is_correct"])
        self.assertEqual(result["question_exposed_aliases"], ["Kniphofia"])

    def test_item_with_every_alias_exposed_is_unscoreable(self) -> None:
        with self.assertRaisesRegex(ValueError, "Every normalized gold alias"):
            score_response(
                "Fiery Face",
                ["Fiery Face"],
                question="Was James II known as Fiery Face?",
            )

    def test_denial_with_alias_is_refusal_and_wrong(self) -> None:
        result = score_response(
            "There is no evidence that Jimi Hendrix did this.",
            ["Jimi Hendrix"],
            question="Who did this?",
        )
        self.assertFalse(result["is_correct"])
        self.assertTrue(result["is_refusal"])
        self.assertEqual(result["refusal_rule"], "no_evidence")

    def test_refusal_overrides_an_alias(self) -> None:
        result = score_response("I cannot answer; perhaps Paris", ["Paris"])
        self.assertFalse(result["is_correct"])
        self.assertTrue(result["is_refusal"])
        self.assertEqual(result["refusal_rule"], "cannot_answer")

    def test_broad_words_are_not_refusals(self) -> None:
        self.assertEqual(detect_refusal("Unknown Pleasures"), (False, None))
        self.assertEqual(detect_refusal("Sorry"), (False, None))

    def test_honorific_and_parenthetical_gold_accepts_bare_surname(self) -> None:
        for output in ("Parkinson", "James Parkinson", "Dr. James Parkinson"):
            with self.subTest(output=output):
                result = score_response(output, ["Dr. (James) Parkinson"])
                self.assertTrue(result["is_correct"])

    def test_derived_gold_match_is_flagged_for_audit(self) -> None:
        result = score_response("Parkinson", ["Dr. (James) Parkinson"])
        self.assertTrue(result["matched_derived_gold"])
        self.assertIn("Parkinson", result["derived_gold_variants"])

    def test_supplied_gold_match_is_not_flagged_as_derived(self) -> None:
        result = score_response("Dr. (James) Parkinson", ["Dr. (James) Parkinson"])
        self.assertTrue(result["is_correct"])
        self.assertFalse(result["matched_derived_gold"])

    def test_leading_preposition_gold_accepts_bare_noun(self) -> None:
        result = score_response("board", ["On a board"])
        self.assertTrue(result["is_correct"])

    def test_gold_expansion_does_not_accept_a_different_answer(self) -> None:
        result = score_response("Egypt", ["On a board"])
        self.assertFalse(result["is_correct"])

    def test_single_word_gold_is_not_expanded(self) -> None:
        self.assertEqual(gold_answer_variants("Alaska"), ["Alaska"])

    def test_expansion_never_drops_a_supplied_alias(self) -> None:
        supplied = ["Dr. (James) Parkinson", "Parkinson's disease"]
        expanded, derived = expand_gold_aliases(supplied)
        for alias in supplied:
            self.assertIn(alias, expanded)
        self.assertNotIn("Dr. (James) Parkinson", derived)

    def test_expansion_can_be_disabled(self) -> None:
        result = score_response(
            "Parkinson", ["Dr. (James) Parkinson"], expand_gold=False
        )
        self.assertFalse(result["is_correct"])

    def test_every_fact_class(self) -> None:
        expected = [
            "NEVER_RECALLED",
            "PROMPT_SENSITIVE_1",
            "PROMPT_SENSITIVE_2",
            "PROMPT_SENSITIVE_3",
            "PROMPT_SENSITIVE_4",
            "STABLE_RECALL",
        ]
        self.assertEqual([fact_class(index) for index in range(6)], expected)


if __name__ == "__main__":
    unittest.main()
