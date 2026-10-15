from __future__ import annotations

import sys
import json
import unittest
from types import SimpleNamespace
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from generate_paraphrases import (
    answer_leak_warnings,
    build_question_set,
    validate_generated_paraphrases,
    request_paraphrases,
)


class ParaphraseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original = "Which city is the capital of France?"
        self.paraphrases = [
            "What is France's capital city?",
            "France has which city as its capital?",
            "Name the city that serves as the capital of France?",
            "Which French city is the national capital?",
        ]

    def test_exactly_four_unique_questions(self) -> None:
        self.assertEqual(
            validate_generated_paraphrases(self.original, self.paraphrases),
            self.paraphrases,
        )

    def test_duplicate_original_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_generated_paraphrases(
                self.original, [self.original, *self.paraphrases[:3]]
            )

    def test_original_is_v0_and_unchanged(self) -> None:
        fact = {
            "fact_id": "q__hash",
            "question_id": "q",
            "question": self.original,
            "answer_aliases": ["Paris"],
            "split": "validation",
        }
        row = build_question_set(
            fact,
            self.paraphrases,
            model="test-model",
            response_id="resp_test",
            usage=None,
        )
        self.assertEqual(len(row["variants"]), 5)
        self.assertEqual(row["variants"][0]["question"], self.original)
        self.assertEqual(row["variants"][0]["source"], "original")

    def test_new_answer_leak_is_a_review_warning(self) -> None:
        warnings = answer_leak_warnings(
            self.original,
            ["Is Paris the capital city of France?", *self.paraphrases[:3]],
            ["Paris"],
        )
        self.assertEqual(warnings[0]["variant_id"], "v1")
        self.assertEqual(warnings[0]["warning"], "NEW_GOLD_ALIAS_IN_QUESTION")

    def test_responses_api_request_is_structured_and_not_stored(self) -> None:
        calls = []

        class FakeResponses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    output_text=json.dumps({"paraphrases": self_paraphrases}),
                    id="resp_test",
                    usage=None,
                )

        self_paraphrases = self.paraphrases
        client = SimpleNamespace(responses=FakeResponses())
        paraphrases, response_id, usage = request_paraphrases(
            client,
            model="test-model",
            question=self.original,
            max_retries=1,
        )
        self.assertEqual(paraphrases, self.paraphrases)
        self.assertEqual(response_id, "resp_test")
        self.assertIsNone(usage)
        self.assertFalse(calls[0]["store"])
        self.assertEqual(calls[0]["text"]["format"]["type"], "json_schema")
        self.assertEqual(
            calls[0]["text"]["format"]["schema"]["properties"]
            ["paraphrases"]["items"]["pattern"],
            r"\?$",
        )
        self.assertNotIn("Paris", calls[0]["input"])

    def test_retry_includes_local_validation_feedback(self) -> None:
        calls = []

        class FakeResponses:
            def create(self, **kwargs):
                calls.append(kwargs)
                values = self_paraphrases if len(calls) > 1 else [
                    "Missing punctuation",
                    *self_paraphrases[:3],
                ]
                return SimpleNamespace(
                    output_text=json.dumps({"paraphrases": values}),
                    id="resp_test",
                    usage=None,
                )

        self_paraphrases = self.paraphrases
        client = SimpleNamespace(responses=FakeResponses())
        result, _, _ = request_paraphrases(
            client,
            model="test-model",
            question=self.original,
            max_retries=2,
        )
        self.assertEqual(result, self.paraphrases)
        self.assertEqual(len(calls), 2)
        self.assertIn("failed local validation", calls[1]["input"])


if __name__ == "__main__":
    unittest.main()
