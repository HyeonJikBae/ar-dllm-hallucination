from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from experiment_utils import make_fact_id
from prepare_triviaqa import (
    is_unaided_recall_scoreable,
    load_local_facts,
    select_facts,
)


class PrepareTriviaQATest(unittest.TestCase):
    def test_fact_with_only_question_exposed_alias_is_not_scoreable(self) -> None:
        self.assertFalse(
            is_unaided_recall_scoreable(
                {
                    "question": "Was James II known as Fiery Face?",
                    "answer_aliases": ["Fiery Face"],
                }
            )
        )

    def test_duplicate_question_id_gets_distinct_fact_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch_path = Path(directory) / "batch_0.json"
            rows = [
                {
                    "question_id": "bb_956",
                    "question": "What happened in June?",
                    "correct_answer": ["Alpha"],
                    "split": "validation",
                    "greedy_completion": "ignored",
                },
                {
                    "question_id": "bb_956",
                    "question": "What happened in November?",
                    "correct_answer": ["Beta", "BETA"],
                    "split": "validation",
                },
            ]
            batch_path.write_text(json.dumps(rows), encoding="utf-8")
            facts, paths = load_local_facts(Path(directory))

        self.assertEqual(len(paths), 1)
        self.assertEqual(len(facts), 2)
        self.assertNotEqual(facts[0]["fact_id"], facts[1]["fact_id"])
        self.assertEqual(
            facts[0]["fact_id"], make_fact_id("bb_956", "What happened in June?")
        )
        self.assertNotIn("greedy_completion", facts[0])

    def test_sampling_is_reproducible(self) -> None:
        facts = [{"fact_id": str(index)} for index in range(20)]
        first = select_facts(facts, 5, 42)
        second = select_facts(facts, 5, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)


if __name__ == "__main__":
    unittest.main()
