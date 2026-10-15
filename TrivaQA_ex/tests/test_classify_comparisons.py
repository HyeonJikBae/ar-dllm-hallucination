from __future__ import annotations

import unittest
import sys
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from classify_comparisons import knowledge_status, paired_output_category


class ComparisonClassificationTest(unittest.TestCase):
    def test_requested_knowledge_status_boundaries(self) -> None:
        self.assertEqual(knowledge_status(0), "NO_KNOWLEDGE_OBSERVED")
        self.assertEqual(knowledge_status(1), "SINGLE_HIT_CHANCE_CANDIDATE")
        for count in (2, 3, 4):
            self.assertEqual(
                knowledge_status(count), "QUESTION_SENSITIVE_KNOWLEDGE"
            )
        self.assertEqual(knowledge_status(5), "CONSISTENT_KNOWLEDGE")

    def test_all_six_pair_categories(self) -> None:
        cases = [
            ((False, False, "x", "x"), "BOTH_WRONG_SAME_FORM"),
            ((False, False, "x", "y"), "BOTH_WRONG_DIFFERENT_FORM"),
            ((True, False, "x", "y"), "DREAM_CORRECT_QWEN_WRONG"),
            ((False, True, "x", "y"), "DREAM_WRONG_QWEN_CORRECT"),
            ((True, True, "x", "x"), "BOTH_CORRECT_SAME_FORM"),
            ((True, True, "x", "y"), "BOTH_CORRECT_DIFFERENT_FORM"),
        ]
        for arguments, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(paired_output_category(*arguments)[0], expected)


if __name__ == "__main__":
    unittest.main()
