from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from experiment_utils import write_jsonl
from score_results import analyze


def question_set() -> dict:
    questions = [
        "What is the capital of France?",
        "Which city is France's capital?",
        "France has which capital city?",
        "Name France's national capital?",
        "What city serves as the French capital?",
    ]
    return {
        "fact_id": "q1__hash",
        "question_id": "q1",
        "question": questions[0],
        "canonical_answer": "Paris",
        "answer_aliases": ["Paris", "City of Light"],
        "variants": [
            {
                "variant_id": f"v{index}",
                "variant_index": index,
                "question": question,
                "source": "original" if index == 0 else "openai_paraphrase",
            }
            for index, question in enumerate(questions)
        ],
    }


def cache_rows(model: str, outputs: list[str]) -> list[dict]:
    fact = question_set()
    return [
        {
            "model_name": model,
            "model_id": f"test/{model}",
            "prompt_mode": "plain",
            "fact_id": fact["fact_id"],
            "variant_id": variant["variant_id"],
            "question": variant["question"],
            "raw_output": output,
        }
        for variant, output in zip(fact["variants"], outputs)
    ]


class ScoreResultsTest(unittest.TestCase):
    def test_fact_bins_cross_model_and_refusal_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.jsonl"
            cache_dir = root / "cache"
            output_dir = root / "analysis"
            write_jsonl(questions, [question_set()])
            write_jsonl(
                cache_dir / "dream_generations.jsonl",
                cache_rows("dream", ["Paris", "London", "Paris", "I don't know", "Lyon"]),
            )
            write_jsonl(
                cache_dir / "qwen_generations.jsonl",
                cache_rows("qwen", ["Paris"] * 5),
            )

            summary = analyze(
                questions_path=questions,
                cache_dir=cache_dir,
                output_dir=output_dir,
                expected_models=["dream", "qwen"],
                limit=0,
                require_complete=True,
            )

            self.assertEqual(
                summary["models"]["dream"]["fact_bin_counts"]["2/5"], 1
            )
            self.assertEqual(
                summary["models"]["dream"]["refusal_output_count"], 1
            )
            self.assertEqual(
                summary["models"]["qwen"]["fact_bin_counts"]["5/5"], 1
            )
            self.assertTrue((output_dir / "wrong_outputs.jsonl").exists())
            self.assertTrue((output_dir / "alias_only_candidates.jsonl").exists())
            self.assertTrue((output_dir / "dream_x_qwen_bins.csv").exists())

    def test_official_alias_only_match_is_a_review_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.jsonl"
            cache_dir = root / "cache"
            output_dir = root / "analysis"
            write_jsonl(questions, [question_set()])
            write_jsonl(
                cache_dir / "qwen_generations.jsonl",
                cache_rows("qwen", ["City of Light"] * 5),
            )
            summary = analyze(
                questions_path=questions,
                cache_dir=cache_dir,
                output_dir=output_dir,
                expected_models=["qwen"],
                limit=0,
                require_complete=True,
            )
            self.assertEqual(
                summary["models"]["qwen"]["fact_bin_counts"]["0/5"], 1
            )
            self.assertEqual(
                summary["models"]["qwen"]["official_alias_fact_bin_counts"]["5/5"],
                1,
            )
            self.assertEqual(
                summary["models"]["qwen"]["official_alias_only_correct_output_count"],
                5,
            )

    def test_missing_variant_is_not_zero_of_five(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.jsonl"
            cache_dir = root / "cache"
            output_dir = root / "analysis"
            write_jsonl(questions, [question_set()])
            write_jsonl(
                cache_dir / "llada_generations.jsonl",
                cache_rows("llada", ["London"] * 4),
            )
            summary = analyze(
                questions_path=questions,
                cache_dir=cache_dir,
                output_dir=output_dir,
                expected_models=["llada"],
                limit=0,
                require_complete=False,
            )
            self.assertEqual(summary["missing_generation_count"], 1)
            self.assertEqual(
                summary["models"]["llada"]["fact_class_counts"]["INCOMPLETE"], 1
            )


if __name__ == "__main__":
    unittest.main()
