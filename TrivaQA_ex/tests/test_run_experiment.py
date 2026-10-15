from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment_utils import load_jsonl, write_jsonl
from run_experiment import (
    merge_model_shards,
    parallel_shard_counts,
    shard_tasks,
    task_seed,
)


def make_tasks(fact_count: int = 3) -> list[dict[str, object]]:
    return [
        {
            "fact_id": f"fact-{fact_index}",
            "question_id": f"question-{fact_index}",
            "variant_id": f"v{variant_index}",
            "variant_index": variant_index,
            "question": f"Question {fact_index}, variant {variant_index}?",
        }
        for fact_index in range(fact_count)
        for variant_index in range(5)
    ]


class ParallelShardingTests(unittest.TestCase):
    def test_extra_gpu_is_assigned_to_dream(self) -> None:
        self.assertEqual(
            parallel_shard_counts(["dream", "qwen"], [0, 1, 2], 100),
            {"dream": 2, "qwen": 1},
        )
        self.assertEqual(
            parallel_shard_counts(["dream", "qwen"], [0, 1, 2], 1),
            {"dream": 1, "qwen": 1},
        )

    def test_four_gpus_are_balanced_between_dream_and_qwen(self) -> None:
        self.assertEqual(
            parallel_shard_counts(["dream", "qwen"], [0, 1, 2, 3], 100),
            {"dream": 2, "qwen": 2},
        )

    def test_sharding_keeps_all_variants_of_each_fact_together(self) -> None:
        tasks = make_tasks(4)
        shards = shard_tasks(tasks, 2)
        observed = [
            (task["fact_id"], task["variant_id"])
            for shard in shards
            for task in shard
        ]
        expected = [(task["fact_id"], task["variant_id"]) for task in tasks]
        self.assertCountEqual(observed, expected)
        for fact_index in range(4):
            containing = [
                shard
                for shard in shards
                if any(task["fact_id"] == f"fact-{fact_index}" for task in shard)
            ]
            self.assertEqual(len(containing), 1)
            self.assertEqual(
                sum(task["fact_id"] == f"fact-{fact_index}" for task in containing[0]),
                5,
            )

    def test_merge_validates_and_restores_task_order(self) -> None:
        tasks = make_tasks(2)
        rows = [
            {
                "model_name": "dream",
                "fact_id": task["fact_id"],
                "variant_id": task["variant_id"],
                "resolved_revision": "commit",
            }
            for task in tasks
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "dream_generations.jsonl"
            write_jsonl(first, list(reversed(rows[:5])))
            write_jsonl(second, list(reversed(rows[5:])))
            merge_model_shards(
                model_name="dream",
                tasks=tasks,
                shard_paths=[first, second],
                output_path=output,
            )
            merged = load_jsonl(output)
            self.assertEqual(
                [(row["fact_id"], row["variant_id"]) for row in merged],
                [(task["fact_id"], task["variant_id"]) for task in tasks],
            )

    def test_merge_rejects_duplicate_task(self) -> None:
        tasks = make_tasks(1)
        rows = [
            {
                "model_name": "dream",
                "fact_id": task["fact_id"],
                "variant_id": task["variant_id"],
            }
            for task in tasks
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            write_jsonl(first, rows)
            write_jsonl(second, [rows[0]])
            with self.assertRaisesRegex(ValueError, "Duplicate task"):
                merge_model_shards(
                    model_name="dream",
                    tasks=tasks,
                    shard_paths=[first, second],
                    output_path=root / "output.jsonl",
                )

    def test_task_seed_is_stable_and_variant_specific(self) -> None:
        first = task_seed(42, "dream", "fact", "v0")
        self.assertEqual(first, task_seed(42, "dream", "fact", "v0"))
        self.assertNotEqual(first, task_seed(42, "dream", "fact", "v1"))


if __name__ == "__main__":
    unittest.main()
