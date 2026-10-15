#!/usr/bin/env python3
"""Extract a reproducible fact manifest from the local TriviaQA batches."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from experiment_utils import SCHEMA_VERSION, make_fact_id, utc_now, write_json, write_jsonl
from scoring import aliases_exposed_by_question, normalized_aliases


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    HERE.parents[1]
    / "sae_entities"
    / "dataset"
    / "processed"
    / "triviaqa"
    / "gemma-2b-it"
)
DEFAULT_OUTPUT = HERE / "data" / "triviaqa_facts.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--num-facts",
        type=int,
        default=100,
        help="Number to select reproducibly; use 0 for every available fact.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output manifest.",
    )
    return parser.parse_args()


def clean_aliases(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"correct_answer must be a list, got {type(value).__name__}")
    aliases: list[str] = []
    seen: set[str] = set()
    for item in value:
        alias = str(item).strip()
        if alias and alias not in seen:
            aliases.append(alias)
            seen.add(alias)
    if not aliases:
        raise ValueError("correct_answer has no non-empty aliases")
    return aliases


def load_local_facts(source_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    batch_paths = sorted(source_dir.glob("batch_*.json"))
    if not batch_paths:
        raise FileNotFoundError(f"No batch_*.json files found in {source_dir}")

    facts: list[dict[str, Any]] = []
    seen_facts: dict[str, dict[str, Any]] = {}
    source_index = 0
    for batch_path in batch_paths:
        with batch_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON array in {batch_path}")
        for row_index, row in enumerate(rows):
            for field in ("question_id", "question", "correct_answer", "split"):
                if field not in row:
                    raise ValueError(
                        f"{batch_path}:{row_index} is missing required field {field!r}"
                    )
            question_id = str(row["question_id"]).strip()
            question = str(row["question"]).strip()
            if not question_id or not question:
                raise ValueError(f"{batch_path}:{row_index} has an empty ID or question")
            fact_id = make_fact_id(question_id, question)
            fact = {
                "schema_version": SCHEMA_VERSION,
                "fact_id": fact_id,
                "question_id": question_id,
                "question": question,
                "answer_aliases": clean_aliases(row["correct_answer"]),
                "split": str(row["split"]),
                "source": "trivia_qa/rc",
                "source_index": source_index,
            }
            source_index += 1
            if fact_id in seen_facts:
                if fact != seen_facts[fact_id]:
                    raise ValueError(f"Conflicting duplicate fact: {fact_id}")
                continue
            seen_facts[fact_id] = fact
            facts.append(fact)
    return facts, batch_paths


def select_facts(
    facts: list[dict[str, Any]], num_facts: int, seed: int
) -> list[dict[str, Any]]:
    if num_facts < 0:
        raise ValueError("--num-facts must be 0 or a positive integer")
    if num_facts == 0 or num_facts >= len(facts):
        return list(facts)
    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(facts)), k=num_facts)
    return [facts[index] for index in selected_indices]


def is_unaided_recall_scoreable(fact: dict[str, Any]) -> bool:
    alias_map = normalized_aliases(fact["answer_aliases"])
    exposed = aliases_exposed_by_question(alias_map, fact["question"])
    return bool(set(alias_map).difference(exposed))


def select_scoreable_facts(
    facts: list[dict[str, Any]], num_facts: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exclude items whose question exposes every gold alias; preserve normal samples."""
    excluded = [fact for fact in facts if not is_unaided_recall_scoreable(fact)]
    initially_selected = select_facts(facts, num_facts, seed)
    selected = [fact for fact in initially_selected if is_unaided_recall_scoreable(fact)]
    target_count = len(facts) - len(excluded) if num_facts == 0 else min(
        num_facts, len(facts) - len(excluded)
    )
    if len(selected) < target_count:
        selected_ids = {fact["fact_id"] for fact in selected}
        replacements = [
            fact
            for fact in facts
            if fact["fact_id"] not in selected_ids and is_unaided_recall_scoreable(fact)
        ]
        replacement_rng = random.Random(seed + 1_000_003)
        selected.extend(
            replacement_rng.sample(replacements, target_count - len(selected))
        )
    return selected, excluded


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; pass --overwrite to replace it"
        )

    facts, batch_paths = load_local_facts(args.source_dir)
    selected, excluded_unscoreable = select_scoreable_facts(
        facts, args.num_facts, args.seed
    )
    write_jsonl(args.output, selected)

    question_id_counts = Counter(fact["question_id"] for fact in facts)
    duplicate_question_ids = {
        question_id: count
        for question_id, count in question_id_counts.items()
        if count > 1
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_dir": str(args.source_dir.resolve()),
        "source_batches": [path.name for path in batch_paths],
        "available_facts": len(facts),
        "selected_facts": len(selected),
        "seed": args.seed,
        "requested_num_facts": args.num_facts,
        "excluded_all_aliases_exposed_count": len(excluded_unscoreable),
        "excluded_all_aliases_exposed_fact_ids": [
            fact["fact_id"] for fact in excluded_unscoreable
        ],
        "split_counts": dict(Counter(fact["split"] for fact in facts)),
        "duplicate_question_ids": duplicate_question_ids,
        "fact_id_rule": "question_id + '__' + sha256(question)[:12]",
    }
    write_json(args.output.with_suffix(args.output.suffix + ".meta.json"), metadata)

    print(f"Loaded {len(facts):,} local TriviaQA facts from {len(batch_paths)} batches")
    print(
        "Excluded facts with every gold alias exposed in the question: "
        f"{len(excluded_unscoreable):,}"
    )
    print(f"Selected {len(selected):,} facts -> {args.output}")
    if duplicate_question_ids:
        print(
            "Duplicate source question IDs were safely disambiguated: "
            f"{duplicate_question_ids}"
        )
    print("First examples:")
    for fact in selected[:5]:
        print(
            json.dumps(
                {
                    "fact_id": fact["fact_id"],
                    "question": fact["question"],
                    "answer_aliases": fact["answer_aliases"][:5],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
