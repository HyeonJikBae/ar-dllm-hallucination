#!/usr/bin/env python3
"""Recover TriviaQA answer.value and attach it to frozen local question data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from experiment_utils import load_jsonl, utc_now, write_json, write_jsonl


HERE = Path(__file__).resolve().parent
DEFAULT_FACTS = HERE / "data" / "triviaqa_facts.jsonl"
DEFAULT_QUESTIONS = HERE / "data" / "triviaqa_questions_5.jsonl"
DEFAULT_DATASET = "mandarjoshi/trivia_qa"
DEFAULT_CONFIG = "rc.nocontext"
DEFAULT_REVISION = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--arrow-file",
        type=Path,
        default=None,
        help="Read a cached Hugging Face Arrow file instead of contacting the Hub.",
    )
    parser.add_argument(
        "--skip-questions",
        action="store_true",
        help="Only enrich the fact manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Required because enrichment atomically replaces the selected JSONL files.",
    )
    return parser.parse_args()


def load_raw_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    try:
        from datasets import Dataset, load_dataset
    except ImportError as error:
        raise RuntimeError(
            "datasets is required; install TrivaQA_ex/requirements.txt"
        ) from error
    if args.arrow_file is not None:
        if not args.arrow_file.is_file():
            raise FileNotFoundError(args.arrow_file)
        return Dataset.from_file(str(args.arrow_file))
    return load_dataset(
        args.dataset,
        args.config,
        revision=args.revision,
        split=args.split,
    )


def build_answer_lookup(
    rows: Iterable[dict[str, Any]],
    wanted: set[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["question_id"]).strip(), str(row["question"]).strip())
        if key not in wanted:
            continue
        answer = row.get("answer")
        if not isinstance(answer, dict):
            raise ValueError(f"Raw TriviaQA answer is not an object for {key}")
        value = str(answer.get("value", "")).strip()
        aliases = [str(alias).strip() for alias in answer.get("aliases", [])]
        aliases = [alias for alias in aliases if alias]
        if not value or not aliases:
            raise ValueError(f"Raw TriviaQA answer is empty for {key}")
        recovered = {"canonical_answer": value, "answer_aliases": aliases}
        previous = lookup.get(key)
        if previous is not None and recovered != previous:
            raise ValueError(f"Conflicting raw TriviaQA answers for {key}")
        lookup[key] = recovered
    missing = wanted.difference(lookup)
    if missing:
        preview = ", ".join(repr(key) for key in sorted(missing)[:5])
        raise ValueError(
            f"Could not recover {len(missing)} selected TriviaQA answer(s); first: {preview}"
        )
    return lookup


def enrich_facts(
    facts: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, Any]],
    *,
    dataset: str,
    config: str,
    revision: str,
    split: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for fact in facts:
        key = (str(fact["question_id"]).strip(), str(fact["question"]).strip())
        recovered = lookup[key]
        if fact["answer_aliases"] != recovered["answer_aliases"]:
            raise ValueError(
                f"Local aliases differ from raw TriviaQA for {fact['fact_id']}"
            )
        enriched.append(
            {
                **fact,
                "canonical_answer": recovered["canonical_answer"],
                "answer_source": {
                    "dataset": dataset,
                    "config": config,
                    "revision": revision,
                    "split": split,
                    "field": "answer.value",
                },
            }
        )
    return enriched


def enrich_questions(
    rows: list[dict[str, Any]], facts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {fact["fact_id"]: fact for fact in facts}
    if len(by_id) != len(facts):
        raise ValueError("Fact manifest contains duplicate fact_id values")
    enriched: list[dict[str, Any]] = []
    for row in rows:
        fact = by_id.get(row["fact_id"])
        if fact is None:
            raise ValueError(f"Question set has unknown fact_id {row['fact_id']}")
        for field in ("question_id", "question", "answer_aliases"):
            if row[field] != fact[field]:
                raise ValueError(f"Question metadata differs for {row['fact_id']}: {field}")
        enriched.append(
            {
                **row,
                "canonical_answer": fact["canonical_answer"],
                "answer_source": fact["answer_source"],
            }
        )
    if {row["fact_id"] for row in rows} != set(by_id):
        raise ValueError("Question sets and fact manifest do not contain the same facts")
    return enriched


def main() -> None:
    args = parse_args()
    if not args.overwrite:
        raise ValueError("Pass --overwrite to atomically enrich the frozen JSONL files")
    facts = load_jsonl(args.facts)
    wanted = {(fact["question_id"], fact["question"]) for fact in facts}
    if len(wanted) != len(facts):
        raise ValueError("Fact manifest contains duplicate (question_id, question) keys")
    lookup = build_answer_lookup(load_raw_rows(args), wanted)
    enriched_facts = enrich_facts(
        facts,
        lookup,
        dataset=args.dataset,
        config=args.config,
        revision=args.revision,
        split=args.split,
    )
    write_jsonl(args.facts, enriched_facts)
    if not args.skip_questions:
        questions = enrich_questions(load_jsonl(args.questions), enriched_facts)
        write_jsonl(args.questions, questions)

    alias_counts = [len(fact["answer_aliases"]) for fact in enriched_facts]
    write_json(
        args.facts.with_suffix(args.facts.suffix + ".answer_audit.json"),
        {
            "created_at_utc": utc_now(),
            "fact_count": len(enriched_facts),
            "dataset": args.dataset,
            "config": args.config,
            "revision": args.revision,
            "split": args.split,
            "canonical_field": "answer.value",
            "official_alias_field": "answer.aliases",
            "official_alias_count_min": min(alias_counts),
            "official_alias_count_max": max(alias_counts),
            "official_alias_count_total": sum(alias_counts),
        },
    )
    print(f"Recovered canonical answers for {len(enriched_facts):,} fact(s)")
    print(f"Enriched facts -> {args.facts}")
    if not args.skip_questions:
        print(f"Enriched question sets -> {args.questions}")


if __name__ == "__main__":
    main()
