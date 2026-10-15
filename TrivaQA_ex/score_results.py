#!/usr/bin/env python3
"""Score cached generations and aggregate exact 0/5 through 5/5 fact bins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiment_utils import (
    SCHEMA_VERSION,
    iter_jsonl,
    load_jsonl,
    utc_now,
    validate_question_set,
    write_json,
    write_jsonl,
)
from scoring import fact_class, score_response


HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "data" / "triviaqa_questions_5.jsonl"
DEFAULT_CACHE_DIR = HERE / "results" / "triviaqa_smoke" / "cache"
DEFAULT_OUTPUT_DIR = HERE / "results" / "triviaqa_smoke" / "analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models", nargs="+", choices=("dream", "llada", "qwen"), default=None
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write INCOMPLETE facts instead of failing on missing variants.",
    )
    return parser.parse_args()


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def load_questions(
    questions_path: Path, limit: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if limit < 0:
        raise ValueError("limit must be 0 or positive")
    facts = load_jsonl(questions_path)
    if limit:
        facts = facts[:limit]
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    seen_facts: set[str] = set()
    for fact in facts:
        validate_question_set(fact)
        fact_id = fact["fact_id"]
        if fact_id in seen_facts:
            raise ValueError(f"Duplicate fact_id in questions: {fact_id}")
        seen_facts.add(fact_id)
        for variant in fact["variants"]:
            expected[(fact_id, variant["variant_id"])] = {
                "fact": fact,
                "variant": variant,
            }
    if not facts:
        raise ValueError("No question sets selected")
    return facts, expected


def discover_models(cache_dir: Path) -> list[str]:
    suffix = "_generations.jsonl"
    return sorted(
        path.name[: -len(suffix)]
        for path in cache_dir.glob(f"*{suffix}")
        if path.name.endswith(suffix)
    )


def load_model_cache(
    cache_path: Path,
    model_name: str,
    expected: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not cache_path.exists():
        return records
    model_ids: set[str] = set()
    prompt_modes: set[str] = set()
    for row in iter_jsonl(cache_path):
        if row.get("model_name") != model_name:
            raise ValueError(
                f"Expected model {model_name!r} in {cache_path}, got {row.get('model_name')!r}"
            )
        key = (str(row.get("fact_id")), str(row.get("variant_id")))
        if key not in expected:
            # A --limit analysis intentionally ignores cached facts outside its prefix.
            continue
        if key in records:
            raise ValueError(f"Duplicate cached generation for {model_name}:{key}")
        expected_variant = expected[key]["variant"]
        if row.get("question") != expected_variant["question"]:
            raise ValueError(f"Cached question text changed for {model_name}:{key}")
        expected_hash = hashlib.sha256(
            expected_variant["question"].encode("utf-8")
        ).hexdigest()
        if row.get("question_sha256") not in (None, expected_hash):
            raise ValueError(f"Cached question hash mismatch for {model_name}:{key}")
        if not isinstance(row.get("raw_output"), str):
            raise ValueError(f"Cached raw_output is not a string for {model_name}:{key}")
        model_ids.add(str(row.get("model_id", "")))
        prompt_modes.add(str(row.get("prompt_mode", "")))
        records[key] = row
    if len(model_ids) > 1:
        raise ValueError(f"Multiple model IDs found in {cache_path}: {sorted(model_ids)}")
    if len(prompt_modes) > 1:
        raise ValueError(f"Multiple prompt modes found in {cache_path}: {sorted(prompt_modes)}")
    return records


def make_variant_results(
    facts: list[dict[str, Any]],
    models: list[str],
    caches: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model_name in models:
        cache = caches[model_name]
        for fact in facts:
            for variant in fact["variants"]:
                key = (fact["fact_id"], variant["variant_id"])
                generation = cache.get(key)
                if generation is None:
                    missing.append(
                        {
                            "model_name": model_name,
                            "fact_id": fact["fact_id"],
                            "variant_id": variant["variant_id"],
                        }
                    )
                    continue
                scored_output = generation.get(
                    "scored_output", generation["raw_output"]
                )
                canonical_answer = str(fact.get("canonical_answer", "")).strip()
                if not canonical_answer:
                    raise ValueError(
                        f"{fact['fact_id']} has no canonical_answer; run "
                        "enrich_canonical_answers.py before scoring"
                    )
                scored = score_response(
                    scored_output,
                    [canonical_answer],
                    question=variant["question"],
                )
                official_alias_scored = score_response(
                    scored_output,
                    fact["answer_aliases"],
                    question=variant["question"],
                )
                scored["scored_output"] = scored_output
                scored["raw_output"] = generation["raw_output"]
                results.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "model_name": model_name,
                        "model_id": generation.get("model_id"),
                        "prompt_mode": generation.get("prompt_mode"),
                        "fact_id": fact["fact_id"],
                        "question_id": fact["question_id"],
                        "original_question": fact["question"],
                        "canonical_answer": canonical_answer,
                        "variant_id": variant["variant_id"],
                        "variant_index": variant["variant_index"],
                        "variant_source": variant["source"],
                        "question": variant["question"],
                        **scored,
                        "official_alias_strict_is_correct": official_alias_scored[
                            "strict_is_correct"
                        ],
                        "official_alias_is_correct": official_alias_scored[
                            "is_correct"
                        ],
                        "official_alias_match_type": official_alias_scored["match_type"],
                        "official_alias_matched_alias": official_alias_scored[
                            "matched_alias"
                        ],
                        "official_alias_diagnostic_present": official_alias_scored[
                            "diagnostic_alias_present"
                        ],
                        "official_alias_diagnostic_present_alias": official_alias_scored[
                            "diagnostic_present_alias"
                        ],
                        "official_alias_question_exposed_aliases": official_alias_scored[
                            "question_exposed_aliases"
                        ],
                        "official_alias_matched_derived_gold": official_alias_scored[
                            "matched_derived_gold"
                        ],
                        "official_alias_only_correct": (
                            official_alias_scored["is_correct"]
                            and not scored["is_correct"]
                        ),
                    }
                )
    return results, missing


def aggregate_facts(
    facts: list[dict[str, Any]],
    models: list[str],
    variant_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in variant_results:
        by_key[(row["model_name"], row["fact_id"])].append(row)

    aggregated: list[dict[str, Any]] = []
    for model_name in models:
        for fact in facts:
            rows = sorted(
                by_key[(model_name, fact["fact_id"])],
                key=lambda row: row["variant_index"],
            )
            observed_ids = {row["variant_id"] for row in rows}
            expected_ids = {f"v{index}" for index in range(5)}
            missing_ids = sorted(expected_ids.difference(observed_ids))
            correct_count = sum(bool(row["is_correct"]) for row in rows)
            strict_correct_count = sum(
                bool(row["strict_is_correct"]) for row in rows
            )
            official_alias_correct_count = sum(
                bool(row["official_alias_is_correct"]) for row in rows
            )
            official_alias_strict_correct_count = sum(
                bool(row["official_alias_strict_is_correct"]) for row in rows
            )
            complete = not missing_ids and len(rows) == 5
            classification = fact_class(correct_count) if complete else "INCOMPLETE"
            pattern_by_index = {
                row["variant_index"]: "1" if row["is_correct"] else "0"
                for row in rows
            }
            correctness_pattern = "".join(
                pattern_by_index.get(index, "?") for index in range(5)
            )
            strict_pattern_by_index = {
                row["variant_index"]: "1" if row["strict_is_correct"] else "0"
                for row in rows
            }
            strict_correctness_pattern = "".join(
                strict_pattern_by_index.get(index, "?") for index in range(5)
            )
            official_alias_pattern_by_index = {
                row["variant_index"]: (
                    "1" if row["official_alias_is_correct"] else "0"
                )
                for row in rows
            }
            official_alias_correctness_pattern = "".join(
                official_alias_pattern_by_index.get(index, "?")
                for index in range(5)
            )
            model_id = rows[0].get("model_id") if rows else None
            prompt_mode = rows[0].get("prompt_mode") if rows else None
            aggregated.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model_name": model_name,
                    "model_id": model_id,
                    "prompt_mode": prompt_mode,
                    "fact_id": fact["fact_id"],
                    "question_id": fact["question_id"],
                    "original_question": fact["question"],
                    "canonical_answer": fact["canonical_answer"],
                    "answer_aliases": fact["answer_aliases"],
                    "observed_variant_count": len(rows),
                    "missing_variant_ids": missing_ids,
                    "correct_count": correct_count,
                    "strict_correct_count": strict_correct_count,
                    "official_alias_correct_count": official_alias_correct_count,
                    "official_alias_strict_correct_count": (
                        official_alias_strict_correct_count
                    ),
                    "incorrect_count": len(rows) - correct_count,
                    "official_alias_only_correct_count": sum(
                        bool(row["official_alias_only_correct"]) for row in rows
                    ),
                    "correctness_pattern": correctness_pattern,
                    "strict_correctness_pattern": strict_correctness_pattern,
                    "official_alias_correctness_pattern": (
                        official_alias_correctness_pattern
                    ),
                    "fact_bin": f"{correct_count}/5" if complete else "INCOMPLETE",
                    "fact_class": classification,
                    "strict_fact_bin": (
                        f"{strict_correct_count}/5" if complete else "INCOMPLETE"
                    ),
                    "strict_fact_class": (
                        fact_class(strict_correct_count) if complete else "INCOMPLETE"
                    ),
                    "official_alias_fact_bin": (
                        f"{official_alias_correct_count}/5"
                        if complete
                        else "INCOMPLETE"
                    ),
                    "official_alias_fact_class": (
                        fact_class(official_alias_correct_count)
                        if complete
                        else "INCOMPLETE"
                    ),
                    "is_prompt_sensitive": complete and 1 <= correct_count <= 4,
                    "refusal_count": sum(bool(row["is_refusal"]) for row in rows),
                    "diagnostic_alias_present_in_wrong_count": sum(
                        bool(row["diagnostic_alias_present"]) and not row["is_correct"]
                        for row in rows
                    ),
                    "correct_variant_ids": [
                        row["variant_id"] for row in rows if row["is_correct"]
                    ],
                    "wrong_variant_ids": [
                        row["variant_id"] for row in rows if not row["is_correct"]
                    ],
                }
            )
    return aggregated


def build_summary(
    facts: list[dict[str, Any]],
    models: list[str],
    variant_results: list[dict[str, Any]],
    fact_results: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "fact_count": len(facts),
        "models": {},
        "missing_generation_count": len(missing),
        "primary_scoring": "canonical_answer.value",
        "sensitivity_scoring": "all_official_answer.aliases",
    }
    for model_name in models:
        variants = [row for row in variant_results if row["model_name"] == model_name]
        model_facts = [row for row in fact_results if row["model_name"] == model_name]
        complete_facts = [row for row in model_facts if row["fact_class"] != "INCOMPLETE"]
        bin_order = [f"{index}/5" for index in range(6)]
        bin_counts = Counter(row["fact_bin"] for row in model_facts)
        strict_bin_counts = Counter(row["strict_fact_bin"] for row in model_facts)
        official_alias_bin_counts = Counter(
            row["official_alias_fact_bin"] for row in model_facts
        )
        refusal_by_bin: dict[str, dict[str, int]] = {}
        for fact_bin in bin_order:
            rows = [row for row in complete_facts if row["fact_bin"] == fact_bin]
            refusal_by_bin[fact_bin] = {
                "fact_count": len(rows),
                "facts_with_refusal": sum(row["refusal_count"] > 0 for row in rows),
                "refusal_output_count": sum(row["refusal_count"] for row in rows),
            }
        incorrect_count = sum(not row["is_correct"] for row in variants)
        refusal_count = sum(row["is_refusal"] for row in variants)
        summary["models"][model_name] = {
            "model_id": next(
                (row["model_id"] for row in variants if row.get("model_id")), None
            ),
            "fact_class_counts": dict(
                sorted(Counter(row["fact_class"] for row in model_facts).items())
            ),
            "fact_bin_counts": {
                fact_bin: bin_counts.get(fact_bin, 0) for fact_bin in bin_order
            },
            "strict_fact_bin_counts": {
                fact_bin: strict_bin_counts.get(fact_bin, 0) for fact_bin in bin_order
            },
            "official_alias_fact_bin_counts": {
                fact_bin: official_alias_bin_counts.get(fact_bin, 0)
                for fact_bin in bin_order
            },
            "official_alias_fact_bin_proportions": {
                fact_bin: (
                    official_alias_bin_counts.get(fact_bin, 0) / len(complete_facts)
                    if complete_facts
                    else 0.0
                )
                for fact_bin in bin_order
            },
            "prompt_sensitive_fact_count": sum(
                row["is_prompt_sensitive"] for row in complete_facts
            ),
            "official_alias_prompt_sensitive_fact_count": sum(
                1 <= row["official_alias_correct_count"] <= 4
                for row in complete_facts
            ),
            "complete_fact_count": len(complete_facts),
            "fact_bin_proportions": {
                fact_bin: (
                    bin_counts.get(fact_bin, 0) / len(complete_facts)
                    if complete_facts
                    else 0.0
                )
                for fact_bin in bin_order
            },
            "observed_output_count": len(variants),
            "correct_output_count": sum(row["is_correct"] for row in variants),
            "official_alias_correct_output_count": sum(
                row["official_alias_is_correct"] for row in variants
            ),
            "official_alias_incorrect_output_count": sum(
                not row["official_alias_is_correct"] for row in variants
            ),
            "official_alias_only_correct_output_count": sum(
                row["official_alias_only_correct"] for row in variants
            ),
            "incorrect_output_count": incorrect_count,
            "refusal_output_count": refusal_count,
            "refusal_rate_among_incorrect": (
                refusal_count / incorrect_count if incorrect_count else 0.0
            ),
            "refusal_outputs_outside_0_of_5": sum(
                row["refusal_count"]
                for row in complete_facts
                if row["fact_bin"] != "0/5"
            ),
            "variant_accuracy": {
                f"v{index}": (
                    sum(
                        row["is_correct"]
                        for row in variants
                        if row["variant_id"] == f"v{index}"
                    )
                    / sum(row["variant_id"] == f"v{index}" for row in variants)
                    if any(row["variant_id"] == f"v{index}" for row in variants)
                    else 0.0
                )
                for index in range(5)
            },
            "official_alias_variant_accuracy": {
                f"v{index}": (
                    sum(
                        row["official_alias_is_correct"]
                        for row in variants
                        if row["variant_id"] == f"v{index}"
                    )
                    / sum(row["variant_id"] == f"v{index}" for row in variants)
                    if any(row["variant_id"] == f"v{index}" for row in variants)
                    else 0.0
                )
                for index in range(5)
            },
            "refusal_by_fact_bin": refusal_by_bin,
        }
    return summary


def add_fact_labels_to_wrong_rows(
    variant_results: list[dict[str, Any]],
    fact_results: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    *,
    use_official_alias: bool,
) -> list[dict[str, Any]]:
    labels = {
        (row["model_name"], row["fact_id"]): row for row in fact_results
    }
    answer_metadata = {
        fact["fact_id"]: {
            "canonical_answer": fact["canonical_answer"],
            "answer_aliases": fact["answer_aliases"],
        }
        for fact in facts
    }
    correctness_key = (
        "official_alias_is_correct" if use_official_alias else "is_correct"
    )
    wrong: list[dict[str, Any]] = []
    for row in variant_results:
        if row[correctness_key]:
            continue
        label = labels[(row["model_name"], row["fact_id"])]
        if use_official_alias:
            fact_bin = label["official_alias_fact_bin"]
            fact_class_name = label["official_alias_fact_class"]
            pattern = label["official_alias_correctness_pattern"]
        else:
            fact_bin = label["fact_bin"]
            fact_class_name = label["fact_class"]
            pattern = label["correctness_pattern"]
        wrong.append(
            {
                **row,
                **answer_metadata[row["fact_id"]],
                "fact_bin": fact_bin,
                "fact_class": fact_class_name,
                "correctness_pattern": pattern,
                "canonical_fact_bin": label["fact_bin"],
                "official_alias_fact_bin": label["official_alias_fact_bin"],
            }
        )
    return wrong


def write_cross_model(
    output_dir: Path,
    facts: list[dict[str, Any]],
    fact_results: list[dict[str, Any]],
) -> None:
    indexed = {
        (row["model_name"], row["fact_id"]): row for row in fact_results
    }
    model_names = sorted({row["model_name"] for row in fact_results})
    for left_name, right_name in itertools.combinations(model_names, 2):
        rows: list[dict[str, Any]] = []
        for fact in facts:
            left = indexed[(left_name, fact["fact_id"])]
            right = indexed[(right_name, fact["fact_id"])]
            if left["fact_class"] == "INCOMPLETE" or right["fact_class"] == "INCOMPLETE":
                continue
            rows.append(
                {
                    "fact_id": fact["fact_id"],
                    "question_id": fact["question_id"],
                    "original_question": fact["question"],
                    f"{left_name}_correct_count": left["correct_count"],
                    f"{left_name}_strict_correct_count": left["strict_correct_count"],
                    f"{left_name}_official_alias_correct_count": left[
                        "official_alias_correct_count"
                    ],
                    f"{left_name}_fact_bin": left["fact_bin"],
                    f"{left_name}_fact_class": left["fact_class"],
                    f"{left_name}_pattern": left["correctness_pattern"],
                    f"{left_name}_official_alias_fact_bin": left[
                        "official_alias_fact_bin"
                    ],
                    f"{left_name}_official_alias_fact_class": left[
                        "official_alias_fact_class"
                    ],
                    f"{left_name}_official_alias_pattern": left[
                        "official_alias_correctness_pattern"
                    ],
                    f"{left_name}_refusal_count": left["refusal_count"],
                    f"{right_name}_correct_count": right["correct_count"],
                    f"{right_name}_strict_correct_count": right["strict_correct_count"],
                    f"{right_name}_official_alias_correct_count": right[
                        "official_alias_correct_count"
                    ],
                    f"{right_name}_fact_bin": right["fact_bin"],
                    f"{right_name}_fact_class": right["fact_class"],
                    f"{right_name}_pattern": right["correctness_pattern"],
                    f"{right_name}_official_alias_fact_bin": right[
                        "official_alias_fact_bin"
                    ],
                    f"{right_name}_official_alias_fact_class": right[
                        "official_alias_fact_class"
                    ],
                    f"{right_name}_official_alias_pattern": right[
                        "official_alias_correctness_pattern"
                    ],
                    f"{right_name}_refusal_count": right["refusal_count"],
                    f"{right_name}_minus_{left_name}": (
                        right["correct_count"] - left["correct_count"]
                    ),
                    "same_fact_bin": right["fact_bin"] == left["fact_bin"],
                }
            )
        if rows:
            write_csv(
                output_dir / f"{left_name}_x_{right_name}_bins.csv",
                rows,
                list(rows[0]),
            )


def analyze(
    *,
    questions_path: Path,
    cache_dir: Path,
    output_dir: Path,
    expected_models: list[str] | None,
    limit: int,
    require_complete: bool,
) -> dict[str, Any]:
    facts, expected = load_questions(questions_path, limit)
    models = list(expected_models or discover_models(cache_dir))
    if not models:
        raise ValueError(f"No model caches found in {cache_dir}")
    if len(set(models)) != len(models):
        raise ValueError("Model list contains duplicates")

    caches = {
        model_name: load_model_cache(
            cache_dir / f"{model_name}_generations.jsonl",
            model_name,
            expected,
        )
        for model_name in models
    }
    variant_results, missing = make_variant_results(facts, models, caches)
    if missing and require_complete:
        preview = ", ".join(
            f"{row['model_name']}:{row['fact_id']}:{row['variant_id']}"
            for row in missing[:10]
        )
        raise ValueError(
            f"Missing {len(missing)} expected generations; first missing: {preview}"
        )
    fact_results = aggregate_facts(facts, models, variant_results)
    canonical_wrong_rows = add_fact_labels_to_wrong_rows(
        variant_results,
        fact_results,
        facts,
        use_official_alias=False,
    )
    official_alias_wrong_rows = add_fact_labels_to_wrong_rows(
        variant_results,
        fact_results,
        facts,
        use_official_alias=True,
    )
    alias_only_candidates = [
        row for row in canonical_wrong_rows if row["official_alias_only_correct"]
    ]
    summary = build_summary(
        facts, models, variant_results, fact_results, missing
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "variant_results.jsonl", variant_results)
    write_jsonl(output_dir / "fact_results.jsonl", fact_results)
    write_jsonl(output_dir / "wrong_outputs.jsonl", canonical_wrong_rows)
    write_jsonl(
        output_dir / "official_alias_wrong_outputs.jsonl",
        official_alias_wrong_rows,
    )
    write_jsonl(output_dir / "alias_only_candidates.jsonl", alias_only_candidates)
    write_jsonl(output_dir / "missing_generations.jsonl", missing)
    write_json(output_dir / "summary.json", summary)
    write_csv(
        output_dir / "variant_results.csv",
        variant_results,
        [
            "model_name",
            "model_id",
            "prompt_mode",
            "fact_id",
            "question_id",
            "canonical_answer",
            "variant_id",
            "variant_index",
            "variant_source",
            "question",
            "raw_output",
            "scored_output",
            "extracted_answer",
            "normalized_answer",
            "strict_is_correct",
            "is_correct",
            "matched_alias",
            "official_alias_strict_is_correct",
            "official_alias_is_correct",
            "official_alias_matched_alias",
            "official_alias_only_correct",
            "is_refusal",
            "refusal_rule",
            "matched_derived_gold",
            "official_alias_matched_derived_gold",
            "diagnostic_alias_present",
            "diagnostic_present_alias",
            "question_exposed_aliases",
            "official_alias_question_exposed_aliases",
        ],
    )
    write_csv(
        output_dir / "fact_results.csv",
        fact_results,
        [
            "model_name",
            "model_id",
            "prompt_mode",
            "fact_id",
            "question_id",
            "original_question",
            "canonical_answer",
            "answer_aliases",
            "observed_variant_count",
            "missing_variant_ids",
            "correct_count",
            "strict_correct_count",
            "official_alias_correct_count",
            "official_alias_strict_correct_count",
            "incorrect_count",
            "official_alias_only_correct_count",
            "correctness_pattern",
            "strict_correctness_pattern",
            "official_alias_correctness_pattern",
            "fact_bin",
            "fact_class",
            "strict_fact_bin",
            "strict_fact_class",
            "official_alias_fact_bin",
            "official_alias_fact_class",
            "is_prompt_sensitive",
            "refusal_count",
            "diagnostic_alias_present_in_wrong_count",
            "correct_variant_ids",
            "wrong_variant_ids",
        ],
    )
    write_cross_model(output_dir, facts, fact_results)
    print(
        f"Scored {len(variant_results):,} outputs across {len(facts):,} facts "
        f"and {len(models)} model(s) -> {output_dir}"
    )
    return summary


def main() -> None:
    args = parse_args()
    analyze(
        questions_path=args.questions,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        expected_models=args.models,
        limit=args.limit,
        require_complete=not args.allow_incomplete,
    )


if __name__ == "__main__":
    main()
