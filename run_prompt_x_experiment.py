#!/usr/bin/env python3
"""Estimate robust PopQA factual recall across five completion paraphrases."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import transformers

from popqa_utils import (
    DATASET_ID,
    MODEL_ORDER,
    MODEL_SPECS,
    REQUIRED_COLUMNS,
    answer_set,
    csv_value,
    evaluate_answer,
    load_popqa,
    write_csv,
    write_jsonl,
)
from model_generation import (
    LLADA_MASK_TOKEN_ID,
    clear_cuda_memory,
    generate_answer,
    load_model,
    model_parameter_count,
    preferred_dtype,
)


DEFAULT_NUM_SAMPLES = 50
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_KNOWN_MIN_CORRECT = 4
DEFAULT_UNKNOWN_MAX_CORRECT = 0

# Every group was checked against PopQA's real subject -> object direction.
# Template 1 matches the completed one-prompt experiment wherever possible.
RELATION_TEMPLATE_SETS: dict[str, tuple[str, ...]] = {
    "author": (
        "{subject} was written by",
        "The author of {subject} was",
        "{subject}'s author was",
        "The person who wrote {subject} was",
        "The writer of {subject} was",
    ),
    "capital": (
        "The capital of {subject} is",
        "{subject}'s capital is",
        "The capital city of {subject} is",
        "The city serving as the capital of {subject} is",
        "{subject} has its capital in",
    ),
    "capital of": (
        "{subject} is the capital of",
        "{subject} serves as the capital of",
        "{subject} is the capital city of",
        "{subject} functions as the capital of",
        "The place for which {subject} is the capital is",
    ),
    "color": (
        "The color of {subject} is",
        "{subject}'s color is",
        "The color associated with {subject} is",
        "The identifying color of {subject} is",
        "{subject} has the color",
    ),
    "composer": (
        "{subject} was composed by",
        "The composer of {subject} was",
        "{subject}'s composer was",
        "The person who composed {subject} was",
        "The music for {subject} was composed by",
    ),
    "country": (
        "{subject} is located in the country of",
        "The country where {subject} is located is",
        "{subject} is situated in the country of",
        "The nation containing {subject} is",
        "The country containing {subject} is",
    ),
    "director": (
        "{subject} was directed by",
        "The director of {subject} was",
        "{subject}'s director was",
        "The person who directed {subject} was",
        "The filmmaker who directed {subject} was",
    ),
    "father": (
        "The father of {subject} is",
        "{subject}'s father is",
        "The person identified as {subject}'s father is",
        "{subject}'s paternal parent is",
        "The name of {subject}'s father is",
    ),
    "genre": (
        "The genre of {subject} is",
        "{subject}'s genre is",
        "{subject} belongs to the genre",
        "The genre classification of {subject} is",
        "{subject} is classified in the genre",
    ),
    "mother": (
        "The mother of {subject} is",
        "{subject}'s mother is",
        "The person identified as {subject}'s mother is",
        "{subject}'s maternal parent is",
        "The name of {subject}'s mother is",
    ),
    "occupation": (
        "{subject}'s occupation is",
        "{subject} worked as a",
        "The occupation of {subject} was",
        "{subject}'s profession was",
        "{subject} was professionally a",
    ),
    "place of birth": (
        "{subject} was born in",
        "{subject}'s birthplace is",
        "The city where {subject} was born is",
        "The birthplace of {subject} was",
        "{subject}'s city of birth is",
    ),
    "producer": (
        "{subject} was produced by",
        "The producer of {subject} was",
        "{subject}'s producer was",
        "The person who produced {subject} was",
        "Production of {subject} was handled by",
    ),
    "religion": (
        "The religion of {subject} is",
        "{subject}'s religion is",
        "{subject} follows the religion",
        "The religious affiliation of {subject} is",
        "{subject}'s religious faith is",
    ),
    "screenwriter": (
        "{subject} was written by",
        "The screenwriter of {subject} was",
        "{subject}'s screenwriter was",
        "The screenplay for {subject} was written by",
        "The person who wrote the screenplay for {subject} was",
    ),
    "sport": (
        "{subject} plays the sport of",
        "The sport played by {subject} is",
        "{subject}'s sport is",
        "The athletic sport associated with {subject} is",
        "The game played competitively by {subject} is",
    ),
}

KNOWLEDGE_CASE_MAPPING = {
    ("KNOWN", "KNOWN", "KNOWN"): "ALL_KNOWN",
    ("UNKNOWN", "KNOWN", "KNOWN"): "AR_ONLY_UNKNOWN",
    ("KNOWN", "UNKNOWN", "KNOWN"): "LLADA_ONLY_UNKNOWN",
    ("KNOWN", "KNOWN", "UNKNOWN"): "DREAM_ONLY_UNKNOWN",
    ("KNOWN", "UNKNOWN", "UNKNOWN"): "AR_ONLY_KNOWN",
    ("UNKNOWN", "KNOWN", "UNKNOWN"): "LLADA_ONLY_KNOWN",
    ("UNKNOWN", "UNKNOWN", "KNOWN"): "DREAM_ONLY_KNOWN",
    ("UNKNOWN", "UNKNOWN", "UNKNOWN"): "ALL_UNKNOWN",
}
KNOWLEDGE_CASE_ORDER = (
    "AR_ONLY_KNOWN",
    "AR_ONLY_UNKNOWN",
    "LLADA_ONLY_KNOWN",
    "DREAM_ONLY_KNOWN",
    "LLADA_ONLY_UNKNOWN",
    "DREAM_ONLY_UNKNOWN",
    "ALL_KNOWN",
    "ALL_UNKNOWN",
    "UNCERTAIN_CASE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("results/popqa_knowledge_smoke"))
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--relations", nargs="+", default=None)
    parser.add_argument("--samples-per-relation", type=int, default=None)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--known-min-correct", type=int, default=DEFAULT_KNOWN_MIN_CORRECT)
    parser.add_argument("--unknown-max-correct", type=int, default=DEFAULT_UNKNOWN_MAX_CORRECT)
    parser.add_argument("--reuse-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--sample-source-results", type=Path, default=None,
        help="Reuse selected full PopQA rows from a completed result JSONL (no Hub access)",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_thresholds(known_min: int, unknown_max: int) -> None:
    if not 0 <= unknown_max < known_min <= 5:
        raise ValueError("Require 0 <= unknown_max_correct < known_min_correct <= 5")


def analyze_relations(dataset: Any, output_dir: Path) -> None:
    counts = Counter(dataset["prop"])
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in dataset:
        if len(examples[raw["prop"]]) < 5:
            examples[raw["prop"]].append(dict(raw))
    print(f"{'relation':30} count")
    print("-" * 42)
    for relation in sorted(counts):
        print(f"{relation:30} {counts[relation]}")

    with (output_dir / "relation_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relation", "count", "mapped"))
        writer.writeheader()
        writer.writerows(
            {"relation": relation, "count": counts[relation], "mapped": relation in RELATION_TEMPLATE_SETS}
            for relation in sorted(counts)
        )
    (output_dir / "template_mapping.json").write_text(
        json.dumps(RELATION_TEMPLATE_SETS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "relation_examples.txt").open("w", encoding="utf-8") as handle:
        for relation in sorted(counts):
            handle.write("=" * 70 + f"\nRELATION: {relation}\nCOUNT: {counts[relation]}\n")
            handle.write("TEMPLATES:\n")
            for index, template in enumerate(RELATION_TEMPLATE_SETS.get(relation, ()), 1):
                handle.write(f"  P{index}: {template}\n")
            for sample in examples[relation]:
                handle.write(
                    f"subject: {sample['subj']}\nrelation: {sample['prop']}\n"
                    f"object: {sample['obj']}\noriginal question: {sample['question']}\n\n"
                )


def prepare_samples(
    dataset: Any,
    num_samples: int,
    seed: int,
    requested_relations: list[str] | None,
    samples_per_relation: int | None,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    actual_relations = set(dataset["prop"])
    unknown = set(requested_relations or ()).difference(actual_relations)
    if unknown:
        raise ValueError(f"Unknown PopQA relations: {sorted(unknown)}")
    allowed = set(requested_relations or RELATION_TEMPLATE_SETS)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = []
    usable = 0
    for raw in dataset:
        sample = dict(raw)
        templates = RELATION_TEMPLATE_SETS.get(sample["prop"])
        if templates is None:
            skipped.append(
                {"id": sample["id"], "relation": sample["prop"], "skip_reason": "NO_SAFE_TEMPLATE_SET"}
            )
            continue
        usable += 1
        if sample["prop"] not in allowed:
            continue
        sample["templates"] = [template.format(subject=str(sample["subj"]).strip()) for template in templates]
        sample["template_ids"] = [f"{sample['prop'].replace(' ', '_')}_v{index}" for index in range(1, 6)]
        groups[sample["prop"]].append(sample)
    write_jsonl(output_dir / "skipped_no_template.jsonl", skipped)

    rng = random.Random(seed)
    for relation in groups:
        rng.shuffle(groups[relation])
        if samples_per_relation is not None:
            groups[relation] = groups[relation][:samples_per_relation]
    selected, position = [], 0
    relations = sorted(groups)
    while len(selected) < num_samples:
        added = False
        for relation in relations:
            if position < len(groups[relation]):
                selected.append(groups[relation][position])
                added = True
                if len(selected) == num_samples:
                    break
        if not added:
            break
        position += 1
    if len(selected) < num_samples:
        raise ValueError(f"Only {len(selected)} facts available under the sampling constraints")

    manifest = [
        {
            "id": sample["id"], "subject": sample["subj"], "relation": sample["prop"],
            "gold_answer": sample["obj"], "templates": sample["templates"],
            "template_ids": sample["template_ids"],
        }
        for sample in selected
    ]
    write_jsonl(output_dir / "selected_samples.jsonl", manifest)
    distribution = Counter(sample["prop"] for sample in selected)
    with (output_dir / "selected_relation_distribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("relation", "count"))
        writer.writerows(sorted(distribution.items()))
    print(f"usable facts: {usable}; skipped facts: {len(skipped)}; selected facts: {len(selected)}")
    print(f"selected relation distribution: {dict(sorted(distribution.items()))}")
    return selected, {"total": len(dataset), "usable": usable, "skipped": len(skipped)}


def cache_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / "cache" / f"{model_name}_generations.jsonl"


def generation_key(sample_id: Any, template_id: str) -> str:
    return f"{sample_id}::{template_id}"


def read_generation_cache(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                records[generation_key(record["id"], record["template_id"])] = record
            except KeyError as exc:
                raise RuntimeError(f"Invalid knowledge cache {path}:{line_number}") from exc
    return records


def append_cache(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def restore_completed_samples(output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Restore full PopQA rows for cached-only regrading without Hub access."""
    result_path = output_dir / "popqa_knowledge_results.jsonl"
    samples = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line]
    inventory_path = output_dir / "relation_inventory.csv"
    with inventory_path.open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    total = sum(int(row["count"]) for row in inventory)
    usable = sum(int(row["count"]) for row in inventory if row["mapped"].lower() == "true")
    print(f"restored {len(samples)} selected facts from {result_path}")
    return samples, {"total": total, "usable": usable, "skipped": total - usable}


def load_samples_from_results(
    source_path: Path, num_samples: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reuse the same sampled facts while rebuilding prompts for the active condition."""
    source_rows = [
        json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(source_rows) < num_samples:
        raise ValueError(f"Sample source has {len(source_rows)} rows, need {num_samples}")
    samples = source_rows[:num_samples]
    for sample in samples:
        relation = sample["prop"]
        templates = RELATION_TEMPLATE_SETS.get(relation)
        if templates is None:
            raise ValueError(f"No active template set for relation {relation!r}")
        sample["templates"] = [
            template.format(subject=str(sample["subj"]).strip()) for template in templates
        ]
        sample["template_ids"] = [
            f"{relation.replace(' ', '_')}_v{index}" for index in range(1, 6)
        ]
    inventory_path = source_path.parent / "relation_inventory.csv"
    with inventory_path.open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    total = sum(int(row["count"]) for row in inventory)
    usable = sum(int(row["count"]) for row in inventory if row["mapped"].lower() == "true")
    print(f"reused {len(samples)} sampled facts from {source_path}")
    return samples, {"total": total, "usable": usable, "skipped": total - usable}


def import_matching_single_prompt_cache(
    samples: list[dict[str, Any]], model_name: str, source_dir: Path, target_path: Path,
    cache: dict[str, dict[str, Any]],
) -> int:
    source_path = source_dir / f"{model_name}_generations.jsonl"
    if not source_path.exists():
        return 0
    source = {}
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            source[str(record["id"])] = record
    imported = 0
    for sample in samples:
        key = generation_key(sample["id"], sample["template_ids"][0])
        old = source.get(str(sample["id"]))
        if key in cache or old is None or old.get("prompt") != sample["templates"][0]:
            continue
        record = {
            "id": sample["id"], "template_index": 1,
            "template_id": sample["template_ids"][0], "prompt": sample["templates"][0],
            "raw_output": old["raw_output"], "imported_from": str(source_path),
        }
        append_cache(target_path, record)
        cache[key] = record
        imported += 1
    return imported


def run_generation(
    samples: list[dict[str, Any]], models: list[str], max_new_tokens: int,
    output_dir: Path, reuse_cache_dir: Path | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    caches = {name: read_generation_cache(cache_path(output_dir, name)) for name in MODEL_ORDER}
    for model_name in MODEL_ORDER:
        if reuse_cache_dir is not None:
            imported = import_matching_single_prompt_cache(
                samples, model_name, reuse_cache_dir, cache_path(output_dir, model_name), caches[model_name]
            )
            print(f"{model_name}: imported {imported} matching template-1 generations")
        if model_name not in models:
            continue
        pending = [
            (sample, index, template_id, prompt)
            for sample in samples
            for index, (template_id, prompt) in enumerate(zip(sample["template_ids"], sample["templates"]), 1)
            if generation_key(sample["id"], template_id) not in caches[model_name]
        ]
        if not pending:
            print(f"{model_name}: all {len(samples) * 5} generations restored from cache")
            continue
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA unavailable and {len(pending)} {model_name} generations are missing; "
                "select a GPU with CUDA_VISIBLE_DEVICES or restore a complete cache"
            )
        display_name, model_id = MODEL_SPECS[model_name]
        model = tokenizer = None
        try:
            model, tokenizer = load_model(model_id, model_name)
            print("\n" + "=" * 50 + f"\nMODEL: {display_name}\n" + "=" * 50)
            print(f"model class: {type(model).__module__}.{type(model).__name__}")
            print(f"tokenizer class: {type(tokenizer).__module__}.{type(tokenizer).__name__}")
            print(f"parameter count: {model_parameter_count(model):,}")
            if model_name == "llada":
                print(f"mask token ID: {LLADA_MASK_TOKEN_ID}")
            for number, (sample, index, template_id, prompt) in enumerate(pending, 1):
                raw_output = generate_answer(
                    model_name, model, tokenizer, prompt, max_new_tokens=max_new_tokens
                )
                record = {
                    "id": sample["id"], "template_index": index, "template_id": template_id,
                    "prompt": prompt, "raw_output": raw_output,
                }
                append_cache(cache_path(output_dir, model_name), record)
                caches[model_name][generation_key(sample["id"], template_id)] = record
                print(f"[{display_name} {number}/{len(pending)}] id={sample['id']} prompt={index}")
        except Exception as exc:
            print(f"ERROR for {model_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            raise
        finally:
            del model
            del tokenizer
            gc.collect()
            clear_cuda_memory()
    return caches


def knowledge_label(correct_count: int, known_min: int, unknown_max: int) -> str:
    if correct_count >= known_min:
        return "KNOWN"
    if correct_count <= unknown_max:
        return "UNKNOWN"
    return "UNCERTAIN"


def classify_knowledge_case(labels: tuple[str, str, str]) -> tuple[str, list[str]]:
    uncertain_models = [name for name, label in zip(MODEL_ORDER, labels) if label == "UNCERTAIN"]
    if uncertain_models:
        return "UNCERTAIN_CASE", uncertain_models
    return KNOWLEDGE_CASE_MAPPING[labels], []


def build_rows(
    samples: list[dict[str, Any]], caches: dict[str, dict[str, dict[str, Any]]],
    known_min: int, unknown_max: int,
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        canonical, candidates = answer_set(sample)
        row = dict(sample)
        row.update(
            {
                "subject": sample["subj"], "relation": sample["prop"],
                "gold_answer": canonical, "aliases": candidates[1:],
                "original_question": sample["question"],
            }
        )
        labels = []
        for model_name in MODEL_ORDER:
            prompt_results = []
            for index, (template_id, prompt) in enumerate(
                zip(sample["template_ids"], sample["templates"]), 1
            ):
                record = caches[model_name].get(generation_key(sample["id"], template_id))
                if record is None:
                    prompt_results.append(
                        {
                            "template_index": index, "template_id": template_id, "prompt": prompt,
                            "raw_output": None, "initial_answer_span": None, "correct": None,
                            "match_type": None, "matched_gold": None,
                        }
                    )
                    continue
                evaluation = evaluate_answer(
                    record["raw_output"], canonical, candidates, subject=str(sample["subj"])
                )
                prompt_results.append(
                    {
                        "template_index": index, "template_id": template_id, "prompt": prompt,
                        "raw_output": record["raw_output"],
                        "initial_answer_span": evaluation["answer_span"],
                        "correct": evaluation["is_correct"],
                        "match_type": evaluation["match_type"],
                        "matched_gold": evaluation["matched_gold"],
                    }
                )
            correctness = [result["correct"] for result in prompt_results]
            if any(value is None for value in correctness):
                count, score, label = None, None, "INCOMPLETE"
            else:
                count = sum(bool(value) for value in correctness)
                score = count / 5
                label = knowledge_label(count, known_min, unknown_max)
            row[f"{model_name}_outputs"] = [result["raw_output"] for result in prompt_results]
            row[f"{model_name}_initial_answer_spans"] = [
                result["initial_answer_span"] for result in prompt_results
            ]
            row[f"{model_name}_prompt_results"] = prompt_results
            row[f"{model_name}_prompt_correctness"] = correctness
            row[f"{model_name}_knowledge_correct_count"] = count
            row[f"{model_name}_knowledge_score"] = score
            row[f"{model_name}_knowledge_label"] = label
            labels.append(label)
        if "INCOMPLETE" in labels:
            row["knowledge_case"], row["uncertain_models"] = "INCOMPLETE_MODEL_SET", []
        else:
            row["knowledge_case"], row["uncertain_models"] = classify_knowledge_case(tuple(labels))
        rows.append(row)
    return rows


def write_case_files(rows: list[dict[str, Any]], output_dir: Path) -> None:
    filenames = {
        "ALL_KNOWN": "all_known.jsonl",
        "AR_ONLY_UNKNOWN": "ar_only_unknown.jsonl",
        "LLADA_ONLY_UNKNOWN": "llada_only_unknown.jsonl",
        "DREAM_ONLY_UNKNOWN": "dream_only_unknown.jsonl",
        "AR_ONLY_KNOWN": "ar_only_known.jsonl",
        "LLADA_ONLY_KNOWN": "llada_only_known.jsonl",
        "DREAM_ONLY_KNOWN": "dream_only_known.jsonl",
        "ALL_UNKNOWN": "all_unknown.jsonl",
        "UNCERTAIN_CASE": "uncertain_cases.jsonl",
    }
    for case, filename in filenames.items():
        write_jsonl(output_dir / "cases" / filename, (row for row in rows if row["knowledge_case"] == case))


def relation_statistics(rows: list[dict[str, Any]], output_dir: Path) -> None:
    stats = []
    for relation in sorted({row["relation"] for row in rows}):
        subset = [row for row in rows if row["relation"] == relation]
        stat: dict[str, Any] = {"relation": relation, "num_facts": len(subset)}
        for model_name in MODEL_ORDER:
            labels = Counter(row[f"{model_name}_knowledge_label"] for row in subset)
            for label in ("KNOWN", "UNKNOWN", "UNCERTAIN"):
                stat[f"{model_name}_{label.lower()}_count"] = labels[label]
                stat[f"{model_name}_{label.lower()}_percentage"] = 100 * labels[label] / len(subset)
        cases = Counter(row["knowledge_case"] for row in subset)
        for case in KNOWLEDGE_CASE_ORDER:
            stat[case.lower()] = cases[case]
        stats.append(stat)
    write_csv(output_dir / "relation_knowledge_results.csv", stats)


def make_summary(
    rows: list[dict[str, Any]], dataset_stats: dict[str, int], known_min: int, unknown_max: int
) -> str:
    total = len(rows)
    lines = [
        f"Total facts: {dataset_stats['total']}", f"Usable facts: {dataset_stats['usable']}",
        f"Skipped facts: {dataset_stats['skipped']}", f"Selected facts: {total}",
        f"KNOWN threshold: >= {known_min}/5", f"UNKNOWN threshold: <= {unknown_max}/5",
        "UNCERTAIN: all counts between the two thresholds", "", "Model knowledge labels:",
    ]
    for model_name in MODEL_ORDER:
        labels = Counter(row[f"{model_name}_knowledge_label"] for row in rows)
        lines.append(MODEL_SPECS[model_name][0] + ":")
        for label in ("KNOWN", "UNKNOWN", "UNCERTAIN"):
            lines.append(f"  {label}: {labels[label]}/{total} ({100 * labels[label] / total:.2f}%)")
    cases = Counter(row["knowledge_case"] for row in rows)
    lines.extend(["", "Knowledge case counts:"])
    for case in KNOWLEDGE_CASE_ORDER:
        lines.append(f"{case}: {cases[case]} ({100 * cases[case] / total:.2f}%)")
    return "\n".join(lines) + "\n"


def print_examples(rows: list[dict[str, Any]], seed: int) -> None:
    rng = random.Random(seed)
    for case in KNOWLEDGE_CASE_ORDER:
        candidates = [row for row in rows if row["knowledge_case"] == case]
        rng.shuffle(candidates)
        for row in candidates[:3]:
            print("\n" + "=" * 60 + f"\nKNOWLEDGE CASE: {case}\n" + "=" * 60)
            print(f"Relation: {row['relation']}\nSubject: {row['subject']}\nGold: {row['gold_answer']}")
            for model_name in MODEL_ORDER:
                print(
                    f"{MODEL_SPECS[model_name][0]}: "
                    f"{row[f'{model_name}_knowledge_correct_count']}/5 "
                    f"=> {row[f'{model_name}_knowledge_label']}"
                )
                for result in row[f"{model_name}_prompt_results"]:
                    print(
                        f"  P{result['template_index']}: {result['prompt']}\n"
                        f"    span={result['initial_answer_span']!r} "
                        f"correct={result['correct']} match={result['match_type']}"
                    )


def validate_or_write_config(args: argparse.Namespace) -> None:
    config_path = args.output_dir / "run_config.json"
    config = {
        "dataset": DATASET_ID, "num_samples": args.num_samples, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "relations": args.relations,
        "samples_per_relation": args.samples_per_relation, "models": args.models,
        "known_min_correct": args.known_min_correct,
        "unknown_max_correct": args.unknown_max_correct,
        "sample_source_results": str(args.sample_source_results) if args.sample_source_results else None,
        "templates": RELATION_TEMPLATE_SETS,
    }
    serialized = json.loads(json.dumps(config))
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot safely resume without {config_path}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != serialized:
            raise ValueError(f"Resume configuration mismatch\nold={previous}\nnew={serialized}")
    else:
        if any((args.output_dir / "cache").glob("*_generations.jsonl")):
            raise FileExistsError("Knowledge caches exist; use --resume or a new output directory")
        config_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_thresholds(args.known_min_correct, args.unknown_max_correct)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache").mkdir(exist_ok=True)
    (args.output_dir / "cases").mkdir(exist_ok=True)
    validate_or_write_config(args)

    print(f"transformers version: {transformers.__version__}")
    print(f"torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA device: cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        print("CUDA device: unavailable (cached-only resume remains supported)")
    print(f"dtype: {preferred_dtype()}")
    print(f"KNOWN >= {args.known_min_correct}/5; UNKNOWN <= {args.unknown_max_correct}/5")

    completed_results = args.output_dir / "popqa_knowledge_results.jsonl"
    if args.resume and completed_results.exists():
        samples, dataset_stats = restore_completed_samples(args.output_dir)
    elif args.sample_source_results is not None:
        samples, dataset_stats = load_samples_from_results(
            args.sample_source_results, args.num_samples
        )
    else:
        dataset = load_popqa()
        if not REQUIRED_COLUMNS.issubset(dataset.column_names):
            raise RuntimeError("Unexpected PopQA schema")
        analyze_relations(dataset, args.output_dir)
        samples, dataset_stats = prepare_samples(
            dataset, args.num_samples, args.seed, args.relations,
            args.samples_per_relation, args.output_dir,
        )
    caches = run_generation(
        samples, args.models, args.max_new_tokens, args.output_dir, args.reuse_cache_dir
    )
    rows = build_rows(
        samples, caches, args.known_min_correct, args.unknown_max_correct
    )
    write_jsonl(args.output_dir / "popqa_knowledge_results.jsonl", rows)
    write_csv(args.output_dir / "popqa_knowledge_results.csv", rows)
    write_case_files(rows, args.output_dir)
    relation_statistics(rows, args.output_dir)
    summary = make_summary(
        rows, dataset_stats, args.known_min_correct, args.unknown_max_correct
    )
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print_examples(rows, args.seed)


if __name__ == "__main__":
    main()
