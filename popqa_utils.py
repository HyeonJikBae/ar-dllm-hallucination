#!/usr/bin/env python3
"""Relation-aware PopQA completion case mining for AR and diffusion base LMs."""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import os
import random
import re
import string
import sys
import traceback
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import transformers
from datasets import load_dataset
try:
    from rapidfuzz.fuzz import ratio
except ImportError:
    # Only used for the optional ALL_WRONG answer-similarity subtype. Keep the
    # core loading/evaluation pipeline usable in minimal environments.
    from difflib import SequenceMatcher

    def ratio(left: str, right: str) -> float:
        return 100.0 * SequenceMatcher(None, left, right).ratio()

from model_generation import (
    LLADA_MASK_TOKEN_ID,
    clear_cuda_memory,
    generate_answer,
    load_model,
    model_parameter_count,
    preferred_dtype,
)


DATASET_ID = "akariasai/PopQA"
DEFAULT_NUM_SAMPLES = 2000
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 32
MODEL_ORDER = ("llama", "llada", "dream")
MODEL_SPECS = {
    "llama": ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B"),
    "llada": ("LLaDA-8B-Base", "GSAI-ML/LLaDA-8B-Base"),
    "dream": ("Dream-v0-Base-7B", "Dream-org/Dream-v0-Base-7B"),
}
REQUIRED_COLUMNS = {
    "id", "subj", "prop", "obj", "question", "possible_answers"
}

# Verified against real examples from every relation in PopQA's 14,267-row test
# split. Each relation has one fixed, direction-preserving template.
RELATION_TEMPLATES = {
    "author": ("author_v1", "{subject} was written by"),
    "capital": ("capital_v1", "The capital of {subject} is"),
    "capital of": ("capital_of_v1", "{subject} is the capital of"),
    "color": ("color_v1", "The color of {subject} is"),
    "composer": ("composer_v1", "{subject} was composed by"),
    "country": ("country_v1", "{subject} is located in"),
    "director": ("director_v1", "{subject} was directed by"),
    "father": ("father_v1", "The father of {subject} is"),
    "genre": ("genre_v1", "The genre of {subject} is"),
    "mother": ("mother_v1", "The mother of {subject} is"),
    "occupation": ("occupation_v1", "{subject}'s occupation is"),
    "place of birth": ("place_of_birth_v1", "{subject} was born in"),
    "producer": ("producer_v1", "{subject} was produced by"),
    "religion": ("religion_v1", "The religion of {subject} is"),
    "screenwriter": ("screenwriter_v1", "{subject} was written by"),
    "sport": ("sport_v1", "{subject} plays"),
}

ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("results/popqa_completion_2k"))
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--relations", nargs="+", default=None)
    parser.add_argument("--samples-per-relation", type=int, default=None)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str):
        return [str(value)]
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed if str(item).strip()]
        except (ValueError, TypeError, SyntaxError, json.JSONDecodeError):
            continue
    return [value] if value.strip() else []


def load_popqa() -> Any:
    dataset_dict = load_dataset(DATASET_ID)
    split_name = "test" if "test" in dataset_dict else next(iter(dataset_dict))
    dataset = dataset_dict[split_name]
    print(f"dataset size: {len(dataset)}")
    print(f"column names: {dataset.column_names}")
    print("first 5 examples:")
    for index in range(min(5, len(dataset))):
        print(json.dumps(dataset[index], ensure_ascii=False, sort_keys=True))
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise RuntimeError(f"PopQA schema missing required columns: {sorted(missing)}")
    return dataset


def analyze_relations(dataset: Any, output_dir: Path, examples_per_relation: int = 5) -> None:
    counts = Counter(dataset["prop"])
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in dataset:
        relation = sample["prop"]
        if len(examples[relation]) < examples_per_relation:
            examples[relation].append(dict(sample))

    print("\nrelation inventory:")
    print(f"{'relation':30} count")
    print("-" * 42)
    for relation in sorted(counts):
        print(f"{relation:30} {counts[relation]}")

    with (output_dir / "relation_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("relation", "count", "mapped", "template_id", "template")
        )
        writer.writeheader()
        for relation in sorted(counts):
            template = RELATION_TEMPLATES.get(relation)
            writer.writerow(
                {
                    "relation": relation,
                    "count": counts[relation],
                    "mapped": template is not None,
                    "template_id": template[0] if template else "",
                    "template": template[1] if template else "",
                }
            )

    with (output_dir / "relation_examples.txt").open("w", encoding="utf-8") as handle:
        for relation in sorted(counts):
            handle.write("=" * 70 + "\n")
            handle.write(f"RELATION: {relation}\nCOUNT: {counts[relation]}\n")
            template = RELATION_TEMPLATES.get(relation)
            handle.write(f"TEMPLATE: {template[1] if template else 'NO_RELATION_TEMPLATE'}\n")
            for sample in examples[relation]:
                handle.write(f"subject: {sample['subj']}\n")
                handle.write(f"relation: {sample['prop']}\n")
                handle.write(f"object: {sample['obj']}\n")
                handle.write(f"original question: {sample['question']}\n\n")


def build_relation_prompt(sample: dict[str, Any]) -> tuple[str, str] | None:
    template = RELATION_TEMPLATES.get(sample["prop"])
    if template is None:
        return None
    template_id, template_text = template
    return template_text.format(subject=str(sample["subj"]).strip()), template_id


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_samples(
    dataset: Any,
    requested_relations: list[str] | None,
    num_samples: int,
    samples_per_relation: int | None,
    seed: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    all_relations = set(dataset["prop"])
    unknown_requested = set(requested_relations or []).difference(all_relations)
    if unknown_requested:
        raise ValueError(f"Unknown PopQA relations: {sorted(unknown_requested)}")
    selected_relation_set = set(requested_relations or RELATION_TEMPLATES)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = []
    compatible_count = 0
    for raw_sample in dataset:
        sample = dict(raw_sample)
        prompt_result = build_relation_prompt(sample)
        if prompt_result is None:
            skipped.append(
                {"id": sample["id"], "relation": sample["prop"], "skip_reason": "NO_RELATION_TEMPLATE"}
            )
            continue
        compatible_count += 1
        if sample["prop"] not in selected_relation_set:
            continue
        prompt, template_id = prompt_result
        sample["prompt"] = prompt
        sample["template_id"] = template_id
        groups[sample["prop"]].append(sample)

    write_jsonl(output_dir / "skipped_no_template.jsonl", skipped)
    if not groups:
        raise ValueError("No template-compatible samples remain for the requested relations")
    if num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if samples_per_relation is not None and samples_per_relation < 1:
        raise ValueError("--samples-per-relation must be positive")

    rng = random.Random(seed)
    for relation, samples in groups.items():
        rng.shuffle(samples)
        if samples_per_relation is not None:
            groups[relation] = samples[:samples_per_relation]

    # Round-robin selection keeps relation counts within one while a relation
    # has remaining data, and naturally redistributes shortages (e.g. color).
    relations = sorted(groups)
    selected: list[dict[str, Any]] = []
    position = 0
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
        print(
            f"WARNING: requested {num_samples}, but only {len(selected)} samples "
            "were available under the relation constraints"
        )

    distribution = Counter(sample["prop"] for sample in selected)
    manifest = [
        {
            "id": sample["id"],
            "subject": sample["subj"],
            "relation": sample["prop"],
            "gold_answer": sample["obj"],
            "original_question": sample["question"],
            "prompt": sample["prompt"],
            "template_id": sample["template_id"],
        }
        for sample in selected
    ]
    write_jsonl(output_dir / "selected_samples.jsonl", manifest)
    with (output_dir / "selected_relation_distribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("relation", "count"))
        writer.writerows((relation, distribution[relation]) for relation in sorted(distribution))

    stats = {
        "total_rows": len(dataset),
        "compatible_rows": compatible_count,
        "skipped_rows": len(skipped),
    }
    print(f"template-compatible samples: {compatible_count}")
    print(f"skipped samples: {len(skipped)}")
    print(f"selected samples: {len(selected)}")
    print(f"selected relation distribution: {dict(sorted(distribution.items()))}")
    return selected, stats


def normalize_answer(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    value = value.replace("’", "'")
    value = "".join(" " if char in string.punctuation else char for char in value)
    value = re.sub(r"\s+", " ", value).strip()
    return ARTICLE_RE.sub("", value).strip()


def extract_initial_answer_span(raw_output: str) -> str:
    """Stop at the first strong early boundary without splitting initials."""
    text = raw_output.strip()
    if not text:
        return ""
    strong_positions = [position for mark in ("\n", ";", ":") if (position := text.find(mark)) >= 0]
    limit = min(strong_positions) if strong_positions else len(text)
    scan = text[:limit]
    for match in re.finditer(r"[.!?](?=\s|$|[A-Z])", scan):
        punctuation_index = match.start()
        prefix = scan[:punctuation_index]
        previous = re.search(r"([A-Za-z]+)$", prefix)
        token = previous.group(1) if previous else ""
        if len(token) == 1 or token.lower() in ABBREVIATIONS:
            continue
        limit = min(limit, punctuation_index)
        break
    return text[:limit].strip().rstrip(".,;:")


def answer_set(sample: dict[str, Any]) -> tuple[str, list[str]]:
    canonical = str(sample["obj"]).strip()
    candidates = [canonical]
    candidates.extend(parse_string_list(sample.get("possible_answers")))
    candidates.extend(parse_string_list(sample.get("o_aliases")))
    unique, seen = [], set()
    for candidate in candidates:
        normalized = normalize_answer(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return canonical, unique


def safe_contains(container: str, candidate: str) -> bool:
    if len(candidate) < 2:
        return False
    return re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", container) is not None


def safe_contains_early(container: str, candidate: str, max_start_tokens: int = 6) -> bool:
    """Match only near the start of the asserted span, not in later explanation."""
    match = re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", container)
    if match is None:
        return False
    return len(container[: match.start()].split()) < max_start_tokens


def evaluate_answer(
    raw_output: str,
    canonical: str,
    candidates: list[str],
    subject: str = "",
) -> dict[str, Any]:
    span = extract_initial_answer_span(raw_output)
    normalized_span = normalize_answer(span)
    normalized_canonical = normalize_answer(canonical)
    candidate_pairs = [(candidate, normalize_answer(candidate)) for candidate in candidates]
    normalized_subject = normalize_answer(subject)
    match_type, matched_gold = "none", None

    def usable_candidate(normalized: str) -> bool:
        """Reject subject-derived aliases; repeating the subject cannot prove the object."""
        if normalized == normalized_canonical:
            return True
        return not safe_contains(normalized_subject, normalized)

    if normalized_span and normalized_span == normalized_canonical:
        match_type, matched_gold = "exact", canonical
    else:
        for candidate, normalized in candidate_pairs:
            if (
                normalized != normalized_canonical
                and usable_candidate(normalized)
                and normalized_span == normalized
            ):
                match_type, matched_gold = "alias_exact", candidate
                break
    if match_type == "none":
        for candidate, normalized in candidate_pairs:
            if (
                usable_candidate(normalized)
                and len(normalized) >= 2
                and re.match(rf"^{re.escape(normalized)}(?!\w)", normalized_span)
            ):
                match_type, matched_gold = "prefix", candidate
                break
    if match_type == "none":
        for candidate, normalized in candidate_pairs:
            is_alias = normalized != normalized_canonical
            overlaps_subject = is_alias and safe_contains(normalized_subject, normalized)
            generic_alias = is_alias and set(normalized.split()) < set(normalized_canonical.split())
            if (
                not overlaps_subject
                and not generic_alias
                and safe_contains_early(normalized_span, normalized)
            ):
                match_type, matched_gold = "safe_substring", candidate
                break

    return {
        "answer_span": span,
        "normalized_answer_span": normalized_span,
        "is_correct": match_type != "none",
        "match_type": match_type,
        "matched_gold": matched_gold,
    }


def cache_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / "cache" / f"{model_name}_generations.jsonl"


def read_cache(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[str(record["id"])] = record
            except (json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(f"Invalid cache {path}:{line_number}: {exc}") from exc
    return records


def append_cache(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_generation(
    samples: list[dict[str, Any]], models: list[str], max_new_tokens: int, output_dir: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    caches = {name: read_cache(cache_path(output_dir, name)) for name in MODEL_ORDER}
    for model_name in MODEL_ORDER:
        if model_name not in models:
            continue
        pending = [sample for sample in samples if str(sample["id"]) not in caches[model_name]]
        if not pending:
            print(f"{model_name}: restored all {len(samples)} outputs from cache")
            continue
        display_name, model_id = MODEL_SPECS[model_name]
        model = tokenizer = None
        try:
            model, tokenizer = load_model(model_id, model_name)
            print("\n" + "=" * 50)
            print(f"MODEL: {display_name}")
            print("=" * 50)
            print(f"model class: {type(model).__module__}.{type(model).__name__}")
            print(f"tokenizer class: {type(tokenizer).__module__}.{type(tokenizer).__name__}")
            print(f"parameter count: {model_parameter_count(model):,}")
            if model_name == "llada":
                print(f"mask token ID: {LLADA_MASK_TOKEN_ID}")
            for number, sample in enumerate(pending, 1):
                raw_output = generate_answer(
                    model_name, model, tokenizer, sample["prompt"], max_new_tokens=max_new_tokens
                )
                record = {"id": sample["id"], "prompt": sample["prompt"], "raw_output": raw_output}
                append_cache(cache_path(output_dir, model_name), record)
                caches[model_name][str(sample["id"])] = record
                print(f"[{display_name} {number}/{len(pending)}] id={sample['id']}")
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


def spans_similar(left: str, right: str) -> bool:
    left_norm, right_norm = normalize_answer(left), normalize_answer(right)
    if not left_norm or not right_norm:
        return False
    return left_norm == right_norm or ratio(left_norm, right_norm) >= 85


def classify_case(
    llama_correct: bool, llada_correct: bool, dream_correct: bool
) -> str:
    """Map the three correctness booleans to one of exactly eight patterns."""
    pattern = (int(llama_correct), int(llada_correct), int(dream_correct))
    mapping = {
        (1, 1, 1): "ALL_CORRECT",
        (0, 1, 1): "AR_ONLY_WRONG",
        (1, 0, 1): "LLADA_ONLY_WRONG",
        (1, 1, 0): "DREAM_ONLY_WRONG",
        (1, 0, 0): "AR_ONLY_CORRECT",
        (0, 1, 0): "LLADA_ONLY_CORRECT",
        (0, 0, 1): "DREAM_ONLY_CORRECT",
        (0, 0, 0): "ALL_WRONG",
    }
    return mapping[pattern]


def categorize(row: dict[str, Any]) -> dict[str, Any]:
    correctness = [row.get(f"{name}_correct") for name in MODEL_ORDER]
    if any(value is None for value in correctness):
        return {
            "num_correct": None,
            "coarse_case_type": "INCOMPLETE_MODEL_SET",
            "case_type": "INCOMPLETE_MODEL_SET",
            "all_wrong_subtype": None,
        }

    llama, llada, dream = map(bool, correctness)
    num_correct = sum((llama, llada, dream))
    coarse_mapping = {
        3: "ALL_CORRECT",
        2: "TWO_CORRECT_ONE_WRONG",
        1: "ONE_CORRECT_TWO_WRONG",
        0: "ALL_WRONG",
    }
    case_type = classify_case(llama, llada, dream)
    all_wrong_subtype = None
    if case_type == "ALL_WRONG":
        pairwise_similar = (
            spans_similar(row["llama_answer_span"], row["llada_answer_span"]),
            spans_similar(row["llama_answer_span"], row["dream_answer_span"]),
            spans_similar(row["llada_answer_span"], row["dream_answer_span"]),
        )
        all_wrong_subtype = (
            "SAME_OR_SIMILAR_HALLUCINATION"
            if all(pairwise_similar)
            else "DIFFERENT_HALLUCINATIONS"
        )
    return {
        "num_correct": num_correct,
        "coarse_case_type": coarse_mapping[num_correct],
        "case_type": case_type,
        "all_wrong_subtype": all_wrong_subtype,
    }


def build_result_rows(
    samples: list[dict[str, Any]], caches: dict[str, dict[str, dict[str, Any]]]
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
        for model_name in MODEL_ORDER:
            cached = caches[model_name].get(str(sample["id"]))
            if cached:
                raw_output = cached["raw_output"]
                evaluation = evaluate_answer(
                    raw_output, canonical, candidates, subject=str(sample["subj"])
                )
                row[f"{model_name}_output"] = raw_output
                row[f"{model_name}_answer_span"] = evaluation["answer_span"]
                row[f"{model_name}_correct"] = evaluation["is_correct"]
                row[f"{model_name}_match_type"] = evaluation["match_type"]
                row[f"{model_name}_matched_gold"] = evaluation["matched_gold"]
            else:
                for suffix in ("output", "answer_span", "correct", "match_type", "matched_gold"):
                    row[f"{model_name}_{suffix}"] = None
        row.update(categorize(row))
        rows.append(row)
    return rows


def csv_value(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: csv_value(value) for key, value in row.items()} for row in rows)


def write_case_pools(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    case_dir = output_dir / "cases"
    filters: dict[str, Callable[[dict[str, Any]], bool]] = {
        "ar_correct_both_dllm_wrong.jsonl": lambda r: r["case_type"] == "AR_ONLY_CORRECT",
        "ar_wrong_both_dllm_correct.jsonl": lambda r: r["case_type"] == "AR_ONLY_WRONG",
        "ar_correct_llada_wrong.jsonl": lambda r: r["llama_correct"] is True and r["llada_correct"] is False,
        "ar_correct_dream_wrong.jsonl": lambda r: r["llama_correct"] is True and r["dream_correct"] is False,
        "ar_wrong_llada_correct.jsonl": lambda r: r["llama_correct"] is False and r["llada_correct"] is True,
        "ar_wrong_dream_correct.jsonl": lambda r: r["llama_correct"] is False and r["dream_correct"] is True,
        "llada_vs_dream_disagreement.jsonl": lambda r: r["llada_correct"] is not None and r["dream_correct"] is not None and r["llada_correct"] != r["dream_correct"],
        "all_wrong_different_answers.jsonl": lambda r: r["all_wrong_subtype"] == "DIFFERENT_HALLUCINATIONS",
        "all_wrong_similar_answers.jsonl": lambda r: r["all_wrong_subtype"] == "SAME_OR_SIMILAR_HALLUCINATION",
        "all_correct_control.jsonl": lambda r: r["case_type"] == "ALL_CORRECT",
        "all_wrong_control.jsonl": lambda r: r["case_type"] == "ALL_WRONG",
    }
    for filename, predicate in filters.items():
        write_jsonl(case_dir / filename, (row for row in rows if predicate(row)))


def relation_statistics(rows: list[dict[str, Any]], output_dir: Path) -> None:
    statistics = []
    for relation in sorted({row["relation"] for row in rows}):
        subset = [row for row in rows if row["relation"] == relation]
        count = len(subset)
        stat: dict[str, Any] = {"relation": relation, "num_samples": count}
        for model_name in MODEL_ORDER:
            correct = sum(row[f"{model_name}_correct"] is True for row in subset)
            stat[f"{model_name}_accuracy"] = correct / count if count else 0.0
        stat["ar_correct_llada_wrong"] = sum(
            row["llama_correct"] is True and row["llada_correct"] is False for row in subset
        )
        stat["ar_correct_dream_wrong"] = sum(
            row["llama_correct"] is True and row["dream_correct"] is False for row in subset
        )
        stat["ar_wrong_llada_correct"] = sum(
            row["llama_correct"] is False and row["llada_correct"] is True for row in subset
        )
        stat["ar_wrong_dream_correct"] = sum(
            row["llama_correct"] is False and row["dream_correct"] is True for row in subset
        )
        stat["all_correct"] = sum(row["case_type"] == "ALL_CORRECT" for row in subset)
        stat["all_wrong"] = sum(row["case_type"] == "ALL_WRONG" for row in subset)
        statistics.append(stat)
    write_csv(output_dir / "relation_results.csv", statistics)


def write_manual_review(rows: list[dict[str, Any]], output_dir: Path, seed: int) -> None:
    chosen = list(rows)
    random.Random(seed).shuffle(chosen)
    chosen = chosen[: min(50, len(chosen))]
    with (output_dir / "evaluator_manual_review.txt").open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write("=" * 70 + "\n")
            handle.write(f"ID: {row['id']}\nRelation: {row['relation']}\n")
            handle.write(f"Prompt: {row['prompt']}\nGold: {row['gold_answer']}\n")
            handle.write(f"Aliases: {row['aliases']}\n")
            for model_name in MODEL_ORDER:
                handle.write(f"\n{MODEL_SPECS[model_name][0]} raw: {row[f'{model_name}_output']}\n")
                handle.write(f"span: {row[f'{model_name}_answer_span']}\n")
                handle.write(
                    f"judgment: {row[f'{model_name}_correct']} / "
                    f"{row[f'{model_name}_match_type']} / matched={row[f'{model_name}_matched_gold']}\n"
                )
            handle.write(f"case: {row['case_type']}\n\n")


def summary_text(rows: list[dict[str, Any]], dataset_stats: dict[str, int]) -> str:
    total = len(rows)
    distribution = Counter(row["relation"] for row in rows)
    case_counts = Counter(row["case_type"] for row in rows)
    coarse_counts = Counter(row["coarse_case_type"] for row in rows)
    subtype_counts = Counter(
        row["all_wrong_subtype"] for row in rows if row["all_wrong_subtype"] is not None
    )
    lines = [
        f"Total PopQA rows: {dataset_stats['total_rows']}",
        f"Template-compatible rows: {dataset_stats['compatible_rows']}",
        f"Skipped rows: {dataset_stats['skipped_rows']}",
        f"Selected rows: {total}", "", "Selected relation distribution:",
    ]
    lines.extend(f"{relation}: {distribution[relation]}" for relation in sorted(distribution))
    lines.extend(["", "Accuracy:"])
    for model_name in MODEL_ORDER:
        evaluated = [row for row in rows if row[f"{model_name}_correct"] is not None]
        correct = sum(row[f"{model_name}_correct"] is True for row in evaluated)
        accuracy = 100 * correct / len(evaluated) if evaluated else 0.0
        lines.append(f"{MODEL_SPECS[model_name][0]}: {correct}/{len(evaluated)} ({accuracy:.2f}%)")
    lines.extend(["", "Coarse cases:"])
    for coarse in (
        "ALL_CORRECT", "TWO_CORRECT_ONE_WRONG", "ONE_CORRECT_TWO_WRONG", "ALL_WRONG",
    ):
        percentage = 100 * coarse_counts[coarse] / total if total else 0.0
        lines.append(f"{coarse}: {coarse_counts[coarse]} ({percentage:.2f}%)")
    lines.extend(["", "Exact behavioral patterns:"])
    for case_type in (
        "AR_ONLY_CORRECT", "AR_ONLY_WRONG", "LLADA_ONLY_WRONG",
        "DREAM_ONLY_WRONG", "LLADA_ONLY_CORRECT", "DREAM_ONLY_CORRECT",
        "ALL_CORRECT", "ALL_WRONG",
    ):
        percentage = 100 * case_counts[case_type] / total if total else 0.0
        lines.append(f"{case_type}: {case_counts[case_type]} ({percentage:.2f}%)")
    lines.extend(["", "ALL_WRONG subtypes:"])
    for subtype in ("SAME_OR_SIMILAR_HALLUCINATION", "DIFFERENT_HALLUCINATIONS"):
        percentage = 100 * subtype_counts[subtype] / total if total else 0.0
        lines.append(f"{subtype}: {subtype_counts[subtype]} ({percentage:.2f}%)")
    return "\n".join(lines) + "\n"


def print_examples(rows: list[dict[str, Any]], seed: int) -> None:
    categories = (
        "AR_ONLY_CORRECT", "AR_ONLY_WRONG",
        "LLADA_ONLY_WRONG", "DREAM_ONLY_WRONG",
        "LLADA_ONLY_CORRECT", "DREAM_ONLY_CORRECT",
        "ALL_CORRECT", "ALL_WRONG",
    )
    rng = random.Random(seed)
    for category in categories:
        candidates = [row for row in rows if row["case_type"] == category]
        rng.shuffle(candidates)
        for row in candidates[:5]:
            print("\n" + "=" * 50)
            print(f"CASE: {category}\n" + "=" * 50)
            print(f"Relation: {row['relation']}\nPrompt: {row['prompt']}\nGold: {row['gold_answer']}")
            for model_name in MODEL_ORDER:
                print(f"{MODEL_SPECS[model_name][0]}: {row[f'{model_name}_output']}")
                print(
                    f"Span: {row[f'{model_name}_answer_span']}\n"
                    f"Correct: {row[f'{model_name}_correct']} ({row[f'{model_name}_match_type']})"
                )


def validate_or_write_config(args: argparse.Namespace) -> None:
    config_path = args.output_dir / "run_config.json"
    config = {
        "dataset": DATASET_ID, "num_samples": args.num_samples, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "relations": args.relations,
        "samples_per_relation": args.samples_per_relation, "models": args.models,
        "templates": RELATION_TEMPLATES,
    }
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot safely resume without {config_path}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        # JSON converts template tuples to lists, so compare serialized forms.
        if previous != json.loads(json.dumps(config)):
            raise ValueError(f"Resume configuration mismatch\nold={previous}\nnew={config}")
    else:
        existing_caches = list((args.output_dir / "cache").glob("*_generations.jsonl"))
        if existing_caches:
            raise FileExistsError("Generation caches exist; use --resume or another --output-dir")
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache").mkdir(exist_ok=True)
    (args.output_dir / "cases").mkdir(exist_ok=True)
    validate_or_write_config(args)

    print(f"transformers version: {transformers.__version__}")
    print(f"torch version: {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a GPU with CUDA_VISIBLE_DEVICES")
    print(f"CUDA device: cuda:0 ({torch.cuda.get_device_name(0)})")
    print(f"dtype: {preferred_dtype()}")
    print(f"max_new_tokens/steps/block_length: {args.max_new_tokens}")
    print("Llama=greedy; LLaDA=low_confidence; Dream=entropy, alg_temp=0.0")

    dataset = load_popqa()
    analyze_relations(dataset, args.output_dir)
    samples, dataset_stats = prepare_samples(
        dataset, args.relations, args.num_samples, args.samples_per_relation,
        args.seed, args.output_dir,
    )
    caches = run_generation(samples, args.models, args.max_new_tokens, args.output_dir)
    rows = build_result_rows(samples, caches)
    write_jsonl(args.output_dir / "popqa_completion_all_results.jsonl", rows)
    write_csv(args.output_dir / "popqa_completion_all_results.csv", rows)
    write_case_pools(args.output_dir, rows)
    relation_statistics(rows, args.output_dir)
    write_manual_review(rows, args.output_dir, args.seed)
    summary = summary_text(rows, dataset_stats)
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print_examples(rows, args.seed)


if __name__ == "__main__":
    main()
