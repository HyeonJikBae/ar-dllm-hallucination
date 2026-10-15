#!/usr/bin/env python3
"""Build fact-level knowledge-status and paired Dream/Qwen output tables."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from experiment_utils import load_jsonl, utc_now, write_json
from score_results import write_csv


HERE = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = HERE / "results" / "dream_qwen2_5_full" / "analysis"

KNOWLEDGE_STATUS_ORDER = (
    "NO_KNOWLEDGE_OBSERVED",
    "SINGLE_HIT_CHANCE_CANDIDATE",
    "QUESTION_SENSITIVE_KNOWLEDGE",
    "CONSISTENT_KNOWLEDGE",
)

KNOWLEDGE_STATUS_KO = {
    "NO_KNOWLEDGE_OBSERVED": "지식이 관측되지 않음 (0/5)",
    "SINGLE_HIT_CHANCE_CANDIDATE": "우연 정답 후보 (1/5)",
    "QUESTION_SENSITIVE_KNOWLEDGE": "질문 형태 민감 지식 (2/5~4/5)",
    "CONSISTENT_KNOWLEDGE": "일관된 지식 (5/5)",
}

PAIR_CATEGORY_ORDER = (
    "BOTH_WRONG_SAME_FORM",
    "BOTH_WRONG_DIFFERENT_FORM",
    "DREAM_CORRECT_QWEN_WRONG",
    "DREAM_WRONG_QWEN_CORRECT",
    "BOTH_CORRECT_SAME_FORM",
    "BOTH_CORRECT_DIFFERENT_FORM",
)

PAIR_CATEGORY_KO = {
    "BOTH_WRONG_SAME_FORM": "두 모델 모두 오답 · 같은 출력 형태",
    "BOTH_WRONG_DIFFERENT_FORM": "두 모델 모두 오답 · 다른 출력 형태",
    "DREAM_CORRECT_QWEN_WRONG": "Dream 정답 · Qwen 오답",
    "DREAM_WRONG_QWEN_CORRECT": "Dream 오답 · Qwen 정답",
    "BOTH_CORRECT_SAME_FORM": "두 모델 모두 정답 · 같은 출력 형태",
    "BOTH_CORRECT_DIFFERENT_FORM": "두 모델 모두 정답 · 다른 출력 형태",
}

PAIR_OUTCOME_KO = {
    "BOTH_WRONG": "두 모델 모두 오답",
    "EXACTLY_ONE_CORRECT": "한 모델만 정답",
    "BOTH_CORRECT": "두 모델 모두 정답",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--left-model", default="dream")
    parser.add_argument("--right-model", default="qwen")
    parser.add_argument(
        "--correctness-field",
        default="official_alias_is_correct",
        help="Variant-result boolean used as the accepted correctness decision.",
    )
    parser.add_argument(
        "--fact-count-field",
        default="official_alias_correct_count",
        help="Fact-result 0..5 field corresponding to --correctness-field.",
    )
    parser.add_argument(
        "--fact-pattern-field",
        default="official_alias_correctness_pattern",
    )
    return parser.parse_args()


def knowledge_status(correct_count: int) -> str:
    if correct_count == 0:
        return "NO_KNOWLEDGE_OBSERVED"
    if correct_count == 1:
        return "SINGLE_HIT_CHANCE_CANDIDATE"
    if 2 <= correct_count <= 4:
        return "QUESTION_SENSITIVE_KNOWLEDGE"
    if correct_count == 5:
        return "CONSISTENT_KNOWLEDGE"
    raise ValueError(f"Expected a correct count from 0 through 5, got {correct_count}")


def paired_output_category(
    left_correct: bool,
    right_correct: bool,
    left_normalized: str,
    right_normalized: str,
) -> tuple[str, bool]:
    same_form = left_normalized == right_normalized
    if not left_correct and not right_correct:
        category = (
            "BOTH_WRONG_SAME_FORM"
            if same_form
            else "BOTH_WRONG_DIFFERENT_FORM"
        )
    elif left_correct and not right_correct:
        category = "DREAM_CORRECT_QWEN_WRONG"
    elif not left_correct and right_correct:
        category = "DREAM_WRONG_QWEN_CORRECT"
    else:
        category = (
            "BOTH_CORRECT_SAME_FORM"
            if same_form
            else "BOTH_CORRECT_DIFFERENT_FORM"
        )
    return category, same_form


def pair_outcome(category: str) -> str:
    if category.startswith("BOTH_WRONG"):
        return "BOTH_WRONG"
    if category.startswith("BOTH_CORRECT"):
        return "BOTH_CORRECT"
    return "EXACTLY_ONE_CORRECT"


def build_fact_tables(
    fact_results: list[dict[str, Any]],
    models: tuple[str, str],
    count_field: str,
    pattern_field: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    selected = [row for row in fact_results if row["model_name"] in models]
    detailed: list[dict[str, Any]] = []
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected:
        count = int(row[count_field])
        status = knowledge_status(count)
        enriched = {
            "model_name": row["model_name"],
            "fact_id": row["fact_id"],
            "question_id": row["question_id"],
            "original_question": row["original_question"],
            "canonical_answer": row["canonical_answer"],
            "correct_count": count,
            "fact_bin": f"{count}/5",
            "correctness_pattern": row[pattern_field],
            "knowledge_status": status,
            "knowledge_status_ko": KNOWLEDGE_STATUS_KO[status],
            "refusal_count": row["refusal_count"],
        }
        key = (row["model_name"], row["fact_id"])
        if key in indexed:
            raise ValueError(f"Duplicate fact result: {key}")
        indexed[key] = enriched
        detailed.append(enriched)

    fact_ids_by_model = {
        model: {row["fact_id"] for row in detailed if row["model_name"] == model}
        for model in models
    }
    if fact_ids_by_model[models[0]] != fact_ids_by_model[models[1]]:
        raise ValueError("The two models do not contain identical fact IDs")
    fact_ids = sorted(fact_ids_by_model[models[0]])

    summary = []
    for model in models:
        model_rows = [row for row in detailed if row["model_name"] == model]
        counts = Counter(row["knowledge_status"] for row in model_rows)
        for status in KNOWLEDGE_STATUS_ORDER:
            summary.append(
                {
                    "model_name": model,
                    "knowledge_status": status,
                    "knowledge_status_ko": KNOWLEDGE_STATUS_KO[status],
                    "fact_count": counts.get(status, 0),
                    "proportion": counts.get(status, 0) / len(model_rows),
                }
            )

    cross = []
    for fact_id in fact_ids:
        left = indexed[(models[0], fact_id)]
        right = indexed[(models[1], fact_id)]
        cross.append(
            {
                "fact_id": fact_id,
                "question_id": left["question_id"],
                "original_question": left["original_question"],
                "canonical_answer": left["canonical_answer"],
                f"{models[0]}_correct_count": left["correct_count"],
                f"{models[0]}_knowledge_status": left["knowledge_status"],
                f"{models[1]}_correct_count": right["correct_count"],
                f"{models[1]}_knowledge_status": right["knowledge_status"],
                "same_knowledge_status": (
                    left["knowledge_status"] == right["knowledge_status"]
                ),
            }
        )
    cross_counts = Counter(
        (
            row[f"{models[0]}_knowledge_status"],
            row[f"{models[1]}_knowledge_status"],
        )
        for row in cross
    )
    cross_matrix = [
        {
            f"{models[0]}_knowledge_status": left_status,
            f"{models[1]}_knowledge_status": right_status,
            "fact_count": cross_counts.get((left_status, right_status), 0),
            "proportion": cross_counts.get((left_status, right_status), 0)
            / len(fact_ids),
        }
        for left_status in KNOWLEDGE_STATUS_ORDER
        for right_status in KNOWLEDGE_STATUS_ORDER
    ]
    return detailed, summary, cross, cross_matrix


def build_pair_tables(
    variant_results: list[dict[str, Any]],
    fact_status_rows: list[dict[str, Any]],
    models: tuple[str, str],
    correctness_field: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in variant_results:
        if row["model_name"] not in models:
            continue
        key = (row["model_name"], row["fact_id"], row["variant_id"])
        if key in indexed:
            raise ValueError(f"Duplicate variant result: {key}")
        indexed[key] = row
    fact_status = {
        (row["model_name"], row["fact_id"]): row["knowledge_status"]
        for row in fact_status_rows
    }
    task_ids = sorted({(key[1], key[2]) for key in indexed})
    detailed = []
    for fact_id, variant_id in task_ids:
        left_key = (models[0], fact_id, variant_id)
        right_key = (models[1], fact_id, variant_id)
        if left_key not in indexed or right_key not in indexed:
            raise ValueError(f"Missing paired output for {fact_id}:{variant_id}")
        left = indexed[left_key]
        right = indexed[right_key]
        if left["question"] != right["question"]:
            raise ValueError(f"Paired question text differs for {fact_id}:{variant_id}")
        left_correct = bool(left[correctness_field])
        right_correct = bool(right[correctness_field])
        category, same_form = paired_output_category(
            left_correct,
            right_correct,
            str(left["normalized_answer"]),
            str(right["normalized_answer"]),
        )
        outcome = pair_outcome(category)
        detailed.append(
            {
                "fact_id": fact_id,
                "question_id": left["question_id"],
                "variant_id": variant_id,
                "variant_index": left["variant_index"],
                "question": left["question"],
                "canonical_answer": left["canonical_answer"],
                f"{models[0]}_knowledge_status": fact_status[(models[0], fact_id)],
                f"{models[1]}_knowledge_status": fact_status[(models[1], fact_id)],
                f"{models[0]}_output": left["scored_output"],
                f"{models[0]}_normalized_output": left["normalized_answer"],
                f"{models[0]}_is_correct": left_correct,
                f"{models[1]}_output": right["scored_output"],
                f"{models[1]}_normalized_output": right["normalized_answer"],
                f"{models[1]}_is_correct": right_correct,
                "same_normalized_output_form": same_form,
                "pair_outcome": outcome,
                "pair_outcome_ko": PAIR_OUTCOME_KO[outcome],
                "pair_category": category,
                "pair_category_ko": PAIR_CATEGORY_KO[category],
            }
        )

    counts = Counter(row["pair_category"] for row in detailed)
    summary = [
        {
            "pair_category": category,
            "pair_category_ko": PAIR_CATEGORY_KO[category],
            "question_pair_count": counts.get(category, 0),
            "proportion": counts.get(category, 0) / len(detailed),
        }
        for category in PAIR_CATEGORY_ORDER
    ]
    outcome_counts = Counter(row["pair_outcome"] for row in detailed)
    outcome_summary = [
        {
            "pair_outcome": outcome,
            "pair_outcome_ko": PAIR_OUTCOME_KO[outcome],
            "question_pair_count": outcome_counts.get(outcome, 0),
            "proportion": outcome_counts.get(outcome, 0) / len(detailed),
        }
        for outcome in ("BOTH_WRONG", "EXACTLY_ONE_CORRECT", "BOTH_CORRECT")
    ]
    return detailed, summary, outcome_summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.analysis_dir / "classification_tables"
    models = (args.left_model, args.right_model)
    facts = load_jsonl(args.analysis_dir / "fact_results.jsonl")
    variants = load_jsonl(args.analysis_dir / "variant_results.jsonl")
    fact_detailed, fact_summary, fact_cross, fact_cross_matrix = build_fact_tables(
        facts,
        models,
        args.fact_count_field,
        args.fact_pattern_field,
    )
    pair_detailed, pair_summary, pair_outcome_summary = build_pair_tables(
        variants,
        fact_detailed,
        models,
        args.correctness_field,
    )

    write_csv(output_dir / "knowledge_status_by_fact.csv", fact_detailed, list(fact_detailed[0]))
    write_csv(output_dir / "knowledge_status_summary.csv", fact_summary, list(fact_summary[0]))
    write_csv(output_dir / "knowledge_status_cross_model.csv", fact_cross, list(fact_cross[0]))
    write_csv(
        output_dir / "knowledge_status_cross_matrix.csv",
        fact_cross_matrix,
        list(fact_cross_matrix[0]),
    )
    write_csv(output_dir / "model_pair_by_question.csv", pair_detailed, list(pair_detailed[0]))
    write_csv(output_dir / "model_pair_category_summary.csv", pair_summary, list(pair_summary[0]))
    write_csv(
        output_dir / "model_pair_outcome_summary.csv",
        pair_outcome_summary,
        list(pair_outcome_summary[0]),
    )
    write_json(
        output_dir / "classification_metadata.json",
        {
            "created_at_utc": utc_now(),
            "models": list(models),
            "correctness_field": args.correctness_field,
            "fact_count_field": args.fact_count_field,
            "fact_pattern_field": args.fact_pattern_field,
            "fact_rows": len(fact_detailed),
            "paired_question_rows": len(pair_detailed),
            "same_form_rule": "normalized_answer exact equality",
            "epistemic_caveat": (
                "Knowledge labels are operational categories over five prompts, "
                "not direct proof of internal knowledge or chance."
            ),
        },
    )
    print(f"Fact knowledge-status rows: {len(fact_detailed):,}")
    print(f"Paired question rows: {len(pair_detailed):,}")
    print(f"Classification tables -> {output_dir}")


if __name__ == "__main__":
    main()
