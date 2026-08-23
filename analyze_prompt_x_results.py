#!/usr/bin/env python3
"""Analyze Prompt X completion outputs at fact and individual-output levels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_prompt_o_results import (
    DISPLAY,
    MODELS,
    is_severe_failure,
    knowledge_case,
    knowledge_label,
    load_prior_decisions,
    pct,
)
from popqa_utils import normalize_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/popqa_knowledge_2k/popqa_knowledge_results.jsonl"),
        help="Prompt X result JSONL containing one row per PopQA fact",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/popqa_knowledge_2k/final_analysis"),
    )
    parser.add_argument(
        "--reviewed-classifications",
        type=Path,
        default=None,
        help="Optional reviewed CSV whose unanimous identical-span decisions are reused",
    )
    parser.add_argument("--expected-rows", type=int, default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    facts = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not facts:
        raise ValueError(f"No rows found in {args.input}")
    if args.expected_rows is not None and len(facts) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows:,} rows, found {len(facts):,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviewed = load_prior_decisions(args.reviewed_classifications)
    output_rows: list[dict[str, Any]] = []
    fact_rows: list[dict[str, Any]] = []
    label_counts = {model: Counter() for model in MODELS}
    score_counts = {model: Counter() for model in MODELS}
    case_counts: Counter[str] = Counter()

    for fact in facts:
        counts: dict[str, int] = {}
        labels: dict[str, str] = {}
        for model in MODELS:
            prompt_results = fact[f"{model}_prompt_results"]
            if len(prompt_results) != 5:
                raise ValueError(
                    f"fact={fact['id']} model={model}: expected 5 prompt results, "
                    f"found {len(prompt_results)}"
                )
            correct_count = 0
            for result in prompt_results:
                span = str(result.get("initial_answer_span") or "").strip()
                correct = bool(result.get("correct"))
                correct_count += correct
                if correct:
                    final_label = "CORRECT"
                    reason = "gold answer or accepted alias"
                elif is_severe_failure(span):
                    final_label = "NO_ANSWER_OR_FAILURE"
                    reason = "no concrete answer or severe generation failure"
                else:
                    prior = reviewed.get((model, fact["prop"], normalize_answer(span)))
                    if prior == "NO_ANSWER_OR_FAILURE":
                        final_label = prior
                        reason = "identical normalized response was previously reviewed as failure"
                    else:
                        final_label = "HALLUCINATION"
                        reason = "concrete incorrect factual answer"
                label_counts[model][final_label] += 1
                output_rows.append(
                    {
                        "fact_id": fact["id"],
                        "relation": fact["prop"],
                        "subject": fact["subj"],
                        "gold_answer": fact["obj"],
                        "model": model,
                        "template_index": result["template_index"],
                        "prompt": result["prompt"],
                        "initial_answer_span": span,
                        "correct": correct,
                        "final_label": final_label,
                        "final_reason": reason,
                    }
                )
            counts[model] = correct_count
            labels[model] = knowledge_label(correct_count)
            score_counts[model][correct_count] += 1

        case = knowledge_case(labels)
        case_counts[case] += 1
        fact_rows.append(
            {
                "fact_id": fact["id"],
                "relation": fact["prop"],
                "subject": fact["subj"],
                "gold_answer": fact["obj"],
                "llama_correct_count": counts["llama"],
                "llada_correct_count": counts["llada"],
                "dream_correct_count": counts["dream"],
                "llama_label": labels["llama"],
                "llada_label": labels["llada"],
                "dream_label": labels["dream"],
                "knowledge_case": case,
            }
        )

    output_fields = list(output_rows[0])
    fact_fields = list(fact_rows[0])
    write_csv(args.output_dir / "classified_all_outputs_3way.csv", output_rows, output_fields)
    write_csv(args.output_dir / "fact_results.csv", fact_rows, fact_fields)
    with (args.output_dir / "classified_all_outputs_3way.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    strong_ar = [
        row for row in fact_rows
        if (row["llama_correct_count"], row["llada_correct_count"], row["dream_correct_count"])
        == (5, 0, 0)
    ]
    strong_reverse = [
        row for row in fact_rows
        if row["llama_correct_count"] == 0
        and row["llada_correct_count"] >= 4
        and row["dream_correct_count"] >= 4
    ]
    write_csv(args.output_dir / "strong_ar_5_0_0_candidates.csv", strong_ar, fact_fields)
    write_csv(
        args.output_dir / "strong_reverse_0_4plus_4plus_candidates.csv",
        strong_reverse,
        fact_fields,
    )

    output_total = len(facts) * 5
    report = [
        "# Prompt X completion 전체 분석 보고서", "", "## 분석 범위", "",
        f"- PopQA 사실 {len(facts):,}개", "- 사실당 completion paraphrase 5개",
        f"- 총 출력 {output_total * len(MODELS):,}개", "",
        "## Fact-level knowledge 판정", "",
        "| 모델 | KNOWN | UNKNOWN | UNCERTAIN |", "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        known = score_counts[model][4] + score_counts[model][5]
        unknown = score_counts[model][0]
        uncertain = sum(score_counts[model][n] for n in (1, 2, 3))
        report.append(
            f"| {DISPLAY[model]} | {known} ({pct(known,len(facts))}) | "
            f"{unknown} ({pct(unknown,len(facts))}) | "
            f"{uncertain} ({pct(uncertain,len(facts))}) |"
        )
    report += [
        "", "## 출력 단위 3분류", "",
        "| 모델 | 정답 | 할루시네이션 | 답 미제시·생성 붕괴 |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        counts = label_counts[model]
        report.append(
            f"| {DISPLAY[model]} | {counts['CORRECT']:,} ({pct(counts['CORRECT'],output_total)}) | "
            f"{counts['HALLUCINATION']:,} ({pct(counts['HALLUCINATION'],output_total)}) | "
            f"{counts['NO_ANSWER_OR_FAILURE']:,} "
            f"({pct(counts['NO_ANSWER_OR_FAILURE'],output_total)}) |"
        )
    report += [
        "", "## Case 분포", "", "| Case | 개수 | 비율 |", "|---|---:|---:|",
    ]
    for case in (
        "AR_ONLY_KNOWN", "AR_ONLY_UNKNOWN", "LLADA_ONLY_KNOWN",
        "DREAM_ONLY_KNOWN", "LLADA_ONLY_UNKNOWN", "DREAM_ONLY_UNKNOWN",
        "ALL_KNOWN", "ALL_UNKNOWN", "UNCERTAIN_CASE",
    ):
        report.append(f"| `{case}` | {case_counts[case]} | {pct(case_counts[case],len(facts))} |")
    report += [
        "", "## 최대 대비 후보", "",
        f"- `5/5–0/5–0/5`: {len(strong_ar)}개",
        f"- `0/5–4+/5–4+/5`: {len(strong_reverse)}개", "",
        "> 자동 분류는 screening 결과다. 복수 정답, entity ambiguity 및 output failure를 수동 검토한 뒤 최종 사례를 확정해야 한다.", "",
    ]
    text = "\n".join(report)
    (args.output_dir / "FINAL_ANALYSIS_REPORT_KO.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
