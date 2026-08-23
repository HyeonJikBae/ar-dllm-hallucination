#!/usr/bin/env python3
"""Analyze instructed PopQA QA outputs with diffusion role-suffix cleanup."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from popqa_utils import answer_set, normalize_answer, safe_contains


MODELS = ("llama", "llada", "dream")
DISPLAY = {"llama": "Llama-3.1-8B", "llada": "LLaDA-8B-Base", "dream": "Dream-v0-Base-7B"}
ROLE_SUFFIX = {"llada": "input", "dream": "user"}
LABELS = ("CORRECT", "HALLUCINATION", "NO_ANSWER_OR_FAILURE")

EXPLICIT_NONANSWER = re.compile(
    r"(?i)\b(?:unknown|not known|not provided|not specified|not mentioned|"
    r"cannot (?:be )?(?:determine|answer)|can't (?:determine|answer)|"
    r"no (?:information|answer)|insufficient information|n/?a)\b"
)
GENERIC_PATTERNS = re.compile(
    r"(?i)^(?:a|an|the)?\s*(?:person|man|woman|place|city|country|region|"
    r"writer|author|composer|director|producer|actor|artist|athlete|player|"
    r"professional|religion|sport|genre|color|colour|occupation|profession|"
    r"member of (?:the )?(?:band|family|team)|same name)(?:\b.*)?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/popqa_qa_instructed_knowledge_2k/popqa_knowledge_results.jsonl"),
        help="Completed experiment JSONL containing one row per PopQA fact",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/popqa_qa_instructed_knowledge_2k/final_analysis_rechecked"),
        help="Directory for corrected fact labels, output labels, candidates, and report",
    )
    parser.add_argument(
        "--completion-classifications",
        type=Path,
        default=None,
        help="Optional reviewed completion-condition labels to reuse for identical spans",
    )
    parser.add_argument(
        "--expected-rows", type=int, default=None,
        help="Fail unless the input has exactly this many facts (for run-integrity checks)",
    )
    return parser.parse_args()


def strip_role_suffix(span: str, model: str) -> tuple[str, bool]:
    """Remove one model-specific leaked role marker, never repeated stripping."""
    original = span.strip()
    cleaned = original
    marker = ROLE_SUFFIX.get(model)
    if marker:
        cleaned = re.sub(
            rf"(?i)[.\s_-]*{marker}$", "", cleaned, count=1
        ).strip().rstrip(".,;:")
    # Custom-code generations can leak the next prompt boundary directly onto
    # the answer (e.g. "Martin ScorseseQ" or "GunnedahQuestion").
    cleaned = re.sub(r"(?:Question|Answer)$", "", cleaned, count=1).strip().rstrip(".,;:")
    cleaned = re.sub(r"Q$", "", cleaned, count=1).strip().rstrip(".,;:")
    return cleaned, cleaned != original


def is_severe_failure(span: str) -> bool:
    value = span.strip()
    if not value:
        return True
    norm = normalize_answer(value)
    if not norm or norm in {"user", "input", "assistant", "question", "answer"}:
        return True
    if EXPLICIT_NONANSWER.search(value) or GENERIC_PATTERNS.fullmatch(value):
        return True
    tokens = norm.split()
    if len(tokens) >= 12:
        trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
        if trigrams and len(set(trigrams)) / len(trigrams) < 0.45:
            return True
    if len(tokens) >= 8 and len(set(tokens)) <= 3:
        return True
    return False


def evaluate_qa_answer(
    span: str, canonical: str, candidates: list[str], subject: str
) -> tuple[bool, str | None]:
    """Conservatively recognize an answer embedded in a natural QA sentence."""
    normalized_span = normalize_answer(span)
    normalized_canonical = normalize_answer(canonical)
    normalized_subject = normalize_answer(subject)
    if not normalized_span:
        return False, None
    pairs = sorted(
        ((candidate, normalize_answer(candidate)) for candidate in candidates),
        key=lambda pair: len(pair[1]), reverse=True,
    )
    for candidate, normalized in pairs:
        if not normalized:
            continue
        is_canonical = normalized == normalized_canonical
        if not is_canonical and safe_contains(normalized_subject, normalized):
            continue
        if normalized_span == normalized:
            return True, candidate
        # Do not treat a target mention inside the repeated question subject as
        # an answer (e.g. target Saanen in "Saanen District ... is Kuopio").
        answer_content = normalized_span
        if normalized_subject:
            answer_content = re.sub(
                rf"(?<!\w){re.escape(normalized_subject)}(?!\w)", " ", answer_content
            )
        match = re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", answer_content)
        if match is None:
            continue
        preceding = answer_content[: match.start()].split()[-4:]
        if any(word in {"not", "isnt", "wasnt", "never", "neither"} for word in preceding):
            continue
        # A short alias wholly contained in the canonical answer is unsafe in a
        # longer sentence (e.g. "football" inside "National Football League").
        alias_is_generic_subset = (
            not is_canonical
            and set(normalized.split()) < set(normalized_canonical.split())
        )
        if alias_is_generic_subset:
            continue
        return True, candidate
    return False, None


def load_prior_decisions(path: Path | None) -> dict[tuple[str, str, str], str]:
    """Reuse unanimous decisions for identical old answer spans when available."""
    votes: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["final_label"] == "CORRECT":
                continue
            key = (row["model"], row["relation"], normalize_answer(row["initial_answer_span"]))
            votes[key][row["final_label"]] += 1
    decisions = {}
    for key, counts in votes.items():
        if len(counts) == 1 and key[2]:
            decisions[key] = next(iter(counts))
    return decisions


def knowledge_label(correct_count: int) -> str:
    if correct_count >= 4:
        return "KNOWN"
    if correct_count == 0:
        return "UNKNOWN"
    return "UNCERTAIN"


def knowledge_case(labels: dict[str, str]) -> str:
    if any(value == "UNCERTAIN" for value in labels.values()):
        return "UNCERTAIN_CASE"
    pattern = tuple(labels[m] == "KNOWN" for m in MODELS)
    mapping = {
        (True, True, True): "ALL_KNOWN",
        (True, False, False): "AR_ONLY_KNOWN",
        (False, True, True): "AR_ONLY_UNKNOWN",
        (False, True, False): "LLADA_ONLY_KNOWN",
        (False, False, True): "DREAM_ONLY_KNOWN",
        (True, False, True): "LLADA_ONLY_UNKNOWN",
        (True, True, False): "DREAM_ONLY_UNKNOWN",
        (False, False, False): "ALL_UNKNOWN",
    }
    return mapping[pattern]


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.2f}%"


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows:,} rows, found {len(rows):,}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior = load_prior_decisions(args.completion_classifications)

    output_rows: list[dict[str, Any]] = []
    fact_rows: list[dict[str, Any]] = []
    label_counts = {model: Counter() for model in MODELS}
    raw_correct = Counter()
    corrected_correct = Counter()
    suffix_counts = Counter()
    score_counts = {model: Counter() for model in MODELS}
    case_counts = Counter()

    for fact in rows:
        canonical, candidates = answer_set(fact)
        per_model_counts: dict[str, int] = {}
        per_model_labels: dict[str, str] = {}
        for model in MODELS:
            count = 0
            for result in fact[f"{model}_prompt_results"]:
                original_span = result["initial_answer_span"]
                cleaned_span, stripped = strip_role_suffix(original_span, model)
                raw_is_correct = bool(result["correct"])
                rescored_correct, rescored_match = evaluate_qa_answer(
                    cleaned_span, canonical, candidates, str(fact["subj"])
                )
                # The stored matcher can mistake a target embedded in the
                # repeated subject for an answer; the audited QA matcher is the
                # sole source of the final correctness label.
                is_correct = rescored_correct
                raw_correct[model] += raw_is_correct
                corrected_correct[model] += is_correct
                suffix_counts[model] += stripped
                count += is_correct

                if is_correct:
                    label, reason = "CORRECT", "gold answer or accepted alias after role-suffix cleanup"
                elif is_severe_failure(cleaned_span):
                    label, reason = "NO_ANSWER_OR_FAILURE", "no concrete answer or severe generation failure"
                else:
                    old = prior.get((model, fact["prop"], normalize_answer(cleaned_span)))
                    if old == "NO_ANSWER_OR_FAILURE":
                        label, reason = old, "same normalized response was previously reviewed as non-answer/failure"
                    else:
                        label, reason = "HALLUCINATION", "concrete incorrect factual answer"
                label_counts[model][label] += 1
                output_rows.append(
                    {
                        "fact_id": fact["id"],
                        "relation": fact["prop"],
                        "subject": fact["subj"],
                        "gold_answer": fact["obj"],
                        "model": model,
                        "template_index": result["template_index"],
                        "prompt": result["prompt"],
                        "original_initial_answer_span": original_span,
                        "cleaned_initial_answer_span": cleaned_span,
                        "role_suffix_removed": stripped,
                        "stored_correct": raw_is_correct,
                        "corrected_correct": is_correct,
                        "rescored_matched_gold": rescored_match,
                        "final_label": label,
                        "final_reason": reason,
                    }
                )
            per_model_counts[model] = count
            per_model_labels[model] = knowledge_label(count)
            score_counts[model][count] += 1
        case = knowledge_case(per_model_labels)
        case_counts[case] += 1
        fact_rows.append(
            {
                "fact_id": fact["id"], "relation": fact["prop"], "subject": fact["subj"],
                "gold_answer": fact["obj"], "llama_correct_count": per_model_counts["llama"],
                "llada_correct_count": per_model_counts["llada"],
                "dream_correct_count": per_model_counts["dream"],
                "llama_label": per_model_labels["llama"], "llada_label": per_model_labels["llada"],
                "dream_label": per_model_labels["dream"], "knowledge_case": case,
            }
        )

    def write_csv(
        path: Path, records: list[dict[str, Any]], fieldnames: list[str] | None = None
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            names = fieldnames or (list(records[0]) if records else [])
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader(); writer.writerows(records)

    write_csv(args.output_dir / "classified_all_outputs_3way.csv", output_rows)
    write_csv(args.output_dir / "corrected_fact_results.csv", fact_rows)
    strong_rows = [
        row for row in fact_rows
        if (row["llama_correct_count"], row["llada_correct_count"], row["dream_correct_count"])
        == (5, 0, 0)
    ]
    reverse_rows = [
        row for row in fact_rows
        if row["llama_correct_count"] == 0
        and row["llada_correct_count"] >= 4
        and row["dream_correct_count"] >= 4
    ]
    write_csv(
        args.output_dir / "strong_ar_5_0_0_candidates.csv",
        strong_rows,
        list(fact_rows[0]),
    )
    # Preserve a header even when the strict reverse set is empty.
    if reverse_rows:
        write_csv(args.output_dir / "strong_reverse_0_4plus_4plus_candidates.csv", reverse_rows)
    else:
        with (args.output_dir / "strong_reverse_0_4plus_4plus_candidates.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fact_rows[0]))
            writer.writeheader()
    with (args.output_dir / "classified_all_outputs_3way.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = [
        "# 의문형 지시 프롬프트 전체 분석 보고서", "", "## 분석 범위", "",
        f"- PopQA 사실 {len(rows):,}개", "- 사실당 질문형 paraphrase 5개", "- 모델 3개",
        f"- 총 출력 {len(rows) * 5 * len(MODELS):,}개", "- KNOWN: 4~5/5, UNKNOWN: 0/5, UNCERTAIN: 1~3/5", "",
        "## 역할 토큰 잔여물 보정", "",
        "Diffusion LM 출력 끝의 `input`/`user` 문자열을 생성 내용과 분리하고, 문장형 QA 출력 안의 canonical answer 또는 안전한 alias를 인식해 재채점했다.", "",
        "| 모델 | 저장 정답 | 보정 정답 | 역할 접미사 발견 |", "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        report.append(f"| {DISPLAY[model]} | {raw_correct[model]:,} | {corrected_correct[model]:,} | {suffix_counts[model]:,} |")
    report += ["", "## 보정 후 5회 점수 분포", "", "| 모델 | 5/5 | 4/5 | 3/5 | 2/5 | 1/5 | 0/5 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        report.append("| " + DISPLAY[model] + " | " + " | ".join(str(score_counts[model][n]) for n in (5,4,3,2,1,0)) + " |")
    report += ["", "## 보정 후 지식 판정", "", "| 모델 | KNOWN | UNCERTAIN | UNKNOWN |", "|---|---:|---:|---:|"]
    for model in MODELS:
        known=sum(score_counts[model][n] for n in (4,5)); uncertain=sum(score_counts[model][n] for n in (1,2,3)); unknown=score_counts[model][0]
        report.append(f"| {DISPLAY[model]} | {known} ({pct(known,len(rows))}) | {uncertain} ({pct(uncertain,len(rows))}) | {unknown} ({pct(unknown,len(rows))}) |")
    report += ["", "## 보정 후 사실 단위 case", "", "| Case | 개수 | 비율 |", "|---|---:|---:|"]
    order=("AR_ONLY_KNOWN","AR_ONLY_UNKNOWN","LLADA_ONLY_KNOWN","DREAM_ONLY_KNOWN","LLADA_ONLY_UNKNOWN","DREAM_ONLY_UNKNOWN","ALL_KNOWN","ALL_UNKNOWN","UNCERTAIN_CASE")
    for case in order:
        report.append(f"| `{case}` | {case_counts[case]} | {pct(case_counts[case],len(rows))} |")
    report += ["", "## 출력 단위 최종 3분류", "", "| 모델 | 정답 | 할루시네이션 | 답 미제시·생성 붕괴 |", "|---|---:|---:|---:|"]
    for model in MODELS:
        c=label_counts[model]
        output_total = len(rows) * 5
        report.append(f"| {DISPLAY[model]} | {c['CORRECT']:,} ({pct(c['CORRECT'],output_total)}) | {c['HALLUCINATION']:,} ({pct(c['HALLUCINATION'],output_total)}) | {c['NO_ANSWER_OR_FAILURE']:,} ({pct(c['NO_ANSWER_OR_FAILURE'],output_total)}) |")
    report += [
        "", "## 최대 대비 후보", "",
        f"- `Llama 5/5, LLaDA 0/5, Dream 0/5`: **{len(strong_rows)}개**",
        f"- `Llama 0/5, LLaDA >=4/5, Dream >=4/5`: **{len(reverse_rows)}개**",
        "- 최대 AR 우위 후보는 자동 매칭 오류와 PopQA 관계 모호성을 사람이 검토한 뒤 최종 사례군으로 확정해야 한다.",
    ]
    report += ["", "## 해석 시 주의", "", "- 역할 접미사 보정과 문장 내부 정답 인식은 특히 QA 출력 평가에 필수적이다.", "- 할루시네이션/실패 구분은 이전 평서문 최종 정의와 동일한 운영적 자동 분류다.", "- 구체적 오답은 할루시네이션, 무응답·일반적 표현·심한 반복은 생성 실패로 센다.", "- 최종 논문 사례군은 자동 선별 후 사람이 원문 출력을 검토해야 한다.", ""]
    (args.output_dir / "FINAL_ANALYSIS_REPORT_KO.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
