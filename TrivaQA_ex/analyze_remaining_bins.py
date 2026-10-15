#!/usr/bin/env python3
"""Inventory and audit the automatic 3/5, 2/5, 1/5, and 0/5 bins.

This script does not modify the accepted automatic scores.  It creates a
separate audit layer containing the manually reviewed 3/5--1/5 cases and a
complete structural inventory of every 0/5 model-fact and its outputs.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "results" / "dream_qwen2_5_full" / "analysis"
OUTPUT = ANALYSIS / "knowledge_bin_extension"


AUDITS: dict[tuple[str, str], dict[str, str]] = {
    # Automatic 3/5: 9 model-facts.
    ("dream", "odql_13017__735a489e6be8"): {
        "manual_pattern": "11010",
        "tier": "CORE",
        "audit_class": "ROBUST_ASSOCIATIVE_COMPETITOR",
        "assessment_ko": "동치 질문에서 v2·v4가 동일 경쟁자 Mammoth를 반복한다.",
        "action_ko": "핵심 hidden-state/logit 비교",
    },
    ("dream", "qf_2121__a630bba15b58"): {
        "manual_pattern": "11001",
        "tier": "CORE",
        "audit_class": "ROBUST_PERSON_COMPETITOR",
        "assessment_ko": "동치 질문 v2·v3에서 Guy Pearce가 반복된다.",
        "action_ko": "핵심 hidden-state/logit 비교",
    },
    ("dream", "qw_10823__e181c7f9b649"): {
        "manual_pattern": "10111",
        "tier": "QUALIFIED",
        "audit_class": "SCORER_FALSE_NEGATIVE_MOVES_TO_4_OF_5",
        "assessment_ko": "v0은 피격을 서술한 의미 정답이다. 실제 target 실패는 pneumonia 하나다.",
        "action_ko": "4/5로 재분류하고 부가 인명 hallucination을 별도 보존",
    },
    ("dream", "sfq_19011__697b024818fc"): {
        "manual_pattern": "target=10110;literal=11111",
        "tier": "QUALIFIED",
        "audit_class": "TARGET_HIERARCHY_DRIFT",
        "assessment_ko": "Sporting dog는 넓어진 질문에는 유효하다. raw 3/5는 지식률이 아니다.",
        "action_ko": "일반 3/5 분석에서는 제외하고 hierarchy drift로만 사용",
    },
    ("dream", "sfq_7552__2766dae6d17b"): {
        "manual_pattern": "11010",
        "tier": "CORE",
        "audit_class": "ROBUST_ROLE_BINDING_FAILURE",
        "assessment_ko": "actor를 묻는 v2·v4에서 role entity Jim Rockford를 반복 복사한다.",
        "action_ko": "핵심 role-binding 분석",
    },
    ("dream", "tc_1243__352dbec1b430"): {
        "manual_pattern": "11001",
        "tier": "QUALIFIED",
        "audit_class": "RELATION_CUE_WEAKENING_AND_SOURCE_TYPO",
        "assessment_ko": "방향성/가족 단서가 약한 v2·v3에서 실패하며 Joey는 Joely의 원문 오타다.",
        "action_ko": "이름 수정·질문 재생성 후 relation 분석",
    },
    ("dream", "wh_2691__fc85150cefc4"): {
        "manual_pattern": "valid(v0-v3)=1011;v4=invalid",
        "tier": "QUALIFIED",
        "audit_class": "ONE_VALID_FAILURE_PLUS_RELATION_DRIFT",
        "assessment_ko": "v1 Tinker Bell은 실제 실패지만 v4 created는 원 관계를 훼손한다.",
        "action_ko": "v1만 사용하고 v4 재생성",
    },
    ("qwen", "qb_7132__01d756cb549c"): {
        "manual_pattern": "11100",
        "tier": "CORE",
        "audit_class": "ROBUST_TITLE_COMPLETION_FAILURE",
        "assessment_ko": "동치 title-completion에서 Lindisfarne와 Shannon으로 실패한다.",
        "action_ko": "핵심 entity-copy/title 분석",
    },
    ("qwen", "sfq_25386__9434d990b24b"): {
        "manual_pattern": "benchmark=10011",
        "tier": "QUALIFIED",
        "audit_class": "TEMPORALLY_STALE_RELATION",
        "assessment_ko": "Reading FC가 두 번 반복되지만 AFC Wimbledon–Kingsmeadow 현재형 관계는 낡았다.",
        "action_ko": "benchmark-world 분석에만 사용하거나 시점을 명시해 재생성",
    },

    # Automatic 2/5: 5 model-facts.
    ("dream", "sfq_3251__eb8293631a4b"): {
        "manual_pattern": "valid(v0-v3)=1100;v4=invalid",
        "tier": "QUALIFIED",
        "audit_class": "ORDINAL_TEMPORAL_RELATION_FAILURE",
        "assessment_ko": "모두 올림픽 개최 도시지만 first-to-host-twice 관계가 흔들린다.",
        "action_ko": "v0-v3만 양방향 모델 비교에 사용",
    },
    ("qwen", "qb_7330__b1b53d9801e4"): {
        "manual_pattern": "10100",
        "tier": "CORE",
        "audit_class": "ROBUST_NUMERIC_COMPETITOR",
        "assessment_ko": "동치 질문에서 정답 56 두 번, 동일 경쟁 수치 59 세 번이다.",
        "action_ko": "2/5 최우선 logit/hidden-state 분석",
    },
    ("qwen", "qg_2116__66a41a2baae2"): {
        "manual_pattern": "10001;v1=truncated_nonanswer",
        "tier": "QUALIFIED",
        "audit_class": "TWO_REAL_ERRORS_PLUS_TRUNCATION",
        "assessment_ko": "정답 두 번, genuine wrong 두 번, 답 직전 잘린 non-answer 한 번이다.",
        "action_ko": "v1을 생성 실패로 분리하고 v2·v3만 오류 분석",
    },
    ("qwen", "sfq_19011__697b024818fc"): {
        "manual_pattern": "target=10100;literal=11101",
        "tier": "EXCLUDE",
        "audit_class": "TARGET_HIERARCHY_DRIFT",
        "assessment_ko": "Sporting 두 응답은 broadened 질문에 맞고 v3 Terrier만 명백한 오류다.",
        "action_ko": "2/5 지식 분석에서 제외하고 target level을 고정해 재생성",
    },
    ("qwen", "sfq_3251__eb8293631a4b"): {
        "manual_pattern": "valid(v0-v3)=1010;v4=invalid",
        "tier": "QUALIFIED",
        "audit_class": "ORDINAL_TEMPORAL_RELATION_FAILURE",
        "assessment_ko": "Paris fact 주변의 St. Louis/Los Angeles로 이동하며 v4는 비문이다.",
        "action_ko": "v0-v3만 양방향 모델 비교에 사용",
    },

    # Automatic 1/5: 13 model-facts.
    ("dream", "odql_12511__72ba0f388cf4"): {
        "manual_pattern": "11111",
        "tier": "RECLASSIFY",
        "audit_class": "FOUR_SCORER_FALSE_NEGATIVES",
        "assessment_ko": "bare Wight 네 번은 which island 문맥에서 Isle of Wight의 명백한 축약이다.",
        "action_ko": "실제 5/5로 교정; 추가 paraphrase보다 alias 승인 우선",
    },
    ("dream", "qb_7132__01d756cb549c"): {
        "manual_pattern": "00010",
        "tier": "WEAK_KNOWLEDGE",
        "audit_class": "ANSWER_TYPE_CUE_DEPENDENT",
        "assessment_ko": "place-name cue에서만 Tyne이며 Sea/Hill이 각각 두 번 반복된다.",
        "action_ko": "place-name cue 유/무 각 5개 이상 추가",
    },
    ("dream", "qz_3920__c99305736210"): {
        "manual_pattern": "00001",
        "tier": "WEAK_KNOWLEDGE",
        "audit_class": "FRANCHISE_RELATION_COMPETITOR",
        "assessment_ko": "The Tomb of the Cybermen이 세 번 반복되어 단순 무작위가 아니다.",
        "action_ko": "first/debut/initial relation 표현을 통제해 15개 추가",
    },
    ("dream", "sfq_1740__1d3cdec468d1"): {
        "manual_pattern": "10000",
        "tier": "WEAK_KNOWLEDGE",
        "audit_class": "ORIGINAL_WORDING_DEPENDENT",
        "assessment_ko": "원문만 1972이고 자연어 paraphrase에서 1968이 세 번 반복된다.",
        "action_ko": "사건 순서와 appointment/assume-office 표현을 factorial하게 추가",
    },
    ("dream", "wh_4381__e2162a7442d0"): {
        "manual_pattern": "exact=10000;near_recall=11000",
        "tier": "NEAR_RECALL",
        "audit_class": "NEAR_CORRECT_ENTITY_STRING",
        "assessment_ko": "Raj Quartels는 철자 오류지만 내부 지식 흔적이 강하다.",
        "action_ko": "exact correctness와 entity recall을 분리하고 추가 질문",
    },
    ("qwen", "jp_3497__95236743ce33"): {
        "manual_pattern": "01000",
        "tier": "WEAK_KNOWLEDGE",
        "audit_class": "CANONICAL_PHRASE_DEPENDENT",
        "assessment_ko": "longest coastline에서만 Alaska, 나머지는 Maine 네 번이다.",
        "action_ko": "coastline metric wording을 통제해 15개 추가",
    },
    ("qwen", "odql_1743__5045e8cd8f36"): {
        "manual_pattern": "10000",
        "tier": "WEAK_KNOWLEDGE",
        "audit_class": "ORIGINAL_WORDING_OR_MEMORIZATION_DEPENDENT",
        "assessment_ko": "원문에서 희귀 고유명 Montmorency를 맞히고 paraphrase에서는 실패한다.",
        "action_ko": "책/저자 표기만 미세 변경한 질문 15개 추가",
    },
    ("qwen", "qb_3702__8d8cf76caf05"): {
        "manual_pattern": "01000",
        "tier": "WEAK_KNOWLEDGE",
        "audit_class": "ANSWER_TYPE_CUE_DEPENDENT",
        "assessment_ko": "consumer products cue에서만 Cigarettes이며 나머지는 운동 주변 개념이다.",
        "action_ko": "product cue 유/무의 균형 질문 추가",
    },
    ("qwen", "qf_440__9131f592f5d6"): {
        "manual_pattern": "10000",
        "tier": "CHANCE_CANDIDATE",
        "audit_class": "LEXICAL_GUESSING_DOMINATES",
        "assessment_ko": "원문의 일반명 cave만 맞고 Smoo→Smoky/smooth 형태 추측이 지배한다.",
        "action_ko": "중립적인 지형 유형 질문 15~20개 추가",
    },
    ("qwen", "qz_3920__c99305736210"): {
        "manual_pattern": "00100",
        "tier": "WEAK_KNOWLEDGE",
        "audit_class": "FRANCHISE_RELATION_COMPETITOR",
        "assessment_ko": "The Daleks' Master Plan이 네 번 반복되는 모델 고유 competitor다.",
        "action_ko": "Dream과 같은 추가 relation 질문으로 competitor 비교",
    },
    ("qwen", "sfq_1740__1d3cdec468d1"): {
        "manual_pattern": "10000",
        "tier": "WEAK_KNOWLEDGE",
        "audit_class": "ORIGINAL_WORDING_DEPENDENT",
        "assessment_ko": "원문만 1972이고 네 paraphrase 모두 1971이다.",
        "action_ko": "원문 memorization과 sub-fact 혼동을 진단하는 추가 질문",
    },
    ("qwen", "sfq_696__f19d05e21149"): {
        "manual_pattern": "10000",
        "tier": "WEAK_KNOWLEDGE",
        "audit_class": "ORIGINAL_WORDING_DEPENDENT",
        "assessment_ko": "원문만 Dublin이고 Lisbon이 세 번 반복된다.",
        "action_ko": "연도·길이·port-airport 단서 순서를 교차한 질문 추가",
    },
    ("qwen", "tc_1243__352dbec1b430"): {
        "manual_pattern": "00001",
        "tier": "INVALID",
        "audit_class": "SOURCE_ENTITY_TYPO",
        "assessment_ko": "Joey는 Joely Richardson의 오타이며 son 출력도 이 오타 영향을 받을 수 있다.",
        "action_ko": "entity를 고치고 전 질문 재생성·재실험",
    },
}


KNOWN_ZERO_AUDITS: dict[tuple[str, str], dict[str, str]] = {
    ("dream", "tb_1916__3835a8c1d6f9"): {
        "manual_pattern": "11111",
        "audit_class": "CANONICAL_ALIAS_PARSING_FAILURE",
        "assessment_ko": "Parkinson/James Parkinson을 모두 냈지만 유일 alias 'Dr. (James) Parkinson'이 확장되지 않았다.",
    },
    ("qwen", "tb_1916__3835a8c1d6f9"): {
        "manual_pattern": "11011",
        "audit_class": "CANONICAL_ALIAS_PARSING_FAILURE",
        "assessment_ko": "v2 James만 제외하면 네 번 Parkinson을 회수했지만 alias 표현 때문에 전부 0 처리됐다.",
    },
    ("dream", "odql_10602__996f36f9916c"): {
        "manual_pattern": "DATASET_GOLD_DISPUTED",
        "audit_class": "FACTUAL_GOLD_LOCATION_ERROR",
        "assessment_ko": "모든 답은 brain이고 해부학적으로 olfactory bulb는 뇌 아래쪽 구조다; gold Nose가 부정확하다.",
    },
    ("qwen", "odql_10602__996f36f9916c"): {
        "manual_pattern": "DATASET_GOLD_DISPUTED",
        "audit_class": "FACTUAL_GOLD_LOCATION_ERROR",
        "assessment_ko": "모든 답은 brain이고 해부학적으로 olfactory bulb는 뇌 아래쪽 구조다; gold Nose가 부정확하다.",
    },
    ("dream", "qw_13355__256e8e0fdab6"): {
        "manual_pattern": "at_least=00101",
        "audit_class": "SEMANTIC_ALIAS_GAP",
        "assessment_ko": "v2·v4의 board description은 gold 'On a board'와 의미상 맞다.",
    },
    ("qwen", "qw_13355__256e8e0fdab6"): {
        "manual_pattern": "at_least=00101",
        "audit_class": "SEMANTIC_ALIAS_GAP",
        "assessment_ko": "v2·v4의 bare board는 gold 'On a board'와 의미상 맞다.",
    },
    ("dream", "qb_7330__b1b53d9801e4"): {
        "manual_pattern": "at_least=01000",
        "audit_class": "SEMANTIC_NUMERIC_INTERVAL_FALSE_NEGATIVE",
        "assessment_ko": "v1의 1216 to 1272는 56년 재위 기간을 정확히 제시해 내부 지식 흔적이 직접 관측된다.",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fact_rows(
    facts: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    generation_index: dict[tuple[str, str, str], dict[str, Any]],
    bin_count: int,
) -> list[dict[str, Any]]:
    fact_index = {(r["model_name"], r["fact_id"]): r for r in facts}
    result: list[dict[str, Any]] = []
    for fact in sorted(
        (r for r in facts if r["official_alias_correct_count"] == bin_count),
        key=lambda r: (r["model_name"], r["fact_id"]),
    ):
        model = fact["model_name"]
        other_model = "qwen" if model == "dream" else "dream"
        other = fact_index[(other_model, fact["fact_id"])]
        rows = sorted(
            (
                r
                for r in variants
                if r["model_name"] == model and r["fact_id"] == fact["fact_id"]
            ),
            key=lambda r: r["variant_index"],
        )
        wrong = [r for r in rows if not r["official_alias_is_correct"]]
        wrong_counts = Counter(r["normalized_answer"] for r in wrong)
        modal_output, modal_count = wrong_counts.most_common(1)[0]
        generations = [
            generation_index[(model, fact["fact_id"], r["variant_id"])] for r in rows
        ]
        cutoff_count = sum(
            g["first_eos_index"] is None
            and len(g["continuation_token_ids"]) >= g["max_new_tokens"]
            for g in generations
        )
        audit = AUDITS.get((model, fact["fact_id"]), {})
        zero_audit = KNOWN_ZERO_AUDITS.get((model, fact["fact_id"]), {})
        result.append(
            {
                "automatic_bin": f"{bin_count}/5",
                "model_name": model,
                "fact_id": fact["fact_id"],
                "original_question": fact["original_question"],
                "canonical_answer": fact["canonical_answer"],
                "automatic_pattern": fact["official_alias_correctness_pattern"],
                "manual_pattern": audit.get(
                    "manual_pattern", zero_audit.get("manual_pattern", "NOT_FULLY_ADJUDICATED")
                ),
                "tier": audit.get("tier", "INVENTORY"),
                "audit_class": audit.get(
                    "audit_class", zero_audit.get("audit_class", "STRUCTURAL_INVENTORY")
                ),
                "assessment_ko": audit.get(
                    "assessment_ko", zero_audit.get("assessment_ko", "")
                ),
                "action_ko": audit.get("action_ko", ""),
                "other_model": other_model,
                "other_model_correct_count": other["official_alias_correct_count"],
                "other_model_pattern": other["official_alias_correctness_pattern"],
                "correct_variant_ids": ",".join(
                    r["variant_id"] for r in rows if r["official_alias_is_correct"]
                ),
                "wrong_variant_ids": ",".join(r["variant_id"] for r in wrong),
                "modal_exact_wrong_output": modal_output,
                "modal_exact_wrong_count": modal_count,
                "unique_exact_wrong_output_count": len(wrong_counts),
                "refusal_count": fact["refusal_count"],
                "token_limit_without_eos_count": cutoff_count,
            }
        )
    return result


def variant_rows(
    facts: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    generation_index: dict[tuple[str, str, str], dict[str, Any]],
    bin_count: int,
) -> list[dict[str, Any]]:
    selected = {
        (r["model_name"], r["fact_id"])
        for r in facts
        if r["official_alias_correct_count"] == bin_count
    }
    result = []
    for row in sorted(
        (r for r in variants if (r["model_name"], r["fact_id"]) in selected),
        key=lambda r: (r["model_name"], r["fact_id"], r["variant_index"]),
    ):
        generation = generation_index[
            (row["model_name"], row["fact_id"], row["variant_id"])
        ]
        hit_limit = generation["first_eos_index"] is None and len(
            generation["continuation_token_ids"]
        ) >= generation["max_new_tokens"]
        result.append(
            {
                "automatic_bin": f"{bin_count}/5",
                "model_name": row["model_name"],
                "fact_id": row["fact_id"],
                "canonical_answer": row["canonical_answer"],
                "variant_id": row["variant_id"],
                "variant_source": row["variant_source"],
                "question": row["question"],
                "scored_output": row["scored_output"],
                "automatic_is_correct": row["official_alias_is_correct"],
                "is_refusal": row["is_refusal"],
                "normalized_answer": row["normalized_answer"],
                "generated_token_count": len(generation["continuation_token_ids"]),
                "max_new_tokens": generation["max_new_tokens"],
                "first_eos_index": generation["first_eos_index"],
                "hit_token_limit_without_eos": hit_limit,
            }
        )
    return result


def zero_cross_inventory(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(r["model_name"], r["fact_id"]): r for r in facts}
    fact_ids = sorted(
        {
            r["fact_id"]
            for r in facts
            if r["official_alias_correct_count"] == 0
        }
    )
    result = []
    for fact_id in fact_ids:
        dream = index[("dream", fact_id)]
        qwen = index[("qwen", fact_id)]
        dc = dream["official_alias_correct_count"]
        qc = qwen["official_alias_correct_count"]
        if dc == qc == 0:
            group = "BOTH_ZERO"
        elif dc == 0:
            group = "DREAM_ZERO_QWEN_NONZERO"
        else:
            group = "QWEN_ZERO_DREAM_NONZERO"
        result.append(
            {
                "fact_id": fact_id,
                "original_question": dream["original_question"],
                "canonical_answer": dream["canonical_answer"],
                "zero_group": group,
                "dream_correct_count": dc,
                "dream_pattern": dream["official_alias_correctness_pattern"],
                "qwen_correct_count": qc,
                "qwen_pattern": qwen["official_alias_correctness_pattern"],
                "known_audit_issue": any(
                    (model, fact_id) in KNOWN_ZERO_AUDITS
                    for model in ("dream", "qwen")
                ),
                "dream_refusal_count": dream["refusal_count"],
                "qwen_refusal_count": qwen["refusal_count"],
            }
        )
    return result


def main() -> None:
    facts = load_jsonl(ANALYSIS / "fact_results.jsonl")
    variants = load_jsonl(ANALYSIS / "variant_results.jsonl")
    generations = load_jsonl(
        HERE / "results" / "dream_qwen2_5_full" / "cache" / "dream_generations.jsonl"
    ) + load_jsonl(
        HERE / "results" / "dream_qwen2_5_full" / "cache" / "qwen_generations.jsonl"
    )
    generation_index = {
        (r["model_name"], r["fact_id"], r["variant_id"]): r for r in generations
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)

    bin_fact_tables: dict[int, list[dict[str, Any]]] = {}
    bin_variant_tables: dict[int, list[dict[str, Any]]] = {}
    expected = {3: 9, 2: 5, 1: 13, 0: 77}
    for bin_count in (3, 2, 1, 0):
        frows = fact_rows(facts, variants, generation_index, bin_count)
        vrows = variant_rows(facts, variants, generation_index, bin_count)
        if len(frows) != expected[bin_count] or len(vrows) != expected[bin_count] * 5:
            raise AssertionError((bin_count, len(frows), len(vrows)))
        bin_fact_tables[bin_count] = frows
        bin_variant_tables[bin_count] = vrows
        write_csv(OUTPUT / f"{bin_count}of5_fact_inventory.csv", frows)
        write_csv(OUTPUT / f"{bin_count}of5_variant_inventory.csv", vrows)

    cross = zero_cross_inventory(facts)
    if len(cross) != 48:
        raise AssertionError(len(cross))
    write_csv(OUTPUT / "0of5_cross_model_fact_inventory.csv", cross)

    summary = {
        "automatic_model_fact_counts": {
            str(k): len(bin_fact_tables[k]) for k in (3, 2, 1, 0)
        },
        "automatic_response_counts": {
            str(k): len(bin_variant_tables[k]) for k in (3, 2, 1, 0)
        },
        "audited_tier_counts": {
            str(k): dict(Counter(r["tier"] for r in bin_fact_tables[k]))
            for k in (3, 2, 1)
        },
        "zero_unique_fact_count": len(cross),
        "zero_cross_group_counts": dict(Counter(r["zero_group"] for r in cross)),
        "refusal_counts": {
            str(k): sum(r["refusal_count"] for r in bin_fact_tables[k])
            for k in (3, 2, 1, 0)
        },
        "known_zero_audit_issue_model_facts": len(KNOWN_ZERO_AUDITS),
        "zero_token_limit_without_eos_response_count": sum(
            r["hit_token_limit_without_eos"] for r in bin_variant_tables[0]
        ),
        "zero_token_limit_affected_model_fact_count": sum(
            r["token_limit_without_eos_count"] > 0 for r in bin_fact_tables[0]
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
