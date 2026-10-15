#!/usr/bin/env python3
"""Create the manual audits used for the first two priority analyses.

Part 1 audits every model/fact that the automatic scorer placed in 4/5.
Part 2 audits every paired question for the 12 facts in which at least one
model was question-sensitive and the other was question-sensitive or stable.

The automatic scores are never overwritten.  Manual judgements live in a
separate output directory so that raw, automatic, and audited results remain
distinguishable.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = HERE / "results" / "dream_qwen2_5_full" / "analysis"
DEFAULT_OUTPUT = DEFAULT_ANALYSIS / "priority_case_audit"


PART1_FACTS: dict[tuple[str, str], dict[str, str]] = {
    ("dream", "odql_1396__8dddda2c19d8"): {
        "short_name": "Eve–Adam's rib",
        "failed_variant": "v4",
        "manual_pattern": "11111",
        "tier": "EXCLUDE",
        "audit_class": "SCORER_FALSE_NEGATIVE",
        "mechanism": "possessive ↔ of-phrase alias gap",
        "finding_ko": "'The rib of Adam'은 Adam's rib과 동치다. 지식 실패가 아니라 matcher 누락이다.",
        "recommendation_ko": "4/5 사례에서 제거하고 의미 기준 5/5로 교정",
    },
    ("dream", "odql_4288__9fe6137b224b"): {
        "short_name": "Oberst–Colonel",
        "failed_variant": "v4",
        "manual_pattern": "11110",
        "tier": "CORE",
        "audit_class": "CLEAN_PROMPT_SENSITIVE_FAILURE",
        "mechanism": "same-type competitor substitution",
        "finding_ko": "동치 질문에서 Colonel 대신 같은 군 계급 범주의 General을 냈다.",
        "recommendation_ko": "핵심 내부표현 분석 사례",
    },
    ("dream", "odql_11052__8cf11607ec01"): {
        "short_name": "Commandaria–Cyprus",
        "failed_variant": "v2",
        "manual_pattern": "11011",
        "tier": "CORE",
        "audit_class": "CLEAN_PROMPT_SENSITIVE_FAILURE",
        "mechanism": "question-entity copy / role-binding failure",
        "finding_ko": "섬을 묻는데 질문 속 와인명 Commandaria를 복사했다.",
        "recommendation_ko": "핵심 내부표현 분석 사례; v4의 부가 허위 주장도 별도 표시",
    },
    ("dream", "qw_5761__e1270874009d"): {
        "short_name": "Missouri–New Orleans",
        "failed_variant": "v0",
        "manual_pattern": "01111",
        "tier": "SECONDARY",
        "audit_class": "SOURCE_PREMISE_AMBIGUITY",
        "mechanism": "route-relation retrieval failure",
        "finding_ko": "원문에서 San Francisco를 냈지만 Missouri가 직접 바다에 닿지 않는 축약 전제가 있다.",
        "recommendation_ko": "보조 사례로만 사용",
    },
    ("dream", "qw_11772__6fca5e284e4d"): {
        "short_name": "Christmas Island–Australia",
        "failed_variant": "v3",
        "manual_pattern": "11101",
        "tier": "EXCLUDE",
        "audit_class": "PARAPHRASE_ANSWER_TYPE_DRIFT",
        "mechanism": "country/nation cue removed by generic 'Who'",
        "finding_ko": "v3만 국가가 아니라 사람·직책·기관도 가능한 질문으로 바뀌었다.",
        "recommendation_ko": "v3 재생성 후 재실험",
    },
    ("qwen", "odql_12511__72ba0f388cf4"): {
        "short_name": "Vectis–Isle of Wight",
        "failed_variant": "v1",
        "manual_pattern": "10111",
        "tier": "CORE",
        "audit_class": "PROMPT_SENSITIVE_ASSOCIATIVE_COMPETITOR",
        "mechanism": "crown-dependency association → Isle of Man",
        "finding_ko": "거의 동치인 문장에서 crown dependency의 강한 연상 경쟁자 Isle of Man으로 이동했다.",
        "recommendation_ko": "핵심 사례; v1의 어색한 genitive는 작은 제한으로 명시",
    },
    ("qwen", "jp_1696__b0f3caa4d5fb"): {
        "short_name": "ECTO-1–Ghostbusters",
        "failed_variant": "v2",
        "manual_pattern": "11011",
        "tier": "EXCLUDE",
        "audit_class": "PARAPHRASE_ANSWER_SCOPE_DRIFT",
        "mechanism": "group/team cue removed → individual answer",
        "finding_ko": "v2의 generic 'Who'가 집단 정답을 개인 정답으로 바꿀 수 있게 했다.",
        "recommendation_ko": "동치 paraphrase 분석에서 제외하고 answer-type ambiguity로 분리",
    },
    ("qwen", "qw_8009__df818dda9063"): {
        "short_name": "Solent–strait",
        "failed_variant": "v1",
        "manual_pattern": "10111",
        "tier": "CORE",
        "audit_class": "PROMPT_SENSITIVE_ENTITY_ECHO",
        "mechanism": "awkward metalinguistic construction → subject echo",
        "finding_ko": "분류명 strait 대신 질문의 고유명 Solent만 반복한 non-answer다.",
        "recommendation_ko": "사용 가능하되 v1 문장의 어색함을 명시",
    },
    ("qwen", "wh_2691__fc85150cefc4"): {
        "short_name": "Blue Fairy–Pinocchio",
        "failed_variant": "v4",
        "manual_pattern": "11110",
        "tier": "EXCLUDE",
        "audit_class": "PARAPHRASE_RELATION_DRIFT",
        "mechanism": "brought to life → created",
        "finding_ko": "v4는 Blue Fairy가 생명을 준 관계를 'created'로 바꿔 원 fact를 훼손했다.",
        "recommendation_ko": "v4 재생성 후 재실험",
    },
    ("qwen", "sfq_18839__d81f22a291c7"): {
        "short_name": "Jacob Epstein",
        "failed_variant": "v2",
        "manual_pattern": "11011",
        "tier": "SECONDARY",
        "audit_class": "ANSWER_TYPE_CUE_ABLATION",
        "mechanism": "sculptor cue broadened to artist",
        "finding_ko": "작품 관계는 유지되지만 구체 직업 단서가 사라지자 William Etty로 이동했다.",
        "recommendation_ko": "순수 표면 paraphrase가 아닌 cue-ablation 사례로 분석",
    },
    ("qwen", "wh_4381__e2162a7442d0"): {
        "short_name": "Paul Scott–Raj Quartet",
        "failed_variant": "v2",
        "manual_pattern": "11011",
        "tier": "CORE",
        "audit_class": "CLEAN_PROMPT_SENSITIVE_FAILURE",
        "mechanism": "unsupported title substitution",
        "finding_ko": "핵심 단서와 답 유형이 유지된 동치 질문에서 The Four Guineas를 생성했다.",
        "recommendation_ko": "가장 강한 Qwen 핵심 사례",
    },
}


# Manual response-level corrections or qualifications.  Rows not listed here
# inherit the official-alias target correctness and a clean response label.
MANUAL_RESPONSE_OVERRIDES: dict[tuple[str, str, str], dict[str, Any]] = {
    ("dream", "odql_1396__8dddda2c19d8", "v4"): {
        "manual_target_is_correct": True,
        "manual_literal_is_correct": True,
        "manual_response_quality": "SCORER_FALSE_NEGATIVE_CORRECT",
        "response_note_ko": "The rib of Adam = Adam's rib; 소유격/of-구문 동치 누락",
    },
    ("dream", "odql_11052__8cf11607ec01", "v4"): {
        "manual_target_is_correct": True,
        "manual_response_quality": "TARGET_CORRECT_FALSE_EXTRA_CLAIM",
        "response_note_ko": "Cyprus는 맞지만 'town of Commandaria'라는 부가 주장은 근거가 없고 공식 자료의 여러 생산 마을 설명과 충돌",
    },
    ("dream", "jp_1696__b0f3caa4d5fb", "v2"): {
        "manual_target_is_correct": False,
        "manual_literal_is_correct": False,
        "manual_response_quality": "SCORER_FALSE_POSITIVE_CONTEXT_MENTION",
        "response_note_ko": "명시 답은 Egon Spengler인데 영화명 Ghostbusters가 포함되어 scorer만 정답 처리",
    },
    ("dream", "qw_10823__e181c7f9b649", "v0"): {
        "manual_target_is_correct": True,
        "manual_literal_is_correct": True,
        "manual_response_quality": "SUBSTANTIVELY_CORRECT_IMPRECISE",
        "response_note_ko": "assassination이라는 라벨 대신 피격 사건을 기술; stomach는 abdominal wound의 부정확한 표현",
    },
    ("dream", "qw_10823__e181c7f9b649", "v3"): {
        "manual_target_is_correct": True,
        "manual_response_quality": "TARGET_CORRECT_MALFORMED_EXTRA_NAME",
        "response_note_ko": "assassinated는 맞지만 범인 Leon Czolgosz의 이름을 'Leonolgosz'로 훼손",
    },
    ("dream", "qw_10823__e181c7f9b649", "v4"): {
        "manual_target_is_correct": True,
        "manual_response_quality": "TARGET_CORRECT_MALFORMED_EXTRA_NAME",
        "response_note_ko": "assassinated는 맞지만 범인 Leon Czolgosz의 이름을 'Leonolgosz'로 훼손",
    },
    ("dream", "sfq_19011__697b024818fc", "v1"): {
        "manual_target_is_correct": False,
        "manual_literal_is_correct": True,
        "manual_response_quality": "CORRECT_FOR_DRIFTED_BROADER_CATEGORY",
        "response_note_ko": "Sporting dog는 literal 질문의 유효한 상위 분류지만 gold Spaniel과 다른 target",
    },
    ("qwen", "sfq_19011__697b024818fc", "v1"): {
        "manual_target_is_correct": False,
        "manual_literal_is_correct": True,
        "manual_response_quality": "CORRECT_FOR_DRIFTED_BROADER_CATEGORY",
        "response_note_ko": "Sporting은 literal 질문의 유효한 AKC group이지만 gold Spaniel과 다른 target",
    },
    ("dream", "sfq_19011__697b024818fc", "v4"): {
        "manual_target_is_correct": False,
        "manual_literal_is_correct": True,
        "manual_response_quality": "CORRECT_FOR_DRIFTED_BROADER_CATEGORY",
        "response_note_ko": "Sporting dog는 broadened 질문에는 맞지만 원래 target Spaniel을 묻지 않음",
    },
    ("qwen", "sfq_19011__697b024818fc", "v4"): {
        "manual_target_is_correct": False,
        "manual_literal_is_correct": True,
        "manual_response_quality": "CORRECT_FOR_DRIFTED_BROADER_CATEGORY",
        "response_note_ko": "Sporting은 broadened 질문에는 맞지만 원래 target Spaniel을 묻지 않음",
    },
}


QUESTION_AUDIT: dict[tuple[str, str], dict[str, Any]] = {
    ("jp_1696__b0f3caa4d5fb", "v2"): {
        "question_fidelity": "INVALID_ANSWER_SCOPE_DRIFT",
        "comparable": False,
        "question_note_ko": "group/team 제약이 사라져 개인과 집단을 모두 허용",
    },
    ("odql_11052__8cf11607ec01", "v4"): {
        "question_fidelity": "VALID_BUT_BROADER_LOCATION_WORDING",
        "comparable": True,
        "question_note_ko": "island 대신 location이지만 원 target Cyprus로 답할 수 있음",
    },
    ("qw_5761__e1270874009d", "v0"): {
        "question_fidelity": "SOURCE_PREMISE_COMPRESSED",
        "comparable": True,
        "question_note_ko": "Missouri가 Mississippi에 합류한 뒤 Gulf로 간다는 경로를 생략",
    },
    ("qw_11772__6fca5e284e4d", "v3"): {
        "question_fidelity": "INVALID_ANSWER_TYPE_DRIFT",
        "comparable": False,
        "question_note_ko": "country/nation 제약을 generic Who로 변경",
    },
    ("qw_8009__df818dda9063", "v1"): {
        "question_fidelity": "VALID_BUT_AWKWARD_METALINGUISTIC",
        "comparable": True,
        "question_note_ko": "의도는 분류 질문이나 수동태 표현이 어색하고 약간 underspecified",
    },
    ("wh_2691__fc85150cefc4", "v4"): {
        "question_fidelity": "INVALID_RELATION_DRIFT_FALSE_PREMISE",
        "comparable": False,
        "question_note_ko": "brought/given to life를 created로 바꿈; Pinocchio를 만든 이는 Geppetto",
    },
    ("sfq_18839__d81f22a291c7", "v2"): {
        "question_fidelity": "VALID_FACT_BUT_TYPE_CUE_BROADENED",
        "comparable": True,
        "question_note_ko": "sculptor를 상위 범주 artist로 바꿔 retrieval cue 강도가 달라짐",
    },
    ("sfq_19011__697b024818fc", "v1"): {
        "question_fidelity": "INVALID_TARGET_LEVEL_DRIFT",
        "comparable": False,
        "question_note_ko": "breed family Spaniel에서 더 넓은 kennel group Sporting으로 target이 이동",
    },
    ("sfq_19011__697b024818fc", "v3"): {
        "question_fidelity": "INVALID_AMBIGUOUS_CATEGORY_LEVEL",
        "comparable": False,
        "question_note_ko": "broader dog category가 Spaniel family인지 Sporting group인지 불명확",
    },
    ("sfq_19011__697b024818fc", "v4"): {
        "question_fidelity": "INVALID_TARGET_LEVEL_DRIFT",
        "comparable": False,
        "question_note_ko": "general dog type은 Sporting이라는 상위 group 답을 자연스럽게 허용",
    },
    ("sfq_3251__eb8293631a4b", "v4"): {
        "question_fidelity": "INVALID_UNGRAMMATICAL_AMBIGUITY",
        "comparable": False,
        "question_note_ko": "'for the first time on two different instances'가 모순적이고 비문에 가까움",
    },
}


PART2_FACTS: dict[str, dict[str, str]] = {
    "jp_1696__b0f3caa4d5fb": {
        "short_name": "ECTO-1–Ghostbusters",
        "tier": "EXCLUDE",
        "audit_class": "TARGET_DRIFT_AND_DREAM_FALSE_POSITIVE",
        "usable_variants": "v0,v1,v3,v4",
        "excluded_variants": "v2",
        "finding_ko": "자동 Dream 5/5 대 Qwen 4/5 차이는 v2의 target drift와 Dream scorer false positive가 만든 인공 차이다. 유효 4개에서는 둘 다 안정적이다.",
    },
    "odql_11052__8cf11607ec01": {
        "short_name": "Commandaria–Cyprus",
        "tier": "CORE",
        "audit_class": "ROBUST_DREAM_SPECIFIC_ROLE_BINDING_FAILURE",
        "usable_variants": "v0,v1,v2,v3,v4",
        "excluded_variants": "",
        "finding_ko": "v2에서만 Dream이 질문 속 와인명을 복사하고 Qwen은 Cyprus를 회수한다. v4 Dream은 target은 맞지만 허위 부가 설명을 생성한다.",
    },
    "odql_13017__735a489e6be8": {
        "short_name": "Bos primigenius–Aurochs",
        "tier": "CORE",
        "audit_class": "ROBUST_DREAM_SPECIFIC_ASSOCIATIVE_FAILURE",
        "usable_variants": "v0,v1,v2,v3,v4",
        "excluded_variants": "",
        "finding_ko": "Dream은 v2·v4에서 Mammoth로 이동하고 Qwen은 5개 모두 Aurochs다. 핵심 단서가 모두 유지된 강한 모델 차이다.",
    },
    "odql_1396__8dddda2c19d8": {
        "short_name": "Eve–Adam's rib",
        "tier": "EXCLUDE",
        "audit_class": "DREAM_SCORER_FALSE_NEGATIVE",
        "usable_variants": "v0,v1,v2,v3,v4",
        "excluded_variants": "",
        "finding_ko": "Dream v4도 의미상 정답이므로 두 모델 모두 실제 5/5다. 자동 모델 차이가 사라진다.",
    },
    "qf_2121__a630bba15b58": {
        "short_name": "Moulin Rouge–Ewan McGregor",
        "tier": "CORE",
        "audit_class": "ROBUST_DREAM_SPECIFIC_COMPETITOR_FAILURE",
        "usable_variants": "v0,v1,v2,v3,v4",
        "excluded_variants": "",
        "finding_ko": "Dream은 v2·v3에서 Guy Pearce, Qwen은 전부 Ewan McGregor다. 질문 의미와 actor target이 유지된다.",
    },
    "qw_10823__e181c7f9b649": {
        "short_name": "McKinley–assassination",
        "tier": "QUALIFIED",
        "audit_class": "USABLE_AFTER_MANUAL_RESCORING",
        "usable_variants": "v0,v1,v2,v3,v4",
        "excluded_variants": "",
        "finding_ko": "Dream v0은 사건 서술형 정답이라 false negative다. 수동으로는 Dream 4/5, Qwen 5/5이며 실제 차이는 v1 pneumonia다.",
    },
    "qw_11772__6fca5e284e4d": {
        "short_name": "Christmas Island–Australia",
        "tier": "QUALIFIED",
        "audit_class": "ANSWER_TYPE_DRIFT",
        "usable_variants": "v0,v1,v2,v4",
        "excluded_variants": "v3",
        "finding_ko": "유일한 모델 차이 v3가 country/nation 제약을 제거했다. 유효한 4개에서는 둘 다 Australia다.",
    },
    "qw_8009__df818dda9063": {
        "short_name": "Solent–strait",
        "tier": "CORE",
        "audit_class": "QWEN_ENTITY_ECHO_ON_AWKWARD_PROMPT",
        "usable_variants": "v0,v1,v2,v3,v4",
        "excluded_variants": "",
        "finding_ko": "v1에서 Qwen만 Solent를 되풀이한다. 실제 non-answer지만 v1 문장이 어색해 causal evidence의 강도는 낮춘다.",
    },
    "sfq_19011__697b024818fc": {
        "short_name": "Clumber–Spaniel",
        "tier": "QUALIFIED",
        "audit_class": "MULTI_VARIANT_TARGET_LEVEL_DRIFT",
        "usable_variants": "v0,v2",
        "excluded_variants": "v1,v3,v4",
        "finding_ko": "v1·v3·v4가 breed family에서 더 넓은 dog group으로 target을 바꾼다. Sporting은 literal하게 유효해 자동 3/5·2/5를 지식 차이로 읽을 수 없다.",
    },
    "sfq_3251__eb8293631a4b": {
        "short_name": "Olympics twice–Paris",
        "tier": "CORE",
        "audit_class": "BOTH_SENSITIVE_EXCLUDE_BAD_V4",
        "usable_variants": "v0,v1,v2,v3",
        "excluded_variants": "v4",
        "finding_ko": "유효한 v0–v3에서 v1은 Dream만, v2는 Qwen만 맞고 v3은 서로 다른 오답이다. v4만 비문이라 제외하면 좋은 양방향 비교다.",
    },
    "sfq_7552__2766dae6d17b": {
        "short_name": "Jim Rockford–James Garner",
        "tier": "CORE",
        "audit_class": "ROBUST_DREAM_SPECIFIC_ROLE_BINDING_FAILURE",
        "usable_variants": "v0,v1,v2,v3,v4",
        "excluded_variants": "",
        "finding_ko": "Dream은 v2·v4에서 배우를 묻는데 배역 Jim Rockford를 복사한다. Qwen은 5개 모두 James Garner다.",
    },
    "wh_2691__fc85150cefc4": {
        "short_name": "Blue Fairy–Pinocchio",
        "tier": "QUALIFIED",
        "audit_class": "DREAM_SENSITIVE_AFTER_EXCLUDING_FALSE_PREMISE",
        "usable_variants": "v0,v1,v2,v3",
        "excluded_variants": "v4",
        "finding_ko": "유효한 v0–v3에서 Dream만 v1을 Tinker Bell로 틀린다. v4는 created라는 잘못된 관계라 두 모델 오답을 비교하면 안 된다.",
    },
}


PART2_VARIANT_NOTES: dict[tuple[str, str], str] = {
    ("jp_1696__b0f3caa4d5fb", "v2"): "둘 다 집단 대신 개인을 답함. Dream은 문맥 속 영화명 때문에 자동 false positive.",
    ("odql_11052__8cf11607ec01", "v2"): "Dream만 질문의 와인 entity를 복사; Qwen은 위치 entity 회수.",
    ("odql_11052__8cf11607ec01", "v4"): "둘 다 target 정답. Dream은 존재하지 않는 것으로 보이는 'town of Commandaria'를 덧붙임.",
    ("odql_13017__735a489e6be8", "v2"): "Dream Mammoth 대 Qwen Aurochs.",
    ("odql_13017__735a489e6be8", "v4"): "Dream Mammoth 대 Qwen Aurochs; v2와 동일 경쟁자 재현.",
    ("odql_1396__8dddda2c19d8", "v4"): "Dream 'The rib of Adam'도 의미상 정답; 자동 차이는 사라짐.",
    ("qf_2121__a630bba15b58", "v2"): "Dream Guy Pearce 대 Qwen Ewan McGregor.",
    ("qf_2121__a630bba15b58", "v3"): "Dream이 같은 경쟁자 Guy Pearce를 반복; Qwen은 정답.",
    ("qw_10823__e181c7f9b649", "v0"): "Dream은 assassination 라벨 대신 피격을 서술했으므로 의미상 정답으로 교정.",
    ("qw_10823__e181c7f9b649", "v1"): "Dream pneumonia 대 Qwen assassination; 실제 모델 차이.",
    ("qw_10823__e181c7f9b649", "v3"): "양쪽 모두 assassination target은 맞지만 Dream이 Leon Czolgosz를 'Leonolgosz'로 훼손.",
    ("qw_10823__e181c7f9b649", "v4"): "v3과 같은 target 정답 + 잘못 생성된 범인 이름이 반복됨.",
    ("qw_11772__6fca5e284e4d", "v3"): "질문의 answer type이 변한 유일한 차이이므로 causal 비교 불가.",
    ("qw_8009__df818dda9063", "v1"): "Qwen은 질문의 subject만 되풀이한 non-answer; Dream은 strait 회수.",
    ("sfq_19011__697b024818fc", "v1"): "양쪽 Sporting 답은 broader category 질문에는 유효; 원 gold와 target level이 다름.",
    ("sfq_19011__697b024818fc", "v3"): "Dream Spaniels, Qwen Terrier지만 category level 자체가 모호.",
    ("sfq_19011__697b024818fc", "v4"): "양쪽 Sporting 답은 literal하게 유효해 자동 both-wrong이 오해를 낳음.",
    ("sfq_3251__eb8293631a4b", "v1"): "Dream Paris 정답, Qwen St. Louis 오답.",
    ("sfq_3251__eb8293631a4b", "v2"): "Dream Athens 오답, Qwen Paris 정답.",
    ("sfq_3251__eb8293631a4b", "v3"): "Dream London, Qwen Los Angeles로 서로 다른 오답.",
    ("sfq_3251__eb8293631a4b", "v4"): "질문이 비문/모호하며 둘 다 오답; 제외.",
    ("sfq_7552__2766dae6d17b", "v2"): "Dream은 요구 actor 대신 질문 속 role entity Jim Rockford 복사.",
    ("sfq_7552__2766dae6d17b", "v4"): "Dream이 동일 role-copy 오류를 재현; Qwen은 정답.",
    ("wh_2691__fc85150cefc4", "v1"): "Dream Tinker Bell 대 Qwen Pinocchio; 유효 질문에서 실제 차이.",
    ("wh_2691__fc85150cefc4", "v4"): "brought to life가 created로 바뀐 false premise; 두 오답을 지식 실패로 세면 안 됨.",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def response_audit(row: dict[str, Any]) -> dict[str, Any]:
    key = (row["model_name"], row["fact_id"], row["variant_id"])
    result = {
        "manual_target_is_correct": bool(row["official_alias_is_correct"]),
        "manual_literal_is_correct": bool(row["official_alias_is_correct"]),
        "manual_response_quality": (
            "CORRECT" if row["official_alias_is_correct"] else "INCORRECT"
        ),
        "response_note_ko": "",
    }
    result.update(MANUAL_RESPONSE_OVERRIDES.get(key, {}))
    return result


def question_audit(fact_id: str, variant_id: str) -> dict[str, Any]:
    result = {
        "question_fidelity": "VALID_EQUIVALENT",
        "comparable": True,
        "question_note_ko": "",
    }
    result.update(QUESTION_AUDIT.get((fact_id, variant_id), {}))
    return result


def pair_category(left_correct: bool, right_correct: bool, same_form: bool) -> str:
    if left_correct and right_correct:
        return "BOTH_CORRECT_SAME_FORM" if same_form else "BOTH_CORRECT_DIFFERENT_FORM"
    if left_correct:
        return "DREAM_CORRECT_QWEN_WRONG"
    if right_correct:
        return "DREAM_WRONG_QWEN_CORRECT"
    return "BOTH_WRONG_SAME_FORM" if same_form else "BOTH_WRONG_DIFFERENT_FORM"


def clipped(text: str, limit: int = 110) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def markdown_escape(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def build_part1(
    variants: list[dict[str, Any]], facts: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fact_index = {(r["model_name"], r["fact_id"]): r for r in facts}
    variant_index = {
        (r["model_name"], r["fact_id"], r["variant_id"]): r for r in variants
    }
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for key, audit in PART1_FACTS.items():
        model, fact_id = key
        fact = fact_index[key]
        auto_pattern = fact["official_alias_correctness_pattern"]
        if fact["official_alias_correct_count"] != 4:
            raise ValueError(f"Part-1 case is no longer automatic 4/5: {key}")
        summaries.append(
            {
                "model_name": model,
                "fact_id": fact_id,
                "short_name": audit["short_name"],
                "canonical_answer": fact["canonical_answer"],
                "automatic_pattern": auto_pattern,
                "manual_pattern": audit["manual_pattern"],
                "failed_variant": audit["failed_variant"],
                "tier": audit["tier"],
                "audit_class": audit["audit_class"],
                "mechanism": audit["mechanism"],
                "finding_ko": audit["finding_ko"],
                "recommendation_ko": audit["recommendation_ko"],
                "refusal_count": fact["refusal_count"],
            }
        )
        for i in range(5):
            row = variant_index[(model, fact_id, f"v{i}")]
            qa = question_audit(fact_id, f"v{i}")
            ra = response_audit(row)
            details.append(
                {
                    "model_name": model,
                    "fact_id": fact_id,
                    "short_name": audit["short_name"],
                    "canonical_answer": row["canonical_answer"],
                    "variant_id": row["variant_id"],
                    "variant_source": row["variant_source"],
                    "question": row["question"],
                    "scored_output": row["scored_output"],
                    "automatic_is_correct": row["official_alias_is_correct"],
                    **ra,
                    **qa,
                    "is_refusal": row["is_refusal"],
                }
            )
    if len(summaries) != 11 or len(details) != 55:
        raise AssertionError((len(summaries), len(details)))
    return summaries, details


def build_part2(
    variants: list[dict[str, Any]], facts: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fact_index = {(r["model_name"], r["fact_id"]): r for r in facts}
    variant_index = {
        (r["model_name"], r["fact_id"], r["variant_id"]): r for r in variants
    }
    summaries: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for fact_id, audit in PART2_FACTS.items():
        dream_fact = fact_index[("dream", fact_id)]
        qwen_fact = fact_index[("qwen", fact_id)]
        summaries.append(
            {
                "fact_id": fact_id,
                "short_name": audit["short_name"],
                "canonical_answer": dream_fact["canonical_answer"],
                "dream_automatic_count": dream_fact["official_alias_correct_count"],
                "dream_automatic_pattern": dream_fact[
                    "official_alias_correctness_pattern"
                ],
                "qwen_automatic_count": qwen_fact["official_alias_correct_count"],
                "qwen_automatic_pattern": qwen_fact[
                    "official_alias_correctness_pattern"
                ],
                "tier": audit["tier"],
                "audit_class": audit["audit_class"],
                "usable_variants": audit["usable_variants"],
                "excluded_variants": audit["excluded_variants"],
                "finding_ko": audit["finding_ko"],
            }
        )
        for i in range(5):
            variant_id = f"v{i}"
            dream = variant_index[("dream", fact_id, variant_id)]
            qwen = variant_index[("qwen", fact_id, variant_id)]
            if dream["question"] != qwen["question"]:
                raise AssertionError(f"Unpaired question: {fact_id}/{variant_id}")
            qa = question_audit(fact_id, variant_id)
            da = response_audit(dream)
            qa_resp = response_audit(qwen)
            same_form = dream["normalized_answer"] == qwen["normalized_answer"]
            auto_cat = pair_category(
                bool(dream["official_alias_is_correct"]),
                bool(qwen["official_alias_is_correct"]),
                same_form,
            )
            manual_cat = (
                pair_category(
                    bool(da["manual_target_is_correct"]),
                    bool(qa_resp["manual_target_is_correct"]),
                    same_form,
                )
                if qa["comparable"]
                else f"NOT_COMPARABLE__{qa['question_fidelity']}"
            )
            literal_cat = pair_category(
                bool(da["manual_literal_is_correct"]),
                bool(qa_resp["manual_literal_is_correct"]),
                same_form,
            )
            pairs.append(
                {
                    "fact_id": fact_id,
                    "short_name": audit["short_name"],
                    "canonical_answer": dream["canonical_answer"],
                    "variant_id": variant_id,
                    "question": dream["question"],
                    **qa,
                    "dream_output": dream["scored_output"],
                    "dream_automatic_is_correct": dream[
                        "official_alias_is_correct"
                    ],
                    "dream_manual_target_is_correct": da[
                        "manual_target_is_correct"
                    ],
                    "dream_manual_literal_is_correct": da[
                        "manual_literal_is_correct"
                    ],
                    "dream_response_quality": da["manual_response_quality"],
                    "dream_response_note_ko": da["response_note_ko"],
                    "qwen_output": qwen["scored_output"],
                    "qwen_automatic_is_correct": qwen[
                        "official_alias_is_correct"
                    ],
                    "qwen_manual_target_is_correct": qa_resp[
                        "manual_target_is_correct"
                    ],
                    "qwen_manual_literal_is_correct": qa_resp[
                        "manual_literal_is_correct"
                    ],
                    "qwen_response_quality": qa_resp["manual_response_quality"],
                    "qwen_response_note_ko": qa_resp["response_note_ko"],
                    "same_normalized_output": same_form,
                    "automatic_pair_category": auto_cat,
                    "manual_pair_category": manual_cat,
                    "manual_literal_pair_category": literal_cat,
                    "analysis_note_ko": PART2_VARIANT_NOTES.get(
                        (fact_id, variant_id),
                        "양쪽 모두 target을 회수" if (
                            da["manual_target_is_correct"]
                            and qa_resp["manual_target_is_correct"]
                        ) else "자동 판정과 수동 target 판정이 일치",
                    ),
                }
            )
    if len(summaries) != 12 or len(pairs) != 60:
        raise AssertionError((len(summaries), len(pairs)))
    return summaries, pairs


def render_report(
    part1_summary: list[dict[str, Any]],
    part1_details: list[dict[str, Any]],
    part2_summary: list[dict[str, Any]],
    part2_pairs: list[dict[str, Any]],
) -> str:
    p1_tiers = Counter(r["tier"] for r in part1_summary)
    p2_tiers = Counter(r["tier"] for r in part2_summary)
    auto_pair = Counter(r["automatic_pair_category"] for r in part2_pairs)
    literal_pair = Counter(r["manual_literal_pair_category"] for r in part2_pairs)
    comparable_pair = Counter(
        r["manual_pair_category"] for r in part2_pairs if r["comparable"]
    )

    def broad_pair_counts(counter: Counter[str]) -> tuple[int, int, int]:
        both_correct = sum(v for k, v in counter.items() if k.startswith("BOTH_CORRECT"))
        both_wrong = sum(v for k, v in counter.items() if k.startswith("BOTH_WRONG"))
        one_correct = sum(v for k, v in counter.items() if not k.startswith("BOTH_"))
        return both_correct, one_correct, both_wrong

    auto_both, auto_one, auto_wrong = broad_pair_counts(auto_pair)
    manual_both, manual_one, manual_wrong = broad_pair_counts(literal_pair)
    valid_both, valid_one, valid_wrong = broad_pair_counts(comparable_pair)
    lines = [
        "# 우선순위 실제 사례 수동 감사: Part 1–2",
        "",
        "대상은 `official_alias_is_correct` 기준 결과다. 자동 점수는 보존하고, 질문 동치성·답 target·응답 전체 품질을 별도 수동 필드로 기록했다.",
        "",
        "## 핵심 결론",
        "",
        f"- Part 1 자동 4/5 11건: 핵심 {p1_tiers['CORE']}건, 보조 {p1_tiers['SECONDARY']}건, 제외/재생성 {p1_tiers['EXCLUDE']}건.",
        f"- Part 2 비교 12 facts: 핵심 {p2_tiers['CORE']}건, 조건부 {p2_tiers['QUALIFIED']}건, 현재 causal set 제외 {p2_tiers['EXCLUDE']}건.",
        f"- Part 2의 60쌍은 자동으로 양쪽 정답/한쪽 정답/양쪽 오답이 {auto_both}/{auto_one}/{auto_wrong}였지만, 질문을 문자 그대로 읽은 수동 의미 판정에서는 {manual_both}/{manual_one}/{manual_wrong}로 바뀐다.",
        f"- 의미·target이 비교 가능한 53쌍만 남기면 양쪽 정답/한쪽 정답/양쪽 오답은 {valid_both}/{valid_one}/{valid_wrong}이다. 나머지 7쌍은 paraphrase drift/모호성으로 비교에서 제외했다.",
        "- 가장 반복적인 실제 실패는 (a) 질문 속 entity 복사/역할 결속 실패, (b) 같은 의미권 경쟁 entity로 치환, (c) 특정 lexical cue에 대한 민감성이다.",
        "- 자동 0/1 판정만으로는 scorer 오류와 paraphrase drift를 구별하지 못하므로 hidden-state 분석 전에 이 감사 필터가 필요하다.",
        "",
        "## Part 1. 자동 4/5 사례 11건 전수 감사",
        "",
        "| 모델 | 사례 | 자동→수동 | 실패 v | 등급 | 판정 | 핵심 해석 |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in part1_summary:
        lines.append(
            "| {model_name} | {short_name} | {automatic_pattern}→{manual_pattern} | "
            "{failed_variant} | {tier} | {audit_class} | {finding_ko} |".format(
                **{k: markdown_escape(v) for k, v in row.items()}
            )
        )

    for number, summary in enumerate(part1_summary, 1):
        rows = [
            r
            for r in part1_details
            if r["model_name"] == summary["model_name"]
            and r["fact_id"] == summary["fact_id"]
        ]
        lines.extend(
            [
                "",
                f"### 1-{number}. {summary['model_name']} · {summary['short_name']}",
                "",
                f"- Fact ID: `{summary['fact_id']}`; gold: `{summary['canonical_answer']}`",
                f"- 판정: `{summary['audit_class']}` / `{summary['tier']}`",
                f"- 메커니즘: {summary['mechanism']}",
                f"- 활용: {summary['recommendation_ko']}",
                "",
                "| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for r in rows:
            audit_note = "; ".join(
                x
                for x in [
                    r["question_fidelity"],
                    r["question_note_ko"],
                    r["manual_response_quality"],
                    r["response_note_ko"],
                ]
                if x
            )
            lines.append(
                "| {v} | {q} | {o} | {a} | {m} | {n} |".format(
                    v=r["variant_id"],
                    q=markdown_escape(clipped(r["question"], 135)),
                    o=markdown_escape(clipped(r["scored_output"], 115)),
                    a="✓" if r["automatic_is_correct"] else "✗",
                    m="✓" if r["manual_target_is_correct"] else "✗",
                    n=markdown_escape(audit_note),
                )
            )

    lines.extend(
        [
            "",
            "## Part 2. 지식 통제 모델 비교 12 facts · 60 질문쌍",
            "",
            "여기서 CORE는 현재 질문 전체 또는 명시된 유효 subset을 핵심 비교에 쓸 수 있는 군, QUALIFIED는 재채점·부분 사용·재생성이 필요한 군, EXCLUDE는 관측된 모델 차이가 평가/질문 오류로 사라져 현재 contrast에서 빼야 하는 군이다.",
            "",
            "| 사례 | Dream | Qwen | 등급 | 유효/제외 v | 수동 결론 |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for row in part2_summary:
        use = f"유효 {row['usable_variants']}"
        if row["excluded_variants"]:
            use += f"; 제외 {row['excluded_variants']}"
        lines.append(
            f"| {markdown_escape(row['short_name'])} | "
            f"{row['dream_automatic_count']}/5 `{row['dream_automatic_pattern']}` | "
            f"{row['qwen_automatic_count']}/5 `{row['qwen_automatic_pattern']}` | "
            f"{row['tier']} | {use} | {markdown_escape(row['finding_ko'])} |"
        )

    for number, summary in enumerate(part2_summary, 1):
        rows = [r for r in part2_pairs if r["fact_id"] == summary["fact_id"]]
        lines.extend(
            [
                "",
                f"### 2-{number}. {summary['short_name']}",
                "",
                f"- Fact ID: `{summary['fact_id']}`; gold: `{summary['canonical_answer']}`",
                f"- 판정: `{summary['audit_class']}` / `{summary['tier']}`",
                f"- 결론: {summary['finding_ko']}",
                "",
                "| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for r in rows:
            lines.append(
                "| {v} | {q} | {do} ({da}/{dm}/{dl}) | {qo} ({qa}/{qm}/{ql}) | {f} | {n} |".format(
                    v=r["variant_id"],
                    q=markdown_escape(clipped(r["question"], 128)),
                    do=markdown_escape(clipped(r["dream_output"], 72)),
                    da="✓" if r["dream_automatic_is_correct"] else "✗",
                    dm="✓" if r["dream_manual_target_is_correct"] else "✗",
                    dl="✓" if r["dream_manual_literal_is_correct"] else "✗",
                    qo=markdown_escape(clipped(r["qwen_output"], 72)),
                    qa="✓" if r["qwen_automatic_is_correct"] else "✗",
                    qm="✓" if r["qwen_manual_target_is_correct"] else "✗",
                    ql="✓" if r["qwen_manual_literal_is_correct"] else "✗",
                    f=markdown_escape(r["question_fidelity"]),
                    n=markdown_escape(r["analysis_note_ko"]),
                )
            )

    lines.extend(
        [
            "",
            "## 오류 메커니즘별 실제 사례",
            "",
            "- **role/entity copy:** Dream `Commandaria→Commandaria`, `actor→Jim Rockford`; Qwen `definition→Solent`.",
            "- **same-domain competitor:** Dream `Aurochs→Mammoth`, `Ewan McGregor→Guy Pearce`, `Colonel→General`; Qwen `Isle of Wight→Isle of Man`.",
            "- **unsupported title/entity:** Qwen `Raj Quartet→The Four Guineas`, `Jacob Epstein→William Etty`.",
            "- **데이터/평가 confound:** `The rib of Adam` false negative, Ghostbusters 문맥 문자열 false positive, Christmas Island·Clumber·Pinocchio의 target/relation drift.",
            "",
            "## 사실 확인 메모",
            "",
            "- UNESCO는 Commandaria를 Troodos 남사면의 여러 마을에서 만드는 와인으로 설명한다. 따라서 Dream의 `town of Commandaria`는 target은 맞아도 부가 사실은 신뢰할 수 없다: https://ich.unesco.org/en/RL/commandaria-wine-02288",
            "- 호주 정부는 Christmas Island를 Australian Government가 관할하며 Indian Ocean Territories Administrator가 대표한다고 설명한다. `Administrator of the Northern Territory`는 부정확하다: https://www.infrastructure.gov.au/territories-regions-cities/territories/indian-ocean-territories/christmas-island/governance-administration",
            "- AKC에서 Clumber Spaniel은 Sporting Group에 속한다. 따라서 broader-category 질문에서 `Sporting`은 단순 오답이 아니다: https://www.akc.org/breeder-programs/breed-health-testing-requirements/sporting-group-health-testing-requirements/",
            "- Disney 자료는 Blue Fairy가 Pinocchio를 *brought to life*했다고 설명한다. `created by`는 동일 관계가 아니다: https://d23.com/a-to-z/pinocchio-film/",
            "- 미국 NPS는 McKinley가 암살자의 총탄에 맞고 8일 뒤 상처로 사망했다고 설명한다. Dream v0은 표현이 거칠지만 assassination 사건을 서술한다: https://www.nps.gov/places/mckinley-monument.htm",
            "",
            "## 산출물 해석 주의",
            "",
            "`manual_target_is_correct`는 원래 TriviaQA gold target을 회수했는지, `manual_literal_is_correct`는 drift된 질문 자체에 문자 그대로 맞는지를 구분한다. 예를 들어 Clumber의 `Sporting`은 후자에는 맞지만 전자에는 맞지 않는다. `manual_response_quality`는 target이 포함돼도 허위 부가 설명이 있는 경우를 보존한다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    variants = load_jsonl(DEFAULT_ANALYSIS / "variant_results.jsonl")
    facts = load_jsonl(DEFAULT_ANALYSIS / "fact_results.jsonl")
    part1_summary, part1_details = build_part1(variants, facts)
    part2_summary, part2_pairs = build_part2(variants, facts)

    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(DEFAULT_OUTPUT / "part1_4of5_fact_audit.csv", part1_summary)
    write_csv(DEFAULT_OUTPUT / "part1_4of5_variant_audit.csv", part1_details)
    write_csv(DEFAULT_OUTPUT / "part2_controlled_fact_audit.csv", part2_summary)
    write_csv(DEFAULT_OUTPUT / "part2_controlled_question_pairs.csv", part2_pairs)

    metadata = {
        "part1_fact_count": len(part1_summary),
        "part1_variant_count": len(part1_details),
        "part1_tier_counts": dict(Counter(r["tier"] for r in part1_summary)),
        "part2_fact_count": len(part2_summary),
        "part2_pair_count": len(part2_pairs),
        "part2_tier_counts": dict(Counter(r["tier"] for r in part2_summary)),
        "part2_question_fidelity_counts": dict(
            Counter(r["question_fidelity"] for r in part2_pairs)
        ),
        "part2_automatic_pair_counts": dict(
            Counter(r["automatic_pair_category"] for r in part2_pairs)
        ),
        "part2_manual_pair_counts": dict(
            Counter(r["manual_pair_category"] for r in part2_pairs)
        ),
        "part2_manual_literal_pair_counts": dict(
            Counter(r["manual_literal_pair_category"] for r in part2_pairs)
        ),
        "part2_comparable_pair_count": sum(r["comparable"] for r in part2_pairs),
        "part2_comparable_manual_pair_counts": dict(
            Counter(
                r["manual_pair_category"] for r in part2_pairs if r["comparable"]
            )
        ),
    }
    (DEFAULT_OUTPUT / "audit_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (HERE / "PRIORITY_CASE_ANALYSIS.md").write_text(
        render_report(part1_summary, part1_details, part2_summary, part2_pairs),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
