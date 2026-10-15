"""Conservative TriviaQA alias matching and fact-level labels."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable


ANSWER_WRAPPER_RE = re.compile(
    r"^\s*(?:final\s+answer|answer)\s*[:\-]\s*", re.IGNORECASE
)
ARTICLE_RE = re.compile(r"\b(?:a|an|the)\b", re.IGNORECASE)

# Gold answers in TriviaQA carry surface noise that is not part of the fact:
# honorific titles, optional name parts in parentheses, and a leading
# preposition on phrase answers. Each is stripped to build equivalent gold
# forms; the fact itself is unchanged.
HONORIFIC_RE = re.compile(
    r"^(?:dr|mr|mrs|ms|st|sir|dame|prof|professor|lord|lady|rev|gen|capt|col|sgt)"
    r"\.?\s+",
    re.IGNORECASE,
)
PARENTHETICAL_RE = re.compile(r"\(([^()]*)\)")
LEADING_PREPOSITION_RE = re.compile(
    r"^(?:on|in|at|to|from|by|with|of|as|into|onto|inside|within)\s+",
    re.IGNORECASE,
)
MIN_DERIVED_GOLD_CHARS = 3

# These patterns identify explicit non-answers. Broad words such as "sorry",
# "unknown", and "fictional" are deliberately not refusal signals by themselves.
REFUSAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "dont_know",
        re.compile(r"\bi\s+(?:do\s+not|don['’]?t)\s+(?:know|remember|recall)\b", re.I),
    ),
    (
        "not_sure",
        re.compile(r"\bi(?:['’]m|\s+am)\s+not\s+sure\b", re.I),
    ),
    (
        "no_idea",
        re.compile(r"\bi\s+have\s+no\s+idea\b", re.I),
    ),
    (
        "cannot_answer",
        re.compile(
            r"\bi\s+(?:cannot|can['’]?t)\s+"
            r"(?:answer|determine|provide|recall|give)\b",
            re.I,
        ),
    ),
    (
        "unable_to_answer",
        re.compile(
            r"\bi(?:['’]m|\s+am)\s+(?:not\s+able|unable)\s+to\s+"
            r"(?:answer|determine|provide|recall|give)\b",
            re.I,
        ),
    ),
    (
        "insufficient_knowledge",
        re.compile(
            r"\bi\s+(?:do\s+not|don['’]?t)\s+have\s+"
            r"(?:enough\s+|sufficient\s+|the\s+necessary\s+)?"
            r"(?:information|knowledge|context)\b",
            re.I,
        ),
    ),
    (
        "context_missing",
        re.compile(
            r"\b(?:the\s+)?context\s+(?:does\s+not|doesn['’]?t)\s+"
            r"(?:provide|specify|mention|contain)\b",
            re.I,
        ),
    ),
    (
        "not_enough_information",
        re.compile(r"\bnot\s+enough\s+(?:information|context)\b", re.I),
    ),
    (
        "cannot_be_determined",
        re.compile(r"\b(?:cannot|can['’]?t)\s+be\s+determined\b", re.I),
    ),
    (
        "premise_rejection",
        re.compile(
            r"\b(?:"
            r"(?:the\s+)?premise(?:\s+of\s+the\s+question)?\s+"
            r"(?:is|seems)\s+(?:not\s+correct|incorrect|false|invalid)"
            r"|(?:the\s+)?statement\s+is\s+(?:not\s+correct|incorrect|false|invalid)"
            r"|(?:the\s+)?question\s+is\s+based\s+on\s+a\s+false\s+premise"
            r"|that\s+is\s+not\s+correct"
            r")\b",
            re.I,
        ),
    ),
    (
        "no_evidence",
        re.compile(r"\bthere\s+is\s+no\s+(?:evidence|record)\b", re.I),
    ),
    (
        "explicit_nonexistence",
        re.compile(r"^\s*there\s+is\s+no\s+", re.I),
    ),
    (
        "no_known_answer",
        re.compile(r"\bthere\s+is\s+no\s+(?:known|definitive|reliable)\s+answer\b", re.I),
    ),
)

FACT_CLASS_BY_CORRECT_COUNT = {
    0: "NEVER_RECALLED",
    1: "PROMPT_SENSITIVE_1",
    2: "PROMPT_SENSITIVE_2",
    3: "PROMPT_SENSITIVE_3",
    4: "PROMPT_SENSITIVE_4",
    5: "STABLE_RECALL",
}


def strip_answer_wrapper(text: str) -> str:
    return ANSWER_WRAPPER_RE.sub("", str(text), count=1).strip()


def normalize_answer(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    value = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in value
    )
    value = ARTICLE_RE.sub(" ", value)
    return " ".join(value.split())


def _gold_expansions_once(answer: str) -> list[str]:
    """One round of surface-noise removal from a single gold answer."""
    candidates: list[str] = []
    if PARENTHETICAL_RE.search(answer):
        # "Dr. (James) Parkinson" is gold for both "Dr. James Parkinson"
        # and "Dr. Parkinson"; the parenthesised part is optional, not wrong.
        candidates.append(PARENTHETICAL_RE.sub(r"\1", answer))
        candidates.append(PARENTHETICAL_RE.sub(" ", answer))
    stripped_title = HONORIFIC_RE.sub("", answer)
    if stripped_title != answer:
        candidates.append(stripped_title)
    if len(answer.split()) > 1:
        stripped_preposition = LEADING_PREPOSITION_RE.sub("", answer)
        if stripped_preposition != answer:
            candidates.append(stripped_preposition)
    return candidates


def gold_answer_variants(answer: Any) -> list[str]:
    """Expand one gold answer into equivalent surface forms.

    The original spelling is always first. Derived forms only remove surface
    noise, so a model that supplies the noise back still matches exactly.
    """
    original = str(answer or "").strip()
    variants = [original] if original else []
    seen = {normalize_answer(original)} if original else set()
    frontier = list(variants)
    while frontier:
        current = frontier.pop()
        for candidate in _gold_expansions_once(current):
            cleaned = " ".join(candidate.split())
            normalized = normalize_answer(cleaned)
            if not normalized or normalized in seen:
                continue
            if len(normalized.replace(" ", "")) < MIN_DERIVED_GOLD_CHARS:
                continue
            seen.add(normalized)
            variants.append(cleaned)
            frontier.append(cleaned)
    return variants


def expand_gold_aliases(aliases: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return (all gold forms, forms that were derived rather than supplied)."""
    supplied = [str(alias).strip() for alias in aliases if str(alias).strip()]
    supplied_normalized = {normalize_answer(alias) for alias in supplied}
    expanded = list(supplied)
    derived: list[str] = []
    seen = set(supplied_normalized)
    for alias in supplied:
        for variant in gold_answer_variants(alias):
            normalized = normalize_answer(variant)
            if normalized in seen:
                continue
            seen.add(normalized)
            expanded.append(variant)
            derived.append(variant)
    return expanded, derived


def normalized_aliases(aliases: Iterable[str]) -> dict[str, str]:
    """Map each unique non-empty normalized alias to its first raw spelling."""
    normalized: dict[str, str] = {}
    for raw_alias in aliases:
        raw = str(raw_alias).strip()
        alias = normalize_answer(raw)
        if alias:
            normalized.setdefault(alias, raw)
    return normalized


def aliases_exposed_by_question(
    alias_map: dict[str, str], question: str | None
) -> dict[str, str]:
    """Return aliases already visible as whole tokens in the current question."""
    normalized_question = f" {normalize_answer(question or '')} "
    return {
        normalized: raw
        for normalized, raw in alias_map.items()
        if f" {normalized} " in normalized_question
    }


def regular_plural_aliases(alias_map: dict[str, str]) -> dict[str, str]:
    """Expand only simple regular English plural endings for conservative spans."""
    expanded: dict[str, str] = {}
    for normalized, raw in alias_map.items():
        tokens = normalized.split()
        if not tokens:
            continue
        word = tokens[-1]
        if len(word) < 3 or not word.isalpha() or word.endswith("s"):
            continue
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            plural = f"{word[:-1]}ies"
        elif word.endswith(("ch", "sh", "x", "z")):
            plural = f"{word}es"
        else:
            plural = f"{word}s"
        expanded.setdefault(" ".join([*tokens[:-1], plural]), raw)
    return expanded


def detect_refusal(text: str) -> tuple[bool, str | None]:
    for rule_name, pattern in REFUSAL_PATTERNS:
        if pattern.search(str(text)):
            return True, rule_name
    return False, None


def find_alias_present(
    normalized_output: str, alias_map: dict[str, str]
) -> str | None:
    """Diagnostic only: find a whole-token alias inside a longer response."""
    padded_output = f" {normalized_output} "
    for alias in sorted(alias_map, key=lambda value: (-len(value), value)):
        if f" {alias} " in padded_output:
            return alias_map[alias]
    return None


def find_safe_alias_span(
    normalized_output: str,
    alias_map: dict[str, str],
    question: str | None,
) -> str | None:
    """Find a non-negated whole-token alias not already exposed by the question."""
    output_tokens = normalized_output.split()
    question_text = f" {normalize_answer(question or '')} "
    negations = {"no", "not", "never", "neither", "without", "wrong", "incorrect", "false"}
    for alias in sorted(alias_map, key=lambda value: (-len(value.split()), -len(value), value)):
        if f" {alias} " in question_text:
            continue
        alias_tokens = alias.split()
        if not alias_tokens:
            continue
        width = len(alias_tokens)
        for start in range(len(output_tokens) - width + 1):
            if output_tokens[start : start + width] != alias_tokens:
                continue
            before = output_tokens[max(0, start - 2) : start]
            after = output_tokens[start + width : start + width + 2]
            if negations.intersection(before) or negations.intersection(after):
                continue
            # Do not promote an incomplete one-token prefix such as gold
            # "Nikkei" inside the wrong answer "Nikkei 25".
            if width == 1 and after and after[0].isdigit():
                continue
            return alias_map[alias]
    return None


def score_response(
    raw_output: str,
    aliases: Iterable[str],
    *,
    question: str | None = None,
    expand_gold: bool = True,
) -> dict[str, Any]:
    """Score exact answers and conservative answer spans over all gold aliases."""
    extracted = strip_answer_wrapper(raw_output)
    normalized_output = normalize_answer(extracted)
    if expand_gold:
        all_gold, derived_gold = expand_gold_aliases(aliases)
    else:
        all_gold, derived_gold = [str(a).strip() for a in aliases], []
    derived_normalized = {normalize_answer(alias) for alias in derived_gold}
    alias_map = normalized_aliases(all_gold)
    if not alias_map:
        raise ValueError("At least one non-empty normalized gold alias is required")
    exposed_alias_map = aliases_exposed_by_question(alias_map, question)
    eligible_alias_map = {
        normalized: raw
        for normalized, raw in alias_map.items()
        if normalized not in exposed_alias_map
    }
    if not eligible_alias_map:
        raise ValueError(
            "Every normalized gold alias is already exposed by the question; "
            "this item cannot measure unaided factual recall"
        )

    is_refusal, refusal_rule = detect_refusal(raw_output)
    exact_alias = eligible_alias_map.get(normalized_output)
    diagnostic_alias = find_alias_present(normalized_output, alias_map)
    safe_span_alias = find_safe_alias_span(
        normalized_output,
        eligible_alias_map,
        question,
    )
    safe_inflected_alias = None
    if safe_span_alias is None:
        safe_inflected_alias = find_safe_alias_span(
            normalized_output,
            regular_plural_aliases(eligible_alias_map),
            question,
        )
    strict_is_correct = exact_alias is not None and not is_refusal
    is_correct = (
        strict_is_correct
        or safe_span_alias is not None
        or safe_inflected_alias is not None
    ) and not is_refusal
    matched_alias = (
        exact_alias
        if strict_is_correct
        else safe_span_alias
        if safe_span_alias is not None
        else safe_inflected_alias
    )

    return {
        "raw_output": raw_output,
        "extracted_answer": extracted,
        "normalized_answer": normalized_output,
        "strict_is_correct": strict_is_correct,
        "is_correct": is_correct,
        "match_type": (
            "normalized_exact"
            if strict_is_correct
            else "safe_alias_span"
            if safe_span_alias is not None and is_correct
            else "safe_regular_plural_span"
            if is_correct
            else None
        ),
        "matched_alias": matched_alias if is_correct else None,
        "is_refusal": is_refusal,
        "refusal_rule": refusal_rule,
        "diagnostic_alias_present": diagnostic_alias is not None,
        "diagnostic_present_alias": diagnostic_alias,
        "question_exposed_aliases": list(exposed_alias_map.values()),
        "derived_gold_variants": derived_gold,
        "matched_derived_gold": bool(
            is_correct
            and matched_alias is not None
            and normalize_answer(matched_alias) in derived_normalized
        ),
    }


def fact_class(correct_count: int, total_variants: int = 5) -> str:
    if total_variants != 5:
        raise ValueError("This experiment defines fact classes over exactly five variants")
    if correct_count not in FACT_CLASS_BY_CORRECT_COUNT:
        raise ValueError("correct_count must be between 0 and 5")
    return FACT_CLASS_BY_CORRECT_COUNT[correct_count]
