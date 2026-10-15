#!/usr/bin/env python3
"""Generate four frozen paraphrases for each TriviaQA question with OpenAI."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from experiment_utils import (
    QUESTION_SET_SIZE,
    SCHEMA_VERSION,
    append_jsonl,
    load_jsonl,
    utc_now,
    validate_question_set,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_INPUT = HERE / "data" / "triviaqa_facts.jsonl"
DEFAULT_OUTPUT = HERE / "data" / "triviaqa_questions_5.jsonl"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_PARAPHRASE_MODEL = "gpt-4o-mini"
PARAPHRASE_PROMPT_VERSION = "triviaqa_same_answer_v2"

PARAPHRASE_INSTRUCTIONS = """You rewrite factual trivia questions.

For each supplied question, produce exactly four different English questions that:
- ask for exactly the same fact and have exactly the same answer;
- preserve every constraint, scope, time reference, and direction of the original;
- are standalone and grammatical;
- use meaningfully different wording or syntax;
- do not answer the question, hint at the answer, add choices, or add explanations.
- end every rewritten question with a literal question mark.

Return only the requested structured fields."""

PARAPHRASE_SCHEMA = {
    "type": "json_schema",
    "name": "triviaqa_paraphrases",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "paraphrases": {
                "type": "array",
                "items": {"type": "string", "pattern": r"\?$"},
                "minItems": 4,
                "maxItems": 4,
            }
        },
        "required": ["paraphrases"],
        "additionalProperties": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "OpenAI model ID (or set OPENAI_PARAPHRASE_MODEL in .env); "
            f"defaults to {DEFAULT_PARAPHRASE_MODEL}."
        ),
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N input facts; 0 means all.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalize_surface(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"\s+", " ", value)


def normalize_for_leak_check(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in value
    )
    return " ".join(value.split())


def validate_generated_paraphrases(
    original_question: str, paraphrases: Any
) -> list[str]:
    if not isinstance(paraphrases, list) or len(paraphrases) != 4:
        raise ValueError("The API response must contain exactly four paraphrases")
    cleaned: list[str] = []
    for index, value in enumerate(paraphrases, start=1):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Paraphrase {index} is not a non-empty string")
        question = value.strip()
        if not question.endswith("?"):
            raise ValueError(f"Paraphrase {index} must end with a question mark")
        cleaned.append(question)
    normalized = [normalize_surface(original_question), *map(normalize_surface, cleaned)]
    if len(set(normalized)) != QUESTION_SET_SIZE:
        raise ValueError("Original and generated questions must all be unique")
    return cleaned


def answer_leak_warnings(
    original_question: str,
    paraphrases: list[str],
    aliases: list[str],
) -> list[dict[str, str]]:
    """Flag newly introduced gold strings for review; do not silently drop facts."""
    original = f" {normalize_for_leak_check(original_question)} "
    warnings: list[dict[str, str]] = []
    normalized_aliases: dict[str, str] = {}
    for alias in aliases:
        normalized = normalize_for_leak_check(alias)
        if len(normalized) >= 4:
            normalized_aliases.setdefault(normalized, alias)
    for variant_index, question in enumerate(paraphrases, start=1):
        normalized_question = f" {normalize_for_leak_check(question)} "
        for normalized_alias, raw_alias in normalized_aliases.items():
            needle = f" {normalized_alias} "
            if needle not in original and needle in normalized_question:
                warnings.append(
                    {
                        "variant_id": f"v{variant_index}",
                        "warning": "NEW_GOLD_ALIAS_IN_QUESTION",
                        "alias": raw_alias,
                    }
                )
    return warnings


def build_question_set(
    fact: dict[str, Any],
    paraphrases: list[str],
    *,
    model: str,
    response_id: str | None,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "schema_version": SCHEMA_VERSION,
        "fact_id": fact["fact_id"],
        "question_id": fact["question_id"],
        "question": fact["question"],
        "canonical_answer": fact.get("canonical_answer"),
        "answer_aliases": fact["answer_aliases"],
        "split": fact.get("split", ""),
        "source": fact.get("source", "trivia_qa/rc"),
        "variants": [
            {
                "variant_id": "v0",
                "variant_index": 0,
                "question": fact["question"],
                "source": "original",
            },
            *[
                {
                    "variant_id": f"v{index}",
                    "variant_index": index,
                    "question": question,
                    "source": "openai_paraphrase",
                }
                for index, question in enumerate(paraphrases, start=1)
            ],
        ],
        "paraphrase_generation": {
            "model": model,
            "prompt_version": PARAPHRASE_PROMPT_VERSION,
            "response_id": response_id,
            "usage": usage,
            "created_at_utc": utc_now(),
            "store": False,
        },
        "validation_warnings": answer_leak_warnings(
            fact["question"], paraphrases, fact["answer_aliases"]
        ),
        "needs_manual_semantic_review": True,
        "review_status": "unreviewed",
    }
    validate_question_set(row)
    return row


def response_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        name: getattr(usage, name)
        for name in ("input_tokens", "output_tokens", "total_tokens")
        if getattr(usage, name, None) is not None
    }


def request_paraphrases(
    client: Any,
    *,
    model: str,
    question: str,
    max_retries: int,
) -> tuple[list[str], str | None, dict[str, Any] | None]:
    if max_retries < 1:
        raise ValueError("--max-retries must be positive")
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            retry_feedback = (
                ""
                if last_error is None
                else (
                    "\n\nThe previous draft failed local validation: "
                    f"{type(last_error).__name__}: {last_error}. "
                    "Return a corrected, distinct set."
                )
            )
            response = client.responses.create(
                model=model,
                instructions=PARAPHRASE_INSTRUCTIONS,
                input=(
                    "Rewrite this original question. Do not use or infer any external "
                    f"context beyond its wording.\n\nOriginal question: {question}"
                    f"{retry_feedback}"
                ),
                text={"format": PARAPHRASE_SCHEMA},
                max_output_tokens=500,
                store=False,
            )
            payload = json.loads(response.output_text)
            paraphrases = validate_generated_paraphrases(
                question, payload.get("paraphrases")
            )
            return paraphrases, getattr(response, "id", None), response_usage(response)
        except Exception as error:  # API and schema errors are both retryable here.
            last_error = error
            if attempt == max_retries:
                break
            delay = min(2 ** (attempt - 1), 30)
            print(
                f"Attempt {attempt}/{max_retries} failed "
                f"({type(error).__name__}); retrying in {delay}s"
            )
            time.sleep(delay)
    assert last_error is not None
    raise RuntimeError(
        f"Failed to generate valid paraphrases after {max_retries} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def validate_resume_rows(
    facts_by_id: dict[str, dict[str, Any]],
    completed_rows: list[dict[str, Any]],
    model: str,
) -> set[str]:
    completed: set[str] = set()
    for row in completed_rows:
        validate_question_set(row)
        fact_id = row["fact_id"]
        if fact_id in completed:
            raise ValueError(f"Duplicate completed fact in output: {fact_id}")
        if fact_id not in facts_by_id:
            raise ValueError(f"Output contains a fact absent from current input: {fact_id}")
        fact = facts_by_id[fact_id]
        if (
            row["question"] != fact["question"]
            or row["answer_aliases"] != fact["answer_aliases"]
            or row.get("canonical_answer") != fact.get("canonical_answer")
        ):
            raise ValueError(f"Input changed for completed fact: {fact_id}")
        generation = row.get("paraphrase_generation", {})
        if generation.get("model") != model:
            raise ValueError(
                f"Cannot resume {fact_id}: model changed from "
                f"{generation.get('model')!r} to {model!r}"
            )
        if generation.get("prompt_version") != PARAPHRASE_PROMPT_VERSION:
            raise ValueError(f"Cannot resume {fact_id}: prompt version changed")
        completed.add(fact_id)
    return completed


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be 0 or positive")

    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "python-dotenv is required; install TrivaQA_ex/requirements.txt"
        ) from error
    load_dotenv(args.env_file, override=False)

    model = (
        args.model
        or os.getenv("OPENAI_PARAPHRASE_MODEL")
        or DEFAULT_PARAPHRASE_MODEL
    )
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(f"OPENAI_API_KEY was not found in the environment or {args.env_file}")

    facts = load_jsonl(args.input)
    if args.limit:
        facts = facts[: args.limit]
    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    if len(facts_by_id) != len(facts):
        raise ValueError("Input manifest contains duplicate fact_id values")

    completed: set[str] = set()
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(
                f"{args.output} already exists; pass --resume to continue it"
            )
        completed = validate_resume_rows(
            facts_by_id, load_jsonl(args.output), model
        )
    elif args.resume:
        print(f"Resume requested, but {args.output} does not exist; starting a new file")

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai package is required; install TrivaQA_ex/requirements.txt"
        ) from error
    client = OpenAI()

    pending = [fact for fact in facts if fact["fact_id"] not in completed]
    print(
        f"Question sets: {len(facts):,} total, {len(completed):,} cached, "
        f"{len(pending):,} pending"
    )
    for index, fact in enumerate(pending, start=1):
        paraphrases, response_id, usage = request_paraphrases(
            client,
            model=model,
            question=fact["question"],
            max_retries=args.max_retries,
        )
        row = build_question_set(
            fact,
            paraphrases,
            model=model,
            response_id=response_id,
            usage=usage,
        )
        append_jsonl(args.output, row)
        warning_count = len(row["validation_warnings"])
        print(
            f"[{index}/{len(pending)}] {fact['fact_id']} saved"
            + (f" ({warning_count} review warning(s))" if warning_count else "")
        )

    print(f"Frozen original+4 question sets -> {args.output}")
    print("Manual semantic review is required before the full model run.")


if __name__ == "__main__":
    main()
