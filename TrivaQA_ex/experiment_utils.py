"""Shared schemas and small I/O helpers for the TriviaQA experiment."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = 1
QUESTION_SET_SIZE = 5
QA_INSTRUCTION = (
    "Answer the following question with only the short factual answer. "
    "Do not provide any explanation."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_fact_id(question_id: str, question: str) -> str:
    """Create a stable key even when TriviaQA reuses a question_id."""
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
    return f"{question_id}__{digest}"


def render_plain_prompt(question: str) -> str:
    return f"{QA_INSTRUCTION}\n\nQuestion: {question.strip()}\nAnswer:"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}"
                )
            yield value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    _atomic_text_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_text_write(path, text)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_question_set(row: dict[str, Any]) -> None:
    required = {"fact_id", "question_id", "question", "answer_aliases", "variants"}
    missing = required.difference(row)
    if missing:
        raise ValueError(
            f"Question set {row.get('fact_id', '<unknown>')} is missing {sorted(missing)}"
        )
    variants = row["variants"]
    if not isinstance(variants, list) or len(variants) != QUESTION_SET_SIZE:
        raise ValueError(
            f"{row['fact_id']} must have exactly {QUESTION_SET_SIZE} variants"
        )
    expected_ids = [f"v{index}" for index in range(QUESTION_SET_SIZE)]
    actual_ids = [variant.get("variant_id") for variant in variants]
    if actual_ids != expected_ids:
        raise ValueError(
            f"{row['fact_id']} variant IDs must be {expected_ids}, got {actual_ids}"
        )
    texts = [str(variant.get("question", "")).strip() for variant in variants]
    if any(not text for text in texts):
        raise ValueError(f"{row['fact_id']} contains an empty question variant")
    if texts[0] != str(row["question"]).strip():
        raise ValueError(f"{row['fact_id']} v0 must be the unchanged original question")
    normalized = [" ".join(text.casefold().split()) for text in texts]
    if len(set(normalized)) != QUESTION_SET_SIZE:
        raise ValueError(f"{row['fact_id']} contains duplicate question variants")

