#!/usr/bin/env python3
"""Apply the documented semantic review to the frozen 100-fact paraphrase set."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiment_utils import load_jsonl, utc_now, validate_question_set, write_json, write_jsonl
from generate_paraphrases import answer_leak_warnings


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "data" / "triviaqa_questions_5.jsonl"

# Only variants that changed the requested answer, inverted a direction, or made
# the answer target ambiguous are replaced. v0 is never modified.
REPAIRS: dict[str, dict[str, str]] = {
    "qb_7132__01d756cb549c": {
        "v1": "What final word completes Lindisfarne's 1971 album title 'The Fog on the ___'?",
        "v2": "Lindisfarne released 'The Fog on the' what as a hit album in 1971?",
        "v3": "Which place name fills the blank in the 1971 Lindisfarne album 'The Fog on the ___'?",
        "v4": "In the title of Lindisfarne's 1971 hit album, what word follows 'The Fog on the'?",
    },
    "qw_9494__36c2e0eb46a5": {
        "v2": "Name the country for which Damascus serves as the capital city?",
    },
    "qb_590__b3a587265ac2": {
        "v2": "Which English singer-songwriter, aged eighteen at the time, wrote the Tremeloes hit 'Here Comes My Baby'?",
    },
    "odql_3280__0f3c67297533": {
        "v1": "Identify the African country with Chad to its east, Nigeria and Benin to its south, Mali to its west, and Algeria and Libya to its north?",
        "v4": "Chad forms the eastern border of which African nation that also has Nigeria and Benin to the south, Mali to the west, and Algeria and Libya to the north?",
    },
    "sfq_19011__697b024818fc": {
        "v1": "A Clumber belongs to which general type of dog?",
        "v2": "Complete the dog type: Clumber what?",
        "v3": "Clumber is a breed in which broader dog category?",
        "v4": "Which general dog type includes the Clumber breed?",
    },
    "qb_3702__8d8cf76caf05": {
        "v1": "Which consumer products did psychoanalyst A A Brill call 'Torches of Freedom' in the 1920s as symbols of women's equality?",
        "v3": "What objects did psychoanalyst A A Brill describe as 'Torches of Freedom' in the 1920s to symbolize equality for women?",
    },
    "qw_15839__d37c3243bbf5": {
        "v4": "Who wrote and released the autobiography 'A Journey' on 1 September 2010?",
    },
    "qw_13355__256e8e0fdab6": {
        "v1": "On what kind of playing surface is the ancient game of senet traditionally played?",
        "v2": "What physical surface is traditionally used to play the ancient game of senet?",
        "v3": "Senet is traditionally played on what object?",
        "v4": "The ancient game known as senet uses what playing surface?",
    },
}

REVIEW_NOTES = {
    "qb_7132__01d756cb549c": "Made the missing title word, rather than the full album title, the answer target.",
    "qw_9494__36c2e0eb46a5": "Removed a country/capital inversion.",
    "qb_590__b3a587265ac2": "Restored the musician, rather than age, as the answer target.",
    "odql_3280__0f3c67297533": "Corrected inverted east/west relational wording.",
    "sfq_19011__697b024818fc": "Kept the general dog type as the answer target.",
    "qb_3702__8d8cf76caf05": "Made the objects/products, rather than the phrase itself, the answer target.",
    "qw_15839__d37c3243bbf5": "Removed publisher-person versus publishing-company ambiguity.",
    "qw_13355__256e8e0fdab6": "Made the playing surface, rather than a geographic location, the answer target.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Required when output already exists, including in-place review.",
    )
    return parser.parse_args()


def review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = {row["fact_id"] for row in rows}
    missing_repair_targets = set(REPAIRS).difference(ids)
    if missing_repair_targets:
        raise ValueError(f"Repair targets are absent: {sorted(missing_repair_targets)}")
    reviewed: list[dict[str, Any]] = []
    for row in rows:
        variants = []
        repairs = REPAIRS.get(row["fact_id"], {})
        for variant in row["variants"]:
            replacement = repairs.get(variant["variant_id"])
            variants.append(
                {
                    **variant,
                    "question": replacement or variant["question"],
                    "source": "manual_semantic_repair" if replacement else variant["source"],
                }
            )
        paraphrases = [variant["question"] for variant in variants[1:]]
        warnings = answer_leak_warnings(
            row["question"], paraphrases, row["answer_aliases"]
        )
        if warnings:
            raise ValueError(
                f"Reviewed row still has answer-leak warning(s): {row['fact_id']} {warnings}"
            )
        reviewed_row = {
            **row,
            "variants": variants,
            "validation_warnings": [],
            "needs_manual_semantic_review": False,
            "review_status": "approved",
            "semantic_review": {
                "reviewer": "codex_agent_manual_review",
                "reviewed_at_utc": utc_now(),
                "all_five_same_answer_checked": True,
                "repaired_variant_ids": sorted(repairs),
                "note": REVIEW_NOTES.get(row["fact_id"]),
            },
        }
        validate_question_set(reviewed_row)
        reviewed.append(reviewed_row)
    return reviewed


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    rows = load_jsonl(args.input)
    reviewed = review_rows(rows)
    write_jsonl(args.output, reviewed)
    repaired_variant_count = sum(len(value) for value in REPAIRS.values())
    write_json(
        args.output.with_suffix(args.output.suffix + ".review.json"),
        {
            "reviewed_at_utc": utc_now(),
            "reviewer": "codex_agent_manual_review",
            "fact_count": len(reviewed),
            "repaired_fact_count": len(REPAIRS),
            "repaired_variant_count": repaired_variant_count,
            "repair_notes": REVIEW_NOTES,
        },
    )
    print(
        f"Approved {len(reviewed):,} fact(s); repaired "
        f"{repaired_variant_count} variant(s) across {len(REPAIRS)} fact(s)"
    )
    print(f"Reviewed question sets -> {args.output}")


if __name__ == "__main__":
    main()
