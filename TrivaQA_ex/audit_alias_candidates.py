#!/usr/bin/env python3
"""Record the manual audit of canonical-vs-official-alias disagreements."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from experiment_utils import load_jsonl, utc_now, write_json, write_jsonl


HERE = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = HERE / "results" / "dream_qwen2_5_full" / "analysis"

# The audit is deliberately tied to exact normalized outputs, not just fact IDs.
# A future run producing a different candidate will fail closed and require review.
APPROVED_OUTPUTS: dict[str, tuple[set[str], str]] = {
    "odql_6828__b12b0cc91480": (
        {"uss missouri"},
        "Punctuation-equivalent spelling of U.S.S. Missouri.",
    ),
    "qg_762__b1c1047262a3": (
        {"adam s apple"},
        "Possessive spelling of Adam's apple.",
    ),
    "sfq_11719__1aed954113de": (
        {
            "theater",
            "fourth wall is considered imaginary wall in category of theater",
        },
        "American spelling or an explanatory answer containing theater.",
    ),
    "sfq_7345__c7da2cd4e1f2": (
        {"5"},
        "Numeric form of the canonical word Five.",
    ),
    "qw_10823__e181c7f9b649": (
        {
            "shot and killed by assassin",
            "william mckinley was assassinated by leonolgosz in buffalo new york on september 6 1901",
            "william mckinley was assassinated by leonolgosz on september 6 1901 in buffalo new york",
        },
        "Semantically states that McKinley died by assassination.",
    ),
    "qb_10008__bddc19d45bdf": (
        {
            "clothing",
            "vestiphobia characterized as irrational fear pertains to items that are worn or clothing",
            "vestiphobia describes irrational fear of items that are worn or clothing",
        },
        "Clothing is a valid equivalent of Clothes in this question.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates_path = args.analysis_dir / "alias_only_candidates.jsonl"
    candidates = load_jsonl(candidates_path)
    audited = []
    for row in candidates:
        approved = APPROVED_OUTPUTS.get(row["fact_id"])
        if approved is None or row["normalized_answer"] not in approved[0]:
            raise ValueError(
                "Unreviewed alias-only output encountered: "
                f"{row['model_name']}:{row['fact_id']}:{row['variant_id']} "
                f"{row['normalized_answer']!r}"
            )
        reason = approved[1]
        audited.append(
            {
                "model_name": row["model_name"],
                "fact_id": row["fact_id"],
                "variant_id": row["variant_id"],
                "canonical_answer": row["canonical_answer"],
                "scored_output": row["scored_output"],
                "normalized_answer": row["normalized_answer"],
                "official_alias_matched_alias": row[
                    "official_alias_matched_alias"
                ],
                "output_sha256": hashlib.sha256(
                    row["scored_output"].encode("utf-8")
                ).hexdigest(),
                "verdict": "valid_answer",
                "reason": reason,
                "reviewer": "codex_agent_manual_audit",
            }
        )

    observed_fact_ids = {row["fact_id"] for row in candidates}
    if observed_fact_ids != set(APPROVED_OUTPUTS):
        raise ValueError(
            "Expected reviewed fact set differs from current candidates: "
            f"expected={sorted(APPROVED_OUTPUTS)}, observed={sorted(observed_fact_ids)}"
        )
    output_path = args.analysis_dir / "alias_candidate_audit.jsonl"
    write_jsonl(output_path, audited)
    official_alias_wrong = load_jsonl(
        args.analysis_dir / "official_alias_wrong_outputs.jsonl"
    )
    write_jsonl(
        args.analysis_dir / "adjudicated_wrong_outputs.jsonl",
        official_alias_wrong,
    )
    counts = Counter(row["verdict"] for row in audited)
    write_json(
        args.analysis_dir / "alias_candidate_audit_summary.json",
        {
            "created_at_utc": utc_now(),
            "reviewer": "codex_agent_manual_audit",
            "candidate_count": len(candidates),
            "fact_count": len(observed_fact_ids),
            "verdict_counts": dict(sorted(counts.items())),
            "invalid_official_alias_match_count": counts.get("invalid_answer", 0),
            "accepted_metric": "official_alias",
            "adjudicated_wrong_output_count": len(official_alias_wrong),
            "note": (
                "Every observed canonical-vs-official-alias disagreement was a "
                "valid answer. The audit does not assert that every unused official "
                "TriviaQA alias is valid."
            ),
        },
    )
    print(
        f"Audited {len(audited)} alias-only candidate(s) across "
        f"{len(observed_fact_ids)} fact(s): all valid"
    )
    print(f"Audit -> {output_path}")
    print(
        "Adjudicated wrong outputs -> "
        f"{args.analysis_dir / 'adjudicated_wrong_outputs.jsonl'}"
    )


if __name__ == "__main__":
    main()
