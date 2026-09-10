"""Grade all Qwen PopQA answers with GPT-4o mini through the Batch API.

The script never makes lexical correctness decisions. It only packages five
answers for one fact, submits them to the semantic judge, and validates/merges
the returned judgments.
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI


SOURCE = Path("analysis/inference/qwen2.5-7b_popqa.jsonl")
WORK = Path("analysis/graded/openai_batch")
REQUESTS = WORK / "qwen2.5-7b_popqa_requests.jsonl"
STATE = WORK / "qwen2.5-7b_popqa_batch.json"
RAW_OUTPUT = WORK / "qwen2.5-7b_popqa_batch_output.jsonl"
GRADED = Path("analysis/graded/qwen2.5-7b_popqa_graded.jsonl")
SUMMARY = Path("analysis/graded/qwen2.5-7b_popqa_summary.csv")
VERSIONS = ("question", "v1", "v2", "v3", "v4")
MODEL = "gpt-4o-mini-2024-07-18"
KEY_FILE = Path("/tmp/hallucination_openai_api_key")

RUBRIC = """You are grading factual QA outputs. Judge each of the five answers semantically using the question, gold answer, and aliases. Do not use mere string matching.

verdict:
- correct: semantically answers the question, including valid equivalent wording.
- incorrect: factually wrong, incomplete enough to change the answer, or contains a correct answer followed by false claims.
- uncertain: evidence is insufficient to decide reliably, including a likely bad/underspecified gold.

For correct answers set reason=null. For incorrect answers assign one dominant reason when applicable:
1 related entity substitution: the output denotes a real entity/category and has an explainable relationship to the correct answer or satisfies meaningful question constraints.
2 unrelated entity substitution: the output denotes a real entity/category but no meaningful relationship to the correct answer can be explained.
3 subject copy: the output returns an entity from the question instead of the requested answer.
4 hallucination after correct answer: the correct answer is present, but added explanation contains a false factual claim.
5 recombination: contextual/name fragments were recombined into an entity or proper name that does not exist.
6 surface-form collapse: the intended answer is visibly damaged/truncated/misspelled, or output is empty. A harmless accepted spelling variant is correct, not 6.
If an incorrect response does not fit 1-6, use reason=null. Set gold_issue=true only when the gold is likely wrong, ambiguous, or materially underspecified. Compare sibling variants when this helps identify recombination or surface collapse. Return exactly one decision per supplied version."""

SCHEMA = {
    "name": "qa_grades",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "version": {"type": "string", "enum": list(VERSIONS)},
                        "verdict": {"type": "string", "enum": ["correct", "incorrect", "uncertain"]},
                        "reason": {"type": ["integer", "null"], "enum": [1, 2, 3, 4, 5, 6, None]},
                        "gold_issue": {"type": "boolean"},
                    },
                    "required": ["version", "verdict", "reason", "gold_issue"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    },
}


def load_source():
    rows = [json.loads(line) for line in SOURCE.open(encoding="utf-8") if line.strip()]
    groups = defaultdict(dict)
    for row in rows:
        groups[str(row["sample_id"])][row["version"]] = row
    if len(rows) != 55_225 or len(groups) != 11_045:
        raise ValueError(f"unexpected source dimensions: {len(rows)} rows, {len(groups)} facts")
    for sid, group in groups.items():
        if set(group) != set(VERSIONS):
            raise ValueError(f"incomplete versions for {sid}: {sorted(group)}")
    return rows, groups


def prepare():
    _, groups = load_source()
    WORK.mkdir(parents=True, exist_ok=True)
    with REQUESTS.open("w", encoding="utf-8") as out:
        for sid, group in groups.items():
            items = [
                {
                    "version": version,
                    "question": group[version]["question"],
                    "model_answer": group[version]["model_answer"],
                    "gold_value": group[version]["gold_value"],
                    "gold_aliases": group[version]["gold_aliases"],
                }
                for version in VERSIONS
            ]
            request = {
                "custom_id": f"fact-{sid}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": RUBRIC},
                        {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
                    ],
                    "response_format": {"type": "json_schema", "json_schema": SCHEMA},
                },
            }
            out.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"prepared {len(groups)} requests at {REQUESTS}")


def client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise SystemExit(f"Set OPENAI_API_KEY or create {KEY_FILE} with mode 600")
    return OpenAI(api_key=key)


def submit():
    api = client()
    with REQUESTS.open("rb") as fh:
        uploaded = api.files.create(file=fh, purpose="batch")
    batch = api.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"task": "qwen-popqa-semantic-grading"},
    )
    STATE.write_text(json.dumps({"batch_id": batch.id, "input_file_id": uploaded.id}, indent=2) + "\n")
    print(json.dumps({"batch_id": batch.id, "status": batch.status}))


def status(download=False):
    api = client()
    state = json.loads(STATE.read_text())
    batch = api.batches.retrieve(state["batch_id"])
    print(json.dumps({"batch_id": batch.id, "status": batch.status, "counts": str(batch.request_counts)}))
    if download and batch.output_file_id:
        RAW_OUTPUT.write_bytes(api.files.content(batch.output_file_id).read())
        print(f"downloaded {RAW_OUTPUT}")


def finalize():
    rows, _ = load_source()
    decisions = {}
    failures = []
    for line in RAW_OUTPUT.open(encoding="utf-8"):
        item = json.loads(line)
        sid = item["custom_id"].removeprefix("fact-")
        if item.get("error") or not item.get("response") or item["response"]["status_code"] != 200:
            failures.append(sid)
            continue
        choice = item["response"]["body"]["choices"][0]["message"]
        if choice.get("refusal"):
            failures.append(sid)
            continue
        parsed = json.loads(choice["content"])["decisions"]
        by_version = {d["version"]: d for d in parsed}
        if set(by_version) != set(VERSIONS):
            failures.append(sid)
            continue
        decisions[sid] = by_version
    if failures or len(decisions) != 11_045:
        raise ValueError(f"batch is incomplete: {len(decisions)} facts, failures={failures[:20]}")

    with GRADED.open("w", encoding="utf-8") as out:
        for row in rows:
            grade = decisions[str(row["sample_id"])][row["version"]]
            merged = dict(row)
            merged.update({k: grade[k] for k in ("verdict", "reason", "gold_issue")})
            out.write(json.dumps(merged, ensure_ascii=False) + "\n")

    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(decisions[str(row["sample_id"])][row["version"]])
    distribution = Counter(sum(d["verdict"] == "correct" for d in ds) for ds in grouped.values())
    uncertain_facts = sum(any(d["verdict"] == "uncertain" for d in ds) for ds in grouped.values())
    with SUMMARY.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["correct_out_of_5", "fact_count", "percentage", "uncertain_fact_count"])
        for n in range(6):
            writer.writerow([n, distribution[n], f"{distribution[n] / len(grouped) * 100:.4f}", uncertain_facts])
    print(f"wrote {len(rows)} graded rows to {GRADED} and summary to {SUMMARY}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "submit", "status", "download", "finalize"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "submit":
        submit()
    elif args.action == "status":
        status()
    elif args.action == "download":
        status(download=True)
    else:
        finalize()


if __name__ == "__main__":
    main()
