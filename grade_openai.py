"""Grade model QA answers with GPT-4o mini through the OpenAI Batch API.

The script never makes lexical correctness decisions. It only packages five
answers for one fact, submits them to the semantic judge, and validates/merges
the returned judgments.

Usage (one full cycle):
    OPENAI_API_KEY=... python3 grade_openai.py prepare  --model qwen2.5-7b --dataset popqa
    OPENAI_API_KEY=... python3 grade_openai.py submit   --model qwen2.5-7b --dataset popqa
    OPENAI_API_KEY=... python3 grade_openai.py status   --model qwen2.5-7b --dataset popqa
    OPENAI_API_KEY=... python3 grade_openai.py download --model qwen2.5-7b --dataset popqa
    OPENAI_API_KEY=... python3 grade_openai.py finalize --model qwen2.5-7b --dataset popqa

Run for all four combinations: qwen2.5-7b/dream-7b x popqa/triviaqa.
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI

VERSIONS = ("question", "v1", "v2", "v3", "v4")
MODEL = "gpt-4o-mini-2024-07-18"
KEY_FILE = Path("/tmp/hallucination_openai_api_key")

EXPECTED_FACTS = {"popqa": 11_045, "triviaqa": 10_739}

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


class Paths:
    def __init__(self, model: str, dataset: str, limit: int | None = None):
        tag = f"{model}_{dataset}"
        if limit:
            tag = f"{tag}_pilot{limit}"
        self.source = Path(f"analysis/inference/{model}_{dataset}.jsonl")
        self.work = Path("analysis/graded/openai_batch")
        self.requests = self.work / f"{tag}_requests.jsonl"
        self.state = self.work / f"{tag}_batch.json"
        self.raw_output = self.work / f"{tag}_batch_output.jsonl"
        self.graded = Path(f"analysis/graded/{tag}_graded.jsonl")
        self.summary = Path(f"analysis/graded/{tag}_summary.csv")
        self.expected_facts = limit if limit else EXPECTED_FACTS[dataset]
        self.expected_rows = self.expected_facts * len(VERSIONS)
        self.limit = limit
        self.tag = tag


def load_source(paths: Paths):
    rows = [json.loads(line) for line in paths.source.open(encoding="utf-8") if line.strip()]
    groups = defaultdict(dict)
    for row in rows:
        groups[str(row["sample_id"])][row["version"]] = row
    if paths.limit:
        keep_ids = list(groups)[: paths.limit]
        groups = {sid: groups[sid] for sid in keep_ids}
        rows = [row for row in rows if str(row["sample_id"]) in groups]
    if len(rows) != paths.expected_rows or len(groups) != paths.expected_facts:
        raise ValueError(
            f"unexpected source dimensions: {len(rows)} rows, {len(groups)} facts "
            f"(expected {paths.expected_rows} rows, {paths.expected_facts} facts)"
        )
    for sid, group in groups.items():
        if set(group) != set(VERSIONS):
            raise ValueError(f"incomplete versions for {sid}: {sorted(group)}")
    return rows, groups


def prepare(paths: Paths):
    _, groups = load_source(paths)
    paths.work.mkdir(parents=True, exist_ok=True)
    with paths.requests.open("w", encoding="utf-8") as out:
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
    print(f"prepared {len(groups)} requests at {paths.requests}")


def client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise SystemExit(f"Set OPENAI_API_KEY or create {KEY_FILE} with mode 600")
    return OpenAI(api_key=key, timeout=10, max_retries=0)


def submit(paths: Paths):
    api = client()
    with paths.requests.open("rb") as fh:
        uploaded = api.files.create(file=fh, purpose="batch")
    batch = api.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"task": f"{paths.tag}-semantic-grading"},
    )
    paths.state.write_text(json.dumps({"batch_id": batch.id, "input_file_id": uploaded.id}, indent=2) + "\n")
    print(json.dumps({"batch_id": batch.id, "status": batch.status}))


def status(paths: Paths, download=False):
    api = client()
    state = json.loads(paths.state.read_text())
    batch = api.batches.retrieve(state["batch_id"])
    print(json.dumps({"batch_id": batch.id, "status": batch.status, "counts": str(batch.request_counts)}))
    if download and batch.output_file_id:
        paths.raw_output.write_bytes(api.files.content(batch.output_file_id).read())
        print(f"downloaded {paths.raw_output}")


def query_one(api, items, temperature=0):
    """Synchronous single-fact retry for batch rows that came back malformed."""
    resp = api.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
        ],
        response_format={"type": "json_schema", "json_schema": SCHEMA},
    )
    content = resp.choices[0].message.content
    parsed = json.loads(content)["decisions"]
    return {d["version"]: d for d in parsed}


SINGLE_SCHEMA = {
    "name": "qa_grade_single",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["correct", "incorrect", "uncertain"]},
            "reason": {"type": ["integer", "null"], "enum": [1, 2, 3, 4, 5, 6, None]},
            "gold_issue": {"type": "boolean"},
        },
        "required": ["verdict", "reason", "gold_issue"],
        "additionalProperties": False,
    },
}


def query_one_version(api, item):
    """Last-resort fallback: grade a single version alone (no siblings), so the
    model cannot conflate near-duplicate sibling answers and drop one."""
    resp = api.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": json.dumps([item], ensure_ascii=False)},
        ],
        response_format={"type": "json_schema", "json_schema": SINGLE_SCHEMA},
    )
    content = resp.choices[0].message.content
    return json.loads(content)


def _retry_one_sid(api, sid, group):
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
    by_version = None
    for temp in [0, 0.4, 0.7, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.2]:
        try:
            candidate = query_one(api, items, temperature=temp)
        except Exception:
            continue
        if set(candidate) == set(VERSIONS):
            by_version = candidate
            break
    if by_version is None:
        # last resort: grade each version alone, one call per version
        try:
            by_version = {item["version"]: {**query_one_version(api, item), "version": item["version"]} for item in items}
        except Exception:
            by_version = None
    if by_version is None or set(by_version) != set(VERSIONS):
        return sid, None
    return sid, by_version


def retry_failed(paths: Paths, sids, groups, workers: int = 16):
    import sys, time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api = client()
    fixed = {}
    still_failing = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_retry_one_sid, api, sid, groups[sid]): sid for sid in sids}
        for fut in as_completed(futures):
            sid, by_version = fut.result()
            done += 1
            if by_version is None:
                still_failing.append(sid)
            else:
                fixed[sid] = by_version
            if done % 20 == 0 or done == len(sids):
                elapsed = time.time() - t0
                print(f"  [{done}/{len(sids)}] fixed={len(fixed)} still_failing={len(still_failing)} elapsed={elapsed:.0f}s", file=sys.stderr, flush=True)
    print(f"retried {len(sids)} facts ({workers} workers): fixed={len(fixed)} still_failing={len(still_failing)}")
    return fixed, still_failing


def finalize(paths: Paths):
    rows, groups = load_source(paths)
    decisions = {}
    failures = []
    for line in paths.raw_output.open(encoding="utf-8"):
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

    missing = set(groups) - set(decisions) - set(failures)
    failures = list(failures) + sorted(missing)

    if failures:
        fixed, still_failing = retry_failed(paths, failures, groups)
        decisions.update(fixed)
        failures = still_failing

    if failures or len(decisions) != paths.expected_facts:
        raise ValueError(f"batch is incomplete: {len(decisions)} facts, failures={failures[:20]}")

    paths.graded.parent.mkdir(parents=True, exist_ok=True)
    with paths.graded.open("w", encoding="utf-8") as out:
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
    with paths.summary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["correct_out_of_5", "fact_count", "percentage", "uncertain_fact_count"])
        for n in range(6):
            writer.writerow([n, distribution[n], f"{distribution[n] / len(grouped) * 100:.4f}", uncertain_facts])
    print(f"wrote {len(rows)} graded rows to {paths.graded} and summary to {paths.summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "submit", "status", "download", "finalize"])
    parser.add_argument("--model", required=True, choices=["qwen2.5-7b", "dream-7b"])
    parser.add_argument("--dataset", required=True, choices=["popqa", "triviaqa"])
    parser.add_argument("--limit", type=int, default=None, help="pilot mode: only grade first N facts")
    args = parser.parse_args()
    paths = Paths(args.model, args.dataset, args.limit)
    if args.action == "prepare":
        prepare(paths)
    elif args.action == "submit":
        submit(paths)
    elif args.action == "status":
        status(paths)
    elif args.action == "download":
        status(paths, download=True)
    else:
        finalize(paths)


if __name__ == "__main__":
    main()
