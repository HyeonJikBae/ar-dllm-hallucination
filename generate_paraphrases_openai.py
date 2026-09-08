"""Generate 4 paraphrases per PopQA question using OpenAI (gpt-4o-mini).

Each row is sent as its OWN independent API call, so the model never sees
other rows and cannot build/reuse a cross-row sentence template - this is
the structural fix for the templating problem seen with batch generation.

Usage:
    OPENAI_API_KEY=... python3 generate_paraphrases_openai.py \
        --input data/popqa_paraphrase_chunks/chunk_031.jsonl \
        --output analysis/popqa_paraphrase_out/chunk_031.jsonl

Resumable: rows whose sample_id already exists in --output are skipped.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You paraphrase a single trivia question into 4 distinct versions (v1-v4).

First identify:
- ANSWER_TYPE: what kind of thing the answer must be (e.g. "which country", "what group", "which island", "which actor")
- RELATION: the core predicate linking the subject and the answer (e.g. "is administered by", "is brought to life by", "played the role of")

Then generate v1-v4 following these rules:

MUST KEEP:
1. Keep the ANSWER_TYPE's category (country -> nation OK; country -> who NOT OK).
2. Do not change RELATION to a different relation (same relation in different words is fine). Do NOT flip the direction of the relation (subject/object order) - e.g. "Who is the father of Gao Wei?" -> "Who did Gao Wei father?" is NOT allowed.
3. Keep the same specificity level of the answer category.

MUST CHANGE:
4. Substantially vary word order, verb choice, and modifiers across v1-v4. Do not just reverse word order. Do not just swap one word for a synonym.
5. Vary the sentence-opening structure across v1-v4 - do not default to the same opening pattern (e.g. always "Which individual...") for every question. Mix question forms such as Who/What/Which/Can you name/statement-form-with-question-mark.
6. Every one of v1-v4 must be worded differently from the original question - none of them may be identical or near-identical to it.

FORBIDDEN:
7. Do not add any information not in the original question, especially hints about the answer's form.
8. Do not include the gold answer string or any part of it in the paraphrased question.
9. Do not produce grammatically awkward or unnatural English sentences.

Respond with ONLY a JSON object with exactly these keys: answer_type, relation, v1, v2, v3, v4. No other text."""


def build_user_prompt(question: str) -> str:
    return f"Original question: {question}"


def call_openai(client: OpenAI, question: str, max_retries: int = 5) -> dict:
    delay = 2.0
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question)},
                ],
                temperature=0.9,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            obj = json.loads(content)
            for k in ("answer_type", "relation", "v1", "v2", "v3", "v4"):
                if k not in obj:
                    raise ValueError(f"missing key {k} in response: {obj}")
            return obj
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"failed after {max_retries} retries: {last_err}")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def process_row(client, row):
    sample_id = str(row["id"])
    question = row["question"]
    result = call_openai(client, question)
    return {
        "sample_id": sample_id,
        "question": question,
        "answer_type": result["answer_type"],
        "relation": result["relation"],
        "v1": result["v1"],
        "v2": result["v2"],
        "v3": result["v3"],
        "v4": result["v4"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    rows = load_jsonl(args.input)
    existing = load_jsonl(args.output)
    done_ids = {r["sample_id"] for r in existing}
    todo = [r for r in rows if str(r["id"]) not in done_ids]

    print(f"input rows: {len(rows)}, already done: {len(done_ids)}, todo: {len(todo)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "a", encoding="utf-8")

    completed = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_row, client, row): row for row in todo}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                result = fut.result()
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                completed += 1
                if completed % 25 == 0:
                    print(f"  {completed}/{len(todo)}", end="\r")
            except Exception as e:  # noqa: BLE001
                failed.append((row["id"], str(e)))

    out_f.close()
    print()
    print(f"completed: {completed}, failed: {len(failed)}")
    if failed:
        print("failed ids:", failed[:10])


if __name__ == "__main__":
    main()
