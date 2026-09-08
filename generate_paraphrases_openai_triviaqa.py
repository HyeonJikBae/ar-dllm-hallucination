"""Generate 4 paraphrases per TriviaQA question using OpenAI (gpt-4o-mini).

Each row is sent as its OWN independent API call, so the model never sees
other rows and cannot build/reuse a cross-row sentence template.

Usage:
    OPENAI_API_KEY=... python3 generate_paraphrases_openai_triviaqa.py \
        --input "data/3. triviaqa_remaining.jsonl" \
        --output analysis/triviaqa_paraphrase_out.jsonl

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
- ANSWER_TYPE: what kind of thing the answer must be (e.g. "which person", "which country", "which film", "which year", "what number")
- RELATION: the core predicate/fact linking the subject and the answer, as stated in the original question

Then generate v1-v4 following these rules:

MUST KEEP:
1. Keep the ANSWER_TYPE's category (film -> movie/work OK; film -> person NOT OK).
2. Do not change RELATION to a different relation (same relation in different words is fine). Do NOT flip the direction of the relation (subject/object order).
3. Keep the same specificity level of the answer category (e.g. do not turn "which city" into "which country").
4. Keep every named entity, date, and number from the original question exactly as given (do not add, drop, or substitute any of them).

MUST CHANGE:
5. Substantially vary word order, verb choice, and modifiers across v1-v4. Do not just reverse word order. Do not just swap one word for a synonym.
6. Vary the sentence-opening structure across v1-v4 - do not default to the same opening pattern for every question. Mix question forms such as Who/What/Which/Can you name/statement-form-with-question-mark.
7. Every one of v1-v4 must be worded differently from the original question - none of them may be identical or near-identical to it.

FORBIDDEN:
8. Do not add any information not in the original question, especially hints about the answer's form.
9. Do not include the gold answer or any of its known aliases (given below) or any part of them in the paraphrased question.
10. Do not produce grammatically awkward or unnatural English sentences.

Respond with ONLY a JSON object with exactly these keys: answer_type, relation, v1, v2, v3, v4. No other text."""


def build_user_prompt(question: str, value: str, aliases: list[str]) -> str:
    all_aliases = sorted(set([value] + list(aliases)))
    return (
        f"Original question: {question}\n"
        f"Known answer and aliases to avoid mentioning: {', '.join(all_aliases)}"
    )


def call_openai(client: OpenAI, question: str, value: str, aliases: list[str], max_retries: int = 5) -> dict:
    delay = 2.0
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, value, aliases)},
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
    sample_id = str(row["question_id"])
    question = row["question"]
    answer = row["answer"]
    value = answer.get("value", "")
    aliases = answer.get("aliases", [])
    result = call_openai(client, question, value, aliases)
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
    todo = [r for r in rows if str(r["question_id"]) not in done_ids]

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
                failed.append((row["question_id"], str(e)))

    out_f.close()
    print()
    print(f"completed: {completed}, failed: {len(failed)}")
    if failed:
        print("failed ids:", failed[:10])


if __name__ == "__main__":
    main()
