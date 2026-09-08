"""Run closed-book QA inference over the paraphrased PopQA/TriviaQA datasets.

For each row, queries the model with 5 question versions (original + v1-v4)
and saves the raw model answer alongside the gold answer/aliases so a
separate scoring step can later judge correctness (exact-match / substring
match against gold + aliases) per version.

This script only performs inference - it does not judge correctness.

Both Dream-7B and Qwen2.5-7B are used here as BASE (non-instruct) models, so
prompting uses a plain zero-shot instruction (not a chat template, no few-shot
exemplars) to keep the model's actual knowledge - rather than imitation of
demonstrated examples - the only thing being measured.

Usage:
    python3 run_inference.py \
        --model Qwen/Qwen2.5-7B \
        --dataset popqa \
        --input "data/popqa/4. popqa_paraphrased.jsonl" \
        --output analysis/inference/qwen2.5-7b_popqa.jsonl

    python3 run_inference.py \
        --model Dream-org/Dream-v0-Base-7B \
        --dataset triviaqa \
        --input "data/4. triviaqa_paraphrased.jsonl" \
        --output analysis/inference/dream-7b_triviaqa.jsonl \
        --trust-remote-code

Resumable: (sample_id, version) pairs already present in --output are skipped.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

VERSIONS = ["question", "v1", "v2", "v3", "v4"]

# Both Dream-7B and Qwen2.5-7B are used here as BASE models (no chat template).
# Zero-shot on purpose: this experiment measures whether the model actually
# knows the answer, so no QA exemplars are shown (few-shot examples would mix
# "imitating the demonstrated format" in with "actually knowing the fact" and
# contaminate that measurement). Only a one-line instruction is kept, to keep
# the answer short enough to grade automatically.
INSTRUCTION_PROMPT = "Answer the question with only the direct answer: a short phrase or name, no explanation, no full sentences.\n\n"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_gold(row: dict, dataset: str):
    if dataset == "popqa":
        aliases = json.loads(row.get("o_aliases", "[]"))
        return row["obj"], aliases
    answer = row["answer"]
    return answer["value"], answer.get("aliases", [])


def get_sample_id(row: dict, dataset: str) -> str:
    return str(row["id"]) if dataset == "popqa" else str(row["question_id"])


def build_prompt(question: str) -> str:
    return f"{INSTRUCTION_PROMPT}Question: {question}\nAnswer:"


def extract_answer(generated_text: str) -> str:
    # base models keep rambling past the answer (e.g. onto "Question: ..." for
    # the next few-shot turn) - cut at the first line break or the next
    # "Question:" marker, whichever comes first.
    text = generated_text.split("Question:")[0]
    text = text.split("\n")[0]
    return text.strip()


@torch.inference_mode()
def generate_batch(
    model, tokenizer, prompts: list[str], max_new_tokens: int = 32, diffusion_steps: int = 64
) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    if hasattr(model, "diffusion_generate"):
        # Dream is a diffusion LM: generation works by iteratively denoising a
        # fully-masked span rather than autoregressive next-token decoding, so
        # it uses its own diffusion_generate() entrypoint instead of generate().
        outputs = model.diffusion_generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            steps=diffusion_steps,
            temperature=0.0,
            alg="origin",
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    decoded = []
    for i in range(len(prompts)):
        gen_ids = outputs[i][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        decoded.append(extract_answer(raw))
    return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--dataset", required=True, choices=["popqa", "triviaqa"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--diffusion-steps", type=int, default=64, help="denoising steps, Dream (diffusion LM) only"
    )
    parser.add_argument("--limit", type=int, default=None, help="debug: only process first N rows")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]

    existing = load_jsonl(args.output)
    done_keys = {(r["sample_id"], r["version"]) for r in existing}

    print(f"loading model {args.model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
    except ValueError:
        # some custom architectures (e.g. Dream) only register under the
        # generic AutoModel, not AutoModelForCausalLM
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
    model.eval()

    # build the full (row, version) worklist
    tasks = []
    for row in rows:
        sample_id = get_sample_id(row, args.dataset)
        gold_value, gold_aliases = get_gold(row, args.dataset)
        for version in VERSIONS:
            key = (sample_id, version)
            if key in done_keys:
                continue
            tasks.append(
                {
                    "sample_id": sample_id,
                    "version": version,
                    "question": row[version],
                    "gold_value": gold_value,
                    "gold_aliases": gold_aliases,
                }
            )

    print(f"total tasks: {len(tasks)} (already done: {len(done_keys)})", file=sys.stderr)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "a", encoding="utf-8")

    for i in range(0, len(tasks), args.batch_size):
        batch = tasks[i : i + args.batch_size]
        prompts = [build_prompt(t["question"]) for t in batch]
        answers = generate_batch(
            model, tokenizer, prompts, max_new_tokens=args.max_new_tokens, diffusion_steps=args.diffusion_steps
        )
        for t, answer in zip(batch, answers):
            out_f.write(
                json.dumps(
                    {
                        "sample_id": t["sample_id"],
                        "version": t["version"],
                        "question": t["question"],
                        "model_answer": answer,
                        "gold_value": t["gold_value"],
                        "gold_aliases": t["gold_aliases"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        out_f.flush()
        print(f"  {min(i + args.batch_size, len(tasks))}/{len(tasks)}", end="\r", file=sys.stderr)

    out_f.close()
    print(f"\ndone: {len(tasks)} tasks written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
