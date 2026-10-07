"""Extract residual-stream hidden states for SAE analysis (PopQA).

Both models use the same capture rule: a forward hook on every selected
decoder layer, taking the residual stream AFTER that layer at the last prompt
token (prompts are left-padded, so that is always index -1 of the prompt).

  qwen  : one prompt-only forward (answers already exist in analysis/inference).
          saved tensor per prompt: [L, D]
  dream : re-runs diffusion generation (seed fixed) so the answers match the
          saved hidden states. The prompt forward happens at every denoising
          step with the answer block still partly masked; the hook captures the
          last-prompt-token state at the chosen steps.
          saved tensor per prompt: [S, L, D]; step 0 = answer block fully masked,
          the closest analogue of Qwen's "just before generating" state.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 extract_hidden.py --kind qwen \
        --model Qwen/Qwen2.5-7B --output-dir analysis/hidden/qwen_popqa
    CUDA_VISIBLE_DEVICES=1 python3 extract_hidden.py --kind dream \
        --model Dream-org/Dream-v0-Base-7B --output-dir analysis/hidden/dream_popqa \
        --trust-remote-code

Resumable: (sample_id, version) pairs already in <output-dir>/meta.jsonl are skipped.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import run_inference as ri

DEFAULT_DREAM_STEPS = [0, 1, 2, 4, 8, 16, 32, 63]


def load_model(path, trust_remote_code):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=trust_remote_code
        )
    except ValueError:
        model = AutoModel.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=trust_remote_code
        )
    return model.eval()


class Capture:
    """Collects the last-prompt-token residual of the selected layers."""

    def __init__(self, model, layers, steps):
        self.layers = layers
        self.steps = steps
        self.step_pos = {s: i for i, s in enumerate(steps)} if steps else None
        self.first_layer = min(layers)
        self.step = -1
        self.buf = None
        self.handles = []
        for li in layers:
            self.handles.append(model.model.layers[li].register_forward_hook(self._make_hook(li)))

    def reset(self, batch_size, d_model, device):
        self.step = -1
        n_s = len(self.steps) if self.steps else 1
        self.buf = torch.zeros(batch_size, n_s, len(self.layers), d_model, dtype=torch.bfloat16, device=device)

    def _make_hook(self, li):
        l_idx = self.layers.index(li)

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if li == self.first_layer:
                self.step += 1
            if self.steps is None:
                s_idx = 0
            elif self.step in self.step_pos:
                s_idx = self.step_pos[self.step]
            else:
                return
            self.buf[:, s_idx, l_idx, :] = h[:, self.prompt_last, :].to(torch.bfloat16)

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def run_batch(kind, model, tok, cap, prompts, max_new_tokens, diffusion_steps, seed):
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    cap.prompt_last = inputs["input_ids"].shape[1] - 1
    cap.reset(len(prompts), model.config.hidden_size, model.device)
    answers = None
    if kind == "qwen":
        model(**inputs)
    else:
        torch.manual_seed(seed)
        out = model.diffusion_generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            steps=diffusion_steps,
            temperature=0.0,
            alg="origin",
            mask_token_id=tok.mask_token_id,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        answers = []
        for i in range(len(prompts)):
            raw = tok.decode(out[i][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            answers.append(ri.extract_answer(raw))
    h = cap.buf.cpu()
    if kind == "qwen":
        h = h[:, 0]
    return h, answers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["qwen", "dream"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default="data/popqa/4. popqa_paraphrased.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--layers", default="all", help="comma list of layer indices, or 'all'")
    ap.add_argument("--steps", default=",".join(map(str, DEFAULT_DREAM_STEPS)), help="dream only")
    ap.add_argument("--diffusion-steps", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N facts only")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "meta.jsonl")
    done = {(r["sample_id"], r["version"]) for r in ri.load_jsonl(meta_path)}
    n_shards = len([f for f in os.listdir(args.output_dir) if f.startswith("shard_")])

    rows = ri.load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    tasks = []
    for row in rows:
        sid = ri.get_sample_id(row, "popqa")
        gold, aliases = ri.get_gold(row, "popqa")
        for v in ri.VERSIONS:
            if (sid, v) not in done:
                tasks.append(dict(sample_id=sid, version=v, question=row[v], gold_value=gold, gold_aliases=aliases))
    print(f"tasks: {len(tasks)} (done {len(done)})", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load_model(args.model, args.trust_remote_code)

    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    steps = [int(x) for x in args.steps.split(",")] if args.kind == "dream" else None
    cap = Capture(model, layers, steps)

    cfg = dict(kind=args.kind, model=args.model, layers=layers, steps=steps, diffusion_steps=args.diffusion_steps,
               max_new_tokens=args.max_new_tokens, seed=args.seed, batch_size=args.batch_size,
               hidden_size=model.config.hidden_size)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    meta_f = open(meta_path, "a", encoding="utf-8")
    for b in range(0, len(tasks), args.batch_size):
        batch = tasks[b : b + args.batch_size]
        prompts = [ri.build_prompt(t["question"]) for t in batch]
        h, answers = run_batch(args.kind, model, tok, cap, prompts, args.max_new_tokens,
                               args.diffusion_steps, args.seed + n_shards)
        shard = f"shard_{n_shards:06d}.pt"
        torch.save({"keys": [(t["sample_id"], t["version"]) for t in batch], "h": h}, os.path.join(args.output_dir, shard))
        for i, t in enumerate(batch):
            rec = dict(t, shard=shard, idx=i)
            if answers is not None:
                rec["model_answer"] = answers[i]
            meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta_f.flush()
        n_shards += 1
        if n_shards % 20 == 0:
            print(f"shards {n_shards}  ({min(b + args.batch_size, len(tasks))}/{len(tasks)})", file=sys.stderr)
    cap.remove()


if __name__ == "__main__":
    main()
