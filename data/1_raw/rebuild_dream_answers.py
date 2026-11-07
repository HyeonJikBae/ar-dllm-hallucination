"""Rebuild Dream's answers from the saved canvas tokens: the text up to the first special token (<|endoftext|> etc.),
then leading spaces and line breaks removed, then the first line.  `generation` keeps the whole canvas with the special tokens.  (The stored model_answer came from run_inference.extract_answer, which also cut at "Question:" and did not
strip a leading line break first.)  Qwen's full generations were not saved; its stored answers are used as they are.
Usage (from this folder): python3 rebuild_dream_answers.py [--out work]
"""
import argparse, json, os, shutil, sys
import numpy as np
from transformers import AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2_grading"))
from run_grading import first_line  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=HERE); args = ap.parse_args()
cfg = json.load(open(os.path.join(HERE, "dream_hidden_steps", "config.json")))
tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"], trust_remote_code=True)
T = np.asarray(np.load(os.path.join(HERE, "dream_hidden_steps", "tokens.npy"), mmap_mode="r"))
SPECIAL = set(tok.all_special_ids)                                   # <|endoftext|>, <|beginoftext|>, <|mask|>


def until_special(row):
    """the answer ends at the first special token: Dream writes <|endoftext|> after its answer and then fills the
    rest of the 32-slot canvas with a new document (web page text, a math problem)"""
    out = []
    for t in row:
        if t < 0 or t in SPECIAL:
            break
        out.append(int(t))
    return out


full = [tok.decode(until_special(row)) for row in T]
canvas = tok.batch_decode([[t for t in row if t >= 0] for row in T], skip_special_tokens=False)
rows = [json.loads(l) for l in open(os.path.join(HERE, "dream_raw.jsonl"), encoding="utf-8")]   # sample_id / question / gold fields
assert len(rows) == len(full)
os.makedirs(args.out, exist_ok=True)
changed = 0
with open(os.path.join(args.out, "dream_raw.jsonl"), "w", encoding="utf-8") as f:
    for r, g, c in zip(rows, full, canvas):
        a = first_line(g)
        changed += a != (r["model_answer"] or "")
        f.write(json.dumps({**r, "model_answer": a, "generation": c}, ensure_ascii=False) + "\n")
print("dream answers changed:", changed, "->", args.out)
