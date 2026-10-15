"""Experiment 1 verification: re-generate with Dream ONLY the wording picked by a signal (one row per fact, fresh batches)
and grade the new answers. Compared with the stored all-wordings answers (lookup), original wording and random wording.
Usage: CUDA_VISIBLE_DEVICES=0 /home/minyeol/.conda/envs/sae-dream/bin/python sae_analysis/dream_regen_check.py --signal "확신도 + SAE 20개"
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "grounded_dream"), os.path.join(ROOT, "sae")):
    sys.path.insert(0, p)
from string_grader import contains_gold, string_rule  # noqa: E402
D = "sae_analysis/results_twelve/dream_ext"; U = "sae_analysis/results_twelve/unified"

def grade(a, r):
    code = string_rule(a, r["question"], r["gold_value"], r["gold_aliases"], r["relation"])
    return code in ("C-alias", "C-short") or contains_gold(a, r["gold_value"], r["gold_aliases"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--signal", default="확신도 + SAE 20개"); ap.add_argument("--setting", default="all")
    a = ap.parse_args()
    import torch
    from dream_acts import load_dream
    from run_inference import build_prompt, extract_answer
    picks = json.load(open(f"{U}/picks_dream_{a.setting}.json"))[a.signal]
    rows = {(r["sample_id"], r["version"]): r for r in map(json.loads, open(f"{D}/rows.jsonl", encoding="utf-8"))}
    stored = {(d["sample_id"], d["version"]): d["answer"] for d in map(json.loads, open(f"{D}/topk_margin_s32.jsonl", encoding="utf-8")) if d["sample_id"]}
    todo = [rows[(f, v)] for f, v in picks.items()]
    model, tok = load_dream("cuda"); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    new = {}
    for b in range(0, len(todo), 64):
        batch = todo[b:b + 64]
        enc = tok([build_prompt(r["question"]) for r in batch], return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.inference_mode():
            out = model.diffusion_generate(inputs=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=32, steps=32,
                temperature=0.0, alg="topk_margin", alg_temp=0.0, mask_token_id=tok.mask_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for r, o in zip(batch, out):
            new[(r["sample_id"], r["version"])] = extract_answer(tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True))
    ks = list(new)
    y_new = np.array([grade(new[k], rows[k]) for k in ks]); y_old = np.array([grade(stored[k], rows[k]) for k in ks])
    same = np.mean([new[k].strip() == stored[k].strip() for k in ks])
    facts = sorted({f for f, _ in rows}); rng = np.random.default_rng(0)
    orig = np.mean([grade(stored[(f, "question")], rows[(f, "question")]) for f in facts])
    rand = np.mean([np.mean([grade(stored[(f, v)], rows[(f, v)]) for v in ("question", "v1", "v2", "v3", "v4")]) for f in facts])
    print(f"신호: {a.signal} / 설정 {a.setting} / 사실 {len(ks)}개")
    print(f"  다시 생성한 답이 저장된 답과 같은 비율: {same:.1%}")
    print(f"  저장된 답으로 본 정답률 {y_old.mean():.1%} | 다시 생성한 답의 정답률 {y_new.mean():.1%}")
    print(f"  비교: 원래 문구 {orig:.1%}, 무작위 문구(기대값) {rand:.1%}")
    sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
