"""Is Dream's excess of collapse / copy errors a sampler artifact?  Re-decode the same prompts with different Dream samplers.

Input : analysis/graded/sampled_1000_seed42/dream-7b_popqa_graded.jsonl (1,000 PopQA facts x 5 questions, answers produced
        with run_inference.py: alg="origin", 64 steps, 32 new tokens, temperature 0, left padding)
Config: name -> (alg, steps); all with the same prompt, 32-token canvas, temperature 0, alg_temp 0 (origin: fixed seed).
Output: sae_analysis/results_twelve/dream_sampler/<config>.jsonl (resumable), then the evaluation table.
Grading is string-only (grounded_dream/string_grader.py) and is applied identically to the stored answers and every config.

Usage:
    CUDA_VISIBLE_DEVICES=0 /home/minyeol/.conda/envs/sae-dream/bin/python sae_analysis/dream_sampler_compare.py            # generate + eval
    /home/minyeol/.conda/envs/sae-dream/bin/python sae_analysis/dream_sampler_compare.py --eval-only
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "grounded_dream")); sys.path.insert(0, os.path.join(ROOT, "sae"))
from string_grader import contains_gold, string_rule  # noqa: E402

INPUT = "analysis/graded/sampled_1000_seed42/dream-7b_popqa_graded.jsonl"
REASON = "analysis/graded/5. reviewed 5/dream-7b_popqa_reclassified.jsonl"
OUT = "sae_analysis/results_twelve/dream_sampler"
CONFIGS = {  # name: (alg, steps, seed)   # run order = priority
    "entropy_s64": ("entropy", 64, 0),
    "origin_s64": ("origin", 64, 0),
    "entropy_s32": ("entropy", 32, 0),
    "origin_s64_seed1": ("origin", 64, 1),
    "topk_margin_s32": ("topk_margin", 32, 0),
    "maskgit_plus_s32": ("maskgit_plus", 32, 0),
}
CANVAS, BATCH = 32, 64


def load_rows():
    rows = [json.loads(l) for l in open(INPUT, encoding="utf-8")]
    raw = {}
    for l in open("data/popqa/1. popqa_raw.jsonl", encoding="utf-8"):
        x = json.loads(l); raw[str(x["id"])] = x["prop"]
    rs = {(json.loads(l)["sample_id"], json.loads(l)["version"]): json.loads(l).get("reason") for l in open(REASON, encoding="utf-8")}
    for r in rows:
        r["relation"] = raw[r["sample_id"]]
        r["stored_reason"] = rs.get((r["sample_id"], r["version"]))
    return rows


def generate(configs):
    import torch
    from dream_acts import load_dream
    from run_inference import build_prompt, extract_answer

    rows = load_rows()
    model, tok = load_dream("cuda")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    os.makedirs(OUT, exist_ok=True)
    for name in configs:
        alg, steps, seed = CONFIGS[name]
        path = f"{OUT}/{name}.jsonl"
        done = set()
        if os.path.exists(path):
            for l in open(path, encoding="utf-8"):
                if l.strip():
                    d = json.loads(l); done.add((d["sample_id"], d["version"]))
        todo = [r for r in rows if (r["sample_id"], r["version"]) not in done]
        print(f"[{name}] alg={alg} steps={steps} seed={seed}: {len(todo)} to generate", flush=True)
        torch.manual_seed(seed)
        with open(path, "a", encoding="utf-8") as fo:
            for b in range(0, len(todo), BATCH):
                batch = todo[b : b + BATCH]
                enc = tok([build_prompt(r["question"]) for r in batch], return_tensors="pt", padding=True, truncation=True).to("cuda")
                with torch.inference_mode():
                    out = model.diffusion_generate(
                        inputs=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=CANVAS, steps=steps,
                        temperature=0.0, alg=alg, alg_temp=0.0, mask_token_id=tok.mask_token_id,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
                for r, o in zip(batch, out):
                    gen = o[enc["input_ids"].shape[1]:]
                    raw = tok.decode(gen, skip_special_tokens=True)
                    fo.write(json.dumps({"sample_id": r["sample_id"], "version": r["version"], "answer": extract_answer(raw), "raw": raw,
                                         "has_eos": bool((gen == tok.eos_token_id).any())}, ensure_ascii=False) + "\n")
                fo.flush()
                if (b // BATCH) % 20 == 0:
                    print(f"  {b + len(batch)}/{len(todo)}", flush=True)


def evaluate():
    rows = [r for r in load_rows() if not r.get("gold_issue")]
    key = lambda r: (r["sample_id"], r["version"])
    sets = {"저장된 원본(origin, 64)": {key(r): {"answer": r["model_answer"], "has_eos": None} for r in rows}}
    for name in CONFIGS:
        p = f"{OUT}/{name}.jsonl"
        if os.path.exists(p):
            sets[name] = {(d["sample_id"], d["version"]): d for d in map(json.loads, open(p, encoding="utf-8")) if d["sample_id"]}
    rowmap = {key(r): r for r in rows}
    labels = {}
    for name, outs in sets.items():
        lab = {}
        for k, r in rowmap.items():
            if k not in outs:
                continue
            a = outs[k]["answer"]
            code = string_rule(a, r["question"], r["gold_value"], r["gold_aliases"], r["relation"])
            lab[k] = {"code": code, "strict": code in ("C-alias", "C-short"), "contain": code in ("C-alias", "C-short") or contains_gold(a, r["gold_value"], r["gold_aliases"]),
                      "words": len((a or "").split()), "has_eos": outs[k].get("has_eos")}
        labels[name] = lab
    n = len(rowmap)
    print(f"행 {n}개 (gold_issue 제외), 사실 {len({k[0] for k in rowmap})}개. 채점은 모든 설정에 같은 문자열 규칙")
    print(f"\n{'설정':<26}{'행':>6}{'정답(엄격)':>11}{'정답(포함)':>11}{'붕괴':>7}{'복사':>7}{'기권':>7}{'EOS없음':>9}{'평균단어':>9}")
    stored = labels["저장된 원본(origin, 64)"]

    def table_row(name, lab):
        m = len(lab); c = collections.Counter(v["code"] for v in lab.values())
        coll = c["6-empty"] + c["6-loop"] + c["6-leak"]
        noeos = np.mean([not v["has_eos"] for v in lab.values()]) if name in CONFIGS else float("nan")
        print(f"{name:<26}{m:>6}{np.mean([v['strict'] for v in lab.values()]):>11.1%}{np.mean([v['contain'] for v in lab.values()]):>11.1%}"
              f"{coll / m:>7.1%}{c['3-copy'] / m:>7.1%}{c['N-abstain'] / m:>7.1%}{noeos:>9.1%}{np.mean([v['words'] for v in lab.values()]):>9.1f}")

    for name, lab in labels.items():
        if name in CONFIGS:  # 부분 결과에서도 같은 행끼리 비교하도록, 이 설정이 생성한 행만으로 저장된 원본을 다시 센다
            table_row("  (저장된 원본, 같은 행)", {k: v for k, v in stored.items() if k in lab})
        table_row(name, lab)
    base = labels["저장된 원본(origin, 64)"]
    print("\n저장된 원본에서 붕괴/복사로 분류된 행이 다른 설정에서 어떻게 되는가 (같은 행)")
    groups = {"붕괴(빈/반복/누출)": lambda v: v["code"] in ("6-empty", "6-loop", "6-leak"), "복사": lambda v: v["code"] == "3-copy"}
    print(f"{'설정':<26}" + "".join(f"{g + ' 행':>22}{'→ 정답(포함)':>14}{'→ 붕괴/복사 유지':>18}" for g in groups))
    for name, lab in labels.items():
        if name.startswith("저장"):
            continue
        cells = []
        for g, f in groups.items():
            ks = [k for k, v in base.items() if f(v) and k in lab]
            cells.append(f"{len(ks):>22}{np.mean([lab[k]['contain'] for k in ks]):>14.1%}{np.mean([lab[k]['code'] in ('6-empty', '6-loop', '6-leak', '3-copy') for k in ks]):>18.1%}")
        print(f"{name:<26}" + "".join(cells))
    print("\n행 단위 전환 (엄격 채점; 저장된 원본 대비)")
    for name, lab in labels.items():
        if name.startswith("저장"):
            continue
        ks = [k for k in base if k in lab]
        up = sum((not base[k]["strict"]) and lab[k]["strict"] for k in ks); down = sum(base[k]["strict"] and not lab[k]["strict"] for k in ks)
        print(f"  {name:<24}오답→정답 {up:>4}   정답→오답 {down:>4}   순증 {up - down:>+5}")
    print("\n같은 행에서 두 실행의 정규화 답이 일치하는 비율 (재현성; origin은 무작위 순서로 확정)")
    pairs = [("저장된 원본(origin, 64)", "origin_s64"), ("origin_s64", "origin_s64_seed1"), ("entropy_s64", "entropy_s32"), ("origin_s64", "entropy_s64")]
    for a_, b_ in pairs:
        if a_ in sets and b_ in sets:
            from string_grader import norm
            ks = [k for k in rowmap if k in sets[a_] and k in sets[b_]]
            if ks:
                same = np.mean([norm(sets[a_][k]["answer"]) == norm(sets[b_][k]["answer"]) for k in ks])
                agree = np.mean([labels[a_][k]["strict"] == labels[b_][k]["strict"] for k in ks])
                print(f"  {a_:<24}vs {b_:<18} 답 일치 {same:.1%} | 엄격 정답 여부 일치 {agree:.1%} (행 {len(ks)})")
    print("\n사실 단위 맞힌 횟수 k (엄격 채점, 5개 질문 모두 있는 사실)")
    print(f"{'설정':<26}{'0/5':>7}{'1~4/5':>8}{'5/5':>7}")
    facts = sorted({k[0] for k in rowmap})
    for name, lab in labels.items():
        ks = collections.Counter()
        for f in facts:
            rs = [lab.get((f, v)) for v in ("question", "v1", "v2", "v3", "v4")]
            if any(x is None for x in rs):
                continue
            kk = sum(x["strict"] for x in rs)
            ks["0" if kk == 0 else "5" if kk == 5 else "mid"] += 1
        t = sum(ks.values())
        print(f"{name:<26}{ks['0'] / t:>7.1%}{ks['mid'] / t:>8.1%}{ks['5'] / t:>7.1%}")
    json.dump({n: {"strict": float(np.mean([v["strict"] for v in lab.values()])), "contain": float(np.mean([v["contain"] for v in lab.values()])),
                   "codes": dict(collections.Counter(str(v["code"]) for v in lab.values()))} for n, lab in labels.items()},
              open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    if not a.eval_only:
        generate(a.configs)
    evaluate()
    sys.stdout.flush(); os._exit(0)
