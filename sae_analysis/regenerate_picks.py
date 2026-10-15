"""Regenerate the answers of the paraphrases chosen by each signal and check them against the gold answer.

Picks (from combine_signals.py cross-validated scores at the answer-prediction position, per fact):
  all   : highest-scored among the 5 versions     (conf, sae20, conf_sae20)
  para  : highest-scored among v1-v4 only          (conf, sae20, conf_sae20)
  orig  : the original question wording
Generation matches run_inference.py: Qwen2.5-7B (pinned revision), bf16, greedy, left padding, batch 8, 32 new tokens, same prompt.
A regenerated answer identical to the stored answer keeps its stored label; a different answer is graded by normalised exact
match with gold or a gold alias (conservative). Reproduction rate is reported.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 sae_analysis/regenerate_picks.py            # generate (resumable) + evaluate
    python3 sae_analysis/regenerate_picks.py --eval-only
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "TrivaQA_ex")
D = "sae_analysis/features_lens_v2/"
OUT = "sae_analysis/results_twelve/regenerated.jsonl"
MODEL, REVISION = "Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796"
SIGNALS = ["conf", "sae20", "conf_sae20"]


def build_picks():
    meta = [json.loads(l) for l in open(D + "meta.jsonl", encoding="utf-8")]
    Z = np.load("sae_analysis/results_twelve/combine_scores_prompt_last.npz")
    rows = Z["rows"]
    sid = np.array([meta[i]["sample_id"] for i in rows]); ver = np.array([meta[i]["version"] for i in rows])
    picks = {}
    for f in np.unique(sid):
        m = np.where(sid == f)[0]
        mp = m[ver[m] != "question"]
        mo = m[ver[m] == "question"]
        rec = {}
        for s in SIGNALS:
            rec[f"all:{s}"] = int(rows[m[np.argmax(Z[s][m])]])
            if len(mp):
                rec[f"para:{s}"] = int(rows[mp[np.argmax(Z[s][mp])]])
        if len(mo):
            rec["orig"] = int(rows[mo[0]])
        picks[f] = rec
    return meta, rows, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    meta, rows, picks = build_picks()
    final = {}
    for l in open("analysis/graded/7. final/qwen2.5-7b_popqa_final.jsonl", encoding="utf-8"):
        r = json.loads(l)
        final[(r["sample_id"], r["version"])] = r
    need = sorted({i for rec in picks.values() for i in rec.values()})
    done = {}
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            r = json.loads(l)
            done[r["row"]] = r
    todo = [i for i in need if i not in done]
    print(f"picked rows {len(need)}, already generated {len(done)}, to generate {len(todo)}", flush=True)
    if todo and not args.eval_only:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from run_inference import build_prompt, generate_batch

        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True).eval()
        todo.sort(key=lambda i: (meta[i]["sample_id"], meta[i]["version"]))
        with open(OUT, "a", encoding="utf-8") as fo:
            for b in range(0, len(todo), 8):
                chunk = todo[b : b + 8]
                answers = generate_batch(model, tok, [build_prompt(meta[i]["question"]) for i in chunk], max_new_tokens=32)
                for i, a in zip(chunk, answers):
                    fo.write(json.dumps({"row": i, "regen": a}, ensure_ascii=False) + "\n")
                fo.flush()
                if (b // 8) % 20 == 0:
                    print(f"{b + len(chunk)}/{len(todo)}", flush=True)
        done = {}
        for l in open(OUT, encoding="utf-8"):
            r = json.loads(l); done[r["row"]] = r
    evaluate(meta, rows, picks, final, done)


def evaluate(meta, rows, picks, final, done):
    from scoring import normalize_answer, strip_answer_wrapper

    def stored_correct(i):
        return meta[i]["final_label"].startswith("correct")

    def regen_correct(i):
        m = meta[i]
        r = final[(m["sample_id"], m["version"])]
        a = done[i]["regen"]
        if a.strip() == (m["model_answer"] or "").strip():
            return stored_correct(i), True
        gold = {normalize_answer(g) for g in [r["gold_value"], *(r.get("gold_aliases") or [])] if normalize_answer(g)}
        return normalize_answer(strip_answer_wrapper(a)) in gold, False

    same = [regen_correct(i)[1] for i in done]
    print(f"\n재생성한 {len(done)}개 중 저장된 답과 같은 답: {np.mean(same):.1%} (다른 답은 정규화 정확 일치로 채점)")
    sid = np.array([meta[i]["sample_id"] for i in rows]); ver = np.array([meta[i]["version"] for i in rows])
    y = np.array([stored_correct(i) for i in rows]).astype(int)
    print(f"\n{'전략':<26}{'사실':>6}{'재생성 정답':>12}{'저장 라벨 기준':>14}{'무작위':>9}{'차이(재생성)':>13}")
    out = {}
    for pool, label, facts_filter in (("all", "5개 변형 중 고르기", None), ("para", "변형 v1~v4 중 고르기", "para")):
        facts = []
        for f, rec in picks.items():
            m = np.where(sid == f)[0]
            mm = m[ver[m] != "question"] if pool == "para" else m
            if pool == "para" and not (len(mm) >= 2 and y[mm].min() == 0 and y[mm].max() == 1):
                continue
            facts.append((f, mm))
        base = np.mean([y[mm].mean() for f, mm in facts])
        for s in SIGNALS:
            key = f"{pool}:{s}"
            reg = np.mean([regen_correct(picks[f][key])[0] for f, _ in facts if key in picks[f]])
            sto = np.mean([stored_correct(picks[f][key]) for f, _ in facts if key in picks[f]])
            n = sum(key in picks[f] for f, _ in facts)
            print(f"{label + ' · ' + s:<26}{n:>6}{reg:>12.1%}{sto:>14.1%}{base:>9.1%}{reg - base:>+13.1%}")
            out[key] = dict(n=n, regen=float(reg), stored=float(sto), random=float(base))
    fo = [(f, rec["orig"]) for f, rec in picks.items() if "orig" in rec]
    base = np.mean([y[sid == f].mean() for f, _ in fo])
    reg = np.mean([regen_correct(i)[0] for _, i in fo]); sto = np.mean([stored_correct(i) for _, i in fo])
    print(f"{'원래 질문 그대로':<26}{len(fo):>6}{reg:>12.1%}{sto:>14.1%}{base:>9.1%}{reg - base:>+13.1%}")
    out["orig"] = dict(n=len(fo), regen=float(reg), stored=float(sto), random=float(base))
    json.dump(out, open("sae_analysis/results_twelve/regenerated_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
