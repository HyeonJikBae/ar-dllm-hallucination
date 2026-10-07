"""Latents specific to each knowledge state, on the refined PopQA labels.

  K5  all rows of facts answered correctly in 5/5 paraphrases      ("knows")
  W1  type-1 wrong rows in facts with 1-4 correct                  ("knows but wrong, related entity")
  W2  type-2 wrong rows in facts with 1-4 correct                  ("knows but wrong, unrelated entity")
  U   all rows of 0/5 facts                                        ("doesn't know")

For a state S: spec_S[j] = min over the other states O of ( f_S[j] - f_O[j] ), f = fraction of rows with latent j > 0.
Frequencies are RELATION-ADJUSTED: computed inside each PopQA relation (director, capital, ...) and averaged with the
same relation weights for every state, so a latent that merely tracks the question type cannot score high.
Selection on train facts, reported on held-out test facts; "stable" = in the train top-10 in >=4 of 5 splits.
The confident subset drops ambiguous-correct and unsure/assumed-wrong rows.

Usage:
    python3 sae_analysis/state_signals_refined.py --position q_last
"""

import argparse
import collections
import json

import numpy as np

FEAT = __import__("os").environ.get("FEAT", "sae_analysis/features_refined")
STATES = ["K5", "W1", "W2", "U"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--min-rel-rows", type=int, default=10)
    ap.add_argument("--min-freq", type=float, default=0.03)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    meta = [json.loads(l) for l in open(f"{FEAT}/qwen_popqa_meta.jsonl", encoding="utf-8")]
    B = np.load(f"{FEAT}/qwen_popqa_{args.position}.npy") > 0
    ok = np.array([not m["gold_issue"] and m["verdict"] != "uncertain" for m in meta])
    corr = np.array([m["verdict"] == "correct" or m["verdict_suggested"] == "correct" for m in meta])
    rule = np.array([m.get("rule") or "" for m in meta])
    amb = corr & np.array([r.startswith("C-") and r != "C-alias" for r in rule])
    uns = np.array([bool(m.get("rule_unsure")) or m.get("rule") == "1-role-assumed" for m in meta])
    reason = np.array([m["reason"] if m["reason"] is not None else -1 for m in meta])
    sid = np.array([m["sample_id"] for m in meta])
    rel = np.array([m["relation"] for m in meta])

    kf = collections.Counter()
    for s, c, o in zip(sid, corr, ok):
        if o:
            kf[s] += int(c)
    k = np.array([kf.get(s, -1) for s in sid])
    use = ok & ~amb & ~uns
    grp = np.full(len(meta), "", dtype=object)
    for i in np.where(use)[0]:
        if k[i] == 5 and corr[i]:
            grp[i] = "K5"
        elif k[i] == 0:
            grp[i] = "U"
        elif 1 <= k[i] <= 4 and not corr[i] and reason[i] in (1, 2):
            grp[i] = f"W{reason[i]}"
    print(f"[{args.position}] facts per state:",
          {g: len(set(sid[grp == g])) for g in STATES}, "| rows:", {g: int((grp == g).sum()) for g in STATES})
    freq_all = B[np.isin(grp, STATES)].mean(0)
    cols = np.where(freq_all >= args.min_freq)[0]
    Bs = B[:, cols].astype(np.float32)
    print(f"latents analysed (active in >= {args.min_freq:.0%} of rows): {len(cols)}")

    facts = np.array(sorted(set(sid[np.isin(grp, STATES)])))

    def adj_freq(mask_rows):
        """relation-adjusted frequency per state; returns {state: vec}, relations used, weights."""
        rels = []
        for r in sorted(set(rel[mask_rows])):
            if all(((grp == g) & mask_rows & (rel == r)).sum() >= args.min_rel_rows for g in STATES):
                rels.append(r)
        w = np.array([np.isin(grp, STATES)[mask_rows & (rel == r)].sum() for r in rels], dtype=float)
        w /= w.sum()
        out = {g: sum(wi * Bs[(grp == g) & mask_rows & (rel == r)].mean(0) for wi, r in zip(w, rels)) for g in STATES}
        return out, rels

    def spec(F):
        return {S: np.min([F[S] - F[O] for O in STATES if O != S], axis=0) for S in STATES}

    picks = {S: collections.Counter() for S in STATES}
    last = None
    for seed in range(args.splits):
        rng = np.random.default_rng(seed)
        tr_f = set(rng.permutation(facts)[: int(0.6 * len(facts))])
        tr = np.isin(sid, list(tr_f))
        te = ~tr
        Ftr, rels = adj_freq(tr)
        Fte, _ = adj_freq(te)
        sp_tr, sp_te = spec(Ftr), spec(Fte)
        for S in STATES:
            for j in np.argsort(-sp_tr[S])[:10]:
                picks[S][j] += 1
        if seed == 0:
            last = (Ftr, Fte, sp_tr, sp_te, rels)
    Ftr, Fte, sp_tr, sp_te, rels = last
    print(f"relations used for adjustment ({len(rels)}): {rels}")
    names = {"K5": "아는 것 (5/5)", "W1": "아는데 틀림 - 유형1 (관련)", "W2": "아는데 틀림 - 유형2 (무관)", "U": "모르는 것 (0/5)"}
    for S in STATES:
        print(f"\n=== {names[S]}  [{S}]  특이 feature (train 상위, 검증은 test)")
        print("   feature  train  test  stable  |  test 켜짐률 K5 / W1 / W2 / U  | 가장 크게 켜진 질문")
        for j in np.argsort(-sp_tr[S])[: args.top]:
            stable = picks[S][j]
            f = [Fte[g][j] for g in STATES]
            o = np.argsort(-Bs[:, j] * (grp == S))[:1]
            q = meta[o[0]]["question"][:44] if (grp[o[0]] == S) else ""
            print(f"   {cols[j]:6d}  {sp_tr[S][j]:+.3f} {sp_te[S][j]:+.3f}  {stable}/{args.splits}   |  "
                  f"{f[0]:.2f} / {f[1]:.2f} / {f[2]:.2f} / {f[3]:.2f}  | {q}")
        npos = int(((sp_tr[S] > 0.03) & (sp_te[S] > 0.03)).sum())
        print(f"   train과 test 모두 특이도 > 0.03 인 feature 수: {npos}")


if __name__ == "__main__":
    main()
