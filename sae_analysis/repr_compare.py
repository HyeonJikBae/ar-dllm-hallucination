"""Is the failure in the summary score s, in the SAE, or in the layer itself?  (steps 1-2 of the plan)

Rows: logit_lens_extract.py selection (7. final analysis_use rows, guess-risk facts removed, U = sampled 0/5 facts).
Splits are by SUBJECT (PopQA s_uri; facts sharing a subject never straddle train/test), 10 splits 60/40.

Representations, all at layer 20 unless stated:
  s            know-axis score: difference-in-means K5 - U of binary SAE activity, relation-stratified, re-estimated
               on this guess-risk-free row set (step 2)
  SAE full     logistic regression on all SAE latents (activation magnitudes)
  raw          logistic regression on the raw residual stream; also prompt_last at layers 4..28
Tasks:
  K5 vs U                   (fact-level knowledge)
  correct vs wrong, overall (rows of 1-4/5 facts; probe trained on raw rows)
  correct vs wrong, within fact (same rows; probe trained on fact-centred features; AUROC averaged over facts)
Null for the within-fact probe: labels shuffled inside facts.

Usage:
    PYTHONPATH=sae_analysis python3 sae_analysis/repr_compare.py
"""

import collections
import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from twelve_cells import auroc, strat_diff
from know_axis_by_k import fact_matrix

warnings.filterwarnings("ignore")
D = "sae_analysis/features_lens"
F = "sae_analysis/features_refined"
HID_LAYERS = [4, 8, 12, 16, 20, 24, 28]
SPLITS = 10


def within_auc(score, y, sid):
    out = []
    for f in np.unique(sid):
        m = sid == f
        a, b = score[m & (y == 1)], score[m & (y == 0)]
        if len(a) and len(b):
            d = a[:, None] - b[None, :]
            out.append(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)
    return float(np.mean(out)) if out else np.nan


def center(X, sid):
    Xc = X.copy()
    for f in np.unique(sid):
        m = sid == f
        Xc[m] -= X[m].mean(0)
    return Xc


def lr_scores(Xtr, ytr, Xte, C=0.01):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=300).fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def main():
    meta = [json.loads(l) for l in open(f"{D}/meta.jsonl", encoding="utf-8")]
    subj = {}
    for l in open("data/popqa/1. popqa_raw.jsonl"):
        x = json.loads(l)
        subj[str(x["id"])] = x["s_uri"].rsplit("/", 1)[-1]
    sid = np.array([m["sample_id"] for m in meta])
    sub = np.array([subj[s] for s in sid])
    rel = np.array([m["relation"] for m in meta])
    grp = np.array([m["fact_group"] for m in meta])
    corr = np.char.startswith(np.array([m["final_label"] for m in meta]).astype(str), "correct")
    k5, U = grp == "5/5", grp == "0/5"
    part = ~k5 & ~U
    paired_facts = {f for f in np.unique(sid[part]) if corr[part & (sid == f)].any() and (~corr[part & (sid == f)]).any()}
    pp = part & np.isin(sid, list(paired_facts))

    # SAE features for these rows (features_refined is aligned with build_meta rows)
    fm = [json.loads(l) for l in open(f"{F}/qwen_popqa_meta.jsonl", encoding="utf-8")]
    pos_of = {(m["sample_id"], m["version"]): i for i, m in enumerate(fm)}
    idx = np.array([pos_of[(m["sample_id"], m["version"])] for m in meta])
    order = np.argsort(idx)
    reps = {}
    for p in ("q_last", "prompt_last"):
        S = np.load(f"{F}/qwen_popqa_{p}.npy", mmap_mode="r")
        X = np.empty((len(idx), S.shape[1]), np.float16)
        X[order] = S[idx[order]]
        keep = (X > 0).mean(0) >= 0.005
        reps[f"SAE full {p} L20"] = X[:, keep].astype(np.float32)
    H = np.load(f"{D}/hidden_prompt_last.npy", mmap_mode="r")
    reps["raw q_last L20"] = np.load(f"{D}/hidden_q_last_L20.npy").astype(np.float32)
    for i, L in enumerate(HID_LAYERS):
        reps[f"raw prompt_last L{L}"] = np.asarray(H[:, i]).astype(np.float32)

    subjects = np.unique(sub)
    res = collections.defaultdict(list)
    for seed in range(SPLITS):
        rng = np.random.default_rng(seed)
        tr = np.isin(sub, rng.permutation(subjects)[: int(0.6 * len(subjects))])
        te = ~tr
        # ---- s: know axis on binary SAE (q_last and prompt_last), re-estimated here
        for p in ("q_last", "prompt_last"):
            B = (reps[f"SAE full {p} L20"] > 0).astype(np.float32)
            _, XK, rK = fact_matrix(B[tr & k5], sid[tr & k5], rel[tr & k5])
            _, XU, rU = fact_matrix(B[tr & U], sid[tr & U], rel[tr & U])
            d, _ = strat_diff(XK, rK, XU, rU, 3)
            s = B @ d.astype(np.float32)
            res[(f"s {p}", "K5 vs U")].append(auroc(s[te & k5], s[te & U]))
            res[(f"s {p}", "정답 vs 오답 (전체)")].append(auroc(s[te & pp & corr], s[te & pp & ~corr]))
            res[(f"s {p}", "정답 vs 오답 (같은 fact)")].append(within_auc(s[te & pp], corr[te & pp].astype(int), sid[te & pp]))
        # ---- probes
        y = corr.astype(int)
        for name, X in reps.items():
            m_ku = k5 | U
            sc = lr_scores(X[tr & m_ku], k5[tr & m_ku].astype(int), X[te & m_ku])
            res[(name, "K5 vs U")].append(auroc(sc[k5[te & m_ku]], sc[~k5[te & m_ku]]))
            sc = lr_scores(X[tr & pp], y[tr & pp], X[te & pp])
            res[(name, "정답 vs 오답 (전체)")].append(auroc(sc[y[te & pp] == 1], sc[y[te & pp] == 0]))
            Xc = center(X[tr & pp], sid[tr & pp])
            sc = lr_scores(Xc, y[tr & pp], X[te & pp])
            res[(name, "정답 vs 오답 (같은 fact)")].append(within_auc(sc, y[te & pp], sid[te & pp]))
            if name in ("raw prompt_last L20", "raw q_last L20", "raw prompt_last L28"):
                yn = y.copy()
                for f in np.unique(sid[pp]):
                    mm = np.where(pp & (sid == f))[0]
                    yn[mm] = rng.permutation(y[mm])
                sc = lr_scores(Xc, yn[tr & pp], X[te & pp])
                res[(name, "null (같은 fact, 라벨 섞음)")].append(within_auc(sc, yn[te & pp], sid[te & pp]))
        print(f"split {seed + 1}/{SPLITS}", flush=True)

    tasks = ["K5 vs U", "정답 vs 오답 (전체)", "정답 vs 오답 (같은 fact)", "null (같은 fact, 라벨 섞음)"]
    print(f"\nrows {len(meta):,} | K5 {k5.sum():,} U {U.sum():,} | 짝지은 fact {len(paired_facts):,} (행 {pp.sum():,}: 정답 {(pp & corr).sum():,}, 오답 {(pp & ~corr).sum():,})")
    print(f"분할: subject 단위 {len(subjects):,}개, {SPLITS}회. AUROC 평균±sd (test)\n")
    print(f"{'표현':<24}" + "".join(f"{t:>26}" for t in tasks))
    names = ["s q_last", "s prompt_last"] + list(reps)
    out = {}
    for n in names:
        cells = []
        for t in tasks:
            v = res.get((n, t))
            cells.append(f"{np.mean(v):.3f}±{np.std(v):.3f}" if v else "-")
            if v:
                out[f"{n}|{t}"] = [float(np.mean(v)), float(np.std(v))]
        print(f"{n:<24}" + "".join(f"{c:>26}" for c in cells))
    json.dump(out, open("sae_analysis/results_twelve/repr_compare.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
