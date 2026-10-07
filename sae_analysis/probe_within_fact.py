"""Within-fact probe: can raw hidden states tell a correctly answered paraphrase from a wrongly
answered one of the SAME fact?

Rows: kept facts (>=1 of 5 correct), gold_issue excluded. K = correct rows, W = wrong rows of type t.
Only facts with both K and W rows ("paired facts") are used. Facts are split 60/40 into train/test
(20 random splits). Methods, all scored on test facts as the mean over facts of
P(score_W > score_K) over the fact's (W, K) pairs (0.5 = chance):

  lr_centered   logistic regression on hidden states with each train fact's mean subtracted
                (removes fact identity from what the probe can learn)
  lr_raw        logistic regression on raw hidden states of the train facts
  dim_centered  difference-in-means direction on fact-centered hidden states
  sae_dim       the same difference-in-means on binary SAE activity (the earlier SAE approach)
  null          lr_centered with labels shuffled inside each fact (should be ~0.5)

Usage:
    python3 sae_analysis/probe_within_fact.py
"""

import argparse
import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MODELS = {
    "qwen": dict(hid="sae_analysis/features/qwen_{ds}_all_{pos}_hidden.npy",
                 meta="sae_analysis/features/qwen_{ds}_all_meta.jsonl",
                 sae="sae_analysis/features/qwen_{ds}_{pos}.npy",
                 sae_meta="sae_analysis/features/qwen_{ds}_meta.jsonl"),
    "dream": dict(hid="sae_analysis/features_dream/dream_{ds}_plain_{pos}_hidden.npy",
                  meta="sae_analysis/features_dream/dream_{ds}_meta.jsonl",
                  sae="sae_analysis/features_dream_sae/dream_plain_{ds}_{pos}.npy",
                  sae_meta="sae_analysis/features_dream_sae/dream_plain_{ds}_meta.jsonl"),
}


def load_meta(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def within_fact_auc(scores, y, sid):
    aucs = []
    for f in np.unique(sid):
        m = sid == f
        w, k = scores[m & (y == 1)], scores[m & (y == 0)]
        if len(w) and len(k):
            d = w[:, None] - k[None, :]
            aucs.append(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)
    return float(np.mean(aucs))


def center(X, sid):
    Xc = X.copy()
    for f in np.unique(sid):
        m = sid == f
        Xc[m] -= X[m].mean(0)
    return Xc


def run(X, B, y, sid, seeds, C):
    res = {k: [] for k in ("lr_centered", "lr_raw", "dim_centered", "sae_dim", "null")}
    facts = np.unique(sid)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        tr_f = set(rng.permutation(facts)[: int(0.6 * len(facts))])
        tr = np.array([s in tr_f for s in sid])
        te = ~tr
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr_c = center(X[tr], sid[tr])
        sc = StandardScaler().fit(Xtr_c)
        clf = LogisticRegression(C=C, max_iter=500).fit(sc.transform(Xtr_c), y[tr])
        res["lr_centered"].append(within_fact_auc(clf.decision_function(sc.transform(X[te])), y[te], sid[te]))
        sc2 = StandardScaler().fit(X[tr])
        clf2 = LogisticRegression(C=C, max_iter=500).fit(sc2.transform(X[tr]), y[tr])
        res["lr_raw"].append(within_fact_auc(clf2.decision_function(sc2.transform(X[te])), y[te], sid[te]))
        d = Xtr_c[y[tr] == 1].mean(0) - Xtr_c[y[tr] == 0].mean(0)
        res["dim_centered"].append(within_fact_auc(X[te] @ d, y[te], sid[te]))
        Btr_c = center(B[tr], sid[tr])
        ds_ = Btr_c[y[tr] == 1].mean(0) - Btr_c[y[tr] == 0].mean(0)
        res["sae_dim"].append(within_fact_auc(B[te] @ ds_, y[te], sid[te]))
        # null: shuffle labels inside each fact (train and test)
        y_null = y.copy()
        for f in facts:
            m = np.where(sid == f)[0]
            y_null[m] = rng.permutation(y[m])
        if len(np.unique(y_null[tr])) == 2:
            clf3 = LogisticRegression(C=C, max_iter=500).fit(sc.transform(Xtr_c), y_null[tr])
            res["null"].append(within_fact_auc(clf3.decision_function(sc.transform(X[te])), y_null[te], sid[te]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--out", default="sae_analysis/probe_within_fact.json")
    args = ap.parse_args()

    out = []
    print(f"{'model':5s} {'data':8s} {'pos':11s} {'type':4s} {'facts':>5s} {'rows':>9s} | "
          + " ".join(f"{k:>13s}" for k in ("lr_centered", "lr_raw", "dim_centered", "sae_dim", "null")))
    for model, cfg in MODELS.items():
        for ds in ("popqa", "triviaqa"):
            meta = load_meta(cfg["meta"].format(ds=ds))
            smeta = load_meta(cfg["sae_meta"].format(ds=ds))
            key = {(m["sample_id"], m["version"]): i for i, m in enumerate(smeta)}
            for pos in ("q_last", "prompt_last"):
                H = np.load(cfg["hid"].format(ds=ds, pos=pos))
                S = np.load(cfg["sae"].format(ds=ds, pos=pos), mmap_mode="r")
                for t in (1, 2):
                    idx, y, sid = [], [], []
                    for i, m in enumerate(meta):
                        if m["group"] != "kept" or m["gold_issue"]:
                            continue
                        if m["verdict"] == "correct":
                            lab = 0
                        elif m["verdict"] == "incorrect" and m["reason"] == t:
                            lab = 1
                        else:
                            continue
                        idx.append(i); y.append(lab); sid.append(m["sample_id"])
                    idx, y, sid = np.array(idx), np.array(y), np.array(sid)
                    paired = [f for f in np.unique(sid) if (y[sid == f] == 0).any() and (y[sid == f] == 1).any()]
                    m_ = np.isin(sid, paired)
                    idx, y, sid = idx[m_], y[m_], sid[m_]
                    X = H[idx].astype(np.float32)
                    rows = [key[(meta[i]["sample_id"], meta[i]["version"])] for i in idx]
                    B = (np.asarray(S[sorted(rows)]) > 0).astype(np.float32)[np.argsort(np.argsort(rows))]
                    r = run(X, B, y, sid, range(args.splits), args.C)
                    row = {"model": model, "dataset": ds, "pos": pos, "type": t, "facts": len(paired),
                           "rows_K_W": [int((y == 0).sum()), int((y == 1).sum())],
                           **{k: [float(np.mean(v)), float(np.std(v))] for k, v in r.items()}}
                    out.append(row)
                    print(f"{model:5s} {ds:8s} {pos:11s} {t:<4d} {len(paired):5d} {row['rows_K_W'][0]:4d}/{row['rows_K_W'][1]:<4d} | "
                          + " ".join(f"{row[k][0]:.3f}±{row[k][1]:.3f}" for k in ("lr_centered", "lr_raw", "dim_centered", "sae_dim", "null")), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
