"""PopQA prior/guess control: relation, subject popularity, and 'gold is the modal answer of its relation'.

1) how often is gold the relation's most common answer, by k (number of correct paraphrases)
2) within-fact probe AUROC (fact-centered logistic regression, as in probe_within_fact.py) on all
   facts vs facts whose gold is NOT the relation's modal answer vs high/low subject popularity.
"""
import collections
import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probe_within_fact import MODELS, center, load_meta, within_fact_auc

warnings.filterwarnings("ignore")
P = [json.loads(l) for l in open("data/popqa/4. popqa_paraphrased.jsonl", encoding="utf-8")]
info = {str(r["id"]): r for r in P}
modal = {}
for prop in {r["prop"] for r in P}:
    c = collections.Counter(r["obj"].lower() for r in P if r["prop"] == prop)
    modal[prop] = c.most_common(1)[0][0]
is_modal = {sid: info[sid]["obj"].lower() == modal[info[sid]["prop"]] for sid in info}
pop_med = float(np.median([r["s_pop"] for r in P]))

# ---- 1) prior share by k (Qwen graded, all 1000 sampled facts)
full = [json.loads(l) for l in open("analysis/graded/5. reviewed 5/qwen2.5-7b_popqa_reclassified.jsonl")]
byf = collections.defaultdict(list)
for r in full:
    byf[r["sample_id"]].append(r)
kf = {s: sum(x["verdict"] == "correct" for x in rs) for s, rs in byf.items()}
print("Qwen PopQA: share of facts whose gold is the relation's modal answer, by k (correct paraphrases of 5)")
for k in range(6):
    fs = [s for s, v in kf.items() if v == k]
    print(f"  k={k}: facts {len(fs):4d}  modal-gold share {np.mean([is_modal[s] for s in fs]):.2f}  median s_pop {np.median([info[s]['s_pop'] for s in fs]):.0f}")
rel = collections.Counter(info[s]["prop"] for s in kf)
print("  relations in the 1000-fact sample (top 6):", rel.most_common(6))
print("  modal answer of top relations:", {p: modal[p] for p, _ in rel.most_common(6)})

# ---- 2) probe with controls
print("\nwithin-fact AUROC (fact-centered LR; 20 splits)  columns: all | gold not modal | s_pop above median | s_pop below median")
for model, cfg in MODELS.items():
    meta = load_meta(cfg["meta"].format(ds="popqa"))
    for pos in ("q_last", "prompt_last"):
        H = np.load(cfg["hid"].format(ds="popqa", pos=pos)).astype(np.float32)
        for t in (1, 2):
            idx, y, sid = [], [], []
            for i, m in enumerate(meta):
                if m["group"] != "kept" or m["gold_issue"]:
                    continue
                if m["verdict"] == "correct":
                    idx.append(i); y.append(0); sid.append(m["sample_id"])
                elif m["verdict"] == "incorrect" and m["reason"] == t:
                    idx.append(i); y.append(1); sid.append(m["sample_id"])
            idx, y, sid = np.array(idx), np.array(y), np.array(sid)
            paired = {f for f in np.unique(sid) if (y[sid == f] == 0).any() and (y[sid == f] == 1).any()}
            sel = np.isin(sid, list(paired))
            idx, y, sid = idx[sel], y[sel], sid[sel]
            X = H[idx]
            subsets = {
                "all": lambda f: True,
                "non-modal": lambda f: not is_modal[f],
                "pop>med": lambda f: info[f]["s_pop"] > pop_med,
                "pop<=med": lambda f: info[f]["s_pop"] <= pop_med,
            }
            out = []
            for name, fn in subsets.items():
                keepf = np.array([fn(f) for f in sid])
                nfacts = len(set(sid[keepf]))
                aucs = []
                for seed in range(20):
                    rng = np.random.default_rng(seed)
                    facts = np.unique(sid[keepf])
                    tr_f = set(rng.permutation(facts)[: int(0.6 * len(facts))])
                    tr = np.array([s in tr_f for s in sid]) & keepf
                    te = ~np.array([s in tr_f for s in sid]) & keepf
                    if len(np.unique(y[tr])) < 2 or not te.any():
                        continue
                    Xc = center(X[tr], sid[tr]); sc = StandardScaler().fit(Xc)
                    clf = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(Xc), y[tr])
                    try:
                        aucs.append(within_fact_auc(clf.decision_function(sc.transform(X[te])), y[te], sid[te]))
                    except Exception:
                        pass
                out.append(f"{np.mean(aucs):.3f}[{nfacts:3d}]")
            print(f"  {model:5s} {pos:11s} type{t}: " + "  ".join(out))
