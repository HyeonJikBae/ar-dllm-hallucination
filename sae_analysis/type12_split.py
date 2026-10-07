"""Type 1 vs type 2 decodability (relation-controlled), split by how much the model knows the fact (k = correct of 5)."""
import collections, json, os, sys
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
FEAT = os.environ.get("FEAT", "sae_analysis/features_g"); pos = sys.argv[1] if len(sys.argv) > 1 else "q_last"
def auroc(p, n):
    r = rankdata(np.concatenate([p, n])); return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
meta = [json.loads(l) for l in open(f"{FEAT}/qwen_popqa_meta.jsonl")]
B = np.load(f"{FEAT}/qwen_popqa_{pos}.npy") > 0
sid = np.array([m["sample_id"] for m in meta]); rel = np.array([m["relation"] for m in meta])
corr = np.array([m["verdict"] == "correct" or m["verdict_suggested"] == "correct" for m in meta])
ok = np.array([not m["gold_issue"] and m["verdict"] != "uncertain" for m in meta])
kf = collections.Counter()
for s, c, o in zip(sid, corr, ok):
    if o: kf[s] += int(c)
k = np.array([kf.get(s, -1) for s in sid])
typ = np.array([str(m["reason"]) if (m["verdict"] == "incorrect" and m["verdict_suggested"] != "correct" and not m.get("rule_unsure") and m["reason"] in (1, 2) and ok[i]) else "" for i, m in enumerate(meta)])
use = typ != ""
cols = np.where(B[use].mean(0) >= 0.03)[0]; X = B[:, cols].astype(np.float32)
rels = sorted(set(rel[use])); R = (rel[:, None] == np.array(rels)[None, :]).astype(np.float32)
def run(mask, label):
    m = use & mask
    n1, n2 = int((typ[m] == "1").sum()), int((typ[m] == "2").sum()); facts = np.array(sorted(set(sid[m])))
    out = {"rel": [], "full": []}
    for seed in range(5):
        rng = np.random.default_rng(seed); trf = set(rng.permutation(facts)[: int(0.6 * len(facts))])
        tr = m & np.isin(sid, list(trf)); te = m & ~np.isin(sid, list(trf))
        y, yt = typ[tr] == "1", typ[te] == "1"
        if y.sum() < 10 or (~y).sum() < 10 or yt.sum() < 5 or (~yt).sum() < 5: continue
        for nm, F in (("rel", R), ("full", np.hstack([R, X]))):
            lr = LogisticRegression(max_iter=300, C=0.3 if nm == "full" else 1.0).fit(F[tr], y); d = lr.decision_function(F[te])
            out[nm].append(auroc(d[yt], d[~yt]))
    if out["rel"]: print(f"  {label:28s} 유형1 {n1:5d} / 유형2 {n2:5d} 행 | 관계만 {np.mean(out['rel']):.3f} → +SAE {np.mean(out['full']):.3f} ({np.mean(out['full'])-np.mean(out['rel']):+.3f})")
    else: print(f"  {label:28s} 표본 부족 ({n1}/{n2})")
print(f"[{pos}] 유형 1 vs 유형 2 (관계 통제)")
run(np.ones(len(meta), bool), "모든 오답 행")
run(k == 0, "0/5 fact (모른다)")
run((k >= 1) & (k <= 4), "1~4/5 fact (아는데 틀림)")
