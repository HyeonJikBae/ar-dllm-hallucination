import collections, json, os, sys
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
FEAT = os.environ.get("FEAT", "sae_analysis/features_g"); pos = sys.argv[1]
def auroc(p, n):
    r = rankdata(np.concatenate([p, n])); return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
meta = [json.loads(l) for l in open(f"{FEAT}/qwen_popqa_meta.jsonl")]
B = np.load(f"{FEAT}/qwen_popqa_{pos}.npy") > 0
sid = np.array([m["sample_id"] for m in meta]); rel = np.array([m["relation"] for m in meta])
corr = np.array([m["verdict"] == "correct" or m["verdict_suggested"] == "correct" for m in meta]); ok = np.array([not m["gold_issue"] and m["verdict"] != "uncertain" for m in meta])
kf = collections.Counter()
for s, c, o in zip(sid, corr, ok):
    if o: kf[s] += int(c)
k = np.array([kf.get(s, -1) for s in sid])
typ = np.array([str(m["reason"]) if (m["verdict"] == "incorrect" and m["verdict_suggested"] != "correct" and not m.get("rule_unsure") and m["reason"] in (1, 2) and ok[i]) else "" for i, m in enumerate(meta)])
use = typ != ""; cols = np.where(B[use].mean(0) >= 0.03)[0]; X = B[:, cols].astype(np.float32)
rels = sorted(set(rel[use])); R = (rel[:, None] == np.array(rels)[None, :]).astype(np.float32)
m = use & (k >= 1) & (k <= 4); idx = np.where(m)[0]; facts = np.array(sorted(set(sid[m])))
def gain(lab, nsplit, seed0=0):
    g = []
    for seed in range(seed0, seed0 + nsplit):
        rng = np.random.default_rng(seed); trf = set(rng.permutation(facts)[: int(0.6 * len(facts))])
        tr = m & np.isin(sid, list(trf)); te = m & ~np.isin(sid, list(trf))
        y, yt = lab[tr] == "1", lab[te] == "1"
        a = []
        for F, C in ((R, 1.0), (np.hstack([R, X]), 0.3)):
            d = LogisticRegression(max_iter=300, C=C).fit(F[tr], y).decision_function(F[te]); a.append(auroc(d[yt], d[~yt]))
        g.append(a[1] - a[0])
    return np.array(g)
real = gain(typ, 20)
print(f"[{pos}] 1~4/5 fact, 유형1 vs 2: 관계 통제 후 SAE 증가 (AUROC) 평균 {real.mean():+.3f}, 분할 sd {real.std():.3f}, 20분할 중 양수 {int((real>0).sum())}")
rng = np.random.default_rng(0); nulls = []
for t in range(10):
    lab = typ.copy()
    for r in rels:   # 관계 안에서 유형 라벨을 섞는다
        ii = idx[rel[idx] == r]; lab[ii] = rng.permutation(typ[ii])
    nulls.append(gain(lab, 2, seed0=t * 2).mean())
nulls = np.array(nulls)
print(f"   라벨을 관계 안에서 섞은 대조군 10회: 증가 평균 {nulls.mean():+.3f}, 최대 {nulls.max():+.3f}, 최소 {nulls.min():+.3f}")
print(f"   실제값이 대조군 최대를 넘는가: {'예' if real.mean()>nulls.max() else '아니오'}")
