"""How well does the step at which the answer's first slot gets filled (earlier = more likely correct) separate correct from wrong?  No fitting, so all rows are used.
Pooled AUROC over rows, and within-fact AUROC (facts 1/5..4/5, mean over facts with both kinds, 95% CI by resampling facts).  Also split by wrong type.
Usage: python3 analysis/5_onoff_vs_strength/commit_auroc.py"""
import json, numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
cor = np.array([r["correct"] for r in rows]); k = np.array([r["k_correct"] for r in rows]); ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); fid = np.array([r["sample_id"] for r in rows])
T = np.load(f"{F}/tokens.npy"); M = np.load(f"{F}/conf_mask.npy", mmap_mode="r")
tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", trust_remote_code=True); SP = set(tok.all_special_ids)
WS = {int(i) for i in np.unique(T) if i >= 0 and int(i) not in SP and tok.decode([int(i)]).strip() == ""}
first = np.full(len(rows), -1)
for b in range(0, len(rows), 4096):
    commit = np.cumprod(np.asarray(M[b:b + 4096]).astype(np.int8), axis=1).sum(1)
    for j in range(len(commit)):
        t = T[b + j]; n = next((i for i, x in enumerate(t) if int(x) in SP), 32); n0 = 0
        while n0 < n and int(t[n0]) in WS: n0 += 1
        if n0 < n: first[b + j] = commit[j, n0]
ok = first >= 0; score = -first.astype(float)           # earlier fill -> higher score -> "correct"
def pooled(m): return roc_auc_score(cor[m], score[m])
def within(m, rng=np.random.default_rng(0)):
    per = [roc_auc_score(cor[m & (fid == u)], score[m & (fid == u)]) for u in np.unique(fid[m]) if cor[m & (fid == u)].min() != cor[m & (fid == u)].max()]
    per = np.array(per); bs = [rng.choice(per, len(per)).mean() for _ in range(500)]; return per.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(per)
mid = (k >= 1) & (k <= 4)
print("pooled AUROC, all rows with a non-empty answer: %.3f" % pooled(ok))
print("pooled AUROC, facts 1/5..4/5 rows:              %.3f" % pooled(ok & mid))
a, lo, hi, n = within(ok & mid); print(f"within-fact AUROC, facts 1/5..4/5 ({n} facts):  {a:.3f} [{lo:.3f}, {hi:.3f}]")
for g in (4, 3, 2, 1):
    a, lo, hi, n = within(ok & (k == g)); print(f"   facts {g}/5 ({n} facts): {a:.3f} [{lo:.3f}, {hi:.3f}]")
for t in ("entity", "broken"):
    m = ok & mid & ((ty == "correct") | (ty == t)); a, lo, hi, n = within(m); print(f"within-fact, correct vs {t} ({n} facts): {a:.3f} [{lo:.3f}, {hi:.3f}]")
