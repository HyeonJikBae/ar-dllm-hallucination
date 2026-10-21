"""Three states: KNOW (correct row) | KNOW-BUT-WRONG (wrong row of a 1-4/5 fact) | DON'T KNOW (wrong row of a 0/5 fact).
Is there a signal for KNOW-BUT-WRONG that separates it from KNOW *and* from DON'T KNOW?   Prompt end, layer 20, step 0.
Wrong rows are compared with wrong rows of the SAME type (broken with broken, entity with entity); all comparisons inside each relation.
Probes (logistic, raw hidden state, fit on dev subjects, scored once on test subjects):
  one-vs-rest   KNOW-BUT-WRONG (1) vs KNOW and DON'T KNOW (0)
  knowledge axis  direction fit on 0/5 rows vs 5/5 rows only (a single "how well is it known" axis)
  relation / question text baselines
Reported: AUROC of KNOW-BUT-WRONG vs KNOW and vs DON'T KNOW. A signal that is specific to KNOW-BUT-WRONG has both numbers clearly above 0.5;
a knowledge axis can only put KNOW-BUT-WRONG on one side of one of them.
Usage (repo root): python3 hyeonjik/analysis/2_know_but_wrong/three_states.py dream|qwen broken|entity
"""
import json, sys, warnings, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); MODEL, TYPE = sys.argv[1], sys.argv[2]; D = "hyeonjik/data"
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
if MODEL == "dream":
    rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
    F = f"{D}/1_raw/dream_hidden_steps"; index = list(range(len(rows))); ok = lambda r: r["analysis_use"] and not r["junk_suffix"]
else:
    rows = [json.loads(l) for l in open(f"{D}/3_classification/full/qwen_final_full.jsonl", encoding="utf-8")]
    F = f"{D}/1_raw/qwen_hidden"; index = json.load(open(f"{F}/rows.json")); ok = lambda r: r["analysis_use"]
def state(r):
    if r["correct"]: return "know" if r["fact_group"] in ("1/5", "2/5", "3/5", "4/5") else ("known5" if r["fact_group"] == "5/5" else None)
    if r["type3"] != TYPE: return None
    return "kbw" if r["fact_group"] in ("1/5", "2/5", "3/5", "4/5") else ("dont" if r["fact_group"] == "0/5" else None)
keep = np.array([i for i in index if ok(rows[i]) and state(rows[i])]); st = np.array([state(rows[i]) for i in keep])
rel = np.array([rows[i]["relation"] for i in keep]); sub = np.array([popqa[rows[i]["sample_id"]] for i in keep]); qtext = [rows[i]["question"] for i in keep]
if MODEL == "dream": X = np.asarray(np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r")[keep, 0], dtype=np.float32)
else: X = np.load(f"{F}/hid.npy")[np.array([{i: j for j, i in enumerate(index)}[i] for i in keep])].astype(np.float32)
rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
print(MODEL, TYPE, {k: int((st == k).sum()) for k in np.unique(st)}, "| test", {k: int((te & (st == k)).sum()) for k in np.unique(st)})
mu, sd = X[dev].mean(0), X[dev].std(0) + 1e-6; Xs = (X - mu) / sd
kbw = (st == "kbw").astype(int); use = np.isin(st, ["know", "kbw", "dont"])
# 1. one-vs-rest probe for KNOW-BUT-WRONG
t = dev & use
s_ovr = Xs @ LogisticRegression(C=0.0003, max_iter=500, class_weight="balanced").fit(Xs[t], kbw[t]).coef_[0]
# 2. knowledge axis: 0/5 rows (any outcome in this type or correct 5/5) vs 5/5 rows; never sees KNOW-BUT-WRONG
ax_t = dev & np.isin(st, ["known5", "dont"])
has_axis = (st[ax_t] == "known5").any()   # Qwen: no 5/5 rows extracted yet
s_axis = Xs @ LogisticRegression(C=0.0003, max_iter=500, class_weight="balanced").fit(Xs[ax_t], (st[ax_t] == "known5").astype(int)).coef_[0] if has_axis else None
# 3. baselines
R1 = (rel[:, None] == np.unique(rel)[None, :]).astype(np.float32)
s_rel = R1 @ LogisticRegression(max_iter=300, class_weight="balanced").fit(R1[t], kbw[t]).coef_[0]
tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000).fit([qtext[i] for i in np.flatnonzero(dev)]); T = tv.transform(qtext).toarray().astype(np.float32)
s_txt = T @ LogisticRegression(max_iter=300, class_weight="balanced").fit(T[t], kbw[t]).coef_[0]
def pair(s, a, b):
    """AUROC of KNOW-BUT-WRONG vs state b inside each relation (weighted by rows)"""
    out, w = [], []
    for r in np.unique(rel):
        m = te & (rel == r); pa, pb = m & (st == a), m & (st == b)
        if pa.sum() >= 8 and pb.sum() >= 8:
            x, y = s[pa], s[pb]; out.append(float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())); w.append(pa.sum() + pb.sum())
    return (np.average(out, weights=w) if out else float("nan")), len(out)
res = {}
print(f"\nKNOW-BUT-WRONG ({TYPE}) vs the two other states, AUROC inside each relation (test subjects)")
print(f"{'signal':26s} {'vs KNOW':>10s} {'vs DONT KNOW':>14s}")
for n, s in (("one-vs-rest probe", s_ovr), ("knowledge axis (0/5 vs 5/5)", s_axis), ("question text", s_txt), ("relation only", s_rel)):
    if s is None: continue
    a, na = pair(s, "kbw", "know"); b, nb = pair(s, "kbw", "dont")
    print(f"{n:26s} {a:10.3f} {b:14.3f}   (relations: {na}, {nb})"); res[n] = [a, b]
json.dump(res, open(f"hyeonjik/analysis/2_know_but_wrong/results_{MODEL}_{TYPE}.json", "w"), indent=1)
