"""One-vs-rest decodability of each wrong-answer type from SAE features, controlling for the question relation.

For type T: positives = confident wrong rows of type T, negatives = confident wrong rows of every other type.
Held-out facts (5 random 60/40 splits by fact). AUROC of (a) relation one-hot only, (b) relation + SAE latents.
Also per-latent relation-adjusted frequency difference (train top vs test).

Usage:  FEAT=sae_analysis/features_g python3 sae_analysis/type_signals.py --position q_last
"""
import argparse, collections, json, os
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

FEAT = os.environ.get("FEAT", "sae_analysis/features_g")
NAMES = {"1": "유형 1 관련", "2": "유형 2 무관", "3": "유형 3 복사", "5": "유형 5 재조합", "6": "유형 6 붕괴", "N": "기권"}


def auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--min-freq", type=float, default=0.03)
    args = ap.parse_args()
    meta = [json.loads(l) for l in open(f"{FEAT}/qwen_popqa_meta.jsonl", encoding="utf-8")]
    B = np.load(f"{FEAT}/qwen_popqa_{args.position}.npy") > 0
    typ = []
    for m in meta:
        t = ""
        if m["verdict"] == "incorrect" and m["verdict_suggested"] != "correct" and not m["gold_issue"] and not m.get("rule_unsure"):
            if m["reason"] in (1, 2, 3, 5, 6):
                t = str(m["reason"])
            elif m["reason"] is None and "N" in (m.get("rule") or "").replace("N-abstain", "N"):
                t = "N"
        typ.append(t)
    typ = np.array(typ); sid = np.array([m["sample_id"] for m in meta]); rel = np.array([m["relation"] for m in meta])
    use = typ != ""
    print(f"[{args.position}] wrong rows used {use.sum()} |", {k: int((typ == k).sum()) for k in NAMES}, "| facts with type 4:", int((np.array([m['reason'] == 4 for m in meta])).sum()), "rows (not analysed)")
    cols = np.where(B[use].mean(0) >= args.min_freq)[0]
    X = B[:, cols].astype(np.float32)
    rels = sorted(set(rel[use])); R = (rel[:, None] == np.array(rels)[None, :]).astype(np.float32)
    facts = np.array(sorted(set(sid[use])))
    res = {k: {"rel": [], "full": []} for k in NAMES}
    top_lat = {k: collections.Counter() for k in NAMES}
    last = {}
    for seed in range(args.splits):
        rng = np.random.default_rng(seed)
        trf = set(rng.permutation(facts)[: int(0.6 * len(facts))])
        tr = use & np.isin(sid, list(trf)); te = use & ~np.isin(sid, list(trf))
        for k in NAMES:
            ytr, yte = (typ[tr] == k), (typ[te] == k)
            if ytr.sum() < 10 or yte.sum() < 5:
                continue
            for name, F in (("rel", R), ("full", np.hstack([R, X]))):
                lr = LogisticRegression(max_iter=300, C=0.3 if name == "full" else 1.0)
                lr.fit(F[tr], ytr)
                res[k][name].append(auroc(lr.decision_function(F[te])[yte], lr.decision_function(F[te])[~yte]))
            # relation-adjusted frequency difference: T vs rest inside relations that have both
            def adj(mask):
                num = np.zeros(X.shape[1]); w = 0
                for r in rels:
                    a = mask & (rel == r) & (typ == k); b = mask & (rel == r) & (typ != k) & use
                    if a.sum() >= 5 and b.sum() >= 5:
                        num += a.sum() * (X[a].mean(0) - X[b].mean(0)); w += a.sum()
                return num / w if w else num
            dtr, dte = adj(tr), adj(te)
            for j in np.argsort(-dtr)[:10]:
                top_lat[k][cols[j]] += 1
            last[k] = (dtr, dte)
    print("\n유형별 AUROC (held-out fact, one-vs-rest)  |  관계만  →  관계+SAE latent  | 증가")
    for k, nm in NAMES.items():
        if not res[k]["rel"]:
            print(f"  {nm:12s} 표본 부족"); continue
        a, b = np.mean(res[k]["rel"]), np.mean(res[k]["full"]); sd = np.std(np.array(res[k]["full"]) - np.array(res[k]["rel"]))
        n = int((typ == k).sum())
        print(f"  {nm:12s} n={n:5d} | {a:.3f} → {b:.3f} | {b-a:+.3f} (±{sd:.3f})")
    print("\ntrain·test 모두 (T - 나머지) 관계보정 차이 > 0.05 인 latent 수 (분할 0 기준)")
    for k, nm in NAMES.items():
        if k in last:
            dtr, dte = last[k]; print(f"  {nm:12s} {int(((dtr > 0.05) & (dte > 0.05)).sum())}개  | 최대 train {dtr.max():+.3f} test(그 latent) {dte[np.argmax(dtr)]:+.3f}")


if __name__ == "__main__":
    main()
