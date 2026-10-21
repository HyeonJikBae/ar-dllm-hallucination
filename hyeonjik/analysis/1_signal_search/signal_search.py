"""Step A: is there a signal that tells "the model knows this fact but answers wrong"?  (Dream and Qwen, same procedure)

Rows    : facts answered correctly in 1-4 of the 5 paraphrases (the model knows the fact), only facts with both a correct and a wrong row.
Label   : correct (1) / wrong (0).  Compared INSIDE each fact: score of a correct row vs a wrong row of the same fact -> within-fact AUROC.
Signals : all gold-free and computed before the answer is written (last prompt token / end of question / end of subject, layer 20, denoising step 0).
  conf       log-prob and margin of the first answer token
  sae-sum    sum of the top-20 SAE latents picked by the within-fact paired t (sign * activation / std)
  sae-lr     logistic regression on the top-K SAE latents (K = 300, 2000), within-fact centred
  raw-lr     logistic regression on the raw hidden state, within-fact centred
Baselines to beat (label-free):
  text       logistic regression on the question text (tf-idf) -> how much is explained by the wording of the paraphrase
  perm       the chosen signal with labels shuffled inside each fact
Protocol: subjects split once into dev / test (50/50, seed 0). Every configuration is scored on dev by 5-fold subject CV;
          the best configuration per signal family is picked on dev, then ALL configurations are scored on test once.
          Test numbers carry 95% CIs from a bootstrap over facts.
Usage (repo root): python3 hyeonjik/analysis/signal_search.py dream|qwen
"""
import collections
import json
import os
import sys
import warnings

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "sae_analysis")
torch.set_num_threads(16)
D = "hyeonjik/data"
MODEL = sys.argv[1]
OUT = f"hyeonjik/analysis/results_signal_{MODEL}.json"


def fact_aucs(score, y, fact):
    """AUROC of correct vs wrong inside every fact (ties 0.5); returns {fact: auc}."""
    res = {}
    order = np.argsort(fact, kind="stable")
    f, s, yy = fact[order], score[order], y[order]
    cuts = np.flatnonzero(np.r_[True, f[1:] != f[:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        p, n = s[a:b][yy[a:b] == 1], s[a:b][yy[a:b] == 0]
        if len(p) and len(n):
            res[f[a]] = float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())
    return res


def centre(X, fact):
    """subtract the fact mean (label-free): leaves only differences between the paraphrases of one fact"""
    X = X.copy()
    for f in np.unique(fact):
        m = fact == f
        X[m] -= X[m].mean(0)
    return X


def load():
    popqa = {}
    for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8"):
        r = json.loads(l)
        popqa[str(r["id"])] = r["s_uri"]
    if MODEL == "dream":
        rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
        F = f"{D}/1_raw/dream_hidden_steps"
        meta = [json.loads(l) for l in open(f"{F}/meta.jsonl", encoding="utf-8")]
        ok = lambda i, r: r["analysis_use"] and not r["junk_suffix"]
        index = list(range(len(rows)))
    else:
        rows = [json.loads(l) for l in open(f"{D}/3_classification/full/qwen_final_full.jsonl", encoding="utf-8")]
        F = f"{D}/1_raw/qwen_hidden"
        index = json.load(open(f"{F}/rows.json"))
        ok = lambda i, r: r["analysis_use"]
    byf = collections.defaultdict(list)
    for i in index:
        r = rows[i]
        if ok(i, r) and r["fact_group"] in ("1/5", "2/5", "3/5", "4/5"):
            byf[r["sample_id"]].append(i)
    keep = [i for f, ix in byf.items() if 0 < sum(rows[j]["correct"] for j in ix) < len(ix) for i in ix]
    keep = np.array(sorted(keep))
    return rows, popqa, F, index, keep


def main():
    rows, popqa, F, index, keep = load()
    y = np.array([rows[i]["correct"] for i in keep], dtype=int)
    fact = np.array([rows[i]["sample_id"] for i in keep])
    sub = np.array([popqa[f] for f in fact])
    qtext = [rows[i]["question"] for i in keep]
    print(f"{MODEL}: rows {len(keep)} | facts {len(np.unique(fact))} | subjects {len(np.unique(sub))} | correct share {y.mean():.2f}", flush=True)

    # --- features (before the answer): confidence and hidden states
    pos = {}
    if MODEL == "dream":
        lp = np.load(f"{F}/conf_lp.npy", mmap_mode="r")
        mg = np.load(f"{F}/conf_mg.npy", mmap_mode="r")
        conf = np.stack([np.asarray(lp[keep, 0, 0]), np.asarray(mg[keep, 0, 0])], 1).astype(np.float32)
        from dream_sae_compare import encode, load_sae
        sae, ub = load_sae("dream-7b_L20_shared-0.2-0.5-0.8_s5_200M")
        enc = lambda x: encode(sae, x, ub)
        for p in ("prompt_last", "q_last"):
            H = np.load(f"{F}/hid_{p}.npy", mmap_mode="r")
            pos[p] = np.asarray(H[keep, 0], dtype=np.float32)
        H = np.load(f"{F}/hid_subj_last.npy", mmap_mode="r")
        meta = [json.loads(l) for l in open(f"{F}/meta.jsonl", encoding="utf-8")]
        found = np.array([meta[i]["subj_found"] for i in keep])
        if found.mean() > 0.5:
            pos["subj_last"] = np.asarray(H[keep, 0], dtype=np.float32)
    else:
        from extract_qwen_features import JumpReLUSAE, SAE_DIR
        pos_of = {i: j for j, i in enumerate(index)}
        ix = np.array([pos_of[i] for i in keep])
        conf = np.stack([np.load(f"{F}/conf_lp.npy")[ix], np.load(f"{F}/conf_mg.npy")[ix]], 1).astype(np.float32)
        sae_q = JumpReLUSAE(SAE_DIR, "cpu")
        enc = lambda x: sae_q.encode(x).float()
        pos["prompt_last"] = np.load(f"{F}/hid.npy")[ix].astype(np.float32)
    acts = {}
    with torch.no_grad():
        for p, X in pos.items():
            acts[p] = np.concatenate([enc(torch.from_numpy(X[b:b + 1024])).numpy() for b in range(0, len(X), 1024)])
    print("positions:", list(pos), "| SAE L0", {p: round(float((a > 0).sum(1).mean()), 1) for p, a in acts.items()}, flush=True)

    # --- split subjects into dev / test, and folds inside dev
    rng = np.random.default_rng(0)
    subs = rng.permutation(np.unique(sub))
    dev_s, test_s = set(subs[: len(subs) // 2]), set(subs[len(subs) // 2:])
    dev = np.array([s in dev_s for s in sub])
    test = ~dev
    folds = np.array_split(rng.permutation(np.array(sorted(dev_s))), 5)

    def paired_t(A, tr):
        d = []
        for f in np.unique(fact[tr]):
            m = tr & (fact == f)
            if 0 < y[m].sum() < m.sum():
                d.append(A[m][y[m] == 1].mean(0) - A[m][y[m] == 0].mean(0))
        d = np.array(d)
        return d.mean(0) / (d.std(0) / np.sqrt(len(d)) + 1e-9)

    # --- every configuration is a function (train mask, eval mask) -> scores on eval rows
    cfgs = {}
    cfgs["conf-lp"] = lambda tr, te: conf[te, 0]
    cfgs["conf-margin"] = lambda tr, te: conf[te, 1]

    def conf_lr(tr, te):
        Z = (conf - conf[tr].mean(0)) / (conf[tr].std(0) + 1e-6)
        return LogisticRegression(max_iter=200).fit(Z[tr], y[tr]).decision_function(Z[te])
    cfgs["conf-lr"] = conf_lr

    def text_lr(tr, te):
        tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000).fit([qtext[i] for i in np.flatnonzero(tr)])
        T = tv.transform(qtext)
        clf = LogisticRegression(C=1.0, max_iter=300)
        # centre inside facts: only the wording differences between paraphrases of one fact remain
        Xc = centre(T.toarray().astype(np.float32), fact)
        return clf.fit(Xc[tr], y[tr]).decision_function(Xc[te])
    cfgs["text-lr"] = text_lr

    for p in acts:
        A = acts[p]
        Ac = centre(np.log1p(A), fact)
        Rc = centre(pos[p], fact)

        def sae_sum(tr, te, A=A):
            t = paired_t(A, tr)
            top = np.argsort(-np.abs(t))[:20]
            sd = A[tr][:, top].std(0) + 1e-6
            return (np.sign(t[top]) * A[te][:, top] / sd).sum(1)
        cfgs[f"sae-sum20|{p}"] = sae_sum
        for K in (300, 2000):
            for C in (0.003, 0.03):
                def sae_lr(tr, te, Ac=Ac, A=A, K=K, C=C):
                    t = paired_t(np.log1p(A), tr)
                    top = np.argsort(-np.abs(t))[:K]
                    X = Ac[:, top]
                    X = X / (X[tr].std(0) + 1e-6)
                    return LogisticRegression(C=C, max_iter=300).fit(X[tr], y[tr]).decision_function(X[te])
                cfgs[f"sae-lr{K}|C{C}|{p}"] = sae_lr
        for C in (0.0003, 0.003):
            def raw_lr(tr, te, Rc=Rc, C=C):
                X = Rc / (Rc[tr].std(0) + 1e-6)
                return LogisticRegression(C=C, max_iter=300).fit(X[tr], y[tr]).decision_function(X[te])
            cfgs[f"raw-lr|C{C}|{p}"] = raw_lr

    # --- 1. score every configuration on dev (5-fold by subject), pick the best per family
    def dev_cv(fn):
        aucs = []
        for fo in folds:
            te = dev & np.isin(sub, fo)
            tr = dev & ~te
            s = np.zeros(len(y))
            s[te] = fn(tr, te)
            aucs.append(np.mean(list(fact_aucs(s[te], y[te], fact[te]).values())))
        return float(np.mean(aucs))

    dev_scores = {}
    for name, fn in cfgs.items():
        dev_scores[name] = dev_cv(fn)
        print(f"  dev {dev_scores[name]:.3f}  {name}", flush=True)
    family = lambda n: n.split("|")[0].rstrip("0123456789").rstrip("-") if n.startswith(("sae-lr", "sae-sum", "raw-lr")) else n
    best = {}
    for n, v in dev_scores.items():
        k = n.split("|")[0][:6] if n.startswith(("sae-lr", "raw-lr", "sae-su")) else n
        if k not in best or v > dev_scores[best[k]]:
            best[k] = n
    print("picked on dev:", best, flush=True)

    # --- 2. score every configuration once on test, bootstrap CI over facts
    def boot(per):
        v = np.array(list(per.values()))
        r = np.random.default_rng(1)
        m = [v[r.integers(0, len(v), len(v))].mean() for _ in range(1000)]
        return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    result = {"rows": int(len(keep)), "facts": int(len(np.unique(fact))), "dev_scores": dev_scores, "picked": best, "test": {}}
    print("\nTEST (within-fact AUROC, mean [95% CI over facts])", flush=True)
    for name, fn in cfgs.items():
        s = np.zeros(len(y))
        s[test] = fn(dev, test)
        per = fact_aucs(s[test], y[test], fact[test])
        m, lo, hi = boot(per)
        tag = "  <- picked" if name in best.values() else ""
        result["test"][name] = {"auc": m, "lo": lo, "hi": hi, "facts": len(per)}
        print(f"  {m:.3f} [{lo:.3f}, {hi:.3f}]  {name}{tag}", flush=True)

    # --- 3. label permutation inside facts for the picked configurations (null)
    print("\nNULL (labels shuffled inside each fact, picked configurations, mean over 20 shuffles)", flush=True)
    y0 = y.copy()
    result["null"] = {}
    for name in sorted(set(best.values())):
        vals = []
        for r in range(20):
            rr = np.random.default_rng(100 + r)
            for f in np.unique(fact):
                m = np.flatnonzero(fact == f)
                y[m] = y0[m][rr.permutation(len(m))]
            s = np.zeros(len(y))
            s[test] = cfgs[name](dev, test)
            vals.append(np.mean(list(fact_aucs(s[test], y[test], fact[test]).values())))
        y[:] = y0
        result["null"][name] = [float(np.mean(vals)), float(np.std(vals))]
        print(f"  {np.mean(vals):.3f} ± {np.std(vals):.3f}  {name}", flush=True)
    json.dump(result, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
