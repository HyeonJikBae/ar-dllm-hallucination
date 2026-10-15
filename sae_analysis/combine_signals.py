"""Does the SAE signal add information to the model's own confidence when choosing a paraphrase?

Row set: logit_lens_extract_v2.py rows (analysis_use, guess-risk facts removed); facts of 1-4/5 with >=1 correct and >=1 wrong row.
Signals, all gold-free:
  conf      log-prob of the model's own first answer token at the last layer
  sae20/5   top-n SAE latents (answer-prediction / question-end position, layer 20), chosen on train subjects by the paired
            t-statistic of correct - wrong activation (minimal_signal.py), summed with weights sign/std
  comb      logistic regression on [conf, sae20] (fact-centred), and on [conf, all latents]
5-fold cross-validation over SUBJECTS: every fact is a test fact exactly once; latent choice, scaling and regression use the
other folds only. Per fact: pick the highest-scoring paraphrase and compare with picking at random (= share correct).
Writes the picks for regenerate_picks.py.

Usage:  PYTHONPATH=sae_analysis python3 sae_analysis/combine_signals.py
"""

import collections
import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_wrong_vs_correct import within_auc
from twelve_cells import FEAT

warnings.filterwarnings("ignore")
D = "sae_analysis/features_lens_v2/"
N_BOOT = 2000


def center(X, f):
    Xc = X.astype(np.float64).copy()
    for u in np.unique(f):
        m = f == u
        Xc[m] -= Xc[m].mean(0)
    return Xc


def lr_scores(Xtr, ytr, ftr, Xte, C):
    sc = StandardScaler().fit(center(Xtr, ftr))
    clf = LogisticRegression(C=C, max_iter=400).fit(sc.transform(center(Xtr, ftr)), ytr)
    return clf.decision_function(sc.transform(Xte))


def main():
    meta = [json.loads(l) for l in open(D + "meta.jsonl", encoding="utf-8")]
    z = np.load(D + "lens.npz")
    conf_all = z["out_lp"][:, -1].astype(np.float64)
    subj = {}
    for l in open("data/popqa/1. popqa_raw.jsonl"):
        x = json.loads(l)
        subj[str(x["id"])] = x["s_uri"].rsplit("/", 1)[-1]
    sid = np.array([m["sample_id"] for m in meta])
    sub = np.array([subj[s] for s in sid])
    ver = np.array([m["version"] for m in meta])
    y_all = np.char.startswith(np.array([m["final_label"] for m in meta]).astype(str), "correct").astype(int)
    grp = np.array([m["fact_group"] for m in meta])
    part = (grp != "0/5") & (grp != "5/5")
    paired = np.array([f for f in np.unique(sid[part]) if y_all[part & (sid == f)].min() == 0 and y_all[part & (sid == f)].max() == 1])
    rows = np.where(part & np.isin(sid, paired))[0]            # rows used (indices into lens meta)
    r_sid, r_sub, r_y, r_ver = sid[rows], sub[rows], y_all[rows], ver[rows]
    fm = [json.loads(l) for l in open(f"{FEAT}/qwen_popqa_meta.jsonl", encoding="utf-8")]
    pos_of = {(m["sample_id"], m["version"]): i for i, m in enumerate(fm)}
    feat_idx = np.array([pos_of[(meta[i]["sample_id"], meta[i]["version"])] for i in rows])
    order = np.argsort(feat_idx)
    f_sub = np.array([r_sub[r_sid == f][0] for f in paired])
    subjects = np.unique(f_sub)
    rng = np.random.default_rng(0)
    fold_of_sub = {s: i % 5 for i, s in enumerate(rng.permutation(subjects))}
    fact_fold = {f: fold_of_sub[s] for f, s in zip(paired, f_sub)}
    row_fold = np.array([fact_fold[f] for f in r_sid])
    kk_fact = {f: int(round(r_y[r_sid == f].mean() * (5 if False else 1) * 0)) for f in paired}
    print(f"사실 {len(paired)}개, 행 {len(rows)}개 (맞힌 {r_y.sum()}, 틀린 {(1 - r_y).sum()})")

    results, picks = {}, {}
    for position in ("prompt_last", "q_last"):
        S = np.load(f"{FEAT}/qwen_popqa_{position}.npy", mmap_mode="r")
        X = np.empty((len(rows), S.shape[1]), np.float16)
        X[order] = S[feat_idx[order]]
        el = np.where((X > 0).mean(0) >= 0.005)[0]
        M = X[:, el].astype(np.float32)
        conf = conf_all[rows]
        fi = {f: i for i, f in enumerate(paired)}
        rf = np.array([fi[f] for f in r_sid])
        Mc, Mw = np.zeros((len(paired), M.shape[1]), np.float32), np.zeros((len(paired), M.shape[1]), np.float32)
        nc, nw = np.bincount(rf[r_y == 1], minlength=len(paired)), np.bincount(rf[r_y == 0], minlength=len(paired))
        np.add.at(Mc, rf[r_y == 1], M[r_y == 1]); np.add.at(Mw, rf[r_y == 0], M[r_y == 0])
        Dm = Mc / nc[:, None] - Mw / nw[:, None]
        scores = {k: np.zeros(len(rows)) for k in ("conf", "sae5", "sae20", "sae_all", "conf+sae5", "conf+sae20", "conf+all")}
        for fold in range(5):
            tr, te = row_fold != fold, row_fold == fold
            ftr = np.array([fact_fold[f] != fold for f in paired])
            Dt = Dm[ftr]
            t = Dt.mean(0) / (Dt.std(0, ddof=1) / np.sqrt(len(Dt)) + 1e-9)
            sd = M[tr].std(0) + 1e-6
            o = np.argsort(-np.abs(t))
            s5 = M[:, o[:5]] @ (np.sign(t[o[:5]]) / sd[o[:5]])
            s20 = M[:, o[:20]] @ (np.sign(t[o[:20]]) / sd[o[:20]])
            scores["conf"][te] = conf[te]
            scores["sae5"][te] = s5[te]; scores["sae20"][te] = s20[te]
            scores["sae_all"][te] = lr_scores(M[tr], r_y[tr], r_sid[tr], M[te], 0.01)
            scores["conf+sae5"][te] = lr_scores(np.c_[conf, s5][tr], r_y[tr], r_sid[tr], np.c_[conf, s5][te], 1.0)
            scores["conf+sae20"][te] = lr_scores(np.c_[conf, s20][tr], r_y[tr], r_sid[tr], np.c_[conf, s20][te], 1.0)
            scores["conf+all"][te] = lr_scores(np.c_[conf, M][tr], r_y[tr], r_sid[tr], np.c_[conf, M][te], 0.01)
            print(f"  [{position}] fold {fold + 1}/5 done", flush=True)
        np.savez(f"sae_analysis/results_twelve/combine_scores_{position}.npz", rows=rows, **{k.replace('+', '_'): v for k, v in scores.items()})
        per = {}
        picks[position] = {}
        for name, sc in scores.items():
            top, bot, base, aucs, rk = [], [], [], [], {}
            for f in paired:
                m = np.where(r_sid == f)[0]
                c = r_y[m]
                j = m[np.argmax(sc[m])]
                top.append(r_y[j]); bot.append(1 - c[np.argmin(sc[m])]); base.append(c.mean())
                aucs.append(within_auc(sc[m], c, np.zeros(len(m))))
                rk[f] = int(rows[j])
            per[name] = dict(top=np.array(top), bot=np.array(bot), base=np.array(base), auc=np.array(aucs))
            picks[position][name] = rk
        # original-question baseline
        orig = {f: np.where((r_sid == f) & (r_ver == "question"))[0] for f in paired}
        have = [f for f in paired if len(orig[f])]
        o_acc = np.mean([r_y[orig[f][0]] for f in have]); o_base = np.mean([r_y[r_sid == f].mean() for f in have])
        results[position] = per
        print(f"\n##### {position} | 5-fold 교차검증 (주어 단위), 사실 {len(paired)}개")
        print(f"  원래 질문(question) 변형만 쓰면: 맞힘 {o_acc:.1%} (그 사실들의 무작위 기준 {o_base:.1%}, 사실 {len(have)}개)")
        print(f"  {'신호':<14}{'점수 1등 변형 맞힘':>20}{'무작위':>9}{'차이 [95% 신뢰구간]':>26}{'꼴찌가 틀림':>12}{'같은 사실 안 AUROC':>20}")
        bs = np.random.default_rng(1).integers(0, len(paired), (N_BOOT, len(paired)))
        for name, v in per.items():
            diff = v["top"] - v["base"]
            lo, hi = np.percentile(diff[bs].mean(1), [2.5, 97.5])
            print(f"  {name:<14}{v['top'].mean():>19.1%}{v['base'].mean():>9.1%}{diff.mean():>+14.1%} [{lo:+.1%}, {hi:+.1%}]{v['bot'].mean():>11.1%}{v['auc'].mean():>19.3f}")
        for a, b in (("conf+sae20", "conf"), ("conf+sae5", "conf"), ("conf+all", "conf"), ("conf+sae20", "sae20")):
            d = per[a]["top"] - per[b]["top"]
            lo, hi = np.percentile(d[bs].mean(1), [2.5, 97.5])
            da = per[a]["auc"] - per[b]["auc"]
            lo2, hi2 = np.percentile(da[bs].mean(1), [2.5, 97.5])
            print(f"  {a} − {b}: 1등 맞힘 {d.mean():+.1%} [{lo:+.1%}, {hi:+.1%}] | AUROC {da.mean():+.3f} [{lo2:+.3f}, {hi2:+.3f}]")
    json.dump({"picks": picks}, open("sae_analysis/results_twelve/combine_picks.json", "w"))
    json.dump({p: {n: {"top": float(v["top"].mean()), "base": float(v["base"].mean()), "bot": float(v["bot"].mean()), "auc": float(v["auc"].mean())}
                   for n, v in per.items()} for p, per in results.items()}, open("sae_analysis/results_twelve/combine_signals.json", "w"), indent=1)


if __name__ == "__main__":
    main()
