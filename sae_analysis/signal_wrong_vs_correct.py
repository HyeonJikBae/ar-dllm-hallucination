"""Where do wrong / correct rows of partially known facts (1-4/5) sit between "doesn't know" (0/5) and "knows" (5/5)?
And can ONE feature separate wrong from correct rows inside the same fact?

Part 1  know-axis (430 latents, K5 - U direction, relation-normalised: 0 = 0/5 mean, 1 = 5/5 mean), held-out subjects.
        Per group: mean position, share of rows on the 0/5 side (< 0.5) vs the 5/5 side (> 0.5).
        U and K5 rows themselves are the reference: even real 0/5 and 5/5 rows are not all on their own side.
Part 2  inside the same fact (facts with >=1 correct and >=1 wrong row among 1-4/5):
        single latent chosen on train subjects (largest paired t-stat of correct - wrong), within-fact AUROC on test subjects.
        Compared with: the 430-latent know-axis score s, and a logistic regression over the same latents.

Usage:  PYTHONPATH=sae_analysis python3 sae_analysis/signal_wrong_vs_correct.py
"""

import collections
import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from know_axis_by_k import fact_matrix
from twelve_cells import FEAT, strat_diff

warnings.filterwarnings("ignore")
SPLITS = 10


def load(position):
    meta = [json.loads(l) for l in open(f"{FEAT}/qwen_popqa_meta.jsonl", encoding="utf-8")]
    subj = {}
    for l in open("data/popqa/1. popqa_raw.jsonl"):
        x = json.loads(l)
        subj[str(x["id"])] = x["s_uri"].rsplit("/", 1)[-1]
    ok = np.array([not m["gold_issue"] and m["verdict"] != "uncertain" for m in meta])
    corr = np.array([m["verdict"] == "correct" or m["verdict_suggested"] == "correct" for m in meta])
    use = ok & ~np.array([bool(m.get("rule_unsure")) for m in meta])
    sid = np.array([m["sample_id"] for m in meta])
    kf = collections.Counter()
    for s, c, o in zip(sid, corr, ok):
        if o:
            kf[s] += int(c)
    k = np.array([kf.get(s, -1) for s in sid])
    keep = np.where(use & (k >= 0))[0]
    S = np.load(f"{FEAT}/qwen_popqa_{position}.npy", mmap_mode="r")
    A = np.asarray(S[keep])
    cols = np.where((A > 0).mean(0) >= 0.002)[0]
    M = A[:, cols].astype(np.float32)
    del A
    freq = (M > 0).mean(0)
    rel = np.array([meta[i]["relation"] for i in keep])
    sub = np.array([subj[meta[i]["sample_id"]] for i in keep])
    return M, freq, corr[keep], sid[keep], sub, rel, k[keep], cols


def within_auc(score, y, sid):
    out = []
    for f in np.unique(sid):
        m = sid == f
        a, b = score[m & (y == 1)], score[m & (y == 0)]
        if len(a) and len(b):
            d = a[:, None] - b[None, :]
            out.append(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)
    return float(np.mean(out)) if out else np.nan


def main():
    for position in ("q_last", "prompt_last"):
        M, freq, corr, sid, sub, rel, k, cols = load(position)
        el = freq >= 0.005
        B = (M[:, el] > 0).astype(np.float32)
        part = (k >= 1) & (k <= 4)
        isK5, isU = corr & (k == 5), (k == 0) & ~corr
        PW, PC = part & ~corr, part & corr
        paired = [f for f in np.unique(sid[part]) if PW[sid == f].any() and PC[sid == f].any()]
        pp = part & np.isin(sid, paired)
        subjects = np.unique(sub)
        res = collections.defaultdict(list)
        chosen = collections.Counter()
        for seed in range(SPLITS):
            rng = np.random.default_rng(seed)
            tr = np.isin(sub, rng.permutation(subjects)[: int(0.6 * len(subjects))])
            te = ~tr
            # ---- part 1: know axis
            _, XK, rK = fact_matrix(B[tr & isK5], sid[tr & isK5], rel[tr & isK5])
            _, XU, rU = fact_matrix(B[tr & isU], sid[tr & isU], rel[tr & isU])
            d, _ = strat_diff(XK, rK, XU, rU, 3)
            s = B @ d.astype(np.float32)
            z = np.full(len(s), np.nan)
            for r in set(rel):
                mr = te & (rel == r)
                if len(set(sid[mr & isK5])) < 3 or len(set(sid[mr & isU])) < 3:
                    continue
                mu, mk = s[mr & isU].mean(), s[mr & isK5].mean()
                if mk > mu:
                    z[mr] = (s[mr] - mu) / (mk - mu)
            v = te & ~np.isnan(z)
            for name, m in (("모두 오답 행 (0/5)", isU), ("모두 정답 행 (5/5)", isK5), ("1~4/5 틀린 행", PW), ("1~4/5 맞힌 행", PC)):
                res[(name, "mean")].append(z[v & m].mean())
                res[(name, "below")].append((z[v & m] < 0.5).mean())
            for kk in (1, 2, 3, 4):
                for name, m in (("틀린", PW), ("맞힌", PC)):
                    res[(f"k={kk} {name}", "mean")].append(z[v & m & (k == kk)].mean())
            gaps = []
            for f in np.unique(sid[v & pp]):
                m = v & (sid == f)
                zc, zw = z[m & PC], z[m & PW]
                if len(zc) and len(zw):
                    gaps.append(zc.mean() - zw.mean())
            res[("fact 안에서 맞힌 행이 더 앎 쪽", "frac")].append(float(np.mean(np.array(gaps) > 0)))
            res[("fact 안 맞힌 − 틀린", "gap")].append(float(np.mean(gaps)))
            # ---- part 2: single latent
            facts = np.array(paired)
            idx = np.where(pp)[0]
            fi = {f: i for i, f in enumerate(facts)}
            row_f = np.array([fi[f] for f in sid[idx]])
            Mc, Mw = np.zeros((len(facts), M.shape[1]), np.float32), np.zeros((len(facts), M.shape[1]), np.float32)
            nc, nw = np.zeros(len(facts)), np.zeros(len(facts))
            for sel, Msum, n in ((corr[idx], Mc, nc), (~corr[idx], Mw, nw)):
                np.add.at(Msum, row_f[sel], M[idx[sel]])
                np.add.at(n, row_f[sel], 1)
            D = Mc / nc[:, None] - Mw / nw[:, None]
            f_sub = np.array([sub[sid == f][0] for f in facts])
            ftr = np.isin(f_sub, subjects[np.isin(subjects, np.unique(sub[tr]))])
            Dt = D[ftr]
            t = Dt.mean(0) / (Dt.std(0, ddof=1) / np.sqrt(len(Dt)) + 1e-9)
            t[~el] = 0
            order = np.argsort(-np.abs(t))
            te_rows = pp & te
            y = corr.astype(int)
            aucs = []
            for rank, j in enumerate(order[:5]):
                a = within_auc(np.sign(t[j]) * M[:, j], y[te_rows], sid[te_rows]) if False else within_auc((np.sign(t[j]) * M[:, j])[te_rows], y[te_rows], sid[te_rows])
                aucs.append(a)
                if rank == 0:
                    chosen[int(cols[j])] += 1
            res[("단일 특징 (train에서 가장 좋은 1개)", "auc")].append(aucs[0])
            res[("단일 특징 상위 5개 각각의 평균", "auc")].append(float(np.mean(aucs)))
            res[("앎 점수 s (430개 합)", "auc")].append(within_auc(s[te_rows], y[te_rows], sid[te_rows]))
            # logistic regression over the same latents, fact-centred training
            tr_rows = pp & tr
            Xt = M[tr_rows][:, el]
            for f in np.unique(sid[tr_rows]):
                m = sid[tr_rows] == f
                Xt[m] -= Xt[m].mean(0)
            sc = StandardScaler().fit(Xt)
            clf = LogisticRegression(C=0.01, max_iter=300).fit(sc.transform(Xt), y[tr_rows])
            res[("430개 + 로지스틱 (학습된 가중치)", "auc")].append(within_auc(clf.decision_function(sc.transform(M[te_rows][:, el])), y[te_rows], sid[te_rows]))
            # descriptive: the chosen latent over the four groups (all rows)
        j_best = max(chosen, key=chosen.get)
        jj = int(np.where(cols == j_best)[0][0])
        fr = lambda m: float((M[m, jj] > 0).mean())
        print(f"\n##### {position} | 1~4/5 사실 중 맞힌 행과 틀린 행이 모두 있는 사실 {len(paired):,}개 (행 {pp.sum():,})")
        print("\n[1] 앎 축 위치 (0 = 모두 오답 평균, 1 = 모두 정답 평균), test 사실, 10회 평균")
        print(f"  {'그룹':<20}{'평균 위치':>10}{'0쪽(<0.5) 비율':>16}")
        for name in ("모두 오답 행 (0/5)", "1~4/5 틀린 행", "1~4/5 맞힌 행", "모두 정답 행 (5/5)"):
            print(f"  {name:<20}{np.mean(res[(name, 'mean')]):>+10.2f}{np.mean(res[(name, 'below')]):>15.1%}")
        print("  k별 평균 위치 (틀린 / 맞힌): " + "  ".join(
            f"{kk}/5 {np.mean(res[(f'k={kk} 틀린', 'mean')]):+.2f} / {np.mean(res[(f'k={kk} 맞힌', 'mean')]):+.2f}" for kk in (1, 2, 3, 4)))
        print(f"  같은 사실 안에서 맞힌 행 − 틀린 행 = {np.mean(res[('fact 안 맞힌 − 틀린', 'gap')]):+.2f}, 맞힌 행이 더 앎 쪽인 사실 {np.mean(res[('fact 안에서 맞힌 행이 더 앎 쪽', 'frac')]):.1%}")
        print("\n[2] 같은 사실 안 맞힘 vs 틀림 AUROC (test 주어 엔티티, 10회 평균±sd; 0.5 = 우연)")
        for name in ("단일 특징 (train에서 가장 좋은 1개)", "단일 특징 상위 5개 각각의 평균", "앎 점수 s (430개 합)", "430개 + 로지스틱 (학습된 가중치)"):
            v = res[(name, "auc")]
            print(f"  {name:<34}{np.mean(v):.3f}±{np.std(v):.3f}")
        print(f"  가장 자주 뽑힌 단일 특징 {j_best} (10회 중 {chosen[j_best]}회). 이 특징이 켜진 비율: "
              f"모두 오답 {fr(isU):.1%}, 1~4/5 틀린 {fr(PW):.1%}, 1~4/5 맞힌 {fr(PC):.1%}, 모두 정답 {fr(isK5):.1%}")
        json.dump({f"{a}|{b}": [float(np.mean(v)), float(np.std(v))] for (a, b), v in res.items()},
                  open(f"sae_analysis/results_twelve/signal_wrong_vs_correct_{position}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
