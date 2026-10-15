"""How few SAE latents separate correct from wrong rows INSIDE the same fact (1-4/5 facts)?

Latents are chosen on train subjects only (paired t-statistic of correct - wrong activation magnitude per fact),
combined as sum_j sign_j * M_j / std_j over the top-n, and scored by within-fact AUROC on held-out subjects.
For a few n, the score is also placed on the know axis exactly like before (0 = 0/5 mean, 1 = 5/5 mean,
normalised inside each relation) to see where 0/5, wrong, correct and 5/5 rows fall.

Usage:  PYTHONPATH=sae_analysis python3 sae_analysis/minimal_signal.py
"""

import collections
import json

import numpy as np

from signal_wrong_vs_correct import load, within_auc
from twelve_cells import auroc

SPLITS = 10
NS = [1, 2, 3, 5, 10, 20, 50, 100]
TABLE_NS = [1, 5, 20]


def main():
    out = {}
    for position in ("q_last", "prompt_last"):
        M, freq, corr, sid, sub, rel, k, cols = load(position)
        el = np.where(freq >= 0.005)[0]
        part = (k >= 1) & (k <= 4)
        isK5, isU = corr & (k == 5), (k == 0) & ~corr
        PW, PC = part & ~corr, part & corr
        paired = np.array([f for f in np.unique(sid[part]) if PW[sid == f].any() and PC[sid == f].any()])
        pp = part & np.isin(sid, paired)
        subjects = np.unique(sub)
        y = corr.astype(int)
        fi = {f: i for i, f in enumerate(paired)}
        idx = np.where(pp)[0]
        row_f = np.array([fi[f] for f in sid[idx]])
        f_sub = np.array([sub[sid == f][0] for f in paired])
        Mc, Mw = np.zeros((len(paired), M.shape[1]), np.float32), np.zeros((len(paired), M.shape[1]), np.float32)
        nc, nw = np.zeros(len(paired)), np.zeros(len(paired))
        for sel, Ms, n in ((corr[idx], Mc, nc), (~corr[idx], Mw, nw)):
            np.add.at(Ms, row_f[sel], M[idx[sel]])
            np.add.at(n, row_f[sel], 1)
        D = Mc / nc[:, None] - Mw / nw[:, None]
        res = collections.defaultdict(list)
        pos = collections.defaultdict(list)
        picks = collections.defaultdict(collections.Counter)
        for seed in range(SPLITS):
            rng = np.random.default_rng(seed)
            trs = set(rng.permutation(subjects)[: int(0.6 * len(subjects))])
            tr = np.array([s in trs for s in sub])
            te = ~tr
            ftr = np.array([s in trs for s in f_sub])
            Dt = D[ftr]
            t = Dt.mean(0) / (Dt.std(0, ddof=1) / np.sqrt(len(Dt)) + 1e-9)
            mask = np.zeros(M.shape[1], bool)
            mask[el] = True
            t[~mask] = 0
            sd = M[tr & part].std(0) + 1e-6
            order = np.argsort(-np.abs(t))
            te_pp = pp & te
            for n in NS + ["all"]:
                sel = order[:n] if n != "all" else el
                w = np.sign(t[sel]) / sd[sel] if n != "all" else t[sel] / sd[sel]
                sc = M[:, sel] @ w
                res[(n, "within")].append(within_auc(sc[te_pp], y[te_pp], sid[te_pp]))
                res[(n, "ku")].append(auroc(sc[te & isK5], sc[te & isU]))
                if n in TABLE_NS:
                    z = np.full(len(sc), np.nan)
                    for r in set(rel):
                        mr = te & (rel == r)
                        if len(set(sid[mr & isK5])) < 3 or len(set(sid[mr & isU])) < 3:
                            continue
                        mu, mk = sc[mr & isU].mean(), sc[mr & isK5].mean()
                        if mk > mu:
                            z[mr] = (sc[mr] - mu) / (mk - mu)
                    v = te & ~np.isnan(z)
                    pos[(n, "U")].append(z[v & isU].mean())
                    pos[(n, "K5")].append(z[v & isK5].mean())
                    for kk in (1, 2, 3, 4):
                        pos[(n, kk, "wrong")].append(z[v & PW & (k == kk)].mean())
                        pos[(n, kk, "correct")].append(z[v & PC & (k == kk)].mean())
            for j in order[:5]:
                picks[1][int(cols[j])] += 1
        print(f"\n##### {position} | 1~4/5 사실 중 맞힌 행과 틀린 행이 모두 있는 사실 {len(paired):,}개")
        print("\n[1] 같은 사실 안 맞힘 vs 틀림 AUROC: 학습용 주어에서 고른 상위 n개 특징 (test 주어, 10회 평균±sd)")
        print(f"  {'특징 수':<8}{'같은 사실 안':>16}{'모두 정답 vs 모두 오답':>24}")
        for n in NS + ["all"]:
            a, b = res[(n, "within")], res[(n, "ku")]
            lab = f"{n}개" if n != "all" else f"{len(el)}개 전부"
            print(f"  {lab:<8}{np.mean(a):>10.3f}±{np.std(a):.3f}{np.mean(b):>18.3f}±{np.std(b):.3f}")
        best = max(np.mean(res[(n, 'within')]) for n in NS + ['all'])
        small = next(n for n in NS + ["all"] if np.mean(res[(n, "within")]) >= best - 0.01)
        print(f"  가장 높은 값 {best:.3f}, 그보다 0.01 이내가 되는 가장 적은 특징 수: {small}")
        for n in TABLE_NS:
            print(f"\n[2] 앎 축 위치 (0 = 0/5 평균, 1 = 5/5 평균, 관계 안 정규화), 특징 {n}개 점수")
            print(f"  기준: 0/5 {np.mean(pos[(n, 'U')]):+.2f}, 5/5 {np.mean(pos[(n, 'K5')]):+.2f}")
            print(f"  {'사실 그룹':<8}{'틀린 행':>10}{'맞힌 행':>10}{'차이':>8}")
            for kk in (1, 2, 3, 4):
                w_, c_ = np.mean(pos[(n, kk, 'wrong')]), np.mean(pos[(n, kk, 'correct')])
                print(f"  {kk}/5{'':<4}{w_:>+10.2f}{c_:>+10.2f}{c_ - w_:>+8.2f}")
        out[position] = {"within": {str(n): [float(np.mean(res[(n, 'within')])), float(np.std(res[(n, 'within')]))] for n in NS + ['all']},
                         "k_u": {str(n): float(np.mean(res[(n, 'ku')])) for n in NS + ['all']},
                         "top_latents": picks[1].most_common(8)}
        print("  자주 뽑힌 특징(상위 5개 후보에 든 횟수): ", picks[1].most_common(6))
    json.dump(out, open("sae_analysis/results_twelve/minimal_signal.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
