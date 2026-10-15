"""Verify the 20-latent table (answer-prediction position, Qwen PopQA final labels).
Latents chosen on train subjects; everything reported on held-out subjects only.
  real      : selection with true correct/wrong labels
  shuffled  : selection after shuffling correct/wrong labels INSIDE each fact (k unchanged) -> what selection alone produces
Per k: mean know-axis position of wrong / correct rows, row-level AUROC (correct vs wrong rows of that k),
and the share of rows on the 'right side' (correct >= 0.5, wrong < 0.5).
Usage: PYTHONPATH=sae_analysis python3 sae_analysis/verify_minimal20.py
"""
import collections
import numpy as np
from signal_wrong_vs_correct import load, within_auc
from twelve_cells import auroc

N, SPLITS = 20, 10
M, freq, corr, sid, sub, rel, k, cols = load("prompt_last")
el = np.where(freq >= 0.005)[0]
part = (k >= 1) & (k <= 4); isK5, isU = corr & (k == 5), (k == 0) & ~corr
paired = np.array([f for f in np.unique(sid[part]) if (part & ~corr)[sid == f].any() and (part & corr)[sid == f].any()])
pp = part & np.isin(sid, paired); subjects = np.unique(sub); y = corr.astype(int)
fi = {f: i for i, f in enumerate(paired)}; idx = np.where(pp)[0]; row_f = np.array([fi[f] for f in sid[idx]])
f_sub = np.array([sub[sid == f][0] for f in paired])

def diffs(lab):
    Mc, Mw = np.zeros((len(paired), M.shape[1]), np.float32), np.zeros((len(paired), M.shape[1]), np.float32)
    nc, nw = np.zeros(len(paired)), np.zeros(len(paired))
    for s_, Ms, n in ((lab, Mc, nc), (~lab, Mw, nw)):
        np.add.at(Ms, row_f[s_], M[idx[s_]]); np.add.at(n, row_f[s_], 1)
    return Mc / nc[:, None] - Mw / nw[:, None]

def shuffled_labels(rng):
    lab = corr[idx].copy()
    for i in range(len(paired)):
        m = np.where(row_f == i)[0]; lab[m] = rng.permutation(lab[m])
    return lab

out = {}
for mode in ("real", "shuffled"):
    R = collections.defaultdict(list)
    for seed in range(SPLITS):
        rng = np.random.default_rng(seed)
        trs = set(rng.permutation(subjects)[: int(0.6 * len(subjects))])
        tr = np.array([s in trs for s in sub]); te = ~tr; ftr = np.array([s in trs for s in f_sub])
        D = diffs(corr[idx] if mode == "real" else shuffled_labels(np.random.default_rng(100 + seed)))
        Dt = D[ftr]; t = Dt.mean(0) / (Dt.std(0, ddof=1) / np.sqrt(len(Dt)) + 1e-9)
        msk = np.zeros(M.shape[1], bool); msk[el] = True; t[~msk] = 0
        sd = M[tr & part].std(0) + 1e-6; sel = np.argsort(-np.abs(t))[:N]
        sc = M[:, sel] @ (np.sign(t[sel]) / sd[sel])
        z = np.full(len(sc), np.nan)
        for r in set(rel):
            mr = te & (rel == r)
            if len(set(sid[mr & isK5])) < 3 or len(set(sid[mr & isU])) < 3: continue
            mu, mk = sc[mr & isU].mean(), sc[mr & isK5].mean()
            if mk > mu: z[mr] = (sc[mr] - mu) / (mk - mu)
        v = te & ~np.isnan(z)
        te_pp = pp & te
        R["within"].append(within_auc(sc[te_pp], y[te_pp], sid[te_pp]))
        R["ku"].append(auroc(sc[te & isK5], sc[te & isU]))
        for kk in (1, 2, 3, 4):
            w, c = z[v & part & ~corr & (k == kk)], z[v & part & corr & (k == kk)]
            R[(kk, "w")].append(w.mean()); R[(kk, "c")].append(c.mean()); R[(kk, "auc")].append(auroc(c, w))
            R[(kk, "cside")].append((c >= 0.5).mean()); R[(kk, "wside")].append((w < 0.5).mean())
            R[(kk, "n")].append((len(c), len(w)))
        R["U_hi"].append((z[v & isU] >= 0.5).mean()); R["K5_hi"].append((z[v & isK5] >= 0.5).mean())
    out[mode] = R
    f = lambda key: f"{np.mean(R[key]):+.2f}±{np.std(R[key]):.2f}"
    print(f"\n===== 피처 20개, 선택 라벨 = {'진짜' if mode == 'real' else '같은 사실 안에서 섞음(대조군)'} | 테스트 주어만, 분할 {SPLITS}회 평균±편차")
    print(f"  같은 사실 안 AUROC {np.mean(R['within']):.3f}±{np.std(R['within']):.3f} | 모두정답 vs 모두오답 AUROC {np.mean(R['ku']):.3f}")
    print(f"  {'그룹':<6}{'오답 평균':>12}{'정답 평균':>12}{'행 단위 AUROC':>16}{'정답≥0.5 비율':>14}{'오답<0.5 비율':>14}{'행 수(정답/오답)':>18}")
    for kk in (1, 2, 3, 4):
        nc, nw = np.mean([a for a, b in R[(kk, 'n')]]), np.mean([b for a, b in R[(kk, 'n')]])
        print(f"  {kk}/5{'':<3}{f((kk,'w')):>12}{f((kk,'c')):>12}{np.mean(R[(kk,'auc')]):>12.3f}±{np.std(R[(kk,'auc')]):.2f}"
              f"{np.mean(R[(kk,'cside')]):>14.1%}{np.mean(R[(kk,'wside')]):>14.1%}{f'{nc:.0f}/{nw:.0f}':>18}")
    print(f"  참고: 0/5 행 중 ≥0.5 {np.mean(R['U_hi']):.1%}, 5/5 행 중 ≥0.5 {np.mean(R['K5_hi']):.1%}")
