"""Pick SAE latents that separate correct from wrong rows with LOW spread (answer-prediction position, Qwen PopQA final labels).
Selection on train subjects only (60%), evaluation on held-out subjects, 10 splits. Rows = 1-4/5 facts.
  A  paired t (current)       : within-fact (correct - wrong) mean / SE across facts
  B  row-level Cohen's d      : (mean correct - mean wrong) / pooled within-class SD over rows -> penalises spread across facts
  C  t then d                 : top-100 by |t|, then top by |d| (consistent within facts AND low spread across facts)
  D  stable d                 : B on 20 bootstrap halves of the train facts; keep latents chosen most often
Score = sum sign_j * M_j / sd_j over the n latents. Reported on test rows:
  row-level AUROC (all 1-4/5 rows pooled, and per k), within-fact AUROC, 0/5 vs 5/5 AUROC,
  precision at the extremes: share correct among the top 10% scores, share wrong among the bottom 10%.
Usage: PYTHONPATH=sae_analysis python3 sae_analysis/select_lowvar.py
"""
import collections
import numpy as np
from signal_wrong_vs_correct import load, within_auc
from twelve_cells import auroc

SPLITS = 10
M, freq, corr, sid, sub, rel, k, cols = load("prompt_last")
el = np.where(freq >= 0.005)[0]
part = (k >= 1) & (k <= 4); isK5, isU = corr & (k == 5), (k == 0) & ~corr
paired = np.array([f for f in np.unique(sid[part]) if (part & ~corr)[sid == f].any() and (part & corr)[sid == f].any()])
pp = part & np.isin(sid, paired); subjects = np.unique(sub); y = corr.astype(int)
fi = {f: i for i, f in enumerate(paired)}; idx = np.where(pp)[0]; row_f = np.array([fi[f] for f in sid[idx]])
f_sub = np.array([sub[sid == f][0] for f in paired])
Me = M[:, el].astype(np.float32)
Mc, Mw = np.zeros((len(paired), len(el)), np.float32), np.zeros((len(paired), len(el)), np.float32)
nc, nw = np.zeros(len(paired)), np.zeros(len(paired))
for s_, Ms, n in ((corr[idx], Mc, nc), (~corr[idx], Mw, nw)):
    np.add.at(Ms, row_f[s_], Me[idx[s_]]); np.add.at(n, row_f[s_], 1)
D = Mc / nc[:, None] - Mw / nw[:, None]
print(f"1~4/5 짝 사실 {len(paired)}개, 후보 피처 {len(el)}개, 행 {pp.sum()}개 (정답 비율 {y[pp].mean():.1%})")

def tstat(rows_f):
    Dt = D[rows_f]; return Dt.mean(0) / (Dt.std(0, ddof=1) / np.sqrt(len(Dt)) + 1e-9)

def cohend(rows):
    a, b = Me[rows & corr], Me[rows & ~corr]
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2)) + 1e-6
    return (a.mean(0) - b.mean(0)) / sp

def evaluate(sc, te, R, key):
    tp = pp & te
    R[(key, "row")].append(auroc(sc[tp & corr], sc[tp & ~corr]))
    R[(key, "within")].append(within_auc(sc[tp], y[tp], sid[tp]))
    R[(key, "ku")].append(auroc(sc[te & isK5], sc[te & isU]))
    for kk in (1, 2, 3, 4):
        m = tp & (k == kk); R[(key, kk)].append(auroc(sc[m & corr], sc[m & ~corr]))
    s = sc[tp]; yy = y[tp]; hi, lo = np.quantile(s, 0.9), np.quantile(s, 0.1)
    R[(key, "top10")].append(yy[s >= hi].mean()); R[(key, "bot10")].append(1 - yy[s <= lo].mean())

R = collections.defaultdict(list); single = collections.defaultdict(list); chosen = collections.defaultdict(collections.Counter)
for seed in range(SPLITS):
    rng = np.random.default_rng(seed)
    trs = set(rng.permutation(subjects)[: int(0.6 * len(subjects))])
    tr = np.array([s in trs for s in sub]); te = ~tr; ftr = np.array([s in trs for s in f_sub])
    t = tstat(ftr); d = cohend(tr & pp); sd = Me[tr & part].std(0) + 1e-6
    boots = collections.Counter()
    trf_idx = np.where(ftr)[0]
    for b in range(20):
        half = set(paired[np.random.default_rng(1000 * seed + b).choice(trf_idx, len(trf_idx) // 2, replace=False)])
        db = cohend(tr & pp & np.isin(sid, list(half)))
        boots.update(np.argsort(-np.abs(db))[:20].tolist())
    stable = np.array([j for j, _ in boots.most_common()])
    for n in (5, 10, 20):
        sels = {"A 같은 사실 안 t (기존)": np.argsort(-np.abs(t))[:n],
                "B 행 단위 효과크기 d": np.argsort(-np.abs(d))[:n],
                "C t 상위100 중 d 상위": (lambda c: c[np.argsort(-np.abs(d[c]))[:n]])(np.argsort(-np.abs(t))[:100]),
                "D 안정적인 d (재표집 20회)": stable[:n]}
        for name, sel in sels.items():
            sgn = np.sign(t[sel]) if name.startswith("A") else np.sign(d[sel])
            sc = Me[:, sel] @ (sgn / sd[sel])
            evaluate(sc, te, R, (name, n))
            if n == 20: chosen[name].update(el[sel].tolist())
    # single latents: test-row AUROC of the 10 best by train |d|
    for j in np.argsort(-np.abs(d))[:10]:
        tp = pp & te; s = Me[:, j] * np.sign(d[j])
        on = Me[tp, j] > 0
        single[int(el[j])].append((auroc(s[tp & corr], s[tp & ~corr]), on.mean(), y[tp][on].mean() if on.any() else np.nan))

print(f"\n테스트 주어만, 분할 {SPLITS}회 평균 (± 편차). 행 단위 AUROC = 사실을 섞은 상태에서 질문 하나 판정력")
print(f"  {'선택 방식':<26}{'n':>3}{'행 단위':>12}{'1/5':>7}{'2/5':>7}{'3/5':>7}{'4/5':>7}{'같은 사실 안':>12}{'0/5vs5/5':>10}{'상위10% 정답':>12}{'하위10% 오답':>12}")
for n in (5, 10, 20):
    for name in ("A 같은 사실 안 t (기존)", "B 행 단위 효과크기 d", "C t 상위100 중 d 상위", "D 안정적인 d (재표집 20회)"):
        key = (name, n); g = lambda x: np.mean(R[(key, x)])
        print(f"  {name:<26}{n:>3}{g('row'):>8.3f}±{np.std(R[(key,'row')]):.2f}{g(1):>7.3f}{g(2):>7.3f}{g(3):>7.3f}{g(4):>7.3f}"
              f"{g('within'):>12.3f}{g('ku'):>10.3f}{g('top10'):>12.1%}{g('bot10'):>12.1%}")
print("\n단일 피처 (학습 주어 |d| 상위 10개에 자주 든 것): 테스트 행 AUROC, 켜지는 비율, 켜졌을 때 정답 비율")
rows = sorted(((lat, v) for lat, v in single.items() if len(v) >= 5), key=lambda x: -np.mean([a for a, _, _ in x[1]]))
for lat, v in rows[:10]:
    a = np.array(v, float)
    print(f"  피처 {lat:>6}: 선택 {len(v):>2}/10회, AUROC {np.nanmean(a[:,0]):.3f}, 켜짐 {np.nanmean(a[:,1]):.1%}, 켜졌을 때 정답 {np.nanmean(a[:,2]):.1%}")
for name, c in chosen.items():
    print(f"  {name}: 20개에 자주 든 피처 {c.most_common(8)}")
