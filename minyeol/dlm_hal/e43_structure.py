import collections, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from common import rows, ty, k, fid, mid, grp, load_X, NF, OUT
X, _ = load_X(); M = np.where(mid & (ty != "abstain"))[0]; corr = ty == "correct"; rel = np.array([r["relation"] for r in rows])
mu = X.mean(0); Xc = X - mu; Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
byf = collections.defaultdict(list)
for i in M: byf[fid[i]].append(int(i))
byf = {f: v for f, v in byf.items() if len(v) >= 2}; fl = list(byf); rng = np.random.default_rng(0); O = {}
cd = lambda a, b: float(1 - Xn[a] @ Xn[b])
# 1) 쌍 유형별 cosine 거리
P = collections.defaultdict(list)
for f, v in byf.items():
    for a in range(len(v)):
        for b in range(a + 1, len(v)):
            i, j = v[a], v[b]; t = "맞힘-맞힘" if corr[i] and corr[j] else ("틀림-틀림" if (not corr[i]) and (not corr[j]) else "맞힘-틀림"); P[t].append(cd(i, j))
byrel = collections.defaultdict(list)
for f in fl: byrel[rel[byf[f][0]]].append(f)
cross = []
for r_, fs in byrel.items():
    if len(fs) < 2: continue
    for _ in range(min(4000, 40 * len(fs))):
        a, b = rng.choice(len(fs), 2, replace=False); i = int(rng.choice(byf[fs[a]])); j = int(rng.choice(byf[fs[b]])); cross.append(cd(i, j))
O["1_쌍별 평균 cosine 거리"] = {**{t: [round(float(np.mean(v)), 4), len(v)] for t, v in P.items()}, "같은 관계 다른 사실": [round(float(np.mean(cross)), 4), len(cross)]}
# 분산 분해
Xm = Xc[M]; fm = {f: Xc[v].mean(0) for f, v in byf.items()}
sst = float((Xc[M] ** 2).sum()); ssw = float(sum(((Xc[v] - fm[f]) ** 2).sum() for f, v in byf.items())); O["1_사실 정체가 설명하는 분산 비율 (R2)"] = round(1 - ssw / sum(((Xc[v]) ** 2).sum() for v in byf.values()), 3)
# 2) k 별 사실 평균 상태 사이 거리 (같은 관계 쌍)
kf = {f: int(k[byf[f][0]]) for f in fl}; Fn = {f: fm[f] / np.linalg.norm(fm[f]) for f in fl}; D = {}
for a in range(1, 5):
    for b in range(a, 5):
        vals = []
        for r_, fs in byrel.items():
            A = [f for f in fs if kf[f] == a]; B = [f for f in fs if kf[f] == b]
            for _ in range(300):
                if not A or not B: break
                f1, f2 = rng.choice(A), rng.choice(B)
                if f1 != f2: vals.append(1 - float(Fn[f1] @ Fn[f2]))
        D[f"k={a} 대 k={b}"] = [round(float(np.mean(vals)), 4), len(vals)]
O["2_같은 관계 사실 평균 사이 cosine 거리 (k 별)"] = D
# 사실 내 (맞힘-틀림) 방향이 k 수준 사이에서 공유되는가
dirs = collections.defaultdict(list)
for f, v in byf.items():
    c = [i for i in v if corr[i]]; w = [i for i in v if not corr[i]]
    if c and w: dirs[kf[f]].append(Xc[c].mean(0) - Xc[w].mean(0))
mk = {a: (lambda m: m / np.linalg.norm(m))(np.mean(dirs[a], 0)) for a in dirs}
O["2_(맞힘-틀림) 평균 방향의 k 사이 cosine"] = {f"k={a} 대 k={b}": round(float(mk[a] @ mk[b]), 3) for a in sorted(mk) for b in sorted(mk) if a < b}
O["2_k 별 (맞힘-틀림) 방향 크기 평균 ‖d_f‖"] = {int(a): round(float(np.mean([np.linalg.norm(d) for d in dirs[a]])), 2) for a in sorted(dirs)}
# 3) 사실 수준 대 문구 수준
loo = np.array([(sum(corr[j] for j in byf[fid[i]]) - corr[i]) / (len(byf[fid[i]]) - 1) for i in M if fid[i] in byf]); Mi = np.array([i for i in M if fid[i] in byf]); y = corr[Mi]
O["3_LOO-k (다른 문구 정답 비율) 로 문구 정오 AUROC"] = round(float(roc_auc_score(y, loo)), 3)
gf = np.array([grp[byf[f][0]] for f in fl]); K = np.array([kf[f] for f in fl]); FM = np.stack([fm[f] for f in fl]); auc_k = []
sc = np.zeros(len(fl))
for g in range(NF):
    tr, te = gf != g, gf == g; mu_, sd_ = FM[tr].mean(0), FM[tr].std(0) + 1e-6
    clf = LogisticRegression(C=0.003, max_iter=500).fit((FM[tr] - mu_) / sd_, (K[tr] >= 3).astype(int)); sc[te] = clf.decision_function((FM[te] - mu_) / sd_)
O["3_사실 평균 상태로 'k>=3' 예측 AUROC (겹밖)"] = round(float(roc_auc_score(K >= 3, sc)), 3)
Dv = np.stack([Xc[i] - fm[fid[i]] for i in Mi]); gM = grp[Mi]; s2 = np.zeros(len(Mi))
for g in range(NF):
    tr, te = gM != g, gM == g; mu_, sd_ = Dv[tr].mean(0), Dv[tr].std(0) + 1e-6
    clf = LogisticRegression(C=0.003, max_iter=500).fit((Dv[tr] - mu_) / sd_, y[tr].astype(int)); s2[te] = clf.decision_function((Dv[te] - mu_) / sd_)
byfi = collections.defaultdict(list)
for j, i in enumerate(Mi): byfi[fid[i]].append(j)
aa = [roc_auc_score(y[v], s2[v]) for v in byfi.values() if 0 < y[v].sum() < len(v)]
O["3_문구 편차 (사실 평균 뺀 것) 로 같은 사실 안 정오 AUROC (겹밖 probe)"] = round(float(np.mean(aa)), 3)
pk = np.array([k[i] / 5 for i in Mi]); p = pk.mean(); O["3_정오 분산 중 사실 수준 (k) 이 설명하는 몫 (ICC 근사)"] = round(float(np.var(pk) / (p * (1 - p))), 3)
_df = lambda o: float(o) if hasattr(o, "item") else str(o)
print(json.dumps(O, ensure_ascii=False, indent=1, default=_df)); json.dump(O, open(f"{OUT}/e43_structure.json", "w"), ensure_ascii=False, indent=1, default=_df)
