"""E17 (PREREG.md, CPU): 차원 축소 대 확대.  층 20 프롬프트 끝 step 0 상태 X 에서 같은 사실 안 정답-오답 신호.
축소: D_f (사실별 정답-오답 차이) 의 PCA, X 의 상위 PC 제거, 수축 LDA, 관계별 방향 SVD.  확대: 층 20 SAE (sae/dream_pilot_200M).
방향은 주어 5 겹 밖에서 만들고 안에서 같은 사실 AUROC (사실마다 AUROC 평균).  usage: python followup_mk/e17_dims.py"""
import collections, json, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.covariance import ledoit_wolf_shrinkage
sys.path.insert(0, "followup_mk"); sys.path.insert(0, "analysis")
from common import rows, grp, ty, fid, mid, unit, load_X, NF, OUT
import sae_io
torch.set_num_threads(24); T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
X, norm = load_X(); O = {}; rel = np.array([r["relation"] for r in rows]); lab = ty == "correct"
idx = np.where(mid & (ty != "abstain"))[0]
byf = collections.defaultdict(list)
for i in idx: byf[fid[i]].append(int(i))
facts = [f for f, v in byf.items() if 0 < lab[v].sum() < len(v)]; fgrp = {f: int(grp[byf[f][0]]) for f in facts}
assert all(len(set(grp[byf[f]])) == 1 for f in facts)
FC = {f: ([i for i in byf[f] if lab[i]], [i for i in byf[f] if not lab[i]]) for f in facts}
O["표본"] = dict(행=int(len(idx)), 두class사실=len(facts)); log(O["표본"])
def fdiff(fs, Z): return np.stack([Z[FC[f][0]].mean(0) - Z[FC[f][1]].mean(0) for f in fs])
TRF = {g: [f for f in facts if fgrp[f] != g] for g in range(NF)}; TEF = {g: [f for f in facts if fgrp[f] == g] for g in range(NF)}
Dtr = {g: fdiff(TRF[g], X) for g in range(NF)}; Bg = {g: Dtr[g].mean(0) for g in range(NF)}
def fact_aucs(score): return np.array([roc_auc_score(lab[byf[f]], score[byf[f]]) for f in facts])
def evaluate(make_v, Z=X):
    sc = np.zeros(len(Z), np.float64)
    for g in range(NF):
        v = make_v(g); te = idx[grp[idx] == g]; sc[te] = Z[te] @ v
    return fact_aucs(sc)
rng = np.random.default_rng(0); BOOT = rng.integers(0, len(facts), (2000, len(facts)))
def cmp(a, base):
    d = a - base; bs = d[BOOT].mean(1)
    return dict(AUROC=round(float(a.mean()), 4), 차이_대B=round(float(d.mean()), 4), CI=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
A = {}; A["B"] = evaluate(lambda g: Bg[g]); log("B", round(float(A["B"].mean()), 4))
DIRS = collections.defaultdict(dict); DIRS["B"] = {g: unit(Bg[g]) for g in range(NF)}
# ---------------- 축소 1: D 의 PCA
Dall = fdiff(facts, X); Dc = Dall - Dall.mean(0)
s = np.linalg.svd(Dc, compute_uv=False); ev = s ** 2 / (s ** 2).sum(); cum = np.cumsum(ev)
su = np.linalg.svd(Dall, compute_uv=False); evu = su ** 2 / (su ** 2).sum(); cumu = np.cumsum(evu)
O["D_PCA"] = dict(사실수=len(facts), 분산50_성분수=int(np.searchsorted(cum, .5) + 1), 분산90_성분수=int(np.searchsorted(cum, .9) + 1), 상위1_10_몫=[round(float(cum[0]), 4), round(float(cum[9]), 4)],
                  비중심_50_90=[int(np.searchsorted(cumu, .5) + 1), int(np.searchsorted(cumu, .9) + 1)],
                  평균차이B_가_D분산에서_차지=round(float(len(facts) * (Dall.mean(0) ** 2).sum() / (Dall ** 2).sum()), 4))
log("D PCA", O["D_PCA"])
UD = {}
for g in range(NF):
    D_ = Dtr[g] - Dtr[g].mean(0); _, _, Vt = np.linalg.svd(D_, full_matrices=False); UD[g] = Vt[:256]
for r in (1, 4, 16, 64, 256):
    A[f"D_PC{r} 제한"] = evaluate(lambda g, r=r: UD[g][:r].T @ (UD[g][:r] @ Bg[g])); log(f"D_PC{r}", round(float(A[f'D_PC{r} 제한'].mean()), 4))
    DIRS[f"D_PC{r}"] = {g: unit(UD[g][:r].T @ (UD[g][:r] @ Bg[g])) for g in range(NF)}
# ---------------- 축소 2: X 의 상위 PC 제거 (라벨 없이, 학습 겹 행에서)
UX = {}
for g in range(NF):
    tr = np.where(grp != g)[0]; sub = np.random.default_rng(g).choice(tr, 20000, replace=False); Z = torch.from_numpy(X[sub] - X[sub].mean(0))
    _, _, V = torch.pca_lowrank(Z, q=210, center=False, niter=4); UX[g] = V[:, :200].numpy().T
for k_ in (10, 50, 200):
    A[f"X_PC{k_} 제거"] = evaluate(lambda g, k_=k_: Bg[g] - UX[g][:k_].T @ (UX[g][:k_] @ Bg[g])); log(f"X_PC{k_} 제거", round(float(A[f'X_PC{k_} 제거'].mean()), 4))
    DIRS[f"XPC{k_}제거"] = {g: unit(Bg[g] - UX[g][:k_].T @ (UX[g][:k_] @ Bg[g])) for g in range(NF)}
O["X_상위PC에_B_몫"] = {k_: round(float(np.mean([np.sum((UX[g][:k_] @ unit(Bg[g])) ** 2) for g in range(NF)])), 4) for k_ in (10, 50, 200)}
# ---------------- 축소 3: 수축 LDA  w = Sigma_w^-1 B
allr = np.arange(len(X)); gk = collections.defaultdict(list)
for i in allr: gk[(fid[i], bool(lab[i]))].append(i)
def resid(rows_):
    R = np.empty((len(rows_), X.shape[1]), np.float32); j = 0
    gg = collections.defaultdict(list)
    for i in rows_: gg[(fid[i], bool(lab[i]))].append(i)
    for v in gg.values():
        if len(v) < 2: continue
        R[j:j + len(v)] = X[v] - X[v].mean(0); j += len(v)
    return R[:j]
cosw = collections.defaultdict(list); LWs = {}
for var in ("mid", "전체"):
    W = collections.defaultdict(dict)
    for g in range(NF):
        rr = idx[grp[idx] != g] if var == "mid" else allr[grp != g]; R = resid(rr); n = len(R)
        Rt = torch.from_numpy(R).double(); S = (Rt.T @ Rt / n).numpy(); mu = np.trace(S) / S.shape[0]
        lw = float(ledoit_wolf_shrinkage(R[np.random.default_rng(g).choice(n, min(n, 20000), replace=False)], assume_centered=True, block_size=1000)); LWs[(var, g)] = lw
        for lam in (0.1, 0.5, 0.9, "LW"):
            l_ = lw if lam == "LW" else lam; Sl = (1 - l_) * S; Sl[np.diag_indices_from(Sl)] += l_ * mu
            w = np.linalg.solve(Sl, Bg[g]); W[lam][g] = w; cosw[f"LDA_{var}_{lam}"].append(float(unit(w) @ unit(Bg[g])))
        log(f"LDA {var} 겹 {g} n={n} LW={lw:.3f}")
    for lam in (0.1, 0.5, 0.9, "LW"):
        nm = f"LDA_{var}_{lam}"; A[nm] = evaluate(lambda g, W=W, lam=lam: W[lam][g]); DIRS[nm] = {g: unit(W[lam][g]) for g in range(NF)}; log(nm, round(float(A[nm].mean()), 4), "cos(w,B)", round(float(np.mean(cosw[nm])), 3))
O["LDA_cos_wB"] = {k_: round(float(np.mean(v)), 3) for k_, v in cosw.items()}; O["LW_수축계수"] = {f"{a}_g{b}": round(v, 3) for (a, b), v in LWs.items()}
# ---------------- 축소 4: 관계별 방향
Brel_all = {}; byr = collections.defaultdict(list)
for f in facts: byr[rel[byf[f][0]]].append(f)
for r_, fs in byr.items():
    if len(fs) >= 15: Brel_all[r_] = unit(fdiff(fs, X).mean(0))
names = sorted(Brel_all); M = np.stack([Brel_all[r_] for r_ in names]); sm = np.linalg.svd(M, compute_uv=False); evm = sm ** 2 / (sm ** 2).sum()
Bu = unit(Dall.mean(0)); cr = np.array([Brel_all[r_] @ Bu for r_ in names])
O["관계방향"] = dict(관계수=len(names), 첫성분몫=round(float(evm[0]), 3), 상위3몫=round(float(evm[:3].sum()), 3), 유효랭크=round(float(np.exp(-(evm * np.log(evm + 1e-12)).sum())), 2),
                  cos_B=dict(최소=round(float(cr.min()), 3), 중앙=round(float(np.median(cr)), 3), 최대=round(float(cr.max()), 3)))
log("관계방향", O["관계방향"])
rel_res = collections.defaultdict(list)    # 같은 사실 집합에서 B, Brel, Brel⊥B
for g in range(NF):
    br = collections.defaultdict(list)
    for f in TRF[g]: br[rel[byf[f][0]]].append(f)
    Bh = unit(Bg[g])
    for r_, fs in br.items():
        if len(fs) < 15: continue
        Br = unit(fdiff(fs, X).mean(0)); Bp = Br - (Br @ Bh) * Bh
        for f in TEF[g]:
            if rel[byf[f][0]] != r_: continue
            v_ = byf[f]; y = lab[v_]
            rel_res["B"].append(roc_auc_score(y, X[v_] @ Bh)); rel_res["Brel"].append(roc_auc_score(y, X[v_] @ Br)); rel_res["Brel⊥B"].append(roc_auc_score(y, X[v_] @ Bp)); rel_res["rel"].append(r_)
rb = np.array(rel_res["B"]); rbs = np.random.default_rng(1).integers(0, len(rb), (2000, len(rb)))
O["관계별_같은사실AUROC"] = {"사실수": len(rb), **{k_: dict(AUROC=round(float(np.mean(rel_res[k_])), 4), 차이_대B=round(float(np.mean(np.array(rel_res[k_]) - rb)), 4),
                         CI=[round(float(np.percentile((np.array(rel_res[k_]) - rb)[rbs].mean(1), 2.5)), 4), round(float(np.percentile((np.array(rel_res[k_]) - rb)[rbs].mean(1), 97.5)), 4)]) for k_ in ("Brel", "Brel⊥B")}, "B": round(float(rb.mean()), 4)}
log("관계별", O["관계별_같은사실AUROC"])
# ---------------- 확대: SAE
SAEs = {"t0.5": os.path.join(sae_io.SAE_ROOT, "dream_pilot_200M", "dream-7b_L20_t0.5_s2_200M"), "shared": sae_io.DREAM}
O["SAE"] = {}
for snm, path in SAEs.items():
    cfg = json.load(open(os.path.join(path, "cfg.json"))); w = sae_io._load(path); enc = sae_io._encoder(w, cfg.get("apply_b_dec_to_input", True)); Wd = w["W_dec"].numpy()
    F = np.zeros((len(X), Wd.shape[0]), np.float32)
    for b0 in range(0, len(idx), 1024):
        ii = idx[b0:b0 + 1024]; F[ii] = enc(torch.from_numpy(X[ii])).numpy()
    xs = idx[:3000]; Xh = F[xs] @ Wd + w["b_dec"].numpy(); fvu = float(((X[xs] - Xh) ** 2).sum() / ((X[xs] - X[xs].mean(0)) ** 2).sum())
    l0 = float((F[idx] > 0).sum(1).mean()); cosrec = float(np.mean(np.sum(X[xs] * Xh, 1) / (np.linalg.norm(X[xs], axis=1) * np.linalg.norm(Xh, axis=1))))
    S_ = dict(설명분산=round(1 - fvu, 4), 재구성cos=round(cosrec, 4), 평균L0=round(l0, 1)); log("SAE", snm, S_)
    DF = fdiff(facts, F); tF = DF.mean(0) / (DF.std(0) / np.sqrt(len(DF)) + 1e-9); S_["|t|>5_특징수"] = int((np.abs(tF) > 5).sum()); S_["|t|>3_특징수"] = int((np.abs(tF) > 3).sum())
    S_["켜진적있는_특징수(이 행들)"] = int((F[idx] > 0).any(0).sum())
    DFg = {}; tg = {}
    for g in range(NF):
        D_ = fdiff(TRF[g], F); DFg[g] = D_.mean(0); tg[g] = DFg[g] / (D_.std(0) / np.sqrt(len(D_)) + 1e-9)
    res = {}
    for k_ in (1, 4, 16, 64, 16384):
        def mk(g, k_=k_):
            top = np.argsort(-np.abs(tg[g]))[:k_]; v = np.zeros(F.shape[1], np.float32); v[top] = DFg[g][top]; return v
        a = evaluate(mk, F); res[f"특징상위{k_}"] = cmp(a, A["B"])
        if k_ <= 64:
            dv = {}
            for g in range(NF):
                top = np.argsort(-np.abs(tg[g]))[:k_]; dv[g] = unit(DFg[g][top] @ Wd[top])
            res[f"특징상위{k_}_디코더방향(잔차공간)"] = cmp(evaluate(lambda g: dv[g]), A["B"]); DIRS[f"SAE_{snm}_top{k_}"] = dv
        log("SAE", snm, k_, res[f"특징상위{k_}"])
    vd = {g: DFg[g] @ Wd for g in range(NF)}; S_["평균특징차_디코더방향_cos_B"] = round(float(np.mean([unit(vd[g]) @ unit(Bg[g]) for g in range(NF)])), 3)
    res["평균특징차_디코더방향(잔차공간)"] = cmp(evaluate(lambda g: vd[g]), A["B"])
    # OMP: 단위 B 를 정규화 디코더 행 k 개로
    Dn = Wd / (np.linalg.norm(Wd, axis=1, keepdims=True) + 1e-9); tgt = Bu.astype(np.float64); sel = []; resid_ = tgt.copy(); coss = {}
    for k_ in range(1, 513):
        j = int(np.argmax(np.abs(Dn @ resid_))); sel.append(j); Ms = Dn[sel].T.astype(np.float64); coef, *_ = np.linalg.lstsq(Ms, tgt, rcond=None); appr = Ms @ coef; resid_ = tgt - appr
        c_ = float(appr @ tgt / (np.linalg.norm(appr) + 1e-12)); coss[k_] = c_
        if c_ >= 0.9: break
    S_["OMP_cos"] = {k_: round(coss[k_], 3) for k_ in (1, 4, 16, 64, 128, 256, 512) if k_ in coss}; S_["OMP_cos0.9_필요k"] = (max(coss) if coss[max(coss)] >= 0.9 else f">{max(coss)}")
    S_["OMP_cos0.5_0.7_필요k"] = [next((k_ for k_ in sorted(coss) if coss[k_] >= t_), None) for t_ in (0.5, 0.7)]
    S_["AUROC"] = res; O["SAE"][snm] = S_; log("SAE 요약", snm, json.dumps({k_: v for k_, v in S_.items() if k_ != "AUROC"}, ensure_ascii=False))
    del F
# ---------------- 정리, 판정
O["같은사실_AUROC"] = {nm: cmp(a, A["B"]) for nm, a in A.items()}
red = [nm for nm in A if nm.startswith(("LDA", "D_PC"))]; best = max(red, key=lambda nm: A[nm].mean()); bc = cmp(A[best], A["B"])
main = O["SAE"]["t0.5"]["AUROC"]; small = {k_: main[f"특징상위{k_}"]["AUROC"] for k_ in (1, 4, 16)}
O["판정"] = {"H17.1": dict(참=bool(bc["차이_대B"] >= 0.03), 최선=best, **bc, 메모="여러 변형 중 최선이라 낙관적"),
             "H17.2": dict(참=bool(max(small.values()) >= O["같은사실_AUROC"]["B"]["AUROC"] - 0.01), SAE="t0.5", 상위k_AUROC=small, B=O["같은사실_AUROC"]["B"]["AUROC"],
                           shared_상위k=[O["SAE"]["shared"]["AUROC"][f"특징상위{k_}"]["AUROC"] for k_ in (1, 4, 16)])}
log("판정", json.dumps(O["판정"], ensure_ascii=False))
np.savez(f"{OUT}/e17_dirs.npz", **{nm: np.stack([DIRS[nm][g] for g in range(NF)]).astype(np.float32) for nm in DIRS})
json.dump(O, open(f"{OUT}/e17_summary.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); log("DONE")
