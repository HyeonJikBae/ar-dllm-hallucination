"""E57 (a) (CPU): 같은 사실 안 probe (E48 층 20, step 0, 프롬프트 끝, PCA64 + 로지스틱) 대 확신·어휘 기준, 잔차 probe, 셀 선택 점검.
행 = E48 과 같음 (E7 행 중 1~4/5, E7 채점 abstain 제외, E7 채점이 섞인 사실).  정오 = E7 채점.  겹 = 주어 5 겹 (grp).
기준 (모두 사실 평균을 뺀 값):
  확신 conf: E7 step 0 칸 0 1 위 log-prob, 1·2 위 margin (라벨 없음).
  어휘 lex: 원래 템플릿과 단어 Jaccard, 단어 수, 글자 수, 문구 판 (v1~v4 더미), 의문사 더미 (E40).
  금 gold (금 라벨 사용, 더 엄격): E44 step 0 칸 0 금 첫 토큰 log2(순위+1), 금 로짓 − 1 위 로짓.
  qwen (부모): 같은 문구 Qwen 금 첫 토큰 log-prob, Qwen 1 위 log-prob, Qwen 정오.
잔차 probe: (1) 점수 잔차: 학습 겹 안의 안쪽 겹밖 probe 점수를 기준에 OLS 로 회귀한 계수로 시험 겹 점수에서 기준 몫을 뺌 (주 지표, H57.1).
            (2) 특징 잔차: PCA64 성분에서 기준으로 설명되는 몫을 학습 겹 OLS 로 빼고 다시 로지스틱.
결합: 로지스틱 [기준, 안쪽 probe 점수] 대 기준 단독의 AUROC 차이."""
import collections, json, sys, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from e57_common import *

t0 = time.time()
cfg = json.load(open(f"{D7}/config.json"))
J = np.array([j for j, r in enumerate(R7) if int(r) in E7T and 1 <= k[int(r)] <= 4 and E7T[int(r)] != "abstain"])
RR = R7[J]; corr = np.array([E7T[int(r)] == "correct" for r in RR]); ff = fid[RR]; gg = grp[RR]; kk = k[RR]
byf = collections.defaultdict(list)
for t, f in enumerate(ff): byf[f].append(t)
mixed = {f: v for f, v in byf.items() if 0 < corr[v].sum() < len(v)}
keep = np.array(sorted(t for v in mixed.values() for t in v)); J, RR, corr, ff, gg, kk = J[keep], RR[keep], corr[keep], ff[keep], gg[keep], kk[keep]
byf = collections.defaultdict(list)
for t, f in enumerate(ff): byf[f].append(t)
hi_f = {f: v for f, v in byf.items() if kk[v[0]] >= 3}
n = len(RR); print("행", n, "사실", len(byf), "AT-high", len(hi_f), flush=True)

def center(X):
    X = np.array(X, dtype=np.float64, copy=True)
    for v in byf.values(): X[v] -= X[v].mean(0)
    return X
def lr(Ztr, ytr, Zte, C=0.01):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    return LogisticRegression(C=C, max_iter=5000).fit((Ztr - mu) / sd, ytr).decision_function((Zte - mu) / sd)
def ols(Btr, Ytr):
    A = np.c_[np.ones(len(Btr)), Btr]; return np.linalg.lstsq(A, Ytr, rcond=None)[0]
def ols_res(W, B, Y): return Y - np.c_[np.ones(len(B)), B] @ W

# ---------------------------------------------------------------- 기준 특징
r_ = RR.astype(int)
LP7 = np.load(f"{D7}/lp.npy", mmap_mode="r")[:, 0, 0].astype(float); MG7 = np.load(f"{D7}/mg.npy", mmap_mode="r")[:, 0, 0].astype(float)
lp0 = np.array([LP7[W7[i]] for i in r_]); mg0 = np.array([MG7[W7[i]] for i in r_])
jac, wc, cc, wh = lexical()
VERS = ["v1", "v2", "v3", "v4"]; WHS = ["what", "who", "where", "which", "when", "in", "how", "whose"]
lex = np.c_[jac[r_], wc[r_], cc[r_], np.array([[ver[i] == v for v in VERS] for i in r_], float), np.array([[wh[i] == w for w in WHS] for i in r_], float)]
gold = np.c_[np.log2(np.array([E44[i]["rank0"] for i in r_]) + 1.0), np.array([E44[i]["gap0"] for i in r_])]
qw = np.c_[np.array([qwen(i, "gold_lp") for i in r_]), np.array([qwen(i, "top1_lp") for i in r_]), q_corr[r_].astype(float)]
assert not np.isnan(qw).any() and not np.isnan(gold).any()
conf = np.c_[lp0, mg0]
SETS = {"conf": conf, "lex": lex, "conf+lex": np.c_[conf, lex], "conf+lex+gold": np.c_[conf, lex, gold], "conf+lex+gold+qwen": np.c_[conf, lex, gold, qw]}
SETS = {kx: center(v) for kx, v in SETS.items()}
SINGLE = {"step0 칸0 logprob": lp0, "step0 칸0 margin": mg0, "템플릿 Jaccard": jac[r_], "단어 수": wc[r_], "글자 수": cc[r_], "원문 여부 (question=1)": (ver[r_] == "question").astype(float),
          "금 첫 토큰 −log2(순위+1) (step0)": -gold[:, 0], "금 로짓 − 1 위 (step0)": gold[:, 1], "Qwen 금 첫 토큰 logprob": qw[:, 0], "Qwen 1 위 logprob": qw[:, 1], "Qwen 정오": qw[:, 2]}

# ---------------------------------------------------------------- probe (E48 셀 재현) + 안쪽 겹밖 점수
H = np.load(f"{D7}/hid.npy", mmap_mode="r"); LI = cfg["layers"].index(20); SI = cfg["steps"].index(0); PI = cfg["positions"].index("prompt_end")
Xc = center(np.asarray(H[J, SI, LI, PI], dtype=np.float32)).astype(np.float32); print(f"X 읽음 {time.time() - t0:.0f}s", flush=True)
probe = np.full(n, np.nan); inner = {}; Zall = {}
for g in range(NF):
    tr, te = np.where(gg != g)[0], np.where(gg == g)[0]
    p = PCA(1024, svd_solver="randomized", random_state=0).fit(Xc[tr]); Ztr, Zte = p.transform(Xc[tr])[:, :64], p.transform(Xc[te])[:, :64]
    probe[te] = lr(Ztr, corr[tr], Zte); Zall[g] = (Ztr, Zte)
    s_in = np.full(len(tr), np.nan)
    for h in sorted(set(gg[tr]) - {g}):
        a, b = np.where(gg[tr] != h)[0], np.where(gg[tr] == h)[0]
        pi_ = PCA(64, svd_solver="randomized", random_state=0).fit(Xc[tr][a]); s_in[b] = lr(pi_.transform(Xc[tr][a]), corr[tr][a], pi_.transform(Xc[tr][b]))
    inner[g] = s_in; print(f"겹 {g} {time.time() - t0:.0f}s", flush=True)
E48S = np.load(f"{OUT}/e48_scores_L20.npy")[SI, PI, 3]
sanity = {"E48 점수와 상관": r4(np.corrcoef(probe, E48S)[0, 1]), "최대 절대 차": r4(np.abs(probe - E48S).max())}
print("sanity", sanity, flush=True)

# ---------------------------------------------------------------- 같은 사실 AUROC
def per_fact(sc, facts=None):
    facts = byf if facts is None else facts
    return {f: roc_auc_score(corr[v], sc[v]) for f, v in facts.items()}
FL = sorted(byf); FH = sorted(hi_f)
def summ(sc, facts=None, B=2000, seed=0):
    pf = per_fact(sc, facts); fs = sorted(pf); a = np.array([pf[f] for f in fs]); rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    return [r4(a.mean()), r4(np.percentile(bs, 2.5)), r4(np.percentile(bs, 97.5))]
def diff(sa, sb, facts=None, B=2000, seed=1):
    pa, pb = per_fact(sa, facts), per_fact(sb, facts); fs = sorted(pa); d = np.array([pa[f] - pb[f] for f in fs]); rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [r4(d.mean()), r4(np.percentile(bs, 2.5)), r4(np.percentile(bs, 97.5))]

O = {"행": int(n), "사실": len(byf), "AT-high 사실": len(hi_f), "sanity (E48 L20 s0 프롬프트끝 PCA64 재현)": sanity}
O["probe (E48 셀)"] = {"전체": summ(probe), "AT-high": summ(probe, hi_f)}
O["단일 기준 같은 사실 AUROC (원래 부호, 맞힘 쪽 큼 >0.5)"] = {nm: summ(v) for nm, v in SINGLE.items()}
pc = center(probe[:, None])[:, 0]
O["probe 와 기준의 같은 사실 상관 (사실 평균 제거 뒤 Pearson)"] = {nm: r4(np.corrcoef(pc, center(v[:, None])[:, 0])[0, 1]) for nm, v in SINGLE.items()}
RES = {}
for nm, Bc in SETS.items():
    sb = np.full(n, np.nan); s_res = np.full(n, np.nan); s_fres = np.full(n, np.nan); s_comb = np.full(n, np.nan); r2 = []
    for g in range(NF):
        tr, te = np.where(gg != g)[0], np.where(gg == g)[0]
        mu, sd = Bc[tr].mean(0), Bc[tr].std(0) + 1e-9; Btr, Bte = (Bc[tr] - mu) / sd, (Bc[te] - mu) / sd
        sb[te] = lr(Btr, corr[tr], Bte, C=1.0)
        W = ols(Btr, inner[g]); s_res[te] = ols_res(W, Bte, probe[te])
        yhat = np.c_[np.ones(len(Btr)), Btr] @ W; r2.append(1 - np.var(inner[g] - yhat) / np.var(inner[g]))
        Ztr, Zte = Zall[g]; WZ = ols(Btr, Ztr); s_fres[te] = lr(ols_res(WZ, Btr, Ztr), corr[tr], ols_res(WZ, Bte, Zte))
        s_comb[te] = lr(np.c_[Btr, inner[g]], corr[tr], np.c_[Bte, probe[te]], C=1.0)
    RES[nm] = {"기준 단독 (로지스틱)": summ(sb), "점수 잔차 probe": summ(s_res), "점수 잔차 probe AT-high": summ(s_res, hi_f), "특징 잔차 probe": summ(s_fres),
               "기준+probe 결합": summ(s_comb), "결합 − 기준 단독": diff(s_comb, sb), "probe − 기준 단독": diff(probe, sb),
               "안쪽 probe 점수 분산 중 기준 설명 몫 R2 (겹 평균)": r4(np.mean(r2))}
    print(nm, RES[nm], flush=True)
O["기준 집합별"] = RES
O["H57.1"] = {"정의": "확신·어휘 (conf+lex) 를 뺀 점수 잔차 probe 의 같은 사실 AUROC >= 0.60", "값": RES["conf+lex"]["점수 잔차 probe"],
              "판정": bool(RES["conf+lex"]["점수 잔차 probe"][0] >= 0.60)}

# ---------------------------------------------------------------- 셀 선택 점검 (사실 id md5 반반)
hf = {f: half[np.where(fid == f)[0][0]] for f in byf}
A_f = {f: v for f, v in byf.items() if hf[f] == 0}; B_f = {f: v for f, v in byf.items() if hf[f] == 1}
METH = ["DM1", "PCA4", "PCA16", "PCA64", "PCA256", "PCA1024", "RAW16", "RAW64", "RAW256"]
cells = {}
for L in cfg["layers"]:
    S = np.load(f"{OUT}/e48_scores_L{L}.npy")
    for si, s in enumerate(cfg["steps"]):
        for pi, pos in enumerate(cfg["positions"]):
            for mi, m in enumerate(METH):
                sc = S[si, pi, mi]; cells[(L, s, pos, m)] = (np.mean(list(per_fact(sc, A_f).values())), np.mean(list(per_fact(sc, B_f).values())))
def sel(pred, a_i, b_i):
    C = {c: v for c, v in cells.items() if pred(c)}; best = max(C, key=lambda c: C[c][a_i]); rank_b = sorted(C, key=lambda c: -C[c][b_i]).index(best) + 1
    return {"고른 셀": "|".join(map(str, best)), "고른 반 AUROC": r4(C[best][a_i]), "다른 반 AUROC": r4(C[best][b_i]), "다른 반에서 그 셀 순위": f"{rank_b}/{len(C)}",
            "다른 반 최고": r4(max(v[b_i] for v in C.values())), "낙관 (고른 반 − 다른 반)": r4(C[best][a_i] - C[best][b_i])}
pre = (20, 0, "prompt_end", "PCA64")
O["셀 선택 점검"] = {"반 크기 (사실)": [len(A_f), len(B_f)],
    "모든 셀 (7 층 x 3 step x 3 위치 x 9 방법)": {"A 에서 고르고 B 평가": sel(lambda c: True, 0, 1), "B 에서 고르고 A 평가": sel(lambda c: True, 1, 0)},
    "step 0 셀만": {"A 에서 고르고 B 평가": sel(lambda c: c[1] == 0, 0, 1), "B 에서 고르고 A 평가": sel(lambda c: c[1] == 0, 1, 0)},
    "미리 정한 셀 L20 s0 프롬프트끝 PCA64": {"A": r4(cells[pre][0]), "B": r4(cells[pre][1])}}
O["걸린 시간 s"] = int(time.time() - t0)
upd_summary("a_probe", O); np.save(f"{OUT}/e57_probe_scores.npy", np.c_[RR, corr, probe])
print(json.dumps({x: O[x] for x in ("sanity (E48 L20 s0 프롬프트끝 PCA64 재현)", "probe (E48 셀)", "H57.1", "셀 선택 점검")}, ensure_ascii=False, indent=1)); print("DONE")
