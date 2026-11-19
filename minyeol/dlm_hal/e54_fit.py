"""E54 (PREREG.md) 준비 (CPU): 층 20/24 프롬프트 끝 step 0 상태 (E44, step 0 = step 1 캔버스) 로 겹밖 (주어 5 겹) 부분공간·지우개를 맞춘다.
- 짝 (a): E51 의 select_pairs 와 같은 규칙 (E7 라벨, 사실당 최대 2 짝) -> H54.1 에서 E51 과 같은 짝을 비교.
- 교환 부분공간: 학습 겹 사실별 D_f = 평균(맞힘) - 평균(틀림) (E44 채점) 의 중심 안 뺀 SVD 상위 64 (r = 1,4,16,64 는 앞부분).
  무작위 대조: 고정 시드 무작위 정규직교 64 (앞부분).
- LEACE rank 1 (Belrose et al. 2023): 같은 사실 중심화 상태에서 c = Cov(x, z), w = Sigma^-1 c.  Sigma 는 등방 (수축 1; 아래 '결과 보기 전 변경') -> w = c.
  적용: h' = h - c (w.h - m) / (w.c),  m = 학습 행 w.h 평균  -> w.h' = m 으로 일정 (그 방향의 정보 제거, 평균 보존).
- INLP: 같은 사실 중심화 상태의 상위 512 PC 공간 안에서 (로지스틱 C 0.01) 방향을 찾아 직교 투영으로 빼기를 반복.  rank r* 는 학습 겹 안의
  안쪽 검증 (학습 4 겹 중 1 겹을 돌아가며) 에서 같은 사실 probe (E48 식 PCA64 + 로지스틱) AUROC <= 0.55 가 되는 첫 반복.  최종은 학습 4 겹 전부로 r* 번.
  적용: h' = h - Q Q^T (h - mu)  (평균 보존).  무작위 대조: rank 1 / rank r* 무작위 정규직교 투영 (같은 식).
- 겹밖 확인: 시험 겹 (grp == g) 에서 지운 뒤 새 probe 의 같은 사실 AUROC (학습 겹에서 맞춘 probe).
출력: followup_mk/e54/fit_L{20,24}.npz, followup_mk/e54/pairs.json, followup_mk/e54/fit_summary.json
usage: python followup_mk/e54_fit.py"""
import collections, json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "16"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "16"); os.environ.setdefault("MKL_NUM_THREADS", "16")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.covariance import ledoit_wolf_shrinkage
sys.path.insert(0, "followup_mk")
from common import rows, ty, k, fid, mid, grp, NF, OUT, D_
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
OD = f"{OUT}/e54"; os.makedirs(OD, exist_ok=True)
WRONG = ("entity", "broken")

# ---------------- E51 짝 (e51_canvas.select_pairs 복사; 스크립트가 최상위 코드를 실행하므로 import 하지 않음)
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); w7 = {int(r): j for j, r in enumerate(R7)}
meta7 = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta7[d["row"]] = d
QW = {}
for l in open(f"{D_}/3_classification/full/qwen_final_full.jsonl", encoding="utf-8"):
    d = json.loads(l); QW[(d["sample_id"], d["version"])] = d
def qroute(r):
    q = QW[(rows[r]["sample_id"], rows[r]["version"])]
    return "D" if q["correct"] else ("H" if q["type3"] in WRONG else None)
TAUS = (2, 4)
def select_pairs():
    MK = np.load(f"{D7}/mask.npy", mmap_mode="r"); e = {t: np.asarray(MK[:, t, 0]) for t in TAUS}
    el = lambda r, t: bool(e[t][w7[r]])
    byf = collections.defaultdict(list)
    for r in R7:
        r = int(r)
        if mid[r]: byf[fid[r]].append(r)
    pairs = []
    for f in sorted(byf):
        rng = np.random.default_rng(int(f) % (2 ** 31))
        Sc = [r for r in byf[f] if meta7[r]["new_type"] == "correct" and el(r, 2)]
        if not Sc: continue
        Fc = {ro: [r for r in byf[f] if meta7[r]["new_type"] in WRONG and qroute(r) == ro and el(r, 2)] for ro in "DH"}
        picks = []
        for ro in "DH":
            c = list(rng.permutation(Fc[ro])) if Fc[ro] else []; c.sort(key=lambda r: -el(int(r), 4))
            picks += [(ro, int(r)) for r in c]
        used = collections.Counter()
        for ro, r in picks[:2]:
            sc = [int(s) for s in rng.permutation(Sc)]; sc.sort(key=lambda s: (used[s], -(el(s, 4) and el(r, 4))))
            s = sc[0]; used[s] += 1
            pairs.append(dict(pid=len(pairs), fact=str(f), route=ro, F=r, S=s, rel=rows[r]["relation"]))
    return pairs

if __name__ == "__main__":
    pairs = select_pairs()
    if os.path.exists(f"{OUT}/e51_pairs.json"):
        p51 = json.load(open(f"{OUT}/e51_pairs.json")); same = [(a["F"], a["S"]) for a in p51] == [(a["F"], a["S"]) for a in pairs]
        log("E51 짝 파일과 같은가:", same, len(p51), len(pairs))
    json.dump(pairs, open(f"{OD}/pairs.json", "w"), ensure_ascii=False)
    log(f"짝 {len(pairs)}: {collections.Counter(p['route'] for p in pairs)}, 사실 {len({p['fact'] for p in pairs})}")

    # ---------------- E44 상태와 라벨
    P = []
    for d in sorted(x for x in os.listdir(f"{OUT}/e44") if x.startswith("shard")):
        D = f"{OUT}/e44/{d}"; R = np.load(f"{D}/rows.npy"); done = np.load(f"{D}/done.npy"); meta = {}
        for l in open(f"{D}/meta.jsonl"): m_ = json.loads(l); meta[m_["row"]] = m_
        P.append((D, R, done, meta))
    lay = json.load(open(f"{P[0][0]}/config.json"))["layers"]
    S = {}
    for L_ in (20, 24):
        li = lay.index(L_); X = {}
        for D, R, done, meta in P:
            H = np.load(f"{D}/hid.npy", mmap_mode="r"); J = np.where(done)[0]
            HJ = np.asarray(H[J, 0, li], dtype=np.float32)
            for j, h in zip(J, HJ): X[int(R[j])] = h
        S[L_] = X
    nt = {}
    for D, R, done, meta in P: nt.update({r: meta[r]["new_type"] for r in meta})
    log("E44 상태", {L_: len(v) for L_, v in S.items()})
    # 섞인 사실 (E44 채점, abstain 제외)
    byf = collections.defaultdict(list)
    for r in S[20]:
        if mid[r] and nt[r] != "abstain": byf[fid[r]].append(r)
    facts = sorted(f for f, v in byf.items() if any(nt[r] == "correct" for r in v) and any(nt[r] in WRONG for r in v))
    fg = {f: int(grp[byf[f][0]]) for f in facts}
    log("섞인 사실", len(facts), "행", sum(len(byf[f]) for f in facts))

    def fact_auc(score, rws):
        a = []
        for f in {fid[r] for r in rws}:
            v = [i for i, r in enumerate(rws) if fid[r] == f]; y = np.array([nt[rws[i]] == "correct" for i in v])
            if 0 < y.sum() < len(y): a.append(roc_auc_score(y, score[v]))
        return float(np.mean(a))
    def centered(X, rws):
        Xc = X.copy(); byf_ = collections.defaultdict(list)
        for i, r in enumerate(rws): byf_[fid[r]].append(i)
        for v in byf_.values(): Xc[v] -= Xc[v].mean(0)
        return Xc
    def probe_auc(Xtr, ytr, Xte, rte, npc=64):
        pca = PCA(npc, svd_solver="randomized", random_state=0).fit(Xtr); clf = LogisticRegression(C=0.01, max_iter=2000).fit(pca.transform(Xtr), ytr)
        return fact_auc(clf.decision_function(pca.transform(Xte)), rte)
    def basis(Xc, nb=512):
        """같은 사실 중심화 상태의 상위 nb PC (정규직교 행).  INLP 는 이 공간 안에서 (속도; 신호는 상위 PC 에 있음, E48)."""
        return PCA(nb, svd_solver="randomized", random_state=0).fit(Xc).components_.astype(np.float32)
    def inlp_step(Y, y, it):
        clf = LogisticRegression(C=0.01, max_iter=3000).fit(Y, y); v = clf.coef_[0].astype(np.float32); return v / np.linalg.norm(v)
    def inlp_dirs(Xc, y, n):
        V = basis(Xc); Y = Xc @ V.T; q = np.zeros((0, V.shape[0]), np.float32)
        for it in range(n):
            v = inlp_step(Y, y, it); v -= q.T @ (q @ v); v /= np.linalg.norm(v); q = np.vstack([q, v[None]]); Y = Y - np.outer(Y @ v, v)
        return q @ V
    MAXR = 96
    summ = {"pairs": dict(n=len(pairs), route=dict(collections.Counter(p["route"] for p in pairs))), "facts": len(facts)}
    for L_ in (20, 24):
        X = S[L_]; out = {}; sm = {}
        rng = np.random.default_rng(5400 + L_); Rnd = np.linalg.qr(rng.standard_normal((3584, MAXR)))[0].T.astype(np.float32)   # [MAXR, d] 정규직교
        out["rand"] = Rnd
        for g in range(NF):
            trf = [f for f in facts if fg[f] != g]; tef = [f for f in facts if fg[f] == g]
            rtr = [r for f in trf for r in byf[f]]; rte = [r for f in tef for r in byf[f]]
            Xtr = np.stack([X[r] for r in rtr]); Xte = np.stack([X[r] for r in rte]); ytr = np.array([nt[r] == "correct" for r in rtr]); yte = np.array([nt[r] == "correct" for r in rte])
            # 교환 부분공간
            Df = np.stack([np.mean([X[r] for r in byf[f] if nt[r] == "correct"], 0) - np.mean([X[r] for r in byf[f] if nt[r] in WRONG], 0) for f in trf])
            U = np.linalg.svd(Df, full_matrices=False)[2][:64].astype(np.float32); out[f"U_{g}"] = U
            # LEACE rank 1
            Xc = centered(Xtr, rtr); z = ytr.astype(np.float32); c = (Xc * (z - z.mean())[:, None]).mean(0)
            sh = ledoit_wolf_shrinkage(Xc); Sig = (1 - sh) * np.cov(Xc, rowvar=False) + sh * np.trace(np.cov(Xc, rowvar=False)) / Xc.shape[1] * np.eye(Xc.shape[1])
            w_lw = np.linalg.solve(Sig, c)
            # 결과 보기 전 변경 (2026-11-19): Ledoit-Wolf 공분산 (수축 ~0.005) 의 LEACE 는 겹밖에서 불안정 (층 20 겹밖 probe AUROC 0.43~0.61, 반대 신호).
            # 진단 (scratchpad diag/leace_variants.py) 에서 겹밖 |AUROC-0.5| 가 가장 작은 등방 공분산 (수축 1) 을 쓴다 = 평균 차이 방향의 평균 보존 직교 투영.
            w = c / float(c @ c) * float(c @ c); mm_ = float((Xtr @ w).mean()); out[f"leace_lw_w_{g}"] = w_lw.astype(np.float32)
            out[f"leace_c_{g}"] = c.astype(np.float32); out[f"leace_w_{g}"] = w.astype(np.float32); out[f"leace_m_{g}"] = np.float32(mm_)
            era = lambda Z: Z - np.outer(Z @ w - mm_, c) / float(w @ c)
            # INLP: 안쪽 검증으로 r*
            # 안쪽 검증: 학습 4 겹 중 (g+1)%5 겹을 검증으로 (비용 때문에 한 겹만; 결과 보기 전 결정).  방향을 하나씩 더하며 검증 AUROC <= 0.55 인 첫 rank.
            gv = [x for x in sorted({fg[f] for f in trf}) if x == (g + 1) % NF][0]
            ri = [r for f in trf if fg[f] != gv for r in byf[f]]; rv = [r for f in trf if fg[f] == gv for r in byf[f]]
            Xi = centered(np.stack([X[r] for r in ri]), ri); yi = np.array([nt[r] == "correct" for r in ri]); Xv = centered(np.stack([X[r] for r in rv]), rv)
            Vi = basis(Xi); Yi = Xi @ Vi.T; Yv = Xv @ Vi.T; qi = np.zeros((0, Vi.shape[0]), np.float32); rstar = [MAXR]; trace = []
            for it in range(MAXR):
                v = inlp_step(Yi, yi, it); v -= qi.T @ (qi @ v); v /= np.linalg.norm(v); qi = np.vstack([qi, v[None]])
                Yi = Yi - np.outer(Yi @ v, v); Yv = Yv - np.outer(Yv @ v, v)
                Qf = qi @ Vi; Zi = Xi - (Xi @ Qf.T) @ Qf; Zv = Xv - (Xv @ Qf.T) @ Qf
                a = probe_auc(Zi, yi, Zv, rv); trace.append(round(a, 4))
                if a <= 0.55: rstar = [it + 1]; break
            r_s = int(rstar[0]); Q = inlp_dirs(Xc, ytr, r_s); mu = Xtr.mean(0)
            out[f"inlp_Q_{g}"] = Q; out[f"inlp_mu_{g}"] = mu.astype(np.float32)
            # 겹밖 확인 (시험 겹, 중심화 상태; probe 는 학습 겹에서 맞춤)
            Xtec = centered(Xte, rte)
            proj = lambda Z, B: Z - (Z @ B.T) @ B
            a_base = probe_auc(Xc, ytr, Xtec, rte); a_le = probe_auc(centered(era(Xtr), rtr), ytr, centered(era(Xte), rte), rte)
            a_in = probe_auc(proj(Xc, Q), ytr, proj(Xtec, Q), rte); a_r1 = probe_auc(proj(Xc, Rnd[:1]), ytr, proj(Xtec, Rnd[:1]), rte)
            a_rr = probe_auc(proj(Xc, Rnd[:r_s]), ytr, proj(Xtec, Rnd[:r_s]), rte)
            dm = lambda Z, Zte: fact_auc(Zte @ (Z[ytr].mean(0) - Z[~ytr].mean(0)), rte)
            sm[g] = dict(n_train=len(rtr), n_test=len(rte), inlp_rank=r_s, inlp_inner_trace=trace, probe_auc=dict(base=round(a_base, 4), leace=round(a_le, 4), inlp=round(a_in, 4), rand1=round(a_r1, 4), rand_r=round(a_rr, 4)),
                         dm_auc=dict(base=round(dm(Xc, Xtec), 4), leace=round(dm(centered(era(Xtr), rtr), centered(era(Xte), rte)), 4)),
                         leace_shrink=round(float(sh), 4), leace_move_norm=round(float(np.linalg.norm(np.outer(Xte @ w - mm_, c) / float(w @ c), axis=1).mean()), 3),
                         state_norm=round(float(np.linalg.norm(Xte, axis=1).mean()), 2),
                         U1_cos_B=round(float(abs(U[0] @ Df.mean(0)) / np.linalg.norm(Df.mean(0))), 4))
            log(L_, g, sm[g])
        np.savez(f"{OD}/fit_L{L_}.npz", **out); summ[f"L{L_}"] = sm
        json.dump(summ, open(f"{OD}/fit_summary.json", "w"), ensure_ascii=False, indent=1)
    log("DONE")
