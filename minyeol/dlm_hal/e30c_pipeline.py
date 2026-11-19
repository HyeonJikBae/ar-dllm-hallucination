"""E30b~E30e (PREREG.md, 사후 계산): PopQA Dream 라우팅 흐름.  repo 루트에서:
  /NAS/minyeol/limit/.venv-gfmrag/bin/python followup_mk/e30c_pipeline.py
흐름: (연속 2 진) LP > a & S1 > b & SC > c,  (3 진) P(1~4/5) > a & SC > c.  문턱은 각 점수 분위 10 개 격자, 주어 5 겹 겹밖 (학습 겹 모집단 가중 이득 최대, 이득 <= 0 이면 '안 함').
특징: LP = −(step 0 칸 0 top-1 logprob), 첫 4 칸 logprob 평균, 칸 0 margin, S1 = 가운데K L27 s0 (e30_scores.npy), SC = 흔들림 (e30_scores.npy).
3 진 분류: 표준화 다항 로지스틱, 모집단 비율 가중 (0/5 44,110 / 1~4/5 5,950 / 5/5 5,165).  0/5 는 재생성해도 변화 0 으로 둠.
재생성 결과: E12 (results_e12_*), E41 (results_e41_*), E25 (results_e16_full_G)."""
import json, glob, numpy as np
from sklearn.linear_model import LogisticRegression
rows = [json.loads(l) for l in open("data/3_classification/full/dream_final_full.jsonl")]
k = np.array([r["k_correct"] for r in rows])
POP = {str(json.loads(l)["id"]): json.loads(l) for l in open("data/1_raw/popqa_raw.jsonl")}
sub = np.array([POP[r["sample_id"]]["s_uri"] for r in rows]); subs = np.random.default_rng(0).permutation(np.unique(sub)); od = {x: j for j, x in enumerate(subs)}; grp = np.array([od[x] % 5 for x in sub])
D = "followup_mk/e7"; R7 = np.load(f"{D}/rows.npy"); done = np.load(f"{D}/done.npy"); meta = {}
for l in open(f"{D}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
J = np.where([done[j] and int(R7[j]) in meta for j in range(len(R7))])[0]; RR = R7[J]; n = len(RR)
kk = k[RR]; g7 = grp[RR]; corr = np.array([meta[int(i)]["new_type"] == "correct" for i in RR]); isz, isf, ism = kk == 0, kk == 5, (kk >= 1) & (kk <= 4)
W = np.where(isz, 44110 / isz.sum(), np.where(isf, 5165 / isf.sum(), 1.0))
lpA = np.asarray(np.load(f"{D}/lp.npy", mmap_mode="r")[J][:, 0]); mgA = np.asarray(np.load(f"{D}/mg.npy", mmap_mode="r")[J][:, 0])
LP = -lpA[:, 0]; S1, SC = np.load("followup_mk/e30_scores.npy")
F = np.column_stack([lpA[:, 0], lpA[:, :4].mean(1), mgA[:, 0], S1]); y = np.where(isz, 0, np.where(ism, 1, 2)); PM = np.zeros(n)
for g in range(5):
    tr, te = g7 != g, g7 == g; mu, sd = F[tr].mean(0), F[tr].std(0)
    PM[te] = LogisticRegression(max_iter=2000).fit((F[tr] - mu) / sd, y[tr], sample_weight=W[tr]).predict_proba((F[te] - mu) / sd)[:, 1]
def load(fs, cond):
    E = {}
    for f in fs:
        for l in open(f):
            try: d = json.loads(l)
            except Exception: continue
            if d["cond"] == cond: E[d["row"]] = d["new_type"] == "correct"
    return np.array([E.get(int(i), corr[j]) if not isz[j] else corr[j] for j, i in enumerate(RR)])
e12, e41 = glob.glob("followup_mk/results_e12_[A-Z].jsonl"), glob.glob("followup_mk/results_e41_[A-Z].jsonl")
RED = {"L20 Brel c1": load(e12, "Brel c1"), "L20 B c0.5 (정답-붕괴)": load(e12, "B c0.5"), "L24 B c0.5 (정답-오답)": load(e41, "B c0.5"),
       "L24 Brel c1": load(e41, "Brel c1"), "L20 Rand c1": load(e12, "Rand c1"), "L20 BrelShuf c1": load(e12, "BrelShuf c1"),
       "변형 Brel c1 (좌표 342)": load(["followup_mk/results_e16_full_G.jsonl"], "변형 Brel c1")}
qs = lambda x: np.unique(np.quantile(x, np.linspace(0, 1, 11))[:-1]); t1, t2, t3, tp = qs(LP), qs(S1), qs(SC), qs(PM)
MASKS = {"연속 2진": [(LP > a) & (S1 > b) & (SC > c) for a in t1 for b in t2 for c in t3], "3진": [(PM > a) & (SC > c) for a in tp for c in t3],
         "5/5만 거르기": [LP > a for a in t1], "5/5 + 흔들림": [(LP > a) & (SC > c) for a in t1 for c in t3]}
mw, mc = ism & ~corr, ism & corr
def run(masks, redo):
    gain = W * (redo.astype(float) - corr); sel = np.zeros(n, bool)
    for g in range(5):
        tr, te = g7 != g, g7 == g; best, bm = -1e9, None
        for m in masks:
            v = gain[tr & m].sum()
            if v > best: best, bm = v, m
        if best > 0: sel[te] = bm[te]
    return sel
out = {}
for rn, redo in RED.items():
    for mn, masks in MASKS.items():
        sel = run(masks, redo); rec = int((sel & mw & redo).sum()); brk = int((sel & mc & ~redo).sum()); fb = float((sel & isf & ~redo).sum() * 5165 / isf.sum())
        out[f"{rn} | {mn}"] = dict(회복=rec, 망가짐=brk, 순=rec - brk, 오답라우팅=int((sel & mw).sum()), 정답라우팅=int((sel & mc).sum()), **{"5/5망가짐_환산": round(fb), "0/5재생성%": round(100 * float((sel & isz).sum()) / isz.sum(), 1)})
        print(rn, "|", mn, out[f"{rn} | {mn}"], flush=True)
json.dump(out, open("followup_mk/e30c_summary.json", "w"), ensure_ascii=False, indent=1)
