"""E53 (PREREG.md) 분석 (CPU): tuned lens / logit lens 의 층 x step 금 첫 토큰 지도, H53.1, H53.2, 렌즈 품질, sanity.
집단 (1~4/5, 이 실행의 채점): C = 맞힘, D = DLM 고유 실패 (Dream entity/broken, 같은 문구 Qwen 맞힘), H = 공유 실패 (Qwen 도 entity/broken).
Dream abstain, Qwen abstain 인 오답은 제외 (E49/E51 과 같음).  참고 집단: 5/5 (E7 5/5 2,000, 렌즈 학습에 쓰지 않음).
CI: 사실 (fid) 군집 부트스트랩 B = 4000.
usage: python followup_mk/e53_analyze.py"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import rows, k, fid, mid, OUT, D_
OD = f"{OUT}/e53"; B = 4000; rng = np.random.default_rng(530)
R = np.load(f"{OD}/rows.npy"); meta = {}
for l in open(f"{OD}/ana_meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
cfg = json.load(open(f"{OD}/ana_config.json")); S = cfg["steps"]; NL = 28
A = {x: np.load(f"{OD}/{x}.npy") for x in ("tl_gold_rank", "ll_gold_rank", "tl_final_rank", "ll_final_rank", "tl_top1", "ll_top1", "tl_kl", "ll_kl", "slot0_gold_rank", "slot0_argmax", "slot0_mask")}
n = len(R); nt = np.array([meta[int(r)]["new_type"] for r in R]); setv = np.array([meta[int(r)]["set"] for r in R]); kk = k[R]; F_ = fid[R]
QW = {}
for l in open(f"{D_}/3_classification/full/qwen_final_full.jsonl", encoding="utf-8"):
    d = json.loads(l); QW[(d["sample_id"], d["version"])] = d
def qlab(r):
    q = QW[(rows[r]["sample_id"], rows[r]["version"])]; return "correct" if q["correct"] else q["type3"]
ql = np.array([qlab(int(r)) for r in R])
wrong = np.isin(nt, ["entity", "broken"]); ism = setv == "mid"
G = {"C": ism & (nt == "correct"), "D": ism & wrong & (ql == "correct"), "H": ism & wrong & np.isin(ql, ["entity", "broken"]), "five": (setv == "five") & (nt == "correct")}
G.update({f"{g}_ATh": G[g] & (kk >= 3) for g in "CDH"}); G.update({f"{g}_ATl": G[g] & (kk <= 2) for g in "CDH"})
out = {"n": {g: int(v.sum()) for g, v in G.items()}, "steps": S}
s0 = S.index(0); s2 = S.index(2)

# ---------------- sanity: E44 (같은 묶음 구성) 와 E7 답 일치
def e44_meta():
    M = {}
    for d in sorted(x for x in os.listdir(f"{OUT}/e44") if x.startswith("shard")):
        for l in open(f"{OUT}/e44/{d}/meta.jsonl"): m_ = json.loads(l); M[m_["row"]] = m_
    return M
M44 = e44_meta(); M7 = {}
for l in open(f"{OUT}/e7/meta.jsonl"): d = json.loads(l); M7[d["row"]] = d
c44 = [int(r) for r in R if int(r) in M44]; c7 = [int(r) for r in R if int(r) in M7]
out["sanity"] = dict(answer_eq_E44=round(np.mean([meta[r]["answer"] == M44[r]["answer"] for r in c44]), 4), grade_eq_E44=round(np.mean([meta[r]["new_type"] == M44[r]["new_type"] for r in c44]), 4),
                     slot0_final_eq_E44=round(np.mean([meta[r]["slot0_final_token"] == M44[r]["slot0_final_token"] for r in c44]), 4), n_E44=len(c44),
                     answer_eq_E7=round(np.mean([meta[r]["answer"] == M7[r]["answer"] for r in c7]), 4), grade_eq_E7=round(np.mean([meta[r]["new_type"] == M7[r]["new_type"] for r in c7]), 4),
                     L27_lens_rank_eq_actual_slot0_rank_s0=round(float((A["ll_gold_rank"][:, s0, 27] == A["slot0_gold_rank"][:, 0]).mean()), 4),
                     L27_top1_eq_slot0_argmax_s0=round(float((A["ll_top1"][:, s0, 27] == A["slot0_argmax"][:, 0]).mean()), 4),
                     tuned_L27_is_identity=bool((A["tl_gold_rank"][:, :, 27] == A["ll_gold_rank"][:, :, 27]).all()))

# ---------------- 렌즈 품질
SC = "${SCRATCH:-/tmp/scratch}/e53/lens"
Qf = f"{OD}/lens_quality_train_holdout.json"
if os.path.exists(f"{SC}/quality.json"):
    import shutil; shutil.copy(f"{SC}/quality.json", Qf)
    for x in ("sweep.json",):
        if os.path.exists(f"{SC}/{x}"): shutil.copy(f"{SC}/{x}", f"{OD}/lens_{x}")
if os.path.exists(Qf):
    Q = json.load(open(Qf))
    out["lens_quality_holdout"] = {l: dict(kl_tuned=Q[l]["tuned"]["kl"], kl_logit=Q[l]["logit"]["kl"], top1_tuned=Q[l]["tuned"]["top1"], top1_logit=Q[l]["logit"]["top1"], best_ep=Q[l]["best_ep"]) for l in sorted(Q, key=int)}
fin_top1 = A["ll_top1"][:, :, 27]
out["lens_quality_analysis_rows"] = {str(l): dict(kl_tuned=round(float(A["tl_kl"][ism][:, :, l].astype(np.float32).mean()), 3), kl_logit=round(float(A["ll_kl"][ism][:, :, l].astype(np.float32).mean()), 3),
                                                  kl_tuned_s0=round(float(A["tl_kl"][ism][:, s0, l].astype(np.float32).mean()), 3), kl_logit_s0=round(float(A["ll_kl"][ism][:, s0, l].astype(np.float32).mean()), 3),
                                                  top1_tuned=round(float((A["tl_top1"][ism][:, :, l] == fin_top1[ism]).mean()), 3), top1_logit=round(float((A["ll_top1"][ism][:, :, l] == fin_top1[ism]).mean()), 3))
                                     for l in range(NL)}
out["lens_quality_analysis_rows_by_step_tuned_kl"] = {str(S[s]): [round(float(A["tl_kl"][ism][:, s, l].astype(np.float32).mean()), 3) for l in range(NL)] for s in range(len(S))}

# ---------------- 부트스트랩
uf, finv = np.unique(F_, return_inverse=True); nF = len(uf)
W = rng.multinomial(nF, np.ones(nF) / nF, size=B).astype(np.float32)   # [B, nF] 사실 재표집 횟수
def fsum(x, m):
    s = np.zeros(nF); c = np.zeros(nF); np.add.at(s, finv[m], x[m]); np.add.at(c, finv[m], 1); return s, c
def gmean(x, m):
    s, c = fsum(x.astype(float), m); return float(s.sum() / c.sum()), (W @ s) / np.maximum(W @ c, 1e-9)
def diff(x, m1, m0):
    a, ba = gmean(x, m1); b, bb = gmean(x, m0); d = ba - bb
    return dict(g1=round(a * 100, 2), g0=round(b * 100, 2), diff=round((a - b) * 100, 2), ci=[round(float(np.percentile(d, 2.5)) * 100, 2), round(float(np.percentile(d, 97.5)) * 100, 2)], n1=int(m1.sum()), n0=int(m0.sum()))
def within(x, m1, m0):
    s1, c1 = fsum(x.astype(float), m1); s0_, c0 = fsum(x.astype(float), m0); ok = (c1 > 0) & (c0 > 0)
    d = np.where(ok, s1 / np.maximum(c1, 1) - s0_ / np.maximum(c0, 1), 0.0); Wk = W * ok[None]
    bd = (Wk @ d) / np.maximum(Wk.sum(1), 1e-9)
    return dict(facts=int(ok.sum()), diff=round(float(d[ok].mean()) * 100, 2), ci=[round(float(np.percentile(bd, 2.5)) * 100, 2), round(float(np.percentile(bd, 97.5)) * 100, 2)])

# ---------------- H53.1: step 0 에서 어떤 층에서도 금 첫 토큰이 top-10 에 없음
for p in ("tl", "ll"):
    never10 = (A[f"{p}_gold_rank"][:, s0, :].min(1) >= 10).astype(float)
    never1 = (A[f"{p}_gold_rank"][:, s0, :].min(1) >= 1).astype(float)
    out[f"H53.1_{p}"] = dict(rate={g: round(float(never10[G[g]].mean()) * 100, 2) for g in G}, H_minus_D=diff(never10, G["H"], G["D"]), H_minus_C=diff(never10, G["H"], G["C"]),
                             D_minus_C=diff(never10, G["D"], G["C"]), within_fact_H_minus_D=within(never10, G["H"], G["D"]),
                             ATh_H_minus_D=diff(never10, G["H_ATh"], G["D_ATh"]), ATl_H_minus_D=diff(never10, G["H_ATl"], G["D_ATl"]),
                             never_top1_rate={g: round(float(never1[G[g]].mean()) * 100, 2) for g in G}, never_top1_H_minus_D=diff(never1, G["H"], G["D"]))
r = out["H53.1_tl"]["H_minus_D"]; out["H53.1_verdict"] = bool(r["ci"][0] > 0)
# 최종 층만 (실제 로짓) 과 중간 층에서만 보이는 비율: 'tuned lens 가 중간 층에서 더 찾는 몫'
tl0 = A["tl_gold_rank"][:, s0, :]; ll0 = A["ll_gold_rank"][:, s0, :]
out["H53.1_extra"] = {g: dict(top10_L27=round(float((tl0[G[g], 27] < 10).mean()) * 100, 2), top10_any_layer_tuned=round(float((tl0[G[g]].min(1) < 10).mean()) * 100, 2),
                              top10_only_before_L27_tuned=round(float(((tl0[G[g], :27].min(1) < 10) & (tl0[G[g], 27] >= 10)).mean()) * 100, 2),
                              first_layer_top10_tuned_median=float(np.median([np.argmax(x < 10) for x in tl0[G[g]] if (x < 10).any()])) if (tl0[G[g]] < 10).any() else None,
                              first_layer_top1_tuned_median=float(np.median([np.argmax(x < 1) for x in tl0[G[g]] if (x < 1).any()])) if (tl0[G[g]] < 1).any() else None,
                              best_layer_rank_median_tuned=float(np.median(tl0[G[g]].min(1))), best_layer_rank_median_logit=float(np.median(ll0[G[g]].min(1))))
                       for g in ("C", "D", "H", "five", "D_ATh", "H_ATh")}

# ---------------- H53.2: 마지막 층 금 순위가 step 0 -> step 2 (첫 확정 뒤) 나빠짐
R27 = A["ll_gold_rank"][:, :, 27]
worse = (R27[:, s2] > R27[:, s0]).astype(float)
out["H53.2"] = dict(rate={g: round(float(worse[G[g]].mean()) * 100, 2) for g in G}, D_minus_C=diff(worse, G["D"], G["C"]), H_minus_C=diff(worse, G["H"], G["C"]), D_minus_H=diff(worse, G["D"], G["H"]),
                    within_fact_D_minus_C=within(worse, G["D"], G["C"]), ATh_D_minus_C=diff(worse, G["D_ATh"], G["C_ATh"]))
out["H53.2_verdict"] = bool(out["H53.2"]["D_minus_C"]["ci"][0] > 0)
# 민감도: step 0 순위 구간별, 1 위를 잃음, 순위 2 배 이상 악화
sens = {}
for nm, msk in (("s0_rank0", R27[:, s0] == 0), ("s0_rank1to9", (R27[:, s0] >= 1) & (R27[:, s0] < 10)), ("s0_rank10plus", R27[:, s0] >= 10)):
    sens[nm] = {g: dict(n=int((G[g] & msk).sum()), worse=round(float(worse[G[g] & msk].mean()) * 100, 2) if (G[g] & msk).any() else None) for g in ("C", "D", "H")}
    if (G["D"] & msk).sum() > 20 and (G["C"] & msk).sum() > 20: sens[nm]["D_minus_C"] = diff(worse, G["D"] & msk, G["C"] & msk)
lost1 = ((R27[:, s0] == 0) & (R27[:, s2] > 0)).astype(float); x2 = (np.log2(R27[:, s2] + 1) - np.log2(R27[:, s0] + 1) >= 1).astype(float)
sens["lost_top1"] = dict(rate={g: round(float(lost1[G[g]].mean()) * 100, 2) for g in ("C", "D", "H")}, D_minus_C=diff(lost1, G["D"], G["C"]))
sens["worse_2x"] = dict(rate={g: round(float(x2[G[g]].mean()) * 100, 2) for g in ("C", "D", "H")}, D_minus_C=diff(x2, G["D"], G["C"]), D_minus_H=diff(x2, G["D"], G["H"]))
for t in (1, 4, 8, 31):
    st = S.index(t) if t in S else None
    if st is not None and t != 0:
        w_ = (R27[:, st] > R27[:, s0]).astype(float); sens[f"worse_s0_to_s{t}"] = dict(rate={g: round(float(w_[G[g]].mean()) * 100, 2) for g in ("C", "D", "H")}, D_minus_C=diff(w_, G["D"], G["C"]))
out["H53.2_sensitivity"] = sens

# ---------------- 지도 (그림용): 집단별 층 x step 의 중앙 금 순위, % top-1, % top-10, 최종 첫 토큰 % top-1
maps = {"layers": list(range(NL)), "steps": S, "groups": {}}
for g in ("C", "D", "H", "five", "C_ATh", "D_ATh", "H_ATh", "C_ATl", "D_ATl", "H_ATl"):
    mg = {}
    for p in ("tl", "ll"):
        X = A[f"{p}_gold_rank"][G[g]]; Fr = A[f"{p}_final_rank"][G[g]]
        mg[p] = dict(median_gold_rank=np.median(X, 0).T.round(1).tolist(), pct_top1=(100 * (X == 0).mean(0)).T.round(1).tolist(), pct_top10=(100 * (X < 10).mean(0)).T.round(1).tolist(),
                     pct_final_top1=(100 * (Fr == 0).mean(0)).T.round(1).tolist())
    maps["groups"][g] = dict(n=int(G[g].sum()), **mg)
maps["note"] = "각 배열은 [층 0..27][step 순서 = steps].  tl = tuned lens, ll = logit lens.  층 27 은 둘 다 실제 칸 0 로짓.  step 0 = step 1 캔버스, 첫 확정은 step 1 끝."
json.dump(maps, open(f"{OUT}/e53_maps.json", "w"), ensure_ascii=False)
# 요약용 작은 표: step 0 과 2 의 층별 % top-10 (tuned)
out["map_digest_tuned_pct_top10"] = {g: {f"s{t}": [maps["groups"][g]["tl"]["pct_top10"][l][S.index(t)] for l in range(0, NL, 3)] + [maps["groups"][g]["tl"]["pct_top10"][27][S.index(t)]] for t in (0, 2, 8)} for g in ("C", "D", "H", "five")}
out["map_digest_layers"] = list(range(0, NL, 3)) + [27]
json.dump(out, open(f"{OUT}/e53_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k_: out[k_] for k_ in ("n", "sanity", "H53.1_verdict", "H53.2_verdict")}, ensure_ascii=False))
print(json.dumps(out["H53.1_tl"], ensure_ascii=False)); print(json.dumps(out["H53.2"], ensure_ascii=False))
