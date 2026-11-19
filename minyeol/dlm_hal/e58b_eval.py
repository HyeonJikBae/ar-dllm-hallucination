"""E58b 평가 (CPU).  PREREG 'E58 주 방법 확인' 의 고정 방법을 DiffuLLaMA (PopQA) 에:
특징 (step 0): 칸 0 top-1 logprob, 칸 0~3 top-1 logprob 평균, 칸 0 top1-top2 로짓 차, 가운데K S1 (마지막 층 (31) 프롬프트 끝, 관계 안 5/5 − 0/5 평균 차이 방향 K, −|투영 − 학습 겹 1~4/5 투영 중앙값|).
3 진 다항 로지스틱 (표준화, 모집단 가중) -> P(1~4/5).  SC = 32 step 중간 답이 최종 답과 다른 step 비율.  라우팅 = P > a 그리고 SC > c (a, c 각 10 분위 격자, 학습 겹 모집단 가중 이득 최대).
라우팅된 행은 steer 실행 결과, 나머지는 base.  모든 맞춤은 주어 5 겹 겹밖.
연속 2 진 (특징별 AND, 해석: E30 식): S1 > a' 그리고 SC > c' (같은 격자, 겹밖).
usage: python followup_mk/e58b_eval.py [--pilot]"""
import collections, glob, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "followup_mk")
from e58b_load import load_base
PILOT = "--pilot" in sys.argv; NB = 2000
R, B, H, k, grp, fid = load_base(PILOT)
rows = [json.loads(l) for l in open("data/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
meta = json.load(open("followup_mk/e58b/rows.json")); pop = meta["pop_rows"]; ns = meta["n"]
S_ = {}
for f in sorted(glob.glob(f"followup_mk/results_e58b_steer{'_pilot' if PILOT else ''}_s*.jsonl")):
    for l in open(f, encoding="utf-8"): o = json.loads(l); S_[o["row"]] = o
A = {}
for l in open("followup_mk/results_e55_ar_s0.jsonl", encoding="utf-8"): o = json.loads(l); A[o["row"]] = o["type"] == "correct"
n = len(R); cat = np.where(k == 0, 0, np.where(k == 5, 2, 1)); W = np.array([pop["zero"] / ns["zero"], 1.0, pop["five"] / ns["five"]])[cat]
b_ok = np.array([B[i]["type"] == "correct" for i in R], float); s_ok = np.array([S_[i]["type"] == "correct" for i in R], float)
lp0 = np.array([B[i]["lp0"] for i in R]); lp4 = np.array([B[i]["lp4"] for i in R]); mg = np.array([B[i]["margin0"] for i in R]); SC = np.array([B[i]["sc"] for i in R])
rel = np.array([rows[i]["relation"] for i in R]); hK = H[:, 1].astype(np.float32)
O = {"n": dict(zero=int((cat == 0).sum()), mid=int((cat == 1).sum()), five=int((cat == 2).sum())), "pop_rows": pop,
     "base_acc": {c: round(100 * float(b_ok[cat == j].mean()), 2) for j, c in enumerate(("zero", "mid", "five"))},
     "steer_all_acc": {c: round(100 * float(s_ok[cat == j].mean()), 2) for j, c in enumerate(("zero", "mid", "five"))}}
# ---- 겹밖 S1, P(1~4/5)
S1 = np.zeros(n); P = np.zeros(n); X0 = np.c_[lp0, lp4, mg]
for g in range(5):   # 겹 g: K 와 1~4/5 중앙값을 학습 겹에서 맞춰 모든 행의 S1 을 계산 -> 학습 행으로 분류기, 시험 행 예측
    tr, te = grp != g, grp == g
    ds = [hK[tr & (cat == 2) & (rel == r)].mean(0) - hK[tr & (cat == 0) & (rel == r)].mean(0) for r in np.unique(rel) if (tr & (cat == 2) & (rel == r)).sum() >= 10 and (tr & (cat == 0) & (rel == r)).sum() >= 10]
    K = np.mean(ds, 0) if ds else hK[tr & (cat == 2)].mean(0) - hK[tr & (cat == 0)].mean(0); K /= np.linalg.norm(K)
    pk = hK @ K; s1g = -np.abs(pk - np.median(pk[tr & (cat == 1)])); S1[te] = s1g[te]
    X = np.c_[X0, s1g]; sc_ = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc_.transform(X[tr]), cat[tr], sample_weight=W[tr]); P[te] = clf.predict_proba(sc_.transform(X[te]))[:, 1]
from sklearn.metrics import roc_auc_score
O["AUROC_P_mid_vs_rest_weighted"] = round(float(roc_auc_score(cat == 1, P, sample_weight=W)), 3)
O["AUROC_SC_wrong_vs_correct_in_mid"] = round(float(roc_auc_score(b_ok[cat == 1] == 0, SC[cat == 1])), 3)
d = s_ok - b_ok   # 다시 생성하면 바뀌는 정오

def route_oof(s1, s2):
    """s1 > a 그리고 s2 > c.  a, c: 학습 겹 각 10 분위 (0, .1, ..., .9) 격자, 학습 겹 모집단 가중 이득 Σ w d 최대.  반환 (라우팅 bool, 겹별 선택)."""
    rt = np.zeros(n, bool); ch = []
    for g in range(5):
        tr, te = grp != g, grp == g; qa = np.quantile(s1[tr], np.arange(10) / 10); qc = np.quantile(s2[tr], np.arange(10) / 10); best = (-np.inf, None)
        for a in qa:
            for c in qc:
                m_ = tr & (s1 > a) & (s2 > c); gain = float((W[m_] * d[m_]).sum())
                if gain > best[0]: best = (gain, (float(a), float(c)))
        a, c = best[1]; rt[te] = (s1[te] > a) & (s2[te] > c); ch.append([round(a, 4), round(c, 4)])
    return rt, ch

def boot(stat, seed=0):
    u, inv = np.unique(fid, return_inverse=True); mem = [np.where(inv == j)[0] for j in range(len(u))]; rng = np.random.default_rng(seed); bs = []
    for _ in range(NB):
        ix = np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))]); v = stat(ix)
        if np.isfinite(v): bs.append(v)
    return [round(float(stat(np.arange(n))), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

def outcome(rt, tag):
    fin = np.where(rt, s_ok, b_ok); m_ = cat == 1; o = {}
    o["routed_rate_%"] = {c: round(100 * float(rt[cat == j].mean()), 2) for j, c in enumerate(("zero", "mid", "five"))}
    o["mid_recovered"] = int(((b_ok == 0) & (fin == 1) & m_).sum()); o["mid_broken"] = int(((b_ok == 1) & (fin == 0) & m_).sum()); o["mid_net"] = o["mid_recovered"] - o["mid_broken"]
    o["H58.1_mid_net_points"] = boot(lambda ix: 100 * (fin[ix][m_[ix]] - b_ok[ix][m_[ix]]).mean())
    f5 = cat == 2; o["five_acc_base_final"] = [round(100 * float(b_ok[f5].mean()), 2), round(100 * float(fin[f5].mean()), 2)]
    o["H58.2_five_loss_points"] = boot(lambda ix: 100 * (b_ok[ix][f5[ix]] - fin[ix][f5[ix]]).mean())
    z = cat == 0; o["zero_regen_rate_%"] = round(100 * float(rt[z].mean()), 2); o["zero_acc_change_points"] = round(100 * float((fin[z] - b_ok[z]).mean()), 3)
    o["H58.3_pop_weighted_net_points"] = boot(lambda ix: 100 * (W[ix] * (fin[ix] - b_ok[ix])).sum() / W[ix].sum())
    o["pop_weighted_regen_rate_%"] = round(100 * float((W * rt).sum() / W.sum()), 2)
    # 경로별 회복 (1~4/5 오답 = entity/broken, LLaMA-2 같은 문구 정오)
    wr = m_ & np.array([B[i]["type"] in ("entity", "broken") for i in R]); dl = np.array([A[i] for i in R])
    o["recovery_by_route_%"] = {nm: dict(n=int((wr & sel).sum()), routed=round(100 * float(rt[wr & sel].mean()), 1), recovered=round(100 * float(fin[wr & sel].mean()), 1),
                                         recovered_if_steered_all=round(100 * float(s_ok[wr & sel].mean()), 1)) for nm, sel in (("dlm", dl), ("shared", ~dl))}
    print(tag, json.dumps(o, ensure_ascii=False)); return o

rt, ch = route_oof(P, SC); O["main"] = outcome(rt, "main"); O["main"]["thresholds_by_fold (a, c)"] = ch
rt2, ch2 = route_oof(S1, SC); O["seqAND_S1_SC"] = outcome(rt2, "seqAND"); O["seqAND_S1_SC"]["thresholds_by_fold"] = ch2
O["steer_all"] = outcome(np.ones(n, bool), "steer_all")
O["verdict"] = {"H58.1": O["main"]["H58.1_mid_net_points"][1] > 0, "H58.2": O["main"]["H58.2_five_loss_points"][0] <= 1.0, "H58.3": O["main"]["H58.3_pop_weighted_net_points"][1] > 0}
print("판정", O["verdict"])
json.dump(O, open(f"followup_mk/e58b_summary{'_pilot' if PILOT else ''}.json", "w"), ensure_ascii=False, indent=1)
