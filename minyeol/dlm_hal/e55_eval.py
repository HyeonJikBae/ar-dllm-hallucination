"""E55 평가 (CPU).  results_e55_dllm[_pilot]_s*.jsonl (DiffuLLaMA: base + 칸0먼저), results_e55_ar[_pilot]_s*.jsonl (LLaMA-2 greedy).
정의 (E49 와 같음): DiffuLLaMA 자기 k/5 가 1~4 인 사실의 오답 (entity/broken, abstain 제외) 행 중 같은 문구를 LLaMA-2 가 맞히면 'DLM 고유', 틀리면 '공유'.
H55.1: step 0 칸 0 금 첫 토큰 1 위 비율 DLM 고유 > 공유 (사실 군집 부트스트랩 하한 > 0).  H55.2: 칸0먼저 오답 복구율 DLM 고유 > 공유 (하한 > 0).
usage: python followup_mk/e55_eval.py [--pilot]
"""
import collections, glob, json, sys
import numpy as np
PILOT = "--pilot" in sys.argv; SF = "_pilot" if PILOT else ""; B = 4000; OUT = "followup_mk"
rows = [json.loads(l) for l in open("data/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
fid_all = np.array([r["sample_id"] for r in rows])

def load(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        for l in open(f, encoding="utf-8"):
            o = json.loads(l); d[o["row"]] = o
    return d
DA = load(f"{OUT}/results_e55_dllm{SF}_A_s*.jsonl"); DB = load(f"{OUT}/results_e55_dllm{SF}_B_s*.jsonl"); A = load(f"{OUT}/results_e55_ar{SF}_s*.jsonl")
D = DA   # A 단계 (base 만, 전체): 정확도와 k/5.  B 단계 (1~4/5 사실 행, base + 칸0먼저 같은 묶음): 경로 분석과 개입.
common_rows = sorted(set(DA) & set(A)); print("행: DiffuLLaMA A", len(DA), "B", len(DB), "LLaMA-2", len(A), "공통", len(common_rows))
S = {}

def kdist(get, R):
    byf = collections.defaultdict(list)
    for i in R: byf[fid_all[i]].append(get(i) == "correct")
    full = {f: v for f, v in byf.items() if len(v) == 5}; kk = {f: int(sum(v)) for f, v in full.items()}
    return kk, {str(j): int(sum(1 for v in kk.values() if v == j)) for j in range(6)}

for name, src, get in [("diffullama", DA, lambda i: DA[i]["base"]["type"]), ("llama2", A, lambda i: A[i]["type"])]:
    R = sorted(src); c = collections.Counter(get(i) for i in R); kk, kd = kdist(get, R)
    S[f"acc_{name}"] = dict(n=len(R), acc=round(100 * c["correct"] / len(R), 2), types={t: int(v) for t, v in c.items()}, k_dist_facts=kd, n_facts_1to4=int(sum(1 <= v <= 4 for v in kk.values())),
                            n_rows_1to4=int(5 * sum(1 <= v <= 4 for v in kk.values())))
    print(name, S[f"acc_{name}"])
kD, _ = kdist(lambda i: D[i]["base"]["type"], common_rows); kA, _ = kdist(lambda i: A[i]["type"], common_rows)
ct = collections.Counter((("0" if kD[f] == 0 else "5" if kD[f] == 5 else "1-4"), ("0" if kA[f] == 0 else "5" if kA[f] == 5 else "1-4")) for f in kD if f in kA)
S["cross_k_facts"] = {f"dllm{a}_ar{b}": int(v) for (a, b), v in sorted(ct.items())}
# 같은 문구 정오 일치 (E49 (2) 와 같은 지표, 두 모델 모두 섞인 사실): P(D 맞힘 | A 맞힘) - P(D 맞힘 | A 틀림) 사실 평균
diffs = []
for f in kD:
    if f in kA and 1 <= kD[f] <= 4 and 1 <= kA[f] <= 4:
        ii = np.where(fid_all == f)[0]; d = np.array([D[i]["base"]["type"] == "correct" for i in ii]); a = np.array([A[i]["type"] == "correct" for i in ii])
        diffs.append(d[a].mean() - d[~a].mean())
S["agreement_both_mixed"] = dict(n_facts=len(diffs), mean_diff=round(float(np.mean(diffs)), 4) if diffs else None)

# ---- 경로 분리
mixed = {f for f, v in kD.items() if 1 <= v <= 4}
RB = sorted(i for i in DB if i in A and fid_all[i] in mixed)
S["phaseB"] = dict(n_rows=len(RB), n_rows_expected=5 * len(mixed), base_answer_same_as_A=round(100 * np.mean([DB[i]["base"]["answer"] == DA[i]["base"]["answer"] for i in RB]), 2) if RB else None,
                   base_correct_same_as_A=round(100 * np.mean([(DB[i]["base"]["type"] == "correct") == (DA[i]["base"]["type"] == "correct") for i in RB]), 2) if RB else None,
                   s0_rank0_same_as_A=round(100 * np.mean([DB[i]["base"]["s0_rank"][0] == DA[i]["base"]["s0_rank"][0] for i in RB]), 2) if RB else None)
print("B 단계 sanity:", S["phaseB"])
D = DB
W = [i for i in RB if D[i]["base"]["type"] in ("entity", "broken")]
route = {i: ("dlm" if A[i]["type"] == "correct" else "shared") for i in W}
gid = {i: D[i]["gold_id"] for i in W}
S["n_wrong_rows_mixed"] = dict(total=len(W), dlm=sum(r == "dlm" for r in route.values()), shared=sum(r == "shared" for r in route.values()),
                               facts=len({fid_all[i] for i in W}), gold_tok_is_bare_space=sum(gid[i] == 29871 for i in W), gold_tok_missing=sum(gid[i] < 0 for i in W),
                               abstain_rows_excluded=sum(1 for i in RB if D[i]["base"]["type"] == "abstain"),
                               types={g: dict(collections.Counter(D[i]["base"]["type"] for i in W if route[i] == g)) for g in ("dlm", "shared")})
print("오답 행 (DiffuLLaMA 1~4/5):", S["n_wrong_rows_mixed"])

def cluster_boot(vals, groups_, fids, g1="dlm", g2="shared", seed=0):
    """사실 군집 부트스트랩: 사실을 복원 추출해 두 집단 비율 (행 가중) 차이.  반환 [차이, 하한, 상한], 집단별 비율."""
    vals = np.asarray(vals, float); groups_ = np.asarray(groups_); fu, fi = np.unique(np.asarray(fids), return_inverse=True); nf = len(fu)
    s1 = np.bincount(fi, weights=vals * (groups_ == g1), minlength=nf); n1 = np.bincount(fi, weights=(groups_ == g1).astype(float), minlength=nf)
    s2 = np.bincount(fi, weights=vals * (groups_ == g2), minlength=nf); n2 = np.bincount(fi, weights=(groups_ == g2).astype(float), minlength=nf)
    rng = np.random.default_rng(seed); w = rng.multinomial(nf, np.ones(nf) / nf, size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"): d = (w @ s1) / (w @ n1) - (w @ s2) / (w @ n2)
    d = d[np.isfinite(d)]; p1 = s1.sum() / max(n1.sum(), 1); p2 = s2.sum() / max(n2.sum(), 1)
    return dict(n=[int(n1.sum()), int(n2.sum())], rate=[round(100 * p1, 2), round(100 * p2, 2)], diff=round(100 * (p1 - p2), 2),
                ci=[round(100 * float(np.percentile(d, 2.5)), 2), round(100 * float(np.percentile(d, 97.5)), 2)], n_facts=nf)

def within_fact(vals, groups_, fids, g1="dlm", g2="shared", seed=1):
    by = collections.defaultdict(lambda: {g1: [], g2: []})
    for v, g, f in zip(vals, groups_, fids): by[f][g].append(v)
    d = np.array([np.mean(x[g1]) - np.mean(x[g2]) for x in by.values() if x[g1] and x[g2]])
    if len(d) == 0: return dict(n_facts=0)
    m = d[np.random.default_rng(seed).integers(0, len(d), (B, len(d)))].mean(1)
    return dict(n_facts=int(len(d)), n_rows=[int(sum(len(x[g1]) for x in by.values() if x[g1] and x[g2])), int(sum(len(x[g2]) for x in by.values() if x[g1] and x[g2]))],
                diff=round(100 * float(d.mean()), 2), ci=[round(100 * float(np.percentile(m, 2.5)), 2), round(100 * float(np.percentile(m, 97.5)), 2)])

def analyze(Wsub, tag):
    g = [route[i] for i in Wsub]; f = [fid_all[i] for i in Wsub]; out = {}
    r0 = np.array([D[i]["base"]["s0_rank"][0] for i in Wsub]); top1 = (r0 == 0).astype(float)
    fin0 = np.array([D[i]["base"]["canvas"][0] for i in Wsub]); gold = np.array([gid[i] for i in Wsub])
    lost = ((r0 == 0) & (fin0 != gold)).astype(float)
    out["H55.1_top1_step0"] = cluster_boot(top1, g, f); out["H55.1_within_fact"] = within_fact(top1, g, f)
    out["top5_step0"] = cluster_boot((r0 < 5).astype(float), g, f); out["lost_top1_then_slot0_not_gold"] = cluster_boot(lost, g, f)
    out["median_rank_step0"] = {k_: float(np.median(r0[np.array(g) == k_])) if (np.array(g) == k_).any() else None for k_ in ("dlm", "shared")}
    w_ = within_fact(np.log2(r0 + 1), g, f); w_["diff"] = round(w_["diff"] / 100, 3) if "diff" in w_ else None; w_["ci"] = [round(c_ / 100, 3) for c_ in w_.get("ci", [])]
    out["log2rank_within_fact"] = w_   # 단위: log2(순위+1)
    rec = np.array([D[i]["s0f"]["type"] == "correct" for i in Wsub], float)
    out["H55.2_slot0first_recovery"] = cluster_boot(rec, g, f); out["H55.2_within_fact"] = within_fact(rec, g, f)
    ar_r0 = np.array([A[i]["s0_rank"] for i in Wsub]); out["llama2_first_pos_gold_top1"] = cluster_boot((ar_r0 == 0).astype(float), g, f)
    s0f_eq_base = np.array([D[i]["s0f"]["tok"] == D[i]["base"]["canvas"][0] for i in Wsub], float); out["s0f_token_equals_base_slot0"] = cluster_boot(s0f_eq_base, g, f)
    s0f_is_gold = np.array([D[i]["s0f"]["tok"] == gid[i] for i in Wsub], float); out["s0f_token_is_gold"] = cluster_boot(s0f_is_gold, g, f)
    S[tag] = out
    for k_, v in out.items(): print(tag, k_, v)

analyze(W, "routes")

# ---- E57 식 층화 (조정자 요청 2026-11-19): LLaMA-2 금 첫 토큰 log 확률 5 분위 x 금 답 토큰 수 (1,2,3,4+), Mantel-Haenszel 차이, 사실 군집 부트스트랩
def qbin(x, q=5):   # e57_common.qbin 과 같음
    x = np.asarray(x, float); out = np.full(len(x), -1); ok = ~np.isnan(x)
    rk = np.argsort(np.argsort(x[ok], kind="stable"), kind="stable"); out[ok] = (rk * q // ok.sum()).astype(int); return out
def mh_rd(y, d, s):   # e57_common.mh_rd 와 같음 (MH 가중)
    y = np.asarray(y, float); d = np.asarray(d, bool); s = np.unique(np.asarray(s), return_inverse=True)[1]; m_ = int(s.max()) + 1
    n1 = np.bincount(s[d], minlength=m_); n0 = np.bincount(s[~d], minlength=m_); y1 = np.bincount(s[d], weights=y[d], minlength=m_); y0 = np.bincount(s[~d], weights=y[~d], minlength=m_)
    v = (n1 > 0) & (n0 > 0)
    if not v.any(): return np.nan, 0.0
    rd = y1[v] / n1[v] - y0[v] / n0[v]; w = n1[v] * n0[v] / (n1[v] + n0[v]); return float((w * rd).sum() / w.sum()), float((n1[v] + n0[v]).sum() / len(y))
def boot_facts(f, stat, B_=2000, seed=0):
    u, inv = np.unique(f, return_inverse=True); members = [np.where(inv == j)[0] for j in range(len(u))]; rng = np.random.default_rng(seed); bs = []
    for _ in range(B_):
        v = stat(np.concatenate([members[j] for j in rng.integers(0, len(u), len(u))]))
        if v is not None and np.isfinite(v): bs.append(v)
    e = stat(np.arange(len(f))); return [round(100 * float(e), 2), round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)]
from transformers import AutoTokenizer
_tk = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
def strat(Wsub, tag):
    Wa = np.array(Wsub); d = np.array([route[i] == "dlm" for i in Wsub]); f = fid_all[Wa]
    ql = np.array([A[i]["s0_gold_logp"] for i in Wsub]); alen = np.array([len(_tk(rows[i]["gold_value"], add_special_tokens=False)["input_ids"]) for i in Wsub]); ab = np.minimum(alen, 4)
    st = np.array([f"{a}_{b}" for a, b in zip(qbin(ql), ab)]); qq = qbin(ql)
    out = {"균형 (DLM 고유 / 공유)": {"LLaMA-2 금 logprob 평균": [round(float(ql[d].mean()), 3), round(float(ql[~d].mean()), 3)], "금 답 토큰 수 평균": [round(float(alen[d].mean()), 2), round(float(alen[~d].mean()), 2)],
                                     "LLaMA-2 금 1 위 %": [round(100 * float(np.mean([A[i]["s0_rank"] == 0 for i in Wa[d]])), 1), round(100 * float(np.mean([A[i]["s0_rank"] == 0 for i in Wa[~d]])), 1)]}}
    for nm, y in [("H55.1 step0 금 1 위", np.array([D[i]["base"]["s0_rank"][0] == 0 for i in Wsub], float)), ("H55.2 칸0먼저 복구", np.array([D[i]["s0f"]["type"] == "correct" for i in Wsub], float))]:
        out[nm] = {"층화 없음": boot_facts(f, lambda ix: y[ix][d[ix]].mean() - y[ix][~d[ix]].mean()),
                   "MH: LLaMA-2 금 logprob 5 분위 x 답 토큰 수": boot_facts(f, lambda ix: mh_rd(y[ix], d[ix], st[ix])[0]), "덮는 행 비율": round(mh_rd(y, d, st)[1], 3),
                   "MH: LLaMA-2 금 logprob 5 분위만": boot_facts(f, lambda ix: mh_rd(y[ix], d[ix], qq[ix])[0])}
    S[tag] = out; print(tag, json.dumps(out, ensure_ascii=False))
strat(W, "stratified_llama2_goldprob_x_len")
W2 = [i for i in W if gid[i] not in (29871, -1)]
if len(W2) < len(W): analyze(W2, "routes_excl_bare_space_gold")
# 칸0먼저의 깨짐 (1~4/5 정답 행), 같은 문구 LLaMA-2 정오별 (E50 H50.4 와 같은 방향의 서술)
C = [i for i in RB if D[i]["base"]["type"] == "correct"]
br = np.array([D[i]["s0f"]["type"] != "correct" for i in C], float); gc = ["ar_wrong" if A[i]["type"] != "correct" else "ar_right" for i in C]
S["slot0first_break_correct_rows"] = cluster_boot(br, gc, [fid_all[i] for i in C], g1="ar_wrong", g2="ar_right")
# k 별 칸0먼저 정확도 변화 (서술)
S["slot0first_acc_by_k"] = {}
for kk_ in range(1, 5):
    R = [i for i in RB if kD.get(fid_all[i]) == kk_]
    if R: S["slot0first_acc_by_k"][str(kk_)] = dict(n=len(R), base=round(100 * np.mean([D[i]["base"]["type"] == "correct" for i in R]), 2), s0f=round(100 * np.mean([D[i]["s0f"]["type"] == "correct" for i in R]), 2))
print("깨짐:", S["slot0first_break_correct_rows"]); print("k 별:", S["slot0first_acc_by_k"])
# 판정
h1 = S["routes"]["H55.1_top1_step0"]; h2 = S["routes"]["H55.2_slot0first_recovery"]
S["acc_diffullama_s0f_phaseB"] = dict(n=len(RB), base=round(100 * np.mean([D[i]["base"]["type"] == "correct" for i in RB]), 2), s0f=round(100 * np.mean([D[i]["s0f"]["type"] == "correct" for i in RB]), 2))
S["verdict"] = {"H55.1": bool(h1["ci"][0] > 0), "H55.2": bool(h2["ci"][0] > 0), "n_facts_1to4_lt_150": S["acc_diffullama"]["n_facts_1to4"] < 150}
print("판정:", S["verdict"])
json.dump(S, open(f"{OUT}/e55_summary{SF}.json", "w"), ensure_ascii=False, indent=1)
