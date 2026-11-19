"""E50 평가 (CPU).  repo 루트에서:  /NAS/minyeol/limit/.venv-gfmrag/bin/python followup_mk/e50_eval.py [--pilot]
경로 (AGENT_BRIEF): 1~4/5 행 중 이번 실행 base 채점이 entity/broken 인 행.  같은 문구 Qwen 이 맞혔으면 'DLM 고유', 아니면 '공유'.
CI: 사실 (sample_id) 군집 부트스트랩, B = 4000, 2.5/97.5 백분위.  같은 W (재표집 가중) 로 집단 간 차이를 계산.
"""
import collections, glob, json, sys
import numpy as np

PILOT = "--pilot" in sys.argv
D_ = "data"; OUT = "followup_mk"; B = 4000; CONDS = ["base", "칸0먼저", "L2R", "entropy"]
rows = [json.loads(l) for l in open(f"{D_}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
qw = {}
for l in open(f"{D_}/3_classification/full/qwen_final_full.jsonl", encoding="utf-8"):
    d = json.loads(l); qw[(d["sample_id"], d["version"])] = d
Q = [qw[(r["sample_id"], r["version"])] for r in rows]
fid_all = np.array([r["sample_id"] for r in rows])

files = [f"{OUT}/results_e50_pilot.jsonl"] if PILOT else sorted(glob.glob(f"{OUT}/results_e50_s*.jsonl"))
res = collections.defaultdict(dict)
for fn in files:
    for l in open(fn, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        res[d["row"]][d["cond"]] = d
R = sorted(i for i, v in res.items() if all(c in v for c in CONDS))
print(f"파일 {files}, 완결 행 {len(R)} (전체 {len(res)})")
meta7 = {}
for l in open(f"{OUT}/e7/meta.jsonl"): d = json.loads(l); meta7[d["row"]] = d

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", revision="6572adb5535263e4d1a337b56942ba48b6dee2a9", trust_remote_code=True)
first = lambda s: (tok(" " + s.strip())["input_ids"][0] if s and s.strip() else -1)

R = np.array(R); n = len(R)
setn = np.array([res[i]["base"]["set"] for i in R]); fidr = fid_all[R]
G = {c: np.array([res[i][c]["grade"] for i in R]) for c in CONDS}
T0 = {c: np.array([res[i][c]["tok0"] for i in R]) for c in CONDS}
S0 = {c: np.array([res[i][c]["slot0_step"] for i in R]) for c in CONDS}
ATXT = {c: np.array([first(res[i][c]["answer"]) for i in R]) for c in CONDS}
qcor = np.array([bool(Q[i]["correct"]) for i in R]); qfirst = np.array([first(Q[i]["model_answer"]) for i in R])
rank0 = np.array([res[i]["base"]["gold_rank0"] for i in R])
e7ans = np.array([res[i]["base"]["answer"] == meta7[i]["answer"] if i in meta7 else False for i in R])
e7gr = np.array([res[i]["base"]["grade"] == meta7[i]["new_type"] if i in meta7 else False for i in R])


def wmat(f_sel):
    """f_sel: 행별 사실 id 배열.  (사실 목록, 행->사실 번호, W [B, nF] 재표집 횟수)."""
    uf, inv = np.unique(f_sel, return_inverse=True); rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(uf), len(uf)), minlength=len(uf)) for _ in range(B)]).astype(np.float32)
    return uf, inv, W


def agg(inv, nf, v):
    return np.bincount(inv, weights=v.astype(float), minlength=nf)


def ci(a):
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def rate_ci(sel, y, W, inv, nf):
    """sel 행에서 y 의 평균 (사실 군집 재표집)."""
    num = agg(inv, nf, (y & sel) if y.dtype == bool else y * sel); den = agg(inv, nf, sel)
    bt = (W @ num) / np.maximum(W @ den, 1e-9)
    return dict(n=int(sel.sum()), rate=round(float(num.sum() / max(den.sum(), 1e-9)), 4), ci=ci(bt)), bt


def diff_ci(selA, selB, y, W, inv, nf):
    a, ba = rate_ci(selA, y, W, inv, nf); b, bb = rate_ci(selB, y, W, inv, nf)
    d = ba - bb
    return dict(A=a, B=b, diff=round(a["rate"] - b["rate"], 4), diff_ci=ci(d), p_le0=round(float((d <= 0).mean()), 4))


def within_fact_diff(selA, selB, y, f_sel):
    """두 집단이 모두 있는 사실만: 사실별 (A 평균 − B 평균) 의 사실 평균, 사실 재표집."""
    byf = collections.defaultdict(lambda: [[], []])
    for j in np.where(selA | selB)[0]: byf[f_sel[j]][0 if selA[j] else 1].append(float(y[j]))
    ds = np.array([np.mean(a) - np.mean(b) for a, b in byf.values() if a and b])
    if len(ds) == 0: return None
    rng = np.random.default_rng(1); bt = ds[rng.integers(0, len(ds), (B, len(ds)))].mean(1)
    return dict(n_fact=len(ds), diff=round(float(ds.mean()), 4), ci=ci(bt))


S = {}
S["sanity"] = dict(n_rows=n, n_by_set={s: int((setn == s).sum()) for s in ["mid", "five", "zero"]},
                   base_eq_E7_answer=round(float(e7ans.mean()), 4), base_eq_E7_answer_n=[int(e7ans.sum()), n],
                   base_eq_E7_grade=round(float(e7gr.mean()), 4),
                   kan0_prefill_eq_base_arg0=float(np.mean([res[i]["칸0먼저"]["prefill_tok"] == res[i]["base"]["arg0"] for i in R])),
                   kan0_final_tok0_eq_prefill=float(np.mean([res[i]["칸0먼저"]["tok0"] == res[i]["칸0먼저"]["prefill_tok"] for i in R])),
                   L2R_slot0_step_values=sorted(set(S0["L2R"].tolist())),
                   L2R_monotone=float(np.mean([all(np.diff(res[i]["L2R"]["csteps"]) >= 0) for i in R])))

# ---- 집합별 정확도, 유형 분포
S["accuracy"] = {}
for s in ["mid", "five", "zero"]:
    sel = setn == s; uf, inv, W = wmat(fidr[sel]); nf = len(uf); out = {}
    for c in CONDS:
        y = G[c][sel] == "correct"; r_, bt = rate_ci(np.ones(sel.sum(), bool), y, W, inv, nf)
        out[c] = dict(acc=r_["rate"], ci=r_["ci"], types={t: int((G[c][sel] == t).sum()) for t in ["correct", "entity", "broken", "abstain"]})
        if c != "base":
            yb = G["base"][sel] == "correct"; d = y.astype(float) - yb
            _, bd = rate_ci(np.ones(sel.sum(), bool), d, W, inv, nf)
            out[c]["minus_base"] = round(float(d.mean()), 4); out[c]["minus_base_ci"] = ci(bd)
            out[c]["recover_of_base_wrong"] = round(float(y[~yb].mean()), 4) if (~yb).any() else None
            out[c]["break_of_base_correct"] = round(float((~y)[yb].mean()), 4) if yb.any() else None
    S["accuracy"][s] = out

# ---- 경로 (1~4/5, base 오답 entity/broken)
mid = setn == "mid"; wrong = mid & np.isin(G["base"], ["entity", "broken"])
dlm = wrong & qcor; shr = wrong & ~qcor
um, invm, Wm = wmat(fidr[mid]); nfm = len(um)
sub = lambda v: v[mid]
S["routes"] = dict(n_dlm=int(dlm.sum()), n_shared=int(shr.sum()), n_wrong=int(wrong.sum()),
                   base_wrong_types={"dlm": {t: int((G["base"][dlm] == t).sum()) for t in ["entity", "broken"]}, "shared": {t: int((G["base"][shr] == t).sum()) for t in ["entity", "broken"]}})
# step 0 금 첫 토큰 순위 (E49 대조: 1 위 27.8 대 12.6)
top1 = rank0 == 0
S["routes"]["gold_rank0_top1"] = diff_ci(sub(dlm), sub(shr), sub(top1), Wm, invm, nfm)
S["routes"]["gold_rank0_median"] = {"dlm": float(np.median(rank0[dlm])), "shared": float(np.median(rank0[shr]))}
S["recovery"] = {}
for c in ["칸0먼저", "L2R", "entropy"]:
    y = G[c] == "correct"
    d = diff_ci(sub(dlm), sub(shr), sub(y), Wm, invm, nfm)
    d["within_fact"] = within_fact_diff(sub(dlm), sub(shr), sub(y), fidr[mid])
    d["by_type"] = {t: diff_ci(sub(dlm & (G["base"] == t)), sub(shr & (G["base"] == t)), sub(y), Wm, invm, nfm) for t in ["entity", "broken"]}
    S["recovery"][c] = d
h1 = S["recovery"]["칸0먼저"]; h2 = S["recovery"]["L2R"]
S["H50.1"] = dict(diff=h1["diff"], ci=h1["diff_ci"], verdict=bool(h1["diff"] >= 0.08 and h1["diff_ci"][0] > 0))
S["H50.2"] = dict(diff=h2["diff"], ci=h2["diff_ci"], verdict=bool(h2["diff"] >= 0.08 and h2["diff_ci"][0] > 0))

# ---- H50.3 첫 토큰이 같은 문구 Qwen 답 첫 토큰과 같음 (1~4/5 전체; Qwen 빈 답은 불일치)
S["first_tok_match_qwen"] = {}
for name, TT in [("tok0", T0), ("answer_text", ATXT)]:
    o = {}
    for c in CONDS:
        y = (TT[c] == qfirst) & (qfirst >= 0); r_, _ = rate_ci(np.ones(mid.sum(), bool), sub(y), Wm, invm, nfm); o[c] = r_
        if c != "base":
            dd = y.astype(float) - ((TT["base"] == qfirst) & (qfirst >= 0)); r2, _ = rate_ci(np.ones(mid.sum(), bool), sub(dd), Wm, invm, nfm)
            o[c + "_minus_base"] = dict(diff=r2["rate"], ci=r2["ci"])
            nz = mid & (qfirst >= 0); un, invn, Wn = wmat(fidr[nz]); r3, _ = rate_ci(np.ones(nz.sum(), bool), dd[nz], Wn, invn, len(un))
            o[c + "_minus_base_qwen_nonempty"] = dict(n=r3["n"], diff=r3["rate"], ci=r3["ci"])
    S["first_tok_match_qwen"][name] = o
h3 = S["first_tok_match_qwen"]["tok0"]["L2R_minus_base"]
S["H50.3"] = dict(base=S["first_tok_match_qwen"]["tok0"]["base"]["rate"], L2R=S["first_tok_match_qwen"]["tok0"]["L2R"]["rate"], diff=h3["diff"], ci=h3["ci"], verdict=bool(h3["ci"][0] > 0))

# ---- H50.4 base 정답 1~4/5 행이 칸0먼저로 깨지는 비율: Qwen 틀림 대 Qwen 맞힘
bc = mid & (G["base"] == "correct")
S["break"] = {}
for c in ["칸0먼저", "L2R", "entropy"]:
    y = G[c] != "correct"
    d = diff_ci(sub(bc & ~qcor), sub(bc & qcor), sub(y), Wm, invm, nfm)
    d["within_fact"] = within_fact_diff(sub(bc & ~qcor), sub(bc & qcor), sub(y), fidr[mid])
    d["break_to"] = {g: {t: int((G[c][bc & (qcor if g == 'qwen_right' else ~qcor)] == t).sum()) for t in ["entity", "broken", "abstain"]} for g in ["qwen_wrong", "qwen_right"]}
    S["break"][c] = d
h4 = S["break"]["칸0먼저"]
S["H50.4"] = dict(qwen_wrong=h4["A"], qwen_right=h4["B"], diff=h4["diff"], ci=h4["diff_ci"], verdict=bool(h4["diff_ci"][0] > 0))

# ---- 칸 0 확정 step 분포 (경로별, base / entropy / 칸0먼저 / L2R)
def dist(v):
    v = np.asarray(v)
    if len(v) == 0: return None
    h = np.bincount(np.clip(v, 0, 31), minlength=32)
    return dict(n=len(v), mean=round(float(v.mean()), 3), q25=float(np.percentile(v, 25)), median=float(np.median(v)), q75=float(np.percentile(v, 75)),
                p_le2=round(float((v <= 2).mean()), 4), p_ge8=round(float((v >= 8).mean()), 4), p_ge16=round(float((v >= 16).mean()), 4), hist=h.tolist())
groups = {"dlm": dlm, "shared": shr, "mid_base_correct_qwen_right": bc & qcor, "mid_base_correct_qwen_wrong": bc & ~qcor, "five": setn == "five", "zero": setn == "zero"}
S["slot0_step"] = {c: {g: dist(S0[c][sel]) for g, sel in groups.items()} for c in CONDS}
# base 칸 0 확정 step 의 경로 차이 (평균, 사실 군집)
S["slot0_step_base_dlm_minus_shared"] = {c: (lambda d: dict(dlm=d["A"]["rate"], shared=d["B"]["rate"], diff=d["diff"], ci=d["diff_ci"]))(diff_ci(sub(dlm), sub(shr), sub(S0[c].astype(float)), Wm, invm, nfm)) for c in ["base", "entropy"]}
# 칸0먼저 의 step 0 argmax 가 금 첫 토큰인 비율 (경로별) 과, 그 때 복구율
S["kan0_prefill_is_gold"] = {g: round(float(top1[sel].mean()), 4) for g, sel in [("dlm", dlm), ("shared", shr)]}
S["recovery_given_step0_argmax_gold"] = {c: {g: dict(n=int((sel & top1).sum()), rate=round(float((G[c][sel & top1] == "correct").mean()), 4) if (sel & top1).any() else None,
                                                 n_not=int((sel & ~top1).sum()), rate_not=round(float((G[c][sel & ~top1] == "correct").mean()), 4) if (sel & ~top1).any() else None) for g, sel in [("dlm", dlm), ("shared", shr)]} for c in ["칸0먼저", "L2R", "entropy"]}
# 칸0먼저 와 L2R 의 칸 0 토큰은 구조상 같다 (step 0 과 step 1 캔버스가 같아 L2R 은 step 1 에 step 0 argmax 로 칸 0 확정).  base 가 step 0 선두를 끝까지 지킨 비율.
arg0 = np.array([res[i]["base"]["arg0"] for i in R])
S["sanity"]["kan0_tok0_eq_L2R_tok0"] = round(float((T0["칸0먼저"] == T0["L2R"]).mean()), 4)
S["base_tok0_eq_step0_argmax"] = {g: round(float((T0["base"][sel] == arg0[sel]).mean()), 4) for g, sel in [("dlm", dlm), ("shared", shr), ("mid_base_correct", bc), ("five", setn == "five"), ("zero", setn == "zero")]}

fo = f"{OUT}/e50_summary{'_pilot' if PILOT else ''}.json"; json.dump(S, open(fo, "w"), ensure_ascii=False, indent=1)
pct = lambda x: f"{100 * x:.1f}"
print(f"sanity: base==E7 답 {S['sanity']['base_eq_E7_answer_n']} ({pct(S['sanity']['base_eq_E7_answer'])}%), 채점 {pct(S['sanity']['base_eq_E7_grade'])}%")
for s in ["mid", "five", "zero"]:
    print(s, {c: (pct(v["acc"]), v.get("minus_base")) for c, v in S["accuracy"][s].items()})
print(f"경로: DLM 고유 {S['routes']['n_dlm']}, 공유 {S['routes']['n_shared']}; step0 금 1 위 {S['routes']['gold_rank0_top1']['A']['rate']} 대 {S['routes']['gold_rank0_top1']['B']['rate']}")
for c, d in S["recovery"].items(): print(f"복구 {c}: DLM {pct(d['A']['rate'])} 공유 {pct(d['B']['rate'])} 차이 {pct(d['diff'])} CI {[pct(x) for x in d['diff_ci']]}; 같은 사실 안 {d['within_fact']}")
for h in ["H50.1", "H50.2", "H50.3", "H50.4"]: print(h, S[h])
print("칸 0 확정 step (base 평균) DLM/공유:", S["slot0_step_base_dlm_minus_shared"])
print("저장:", fo)
