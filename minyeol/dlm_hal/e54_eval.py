"""E54 (PREREG.md) 평가 (CPU).  results_e54_{a,b}_s*.jsonl -> e54_summary.json.
(a) 확정 전 교환.  정방향: F 행 중 같은 묶음 base 가 오답 (entity/broken) 인 행의 조건별 정답률 (= 복구율), 경로 D (DLM 고유) / H (공유).
    rank 곡선: pc{r} (같은 사실 차이 부분공간) 대 rnd{r} (무작위), 같은 행 짝 차이.  역방향: S 행 중 base 정답인 행이 조건에서 틀리는 비율 (= 깨짐).
    H54.1 (결과 보기 전 결정): 주 판정 = 층 20 전체 교환 대 E51 F+중립 캔버스 (tau 2) 복구, 같은 F 행.  (교환 - 캔버스) 의 H - D 차이 하한 > 0 이고 교환 복구 H >= D.
    다른 조합 (층 24, tau 4) 은 견고성으로 보고.
(b) 지우기.  집단 (같은 묶음 base 기준): 1~4/5 전체 (정확도 변화 Δ), 1~4/5 base 정답 (유지율), 1~4/5 base 오답 (복구율), 5/5, 0/5.
    H54.2: LEACE rank 1 의 1~4/5 정확도 변화 |Δ| < 2 점 (네 조건 모두: 층 20/24 x step 0~3/모든 step).
CI: 사실 군집 부트스트랩 B = 4000.
usage: python followup_mk/e54_eval.py [--pilot]"""
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import rows, k, fid, mid, OUT
PIL = "--pilot" in sys.argv; SFX = "_pilot" if PIL else ""
B = 4000; WRONG = ("entity", "broken")
def readj(pat):
    out = []
    for fn in sorted(glob.glob(pat)):
        for l in open(fn, encoding="utf-8"):
            try: out.append(json.loads(l))
            except Exception: pass
    return out

def boot_groups(vals, facts, seed=0):
    """vals: 행별 값 (float), facts: 행별 사실.  사실 군집 부트스트랩 평균 [B]."""
    uf, inv = np.unique(facts, return_inverse=True); s = np.zeros(len(uf)); c = np.zeros(len(uf)); np.add.at(s, inv, vals); np.add.at(c, inv, 1)
    W = np.random.default_rng(seed).multinomial(len(uf), np.ones(len(uf)) / len(uf), size=B)
    return (W @ s) / np.maximum(W @ c, 1e-9), uf, s, c, W
def mean_ci(vals, facts, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0: return None
    bm = boot_groups(vals, facts, seed)[0]
    return dict(n=int(len(vals)), mean=round(float(vals.mean()) * 100, 2), ci=[round(float(np.percentile(bm, 2.5)) * 100, 2), round(float(np.percentile(bm, 97.5)) * 100, 2)])
def diff_ci(v1, f1, v0, f0, seed=0):
    """서로 다른 두 집단의 평균 차이, 공통 사실 재표집."""
    v1, v0 = np.asarray(v1, float), np.asarray(v0, float)
    allf = np.unique(np.concatenate([f1, f0])); idx = {f: j for j, f in enumerate(allf)}
    i1 = np.array([idx[f] for f in f1]); i0 = np.array([idx[f] for f in f0])
    s1 = np.zeros(len(allf)); c1 = np.zeros(len(allf)); s0 = np.zeros(len(allf)); c0 = np.zeros(len(allf))
    np.add.at(s1, i1, v1); np.add.at(c1, i1, 1); np.add.at(s0, i0, v0); np.add.at(c0, i0, 1)
    W = np.random.default_rng(seed).multinomial(len(allf), np.ones(len(allf)) / len(allf), size=B)
    d = (W @ s1) / np.maximum(W @ c1, 1e-9) - (W @ s0) / np.maximum(W @ c0, 1e-9)
    return dict(diff=round(float(v1.mean() - v0.mean()) * 100, 2), ci=[round(float(np.percentile(d, 2.5)) * 100, 2), round(float(np.percentile(d, 97.5)) * 100, 2)], n1=int(len(v1)), n0=int(len(v0)))

out = {}
# ======================= (a)
A = readj(f"{OUT}/results_e54_a{SFX}_s*.jsonl")
if A:
    rec = {}
    for d in A: rec[(d["role"], d["row"], d["donor"], d["cond"])] = d
    pairs = json.load(open(f"{OUT}/e54/pairs.json")); pid = {p["pid"]: p for p in pairs}
    base = {(d["role"], d["row"]): d for d in A if d["cond"] == "base"}
    conds_F = sorted({d["cond"] for d in A if d["role"] == "F" and d["cond"] != "base"}); conds_S = sorted({d["cond"] for d in A if d["role"] == "S" and d["cond"] != "base"})
    oa = {"sanity": {}}
    # sanity: base 가 E7 / E44 와 같은 답, self 교환 = base
    M7 = {}
    for l in open(f"{OUT}/e7/meta.jsonl"): x = json.loads(l); M7[x["row"]] = x
    bl = list(base.values()); oa["sanity"]["base_answer_eq_E7"] = round(np.mean([b["answer"] == M7[b["row"]]["answer"] for b in bl if b["row"] in M7]), 4)
    oa["sanity"]["base_grade_eq_E7"] = round(np.mean([b["new_type"] == M7[b["row"]]["new_type"] for b in bl if b["row"] in M7]), 4)
    for c in ("self20", "self24"):
        xs = [d for d in A if d["cond"] == c]
        if xs: oa["sanity"][f"{c}_answer_eq_base"] = round(np.mean([d["answer"] == base[(d["role"], d["row"])]["answer"] for d in xs]), 4)
    # 정방향
    Fp = [p for p in pairs if ("F", p["F"]) in base]
    Fw = [p for p in Fp if base[("F", p["F"])]["new_type"] in WRONG]
    oa["n_pairs_run"] = len(Fp); oa["n_F_base_wrong"] = dict(collections.Counter(p["route"] for p in Fw)); oa["F_base_wrong_rate"] = round(len(Fw) / max(len(Fp), 1), 4)
    def frec(c, ps): return np.array([rec[("F", p["F"], p["S"], c)]["new_type"] == "correct" for p in ps if ("F", p["F"], p["S"], c) in rec], float)
    def ffac(c, ps): return np.array([p["fact"] for p in ps if ("F", p["F"], p["S"], c) in rec])
    fw = {"all": Fw, "D": [p for p in Fw if p["route"] == "D"], "H": [p for p in Fw if p["route"] == "H"]}
    oa["forward_recovery"] = {c: {g: mean_ci(frec(c, ps), ffac(c, ps)) for g, ps in fw.items()} for c in conds_F}
    oa["forward_D_minus_H"] = {c: diff_ci(frec(c, fw["D"]), ffac(c, fw["D"]), frec(c, fw["H"]), ffac(c, fw["H"])) for c in conds_F}
    # rank 곡선: pc - rnd 같은 행 짝 차이
    curve = {}
    for L_ in (20, 24):
        for r in (1, 4, 16, 64):
            for g, ps in fw.items():
                a = frec(f"pc{r}_{L_}", ps); b = frec(f"rnd{r}_{L_}", ps)
                if len(a) and len(a) == len(b):
                    curve[f"L{L_}_r{r}_{g}"] = dict(pc=round(a.mean() * 100, 2), rnd=round(b.mean() * 100, 2), pc_minus_rnd=mean_ci(a - b, ffac(f"pc{r}_{L_}", ps)))
        for g, ps in fw.items():
            a = frec(f"full{L_}", ps); curve[f"L{L_}_full_{g}"] = round(a.mean() * 100, 2) if len(a) else None
    oa["rank_curve"] = curve
    # 칸 0 금 순위 (step 1 = 교환이 들어간 첫 확정 step 의 forward): 개선 비율
    oa["forward_s1_gold_top1"] = {c: {g: round(float(np.mean([rec[("F", p["F"], p["S"], c)]["s1_rank"] == 0 for p in ps if ("F", p["F"], p["S"], c) in rec])) * 100, 2) for g, ps in fw.items()} for c in ["base"] + conds_F if c != "base"}
    oa["forward_s1_gold_top1"]["base"] = {g: round(float(np.mean([base[("F", p["F"])]["s1_rank"] == 0 for p in ps])) * 100, 2) for g, ps in fw.items()}
    # 역방향 (S 수신): base 정답인 S 의 깨짐 (조건에서 정답 아님), 짝의 F 경로별
    Sp = [p for p in pairs if ("S", p["S"]) in base and base[("S", p["S"])]["new_type"] == "correct"]
    def sbrk(c, ps): return np.array([rec[("S", p["S"], p["F"], c)]["new_type"] != "correct" for p in ps if ("S", p["S"], p["F"], c) in rec], float)
    def sfac(c, ps): return np.array([p["fact"] for p in ps if ("S", p["S"], p["F"], c) in rec])
    sp = {"all": Sp, "D": [p for p in Sp if p["route"] == "D"], "H": [p for p in Sp if p["route"] == "H"]}
    oa["reverse_n_S_base_correct"] = {g: len(v) for g, v in sp.items()}
    oa["reverse_breaking"] = {c: {g: mean_ci(sbrk(c, ps), sfac(c, ps)) for g, ps in sp.items()} for c in conds_S}
    # 역방향에서 S 가 F 의 답 (첫 토큰) 을 따라가는 비율
    oa["reverse_copies_F_answer"] = {c: round(float(np.mean([rec[("S", p["S"], p["F"], c)]["answer"] == base[("F", p["F"])]["answer"] for p in Sp if ("S", p["S"], p["F"], c) in rec and ("F", p["F"]) in base])) * 100, 2) for c in conds_S}
    # H54.1: E51 중립 캔버스와 같은 F 행
    h541 = {}
    for tau in (2, 4):
        E = readj(f"{OUT}/results_e51_t{tau}.jsonl")
        if not E: continue
        neut = {d["row"]: d for d in E if d["cond"] == "neut" and d.get("role") == "F"}
        for L_ in (20, 24):
            v_sw, v_cv, fs, ro = [], [], [], []
            for p in Fw:
                if p["F"] in neut and neut[p["F"]]["base_type"] in WRONG and ("F", p["F"], p["S"], f"full{L_}") in rec:
                    v_sw.append(rec[("F", p["F"], p["S"], f"full{L_}")]["new_type"] == "correct"); v_cv.append(neut[p["F"]]["new_type"] == "correct"); fs.append(p["fact"]); ro.append(p["route"])
            v_sw, v_cv, fs, ro = np.array(v_sw, float), np.array(v_cv, float), np.array(fs), np.array(ro)
            if len(v_sw) < 20: continue
            dd = v_sw - v_cv; mD, mH = ro == "D", ro == "H"
            h541[f"L{L_}_tau{tau}"] = dict(n=dict(D=int(mD.sum()), H=int(mH.sum())), swap=dict(D=mean_ci(v_sw[mD], fs[mD]), H=mean_ci(v_sw[mH], fs[mH])), canvas=dict(D=mean_ci(v_cv[mD], fs[mD]), H=mean_ci(v_cv[mH], fs[mH])),
                                            swap_H_minus_D=diff_ci(v_sw[mH], fs[mH], v_sw[mD], fs[mD]), canvas_D_minus_H=diff_ci(v_cv[mD], fs[mD], v_cv[mH], fs[mH]),
                                            dd_swap_minus_canvas_H_minus_D=diff_ci(dd[mH], fs[mH], dd[mD], fs[mD]))
    oa["H54.1"] = h541
    if "L20_tau2" in h541:
        x = h541["L20_tau2"]; oa["H54.1_verdict"] = bool(x["swap_H_minus_D"]["diff"] >= 0 and x["dd_swap_minus_canvas_H_minus_D"]["ci"][0] > 0)
    else: oa["H54.1_verdict"] = "E51 결과 없음"
    out["a"] = oa

# ======================= (b)
Bd = readj(f"{OUT}/results_e54_b{SFX}_s*.jsonl")
if Bd:
    rec = {(d["row"], d["cond"]): d for d in Bd}; base = {d["row"]: d for d in Bd if d["cond"] == "base"}
    conds = sorted({d["cond"] for d in Bd if d["cond"] != "base"})
    rws = sorted(base); ism = {r: bool(mid[r]) for r in rws}
    grp_ = {"mid": [r for r in rws if mid[r]], "mid_base_correct": [r for r in rws if mid[r] and base[r]["new_type"] == "correct"], "mid_base_wrong": [r for r in rws if mid[r] and base[r]["new_type"] in WRONG],
            "five": [r for r in rws if k[r] == 5], "zero": [r for r in rws if k[r] == 0]}
    ob = {"n": {g: len(v) for g, v in grp_.items()}, "base_acc": {g: round(float(np.mean([base[r]["new_type"] == "correct" for r in v])) * 100, 2) for g, v in grp_.items()}}
    M7 = {}
    for l in open(f"{OUT}/e7/meta.jsonl"): x = json.loads(l); M7[x["row"]] = x
    ob["sanity_base_answer_eq_E7"] = round(np.mean([base[r]["answer"] == M7[r]["answer"] for r in rws if r in M7]), 4)
    res = {}
    for c in conds:
        rc = {}
        for g, v in grp_.items():
            v = [r for r in v if (r, c) in rec]
            if not v: continue
            y1 = np.array([rec[(r, c)]["new_type"] == "correct" for r in v], float); y0 = np.array([base[r]["new_type"] == "correct" for r in v], float); fs = np.array([fid[r] for r in v])
            rc[g] = dict(acc=round(y1.mean() * 100, 2), delta=mean_ci(y1 - y0, fs), changed_answer=round(float(np.mean([rec[(r, c)]["answer"] != base[r]["answer"] for r in v])) * 100, 2))
        res[c] = rc
    ob["by_cond"] = res
    # 지우개 - 무작위 대조 (같은 층, 모든 step; 같은 행 짝 차이) 의 1~4/5 Δ 차이
    cmpd = {}
    for L_ in (20, 24):
        for er, rc_ in ((f"leace{L_}_all", f"rnd1_{L_}_all"), (f"inlp{L_}_all", f"rndr_{L_}_all")):
            v = [r for r in grp_["mid"] if (r, er) in rec and (r, rc_) in rec]
            if v:
                a = np.array([rec[(r, er)]["new_type"] == "correct" for r in v], float) - np.array([rec[(r, rc_)]["new_type"] == "correct" for r in v], float)
                cmpd[f"{er}_minus_{rc_}"] = mean_ci(a, np.array([fid[r] for r in v]))
    ob["eraser_minus_random_mid"] = cmpd
    le = [c for c in conds if c.startswith("leace")]
    ob["H54.2"] = {c: res[c]["mid"]["delta"] for c in le if "mid" in res[c]}
    ob["H54.2_verdict"] = bool(all(abs(x["mean"]) < 2 for x in ob["H54.2"].values())) if ob["H54.2"] else None
    fs_ = os.path.join(OUT, "e54", "fit_summary.json")
    if os.path.exists(fs_):
        fsum = json.load(open(fs_)); ob["fit"] = {L_: {g: dict(inlp_rank=v["inlp_rank"], probe_auc=v["probe_auc"], dm_auc=v["dm_auc"], leace_move_norm=v["leace_move_norm"], state_norm=v["state_norm"]) for g, v in fsum[L_].items()} for L_ in ("L20", "L24") if L_ in fsum}
    out["b"] = ob
json.dump(out, open(f"{OUT}/e54_summary{SFX}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False)[:3000])
