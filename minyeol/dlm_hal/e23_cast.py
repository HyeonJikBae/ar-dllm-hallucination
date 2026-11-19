"""E23 (PREREG.md, CPU): CAST 식 기준선.  step 0 상태 (층 20/27 프롬프트 끝) 만으로 다시 생성할 행을 고르고, E12 의 Brel c1 다시 생성 결과를 써서
같은 다시 생성 비율에서 궤적 감지 ('최종답과 다른 step 비율', e12_sc.npy) 와 이득을 비교.  방향·문턱은 주어 5 겹 겹밖."""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import grp, k, fid, mid, unit, NF, OUT, e7_rows
D7 = f"{OUT}/e7"; _, _, F7 = e7_rows(); R = np.array([int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]); n = len(R); ism = mid[R]; isf = ~ism
R7 = np.load(f"{D7}/rows.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
p7 = {int(r): j for j, r in enumerate(R7)}; J = np.array([p7[i] for i in R]); cfg = json.load(open(f"{D7}/config.json")); LAY = cfg["layers"]
first = np.array([meta[int(i)]["new_type"] == "correct" for i in R]); g = grp[R]; fids = fid[R]
LP = np.asarray(np.load(f"{D7}/lp.npy", mmap_mode="r")[J]).astype(np.float32)
SC_T = np.load(f"{OUT}/e12_sc.npy")
E = {}
for f in sorted(glob.glob(f"{OUT}/results_e12_[A-Z].jsonl")):
    for l in open(f, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        if d["cond"] == "Brel c1": E[d["row"]] = d["new_type"] == "correct"
redo = np.array([E[int(i)] for i in R]); assert len(E) >= n
wrong = ism & ~first
def cast_scores(Xh):
    """CAST 식: 조건 방향 = (1~4/5 오답 평균 - 나머지 평균) 단위 벡터, 점수 = 평균 제거한 상태와의 cosine (겹밖)."""
    s = np.zeros(n)
    for gg in range(NF):
        tr, te = g != gg, g == gg; mu = Xh[tr].mean(0); v = unit(Xh[tr & wrong].mean(0) - Xh[tr & ~wrong].mean(0))
        Z = Xh[te] - mu; s[te] = (Z @ v) / (np.linalg.norm(Z, axis=1) + 1e-9)
    return s
def mid_score(Xh):
    """step 0 최선 학습 없는 점수 (E10): 가운데K = -|h.K - m|, K = 관계 안 5/5 - 0/5 평균 (여기서는 0/5 가 E12 행에 없어 E7 의 0/5 로)."""
    return None
SCORES = {"궤적 (최종답과 다른 step 비율)": SC_T, "첫4칸 logprob s0 (-)": -LP[:, 0, :4].mean(1)}
for l_ in (20, 27):
    Xh = np.asarray(np.load(f"{D7}/hid.npy", mmap_mode="r")[J, 0, LAY.index(l_), 0], dtype=np.float32); SCORES[f"CAST식 cos L{l_} s0"] = cast_scores(Xh)
rng = np.random.default_rng(0)
def gain(sel):
    p = np.where(sel, redo, first); d = (p.astype(float) - first.astype(float))[ism]; return d
def ci(d, B=2000):
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]) * 100; return [round(float(d.mean() * 100), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
O = {"기준": "Brel c1 다시 생성 결과 (E12), 평가 행 전체 (1~4/5 5,950 + 5/5 2,000) 기준 상위 q 를 다시 생성"}
def auc(y, s):
    from sklearn.metrics import roc_auc_score; return round(float(roc_auc_score(y, s)), 3)
for nm, sc in SCORES.items():
    o = {"1~4/5 오답 대 나머지 (평가 행)": auc(wrong.astype(int), sc)}
    for q in (.2, .3, .4, .5):
        sel = sc >= np.quantile(sc, 1 - q); d = gain(sel); o[f"상위{int(q * 100)}%"] = {"이득 [CI]": ci(d), "다시생성 1~4/5 %": round(100 * float(sel[ism].mean()), 1), "다시생성 5/5 %": round(100 * float(sel[isf].mean()), 1),
                                                                                       "5/5 정답 %": round(100 * float(np.where(sel, redo, first)[isf].mean()), 2)}
    O[nm] = o; print(nm, json.dumps(o, ensure_ascii=False), flush=True)
t = O["궤적 (최종답과 다른 step 비율)"]
for q in (".2", ".3", ".4", ".5"):
    pass
dif = {}
for nm in SCORES:
    if nm.startswith("궤적"): continue
    for q in (20, 30, 40, 50):
        a = gain(SCORES["궤적 (최종답과 다른 step 비율)"] >= np.quantile(SCORES["궤적 (최종답과 다른 step 비율)"], 1 - q / 100)); b = gain(SCORES[nm] >= np.quantile(SCORES[nm], 1 - q / 100)); dd = a - b
        bs = np.array([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]) * 100; dif[f"{nm} 상위{q}%"] = [round(float(dd.mean() * 100), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
O["궤적 - 기준선 이득 차이 [CI]"] = dif; print(json.dumps(dif, ensure_ascii=False))
O["H23.1"] = bool(all(v[1] > 0 and v[0] >= 2 for kx, v in dif.items() if "첫4칸" not in kx))
json.dump(O, open(f"{OUT}/e23_summary.json", "w"), ensure_ascii=False, indent=1); print("H23.1", O["H23.1"], "DONE", flush=True)
