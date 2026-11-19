"""E57 (b) (CPU): 경로 분리 (DLM 고유 / 공유) 의 혼동 요인.  층화 뒤에도 E49 (step 0 칸 0 금 첫 토큰 1 위) 와 E49c (순서 개입 복구) 차이가 남는가.
E49 행: Dream 1~4/5 중 E44 실행 채점이 entity/broken 인 행 (1,282 / 1,816; E49 와 같은 정의).  결과 = E44 step 0 칸 0 실제 로짓에서 금 첫 토큰 1 위.
E49c 행: set_S 의 broken/entity/correct 중 E27b·E8 base 가 오답인 행 (265 / 332).  결과 = 각 개입에서 정답.
공변량: 금 답 토큰 수 (Qwen 토크나이저, ' ' + gold_value), 금 첫 토큰 빈도 (11,045 사실의 금 첫 토큰 가운데 같은 토큰 수), 토큰 id (BPE 병합 순서 = 빈도 대리),
        s_pop, o_pop, 관계, 문구 판, Qwen 금 첫 토큰 log-prob (e57_qwen_gold.py, 같은 프롬프트 forward 1 회).
지표: Mantel-Haenszel 위험 차이 (층 가중 n1 n0 / N) 와 층 크기 가중 평균 (두 경로 모두 있는 층만), 모든 공변량 선형 확률 모형의 경로 계수.  CI = 사실 군집 부트스트랩 B 2000."""
import collections, json, sys, time
import numpy as np
sys.path.insert(0, "followup_mk")
from e57_common import *
from common import set_S

t0 = time.time(); B = 2000
gid, alen, gfreq = gold_tokens()
alen_b = np.minimum(alen, 4)

def covs(R):
    """행 R 의 층 변수들 (분위는 R 안에서)."""
    ql = np.array([qwen(int(r), "gold_lp") for r in R])
    C = {"답 토큰 수 (1,2,3,4+)": alen_b[R], "금 첫 토큰 빈도 5 분위": qbin(np.log(gfreq[R])), "금 첫 토큰 id 5 분위": qbin(gid[R].astype(float)),
         "s_pop 5 분위": qbin(np.log1p(s_pop[R])), "o_pop 5 분위": qbin(np.log1p(o_pop[R])), "관계": rel[R], "문구 판": ver[R],
         "Qwen 금 logprob 5 분위": qbin(ql)}
    C["[H57.2] Qwen 금 logprob 5 분위 x 답 토큰 수"] = np.array([f"{a}_{b}" for a, b in zip(C["Qwen 금 logprob 5 분위"], C["답 토큰 수 (1,2,3,4+)"])])
    C["Qwen 금 logprob 10 분위 x 답 토큰 수"] = np.array([f"{a}_{b}" for a, b in zip(qbin(ql, 10), alen_b[R])])
    C["Qwen 금 1 위 여부 x 답 토큰 수"] = np.array([f"{int(qwen(int(r), 'gold_rank') == 0)}_{b}" for r, b in zip(R, alen_b[R])])
    C["Qwen 금 logprob 5 분위 x 답 토큰 수 x 관계"] = np.array([f"{a}_{b}" for a, b in zip(C["[H57.2] Qwen 금 logprob 5 분위 x 답 토큰 수"], rel[R])])
    C = {nm: np.unique(v, return_inverse=True)[1] for nm, v in C.items()}
    return C, ql

def design(R, ql):
    cols = [np.ones(len(R))]
    for v in (rel[R], ver[R], alen_b[R], qbin(np.log(gfreq[R])), qbin(np.log1p(s_pop[R])), qbin(np.log1p(o_pop[R])), qbin(ql)):
        u = sorted(set(v.tolist()))[1:]; cols += [(v == x).astype(float) for x in u]
    cols.append(ql); return np.column_stack(cols)

def analyze(R, d, y, tag):
    C, ql = covs(R); f = fid[R]; out = {"n (DLM 고유 / 공유)": [int(d.sum()), int((~d).sum())]}
    out["층화 없음 (DLM 고유 − 공유)"] = boot_facts(f, lambda ix: y[ix][d[ix]].mean() - y[ix][~d[ix]].mean(), B)
    out["비율 % (DLM 고유 / 공유)"] = [round(100 * float(y[d].mean()), 1), round(100 * float(y[~d].mean()), 1)]
    st = {}
    for nm, s in C.items():
        mh, wavg, cover = mh_rd(y, d, s)
        st[nm] = {"MH 차이": boot_facts(f, lambda ix: mh_rd(y[ix], d[ix], s[ix])[0], B), "층 크기 가중 차이": r4(wavg), "두 경로가 다 있는 층에 든 행 비율": r4(cover)}
    out["층화"] = st
    X = design(R, ql)
    def lpm(ix):
        A = np.c_[d[ix].astype(float), X[ix]]; return np.linalg.lstsq(A, y[ix].astype(float), rcond=None)[0][0]
    out["모든 공변량 선형 확률 모형의 경로 계수"] = boot_facts(f, lpm, B)
    print(tag, json.dumps({k_: (v if k_ != "층화" else {a: b["MH 차이"] for a, b in v.items()}) for k_, v in out.items()}, ensure_ascii=False), f"{time.time() - t0:.0f}s", flush=True)
    return out

def balance(R, d):
    C, ql = covs(R); m = lambda v: [r4(np.nanmean(v[d])), r4(np.nanmean(v[~d]))]
    o = {"답 토큰 수": m(alen[R].astype(float)), "log 금 첫 토큰 빈도": m(np.log(gfreq[R])), "log1p s_pop": m(np.log1p(s_pop[R])), "log1p o_pop": m(np.log1p(o_pop[R])),
         "Qwen 금 logprob": m(ql), "Qwen 금 1 위 %": [round(100 * float(np.mean([qwen(int(r), 'gold_rank') == 0 for r in R[d]])), 1), round(100 * float(np.mean([qwen(int(r), 'gold_rank') == 0 for r in R[~d]])), 1)],
         "원문 (question) %": [round(100 * float((ver[R][d] == "question").mean()), 1), round(100 * float((ver[R][~d] == "question").mean()), 1)],
         "Dream step0 칸0 argmax = Qwen 1 위 %": [round(100 * float(np.mean([E44[int(r)]["argmax0"] == qwen(int(r), "top1_id") for r in R[d]])), 1), round(100 * float(np.mean([E44[int(r)]["argmax0"] == qwen(int(r), "top1_id") for r in R[~d]])), 1)]}
    rc = collections.Counter(rel[R][d]); rs = collections.Counter(rel[R][~d])
    o["관계별 (DLM 고유 / 공유 행 수)"] = {x: [rc[x], rs[x]] for x in sorted(set(rel[R]), key=lambda x: -(rc[x] + rs[x]))}
    return o

O = {"공변량 정의": {"답 토큰 수": "len(Qwen 토크나이저(' ' + gold_value))", "금 첫 토큰 빈도": "11,045 사실의 금 첫 토큰 중 같은 토큰을 가진 사실 수 (사실 단위, 문구 5 배 중복 없음)",
                 "금 첫 토큰 id": "Qwen BPE 토큰 id (작을수록 먼저 병합 = 학습 말뭉치에서 흔함; 거친 대리)", "Qwen 금 logprob": "Qwen2.5-7B 가 같은 프롬프트 끝에서 금 첫 토큰에 주는 log softmax (e57_qwen_gold.py)",
                 "분위": "분석 행 안에서 순위로 5 (또는 10) 등분"}}
# ---------------------------------------------------------------- E49
R49 = np.array(sorted(r for r, x in E44.items() if mid[r] and x["new_type"] in ("entity", "broken")))
d49 = q_corr[R49]; y_top1 = np.array([E44[int(r)]["rank0"] == 0 for r in R49]); y_lost = np.array([E44[int(r)]["lost"] for r in R49]); y_top5 = np.array([E44[int(r)]["rank0"] < 5 for r in R49])
O["E49 균형표 (DLM 고유 / 공유 평균)"] = balance(R49, d49)
O["E49 step0 금 1 위"] = analyze(R49, d49, y_top1, "E49 top1")
O["E49 1 위였다가 첫 칸에서 잃음"] = analyze(R49, d49, y_lost, "E49 lost")
O["E49 top-5"] = analyze(R49, d49, y_top5, "E49 top5")
h = O["E49 step0 금 1 위"]["층화"]["[H57.2] Qwen 금 logprob 5 분위 x 답 토큰 수"]["MH 차이"]
O["H57.2"] = {"정의": "Qwen 금 logprob 5 분위 x 답 토큰 수 층화 뒤 E49 step0 금 1 위 차이 (DLM 고유 − 공유) 하한 > 0", "값 (MH, 사실 군집 CI)": h, "판정": bool(h[1] > 0)}
# 같은 사실 안 (두 경로 모두 있는 사실) + Qwen 층
bf = collections.defaultdict(lambda: [[], []])
for t, r in enumerate(R49): bf[fid[r]][int(d49[t])].append(t)
both = np.array(sorted(t for v in bf.values() if v[0] and v[1] for vv in v for t in vv))
O["E49 같은 사실 안 (두 경로 모두 있는 사실)"] = {"사실": int(len(set(fid[R49[both]]))), "행": int(len(both)),
    "층화 없음 (행 가중)": boot_facts(fid[R49[both]], lambda ix: y_top1[both][ix][d49[both][ix]].mean() - y_top1[both][ix][~d49[both][ix]].mean(), B),
    "사실을 층으로 한 MH": boot_facts(fid[R49[both]], lambda ix: mh_rd(y_top1[both][ix], d49[both][ix], fid[R49[both]][ix])[0], B)}
print("같은 사실", O["E49 같은 사실 안 (두 경로 모두 있는 사실)"], flush=True)

# ---------------------------------------------------------------- E49c
def load(fn):
    R = collections.defaultdict(dict)
    for l in open(f"{OUT}/{fn}", encoding="utf-8"): x = json.loads(l); R[x["cond"]][(x["set"], x["row"])] = x["new_type"]
    return R
E8 = load("results_e8.jsonl"); E27 = load("results_e27b.jsonl")
def wrong_rows(Rc):
    return sorted((s, r) for (s, r), t in Rc["base"].items() if s in ("broken", "entity", "correct") and t in ("entity", "broken"))
W27 = wrong_rows(E27); W8 = wrong_rows(E8); assert W27 == W8, "E8 과 E27b base 오답 행이 다름"
Rc = np.array([r for _, r in W27]); dc = q_corr[Rc]
O["E49c 균형표"] = balance(Rc, dc)
CONDS = [("칸0먼저 (E27b)", E27, "칸0먼저"), ("origin (E8)", E8, "origin"), ("entropy (E8)", E8, "entropy"), ("maskgit_plus (E8)", E8, "maskgit_plus"), ("canvas16 (E8, 대조)", E8, "canvas16"), ("canvas64 (E8, 대조)", E8, "canvas64")]
EC = {}
for nm, Rd, c in CONDS:
    y = np.array([Rd[c][x] == "correct" for x in W27]); EC[nm] = analyze(Rc, dc, y, "E49c " + nm)
O["E49c 순서 개입 복구"] = EC
# 특이성: (순서 개입의 경로 차이) − (캔버스 대조의 경로 차이).  같은 행, 같은 base.  y_c − y_k 의 경로 차이 = 두 경로 차이의 차.
Ys = {nm: np.array([Rd[c][x] == "correct" for x in W27]).astype(float) for nm, Rd, c in CONDS}
Cc, qlc = covs(Rc); s57 = Cc["[H57.2] Qwen 금 logprob 5 분위 x 답 토큰 수"]; Xc_ = design(Rc, qlc); fc = fid[Rc]
DID = {}
for a in [x[0] for x in CONDS[:4]]:
    for b in [x[0] for x in CONDS[4:]]:
        z = Ys[a] - Ys[b]
        def lpm_z(ix):
            A = np.c_[dc[ix].astype(float), Xc_[ix]]; return np.linalg.lstsq(A, z[ix], rcond=None)[0][0]
        DID[f"{a} − {b}"] = {"층화 없음": boot_facts(fc, lambda ix: z[ix][dc[ix]].mean() - z[ix][~dc[ix]].mean(), B),
                             "H57.2 층 MH": boot_facts(fc, lambda ix: mh_rd(z[ix], dc[ix], s57[ix])[0], B), "모든 공변량 LPM": boot_facts(fc, lpm_z, B)}
        print("DiD", a, b, DID[f"{a} − {b}"], flush=True)
O["E49c 특이성 (순서 개입 경로 차이 − 캔버스 대조 경로 차이)"] = DID
# 매개 점검: step 0 Dream 금 1 위 (또는 순위 구간) 로 층화하면 경로 차이가 남는가
rk = np.array([E44[int(r)]["rank0"] for r in Rc]); rkb = np.where(rk == 0, 0, np.where(rk < 5, 1, 2))
MED = {}
for nm, Rd, c in CONDS[:4]:
    y = np.array([Rd[c][x] == "correct" for x in W27]); f = fid[Rc]
    MED[nm] = {"Dream step0 금 순위 구간 (1 위 / 2~5 / 6+) 층 MH": boot_facts(f, lambda ix: mh_rd(y[ix], dc[ix], rkb[ix])[0], B),
               "구간별 복구 % (DLM 고유 / 공유, n)": {b: [round(100 * float(y[(rkb == j) & dc].mean()), 1) if ((rkb == j) & dc).any() else None, round(100 * float(y[(rkb == j) & ~dc].mean()), 1) if ((rkb == j) & ~dc).any() else None,
                                                     int(((rkb == j) & dc).sum()), int(((rkb == j) & ~dc).sum())] for j, b in enumerate(["1 위", "2~5 위", "6 위 이하"])}}
    print("매개", nm, MED[nm], flush=True)
O["E49c 매개 점검 (Dream step0 금 순위로 층화)"] = MED
# E49 도 같은 매개 점검의 거울: Dream step0 1 위 행 안에서 '잃음' 비율이 경로로 다른가
t1 = y_top1
O["E49 step0 1 위 행 안에서 잃음 비율 (DLM 고유 / 공유, n)"] = [round(100 * float(y_lost[t1 & d49].mean()), 1), round(100 * float(y_lost[t1 & ~d49].mean()), 1), int((t1 & d49).sum()), int((t1 & ~d49).sum())]
O["E50"] = "results_e50* 없음 (실행 전): E50 은 판정하지 않음"
O["걸린 시간 s"] = int(time.time() - t0)
upd_summary("b_routes", O); print("H57.2", O["H57.2"]); print("DONE")
