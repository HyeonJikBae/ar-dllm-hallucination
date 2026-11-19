"""E51 평가 (CPU).  usage (repo 루트): python followup_mk/e51_eval.py [--pilot]
단위 = (짝, tau).  주 분석: 두 tau 를 모은 것 (사실 군집 부트스트랩, 같은 짝의 두 tau 는 같은 군집), tau 별도 함께.
힌트 없음 = 비교에 들어가는 캔버스 (두 조건의 주입 캔버스) 가 모두 힌트 없음.  엄격 (짝의 모든 캔버스가 힌트 없음) 도 함께."""
import collections, json, sys
import numpy as np
OUT = "followup_mk"; PIL = "--pilot" in sys.argv; SFX = "_pilot" if PIL else ""; NB = 4000; TAUS = (2, 4); WRONG = ("entity", "broken")
rd = lambda fn: [json.loads(l) for l in open(fn, encoding="utf-8")]
pairs = json.load(open(f"{OUT}/e51_pairs{SFX}.json")); B_ = {d["row"]: d for d in rd(f"{OUT}/results_e51{SFX}_base.jsonl")}
MID = None
O = {"설정": dict(B=NB, 짝_선택=len(pairs), 선택_경로=dict(collections.Counter(p["route"] for p in pairs)), 사실=len({p["fact"] for p in pairs}))}
# ---- base sanity: E7 라벨과 일치
O["sanity_base_대_E7"] = dict(정오일치=f"{sum((d['new_type'] == 'correct') == (d['e7_type'] == 'correct') for d in B_.values())}/{len(B_)}",
                             답일치=f"{sum(d['answer'] == d['e7_answer'] for d in B_.values())}/{len(B_)}")
U = []; SAN = {}
for tau in TAUS:
    R = rd(f"{OUT}/results_e51{SFX}_t{tau}.jsonl"); ix = {}
    for d in R: ix[(d["row"], d["cond"]) if d["cond"] != "swap" else (d["row"], "swap", d["partner"])] = d
    own = [d for d in R if d["cond"] == "own"]
    SAN[f"tau{tau}"] = dict(자기캔버스_이어생성_답이_base_와_같음=f"{sum(d['same_as_base'] for d in own)}/{len(own)}",
                            최종캔버스_같음=f"{sum(d['final_same_as_base'] for d in own)}/{len(own)}",
                            정오_같음=f"{sum((d['new_type'] == 'correct') == (d['base_type'] == 'correct') for d in own)}/{len(own)}")
    for p in pairs:
        F, S = p["F"], p["S"]
        need = [(F, "own"), (F, "swap", S), (F, "neut"), (F, "neutB"), (F, "reset"), (S, "own"), (S, "swap", F), (S, "neut"), (S, "neutB")]
        if not all(kk in ix for kk in need): continue
        g = {("F", c): ix[(F, c)] for c in ("own", "neut", "neutB", "reset")}; g.update({("S", c): ix[(S, c)] for c in ("own", "neut", "neutB")})
        g[("F", "swap")] = ix[(F, "swap", S)]; g[("S", "swap")] = ix[(S, "swap", F)]
        U.append(dict(pid=p["pid"], fact=p["fact"], route=p["route"], tau=tau, g=g))
O["sanity_자기캔버스"] = SAN
N = len(U); fact = np.array([u["fact"] for u in U]); route = np.array([u["route"] for u in U]); tauv = np.array([u["tau"] for u in U])
acc = lambda r, c: np.array([u["g"][(r, c)]["new_type"] == "correct" for u in U], float)
acc0 = lambda r, c: np.array([bool(u["g"][(r, c)].get("final0_is_gold", False)) for u in U], float)   # 최종 칸 0 = 금 첫 토큰 (주입 토큰이 답 문자열에 섞이는 효과를 뺀 결과)
hint = lambda r, c: np.array([u["g"][(r, c)]["hint"] for u in U], bool)
rank = lambda r, c: np.array([u["g"][(r, c)]["s0_rank"] for u in U], float)
logit = lambda r, c: np.array([u["g"][(r, c)]["s0_logit"] for u in U], float)
CONDS = [("F", "own"), ("F", "swap"), ("F", "neut"), ("F", "neutB"), ("F", "reset"), ("S", "own"), ("S", "swap"), ("S", "neut"), ("S", "neutB")]
NAME = {("F", "own"): "F+자기", ("F", "swap"): "F+S캔버스", ("F", "neut"): "F+중립", ("F", "neutB"): "F+경계보존중립", ("F", "reset"): "F+전부마스크(참고)",
        ("S", "own"): "S+자기", ("S", "swap"): "S+F캔버스", ("S", "neut"): "S+중립", ("S", "neutB"): "S+경계보존중립"}
HF_ALL = ~np.any(np.stack([hint(*c) for c in CONDS if c[1] != "reset"]), 0)   # 엄격: 짝의 모든 캔버스 힌트 없음
uf, finv = np.unique(fact, return_inverse=True); rng = np.random.default_rng(0); W = np.stack([np.bincount(rng.integers(0, len(uf), len(uf)), minlength=len(uf)) for _ in range(NB)])


def cb(v, sel, by_route=False):
    """사실 군집 부트스트랩.  by_route: D 평균 - H 평균.  반환 [추정, 하한, 상한, n (D/H)]."""
    v = np.asarray(v, float)
    def agg(mask):
        s = np.bincount(finv[mask], weights=v[mask], minlength=len(uf)); c = np.bincount(finv[mask], minlength=len(uf)).astype(float)
        return s, c
    if not by_route:
        s, c = agg(sel)
        if c.sum() == 0: return None
        bs = (W @ s) / np.maximum(W @ c, 1e-9); return [round(float(s.sum() / c.sum()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), int(c.sum())]
    sD, cD = agg(sel & (route == "D")); sH, cH = agg(sel & (route == "H"))
    if cD.sum() == 0 or cH.sum() == 0: return None
    bs = (W @ sD) / np.maximum(W @ cD, 1e-9) - (W @ sH) / np.maximum(W @ cH, 1e-9)
    return [round(float(sD.sum() / cD.sum() - sH.sum() / cH.sum()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), f"{int(cD.sum())}/{int(cH.sum())}"]


pct = lambda r: None if r is None else [round(100 * r[0], 1), round(100 * r[1], 1), round(100 * r[2], 1), r[3]]
SELS = {"모든tau": np.ones(N, bool), **{f"tau{t}": tauv == t for t in TAUS}}
# ---- 짝 수
O["짝수"] = {f"tau{t}": {ro: dict(전체=int(((tauv == t) & (route == ro)).sum()), 엄격힌트없음=int(((tauv == t) & (route == ro) & HF_ALL).sum()),
                                  F캔버스힌트=int(((tauv == t) & (route == ro) & hint("F", "own")).sum()), S캔버스힌트=int(((tauv == t) & (route == ro) & hint("S", "own")).sum()))
                         for ro in "DH"} for t in TAUS}
# ---- 정답률 표 (조건 x 경로 x tau), 모든 짝 / 엄격 힌트 없음
tab = {}
for hn, hm in (("모든짝", np.ones(N, bool)), ("엄격힌트없음", HF_ALL)):
    for sn, sm in SELS.items():
        for ro in ("D", "H", "전체"):
            rm = np.ones(N, bool) if ro == "전체" else route == ro
            tab[f"{hn}|{sn}|{ro}"] = {NAME[c]: pct(cb(acc(*c), hm & sm & rm)) for c in CONDS}
O["정답률"] = tab


def contrast(a, b, hf, by_route, name, f=None):
    """정답률 차 a - b (짝 단위), 힌트 없음 = a, b 의 주입 캔버스가 모두 힌트 없음."""
    f = f or acc; d = f(*a) - f(*b); res = {}
    for hn, hm in (("힌트없음", ~hint(*a) & ~hint(*b)), ("엄격힌트없음", HF_ALL), ("모든짝", np.ones(N, bool))):
        for sn, sm in SELS.items():
            res[f"{hn}|{sn}"] = pct(cb(d, hm & sm, by_route))
            if not by_route:
                for ro in "DH": res[f"{hn}|{sn}|{ro}"] = pct(cb(d, hm & sm & (route == ro)))
            else:
                for ro in "DH": res[f"{hn}|{sn}|{ro} 자체"] = pct(cb(d, hm & sm & (route == ro)))
    return res


H = {}
H["H51.1 S+F캔버스 - S+중립 (<= -10 이면 참)"] = contrast(("S", "swap"), ("S", "neut"), True, False, "")
H["H51.1b S+F캔버스 - S+경계보존중립"] = contrast(("S", "swap"), ("S", "neutB"), True, False, "")
H["H51.2 (F+중립 - F+자기), D - H (하한 > 0 이면 참)"] = contrast(("F", "neut"), ("F", "own"), True, True, "")
H["H51.2b (F+경계보존중립 - F+자기), D - H"] = contrast(("F", "neutB"), ("F", "own"), True, True, "")
H["H51.3 (F+S캔버스 - F+자기), D - H (힌트 없음, 하한 > 0 이면 참)"] = contrast(("F", "swap"), ("F", "own"), True, True, "")
H["참고 F+전부마스크 - F+자기, D - H"] = contrast(("F", "reset"), ("F", "own"), True, True, "")
H["참고 S+중립 - S+자기"] = contrast(("S", "neut"), ("S", "own"), True, False, "")
H["참고 S+F캔버스 - S+자기"] = contrast(("S", "swap"), ("S", "own"), True, False, "")
H["참고 S+경계보존중립 - S+자기"] = contrast(("S", "neutB"), ("S", "own"), True, False, "")
H["참고 F+S캔버스 - F+중립 (문구 고정, 내용만)"] = contrast(("F", "swap"), ("F", "neut"), True, False, "")
O["대비"] = H
H0 = {}
H0["H51.1 S+F캔버스 - S+중립"] = contrast(("S", "swap"), ("S", "neut"), True, False, "", acc0)
H0["H51.1b S+F캔버스 - S+경계보존중립"] = contrast(("S", "swap"), ("S", "neutB"), True, False, "", acc0)
H0["H51.2 (F+중립 - F+자기), D - H"] = contrast(("F", "neut"), ("F", "own"), True, True, "", acc0)
H0["H51.2b (F+경계보존중립 - F+자기), D - H"] = contrast(("F", "neutB"), ("F", "own"), True, True, "", acc0)
H0["H51.3 (F+S캔버스 - F+자기), D - H"] = contrast(("F", "swap"), ("F", "own"), True, True, "", acc0)
H0["참고 F+전부마스크 - F+자기, D - H"] = contrast(("F", "reset"), ("F", "own"), True, True, "", acc0)
O["대비_첫토큰기준"] = H0
O["첫토큰_정답률"] = {f"{sn}|{ro}": {NAME[c]: pct(cb(acc0(*c), sm & (route == ro))) for c in CONDS} for sn, sm in SELS.items() for ro in "DH"}
# ---- 주입 직후 칸 0 금 첫 토큰 (자기 캔버스 대비 변화)
l2 = lambda r, c: np.log2(rank(r, c) + 1)
imm = {}
for sn, sm in SELS.items():
    for ro in ("D", "H"):
        rm = sm & (route == ro); o = {}
        for c in CONDS:
            own = (c[0], "own"); hf = rm & ~hint(*c) & ~hint(*own)
            o[NAME[c]] = dict(금1위율=pct(cb(rank(*c) == 0, rm)), 금순위중앙=float(np.median(rank(*c)[rm])) if rm.any() else None,
                              dlog2순위_대_자기=cb(l2(*c) - l2(*own), rm), dlog2순위_대_자기_힌트없음=cb(l2(*c) - l2(*own), hf), d금로짓_대_자기=cb(logit(*c) - logit(*own), rm))
        imm[f"{sn}|{ro}"] = o
O["주입직후_칸0"] = imm
# ---- 경계 기록: 자기 캔버스 (tau) 에 줄바꿈/특수 경계가 확정돼 있는지, 위치 - 금 답 토큰 수
bd = {}
for sn, sm in SELS.items():
    for r in ("F", "S"):
        for ro in ("D", "H"):
            rm = sm & (route == ro); us = [u for u, z in zip(U, rm) if z]
            fb = [u["g"][(r, "own")]["first_bound"] for u in us]; gl = [u["g"][(r, "own")]["gold_len"] for u in us]
            rel = [b - g_ for b, g_ in zip(fb, gl) if b is not None]
            bd[f"{sn}|{r}|{ro}"] = dict(n=len(us), 경계있음_pct=round(100 * float(np.mean([b is not None for b in fb])), 1) if us else None,
                                         위치마이너스금길이=dict(collections.Counter(int(np.clip(x, -3, 3)) for x in rel)), 위치중앙=float(np.median(rel)) if rel else None,
                                         경계가_금길이보다_앞=round(100 * float(np.mean([x < 0 for x in rel])), 1) if rel else None)
O["경계"] = bd
# 경계 유무로 나눈 F 복구 (중립, 경계보존중립)
bsplit = {}
hasb = np.array([u["g"][("F", "own")]["first_bound"] is not None for u in U])
for nm_, msk in (("F자기캔버스에경계있음", hasb), ("경계없음", ~hasb)):
    for c in (("F", "neut"), ("F", "neutB"), ("F", "swap")):
        bsplit[f"{nm_}|{NAME[c]}-F+자기"] = pct(cb(acc(*c) - acc("F", "own"), msk)); bsplit[f"{nm_}|{NAME[c]}-F+자기|D-H"] = pct(cb(acc(*c) - acc("F", "own"), msk, True))
O["경계별_F복구"] = bsplit
nbsame = np.array([u["g"][("F", "neutB")]["committed"] == u["g"][("F", "own")]["committed"] for u in U])
O["경계보존중립이_자기캔버스와_같은_짝_pct(F)"] = round(100 * float(nbsame.mean()), 1) if N else None
# ---- 판정
def g_(key, sub): return H[key][sub]
j1 = g_("H51.1 S+F캔버스 - S+중립 (<= -10 이면 참)", "힌트없음|모든tau"); j2 = g_("H51.2 (F+중립 - F+자기), D - H (하한 > 0 이면 참)", "힌트없음|모든tau")
j3 = g_("H51.3 (F+S캔버스 - F+자기), D - H (힌트 없음, 하한 > 0 이면 참)", "힌트없음|모든tau")
O["판정"] = {"H51.1": dict(값=j1, 참=bool(j1 and j1[0] <= -10)), "H51.2": dict(값=j2, 참=bool(j2 and j2[1] > 0)), "H51.3": dict(값=j3, 참=bool(j3 and j3[1] > 0))}
json.dump(O, open(f"{OUT}/e51_summary{SFX}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(dict(sanity=O["sanity_base_대_E7"], own=SAN, 짝수=O["짝수"], 판정=O["판정"]), ensure_ascii=False, indent=1))
for kk in ("모든짝|모든tau|D", "모든짝|모든tau|H", "엄격힌트없음|모든tau|D", "엄격힌트없음|모든tau|H"): print(kk, json.dumps(tab[kk], ensure_ascii=False))
