"""E10b (사후, PREREG 밖): E10 감지가 어디서 막히는지 분해 + step 0 감지 x 초기 조향 게이팅 + 조건별 신뢰구간.  CPU."""
import collections, json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from common import rows, grp, k, fid, unit, NF, OUT, MODEL, REV, extract_answer, is_correct, set_S
D7 = f"{OUT}/e7"; O = {}
R7 = np.load(f"{D7}/rows.npy"); cfg = json.load(open(f"{D7}/config.json")); done = np.load(f"{D7}/done.npy")
meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
J = np.where([done[j] and int(R7[j]) in meta for j in range(len(R7))])[0]; RR = R7[J]
HID = np.load(f"{D7}/hid.npy", mmap_mode="r"); IDS = np.load(f"{D7}/ids.npy", mmap_mode="r")[J]; LP = np.load(f"{D7}/lp.npy", mmap_mode="r")[J].astype(np.float32)
MG = np.load(f"{D7}/mg.npy", mmap_mode="r")[J].astype(np.float32); MK = np.load(f"{D7}/mask.npy", mmap_mode="r")[J]
kk = k[RR]; g7 = grp[RR]; corr = np.array([meta[int(i)]["new_type"] == "correct" for i in RR]); fid7 = fid[RR]; rel = np.array([rows[i]["relation"] for i in RR])
isz, isf, ism = kk == 0, kk == 5, (kk >= 1) & (kk <= 4); mw, mc = ism & ~corr, ism & corr
W = np.where(isz, 44110 / isz.sum(), np.where(isf, 5165 / isf.sum(), 1.0))
LAY, SV, PS = cfg["layers"], cfg["steps"], cfg["positions"]
# ---- 점수 (E10 과 같은 겹밖 방식)
SC = {}
for s in (0, 8):
    for l_ in (20, 27):
        Xh = np.asarray(HID[J, SV.index(s), LAY.index(l_), 0], dtype=np.float32); pk = np.zeros(len(RR)); pb = np.zeros(len(RR)); mk_ = np.zeros(len(RR))
        for g in range(NF):
            tr = g7 != g; te = ~tr; ds = []
            for r_ in np.unique(rel):
                a = tr & isf & (rel == r_); b = tr & isz & (rel == r_)
                if a.sum() >= 10 and b.sum() >= 10: ds.append(Xh[a].mean(0) - Xh[b].mean(0))
            K = unit(np.mean(ds, 0)); byf = collections.defaultdict(lambda: collections.defaultdict(list))
            for j in np.where(tr & ism)[0]: byf[fid7[j]]["c" if corr[j] else "w"].append(j)
            Bw = unit(np.mean([Xh[v["c"]].mean(0) - Xh[v["w"]].mean(0) for v in byf.values() if v["c"] and v["w"]], 0))
            pk[te] = Xh[te] @ K; pb[te] = Xh[te] @ Bw; mk_[te] = np.median(Xh[tr & ism] @ K)
        SC[f"K L{l_} s{s}"] = pk; SC[f"Bw L{l_} s{s}"] = pb; SC[f"가운데K L{l_} s{s}"] = -np.abs(pk - mk_)
    lp = LP[:, s, :4].mean(1); SC[f"첫4칸 logprob s{s}"] = lp; ml = np.zeros(len(RR))
    for g in range(NF): ml[g7 == g] = np.median(lp[(g7 != g) & ism])
    SC[f"가운데logprob s{s}"] = -np.abs(lp - ml)
def auc(pos, neg, s, w=None):
    ix = pos | neg; return round(float(roc_auc_score(pos[ix].astype(int), s[ix], sample_weight=None if w is None else w[ix])), 3)
# 1) 쌍별 분해.  부호: K, logprob, Bw 는 '앎/정답' 이 높음.  가운데 는 '가운데' 가 높음.
dec = {}
for nm, s in SC.items():
    dec[nm] = {"T1(가중)": auc(ism, ~ism, s, W), "1~4/5 대 0/5": auc(ism, isz, s), "1~4/5 대 5/5": auc(ism, isf, s), "5/5 대 0/5": auc(isf, isz, s),
               "아는데틀림 대 0/5": auc(mw, isz, s), "아는데틀림 대 5/5": auc(mw, isf, s), "아는데틀림 대 1~4/5정답": auc(mw, mc, s), "T2(가중,부호그대로)": auc(mw, ~mw, s, W)}
O["쌍별_AUROC"] = dec
for nm in dec: print(nm, dec[nm], flush=True)
# 2) 두 축 (학습 없음): 사실 층위 가운데K x 질문 층위 Bw.  T2 정밀도/재현율 (모집단 가중)
rk = lambda x: (np.argsort(np.argsort(x)) + 0.5) / len(x)
def prec_at(score, fr):
    o = np.argsort(-score); cw = np.cumsum(W[o]); top = o[cw <= fr * W.sum()]
    return dict(정밀도=round(100 * float(W[top][mw[top]].sum() / W[top].sum()), 1), 재현율=round(100 * float(mw[top].sum() / mw.sum()), 1))
base_rate = round(100 * float(W[mw].sum() / W.sum()), 2); pr = {"기저율(모집단 중 아는데틀림 %)": base_rate}
for nm in ("가운데K L27 s0", "가운데K L20 s0", "가운데K L27 s8"):
    pr[nm] = {f"상위{int(fr * 100)}%": prec_at(SC[nm], fr) for fr in (.02, .05, .10, .20)}
for s in (0, 8):
    cmb = rk(SC[f"가운데K L27 s0"]) * rk(-SC[f"Bw L20 s{s}"]); nm = f"가운데K L27 s0 x Bw L20 s{s}(-)"; SC[nm] = cmb
    pr[nm] = {f"상위{int(fr * 100)}%": prec_at(cmb, fr) for fr in (.02, .05, .10, .20)}; pr[nm]["T2"] = auc(mw, ~mw, cmb, W)
O["정밀도"] = pr; print(json.dumps(pr, ensure_ascii=False), flush=True)
# 3) 중간 step 답의 정답률 (궤적, 금 라벨로 채점, 기술용)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True); SP = set(tok.all_special_ids)
lastm = np.where(MK, np.arange(MK.shape[1])[None, :, None], -1).max(1); committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
x0 = lambda s: np.where(MK[:, s], IDS[:, s], committed)
def dc(t):
    t = list(t); t = t[: next((q for q, v in enumerate(t) if v in SP), len(t))]; return extract_answer(tok.decode(t, skip_special_tokens=True))
STS = [0, 1, 2, 4, 8, 16, 24, 31]; acc = {}; ICM = {}
for s in STS:
    A = [dc(r) for r in x0(s)]; ICM[s] = np.array([bool(is_correct({"model_answer": a, "gold_value": rows[i]["gold_value"], "gold_aliases": rows[i]["gold_aliases"]})) for a, i in zip(A, RR)])
for nm, ix in (("0/5", isz), ("1~4/5 정답", mc), ("1~4/5 오답", mw), ("5/5", isf)):
    acc[nm] = {f"s{s}": round(100 * float(ICM[s][ix].mean()), 1) for s in STS}
fc = (~MK[:, :, 0]).argmax(1); fc[MK[:, :, 0].all(1)] = 32
acc["첫칸_확정_step_중앙값"] = {nm: float(np.median(fc[ix])) for nm, ix in (("0/5", isz), ("1~4/5 정답", mc), ("1~4/5 오답", mw), ("5/5", isf))}
O["중간답_정답률"] = acc; print(json.dumps(acc, ensure_ascii=False), flush=True)
# 4) 조건별 1~4/5 가중 정답률 차이의 신뢰구간 (같은 행 짝, E8/E9 base 동일 확인됨)
E = collections.defaultdict(dict); SET = {}
for f in ("results_e8.jsonl", "results_e9.jsonl"):
    for l in open(f"{OUT}/{f}"):
        d = json.loads(l); c = d["cond"] if d["cond"] == "base" or f.endswith("e9.jsonl") else f"[e8]{d['cond']}"; E[c][d["row"]] = d["new_type"] == "correct"; SET[d["row"]] = d["set"]
S = [i for _, i in set_S()]; sets = np.array([SET[i] for i in S]); WS = {"broken": 310 / 310, "entity": 2749 / 300, "correct": 2875 / 300, "zero": 44110 / 300, "five": 5165 / 300}
ws = np.array([WS[x] for x in sets]); mS = np.isin(sets, ["broken", "entity", "correct"]); base = np.array([E["base"][i] for i in S])
wacc = lambda x, msk: float(100 * (x * ws)[msk].sum() / ws[msk].sum())
rng = np.random.default_rng(0); BIX = {st: rng.integers(0, (sets == st).sum(), (2000, (sets == st).sum())) for st in WS}; POS = {st: np.where(sets == st)[0] for st in WS}
def ci(x, msk):
    d = []
    for b in range(2000):
        ix = np.concatenate([POS[st][BIX[st][b]] for st in WS]); d.append(wacc(x[ix], msk[ix]) - wacc(base[ix], msk[ix]))
    return [round(wacc(x, msk) - wacc(base, msk), 2), round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]
cis = {}
for c in E:
    if c == "base": continue
    x = np.array([E[c][i] for i in S]); cis[c] = {"1~4/5 차이[CI]": ci(x, mS), "5/5": round(100 * float(x[sets == "five"].mean()), 1), "0/5": round(100 * float(x[sets == "zero"].mean()), 1)}
O["조건별_CI"] = cis; print(json.dumps(cis, ensure_ascii=False), flush=True)
# 5) 사후 게이팅: step 0 학습 없는 점수 (가운데K L27 s0 와 두 축 조합) 로 고른 행만 초기부터 조향.  문턱은 다른 겹에서 1~4/5 가중 정답률 최대로.
pos7 = {int(i): j for j, i in enumerate(RR)}; gS = grp[S]
def gate(score, cond):
    x = np.array([E[cond][i] for i in S]); sc = np.array([score[pos7[i]] for i in S]); sel = np.zeros(len(S), bool); qs = []
    for g in range(NF):
        tr, te = gS != g, gS == g; bq, bv = 0, -1e9
        for q in np.linspace(0, 1, 51):
            th = np.quantile(sc[tr], 1 - q) if q > 0 else np.inf; v = wacc(np.where(sc >= th, x, base), tr)       # 전체 모집단 가중으로 문턱 (0/5, 5/5 손해 포함)
            if v > bv: bv, bq = v, q
        th = np.quantile(sc[tr], 1 - bq) if bq > 0 else np.inf; sel[te] = sc[te] >= th; qs.append(round(bq, 2))
    p = np.where(sel, x, base); orc = np.where(~base, x, base)
    return {"1~4/5": round(wacc(p, mS), 2), "전체": round(wacc(p, np.ones(len(S), bool)), 3), "5/5": round(100 * float(p[sets == "five"].mean()), 1), "민비율q": qs,
            "모두밀기 1~4/5": round(wacc(x, mS), 2), "모두밀기 전체": round(wacc(x, np.ones(len(S), bool)), 3), "오라클 1~4/5": round(wacc(orc, mS), 2), "오라클 전체": round(wacc(orc, np.ones(len(S), bool)), 3)}
gt = {"base": {"1~4/5": round(wacc(base, mS), 2), "전체": round(wacc(base, np.ones(len(S), bool)), 3)}}
for cond in ("B c0.5", "B c1", "B c1 s0-3", "ACE t27", "Brel c1", "[e8]maskgit_plus", "[e8]entropy"):
    for snm in ("가운데K L27 s0", "가운데K L27 s0 x Bw L20 s0(-)"):
        gt[f"{cond} | {snm}"] = gate(SC[snm], cond)
O["사후_게이팅"] = gt; print(json.dumps(gt, ensure_ascii=False), flush=True)
json.dump(O, open(f"{OUT}/e10b_posthoc.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("DONE")
