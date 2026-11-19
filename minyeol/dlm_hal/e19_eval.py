"""E19 평가 (PREREG.md, CPU): LDA 방향 조향.  기준은 E9 의 B c0.5, Brel c1 (같은 S, 같은 배치 구성)."""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import rows, grp, k, NF, OUT, MODEL, REV, extract_answer, set_S
PIL = "--pilot" in sys.argv; O = {}
E = collections.defaultdict(dict); SET = {}; NON = collections.defaultdict(dict); files = sorted(glob.glob(f"{OUT}/results_e19{'_pilot' if PIL else ''}_*.jsonl"))
if not PIL: files = [f for f in files if "_pilot" not in f]
for f in files:
    for l in open(f, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        E[d["cond"]][d["row"]] = d["new_type"] == "correct"; SET[d["row"]] = d["set"]
        if d.get("n_on") is not None: NON[d["cond"]][d["row"]] = d["n_on"]
for l in open(f"{OUT}/results_e9.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if d["cond"] in ("B c0.5", "Brel c1", "B c1"): E["[E9] " + d["cond"]][d["row"]] = d["new_type"] == "correct"
    if d["cond"] == "base": E["[E9] base"][d["row"]] = d["new_type"] == "correct"
S = [i for i in E["base"]]; sets = np.array([SET[i] for i in S]); gS = grp[S]
WS = {"broken": 310 / 310, "entity": 2749 / 300, "correct": 2875 / 300, "zero": 44110 / 300, "five": 5165 / 300}; ws = np.array([WS[x] for x in sets]); mS = np.isin(sets, ["broken", "entity", "correct"])
base = np.array([E["base"][i] for i in S]); CONDS = [c for c in E if c not in ("base", "[E9] base") and len(E[c]) == len(S)]
O["완료"] = {c: len(E[c]) for c in E}; O["E9base_E19base_일치%"] = round(100 * float(np.mean([E["[E9] base"][i] == E["base"][i] for i in S])), 2); print("파일", files, O["완료"], flush=True)
wacc = lambda x, m: float(100 * (x * ws)[m].sum() / ws[m].sum())
rng = np.random.default_rng(0); POS = {st: np.where(sets == st)[0] for st in WS}; NB = 4000
BIX = {st: rng.integers(0, len(POS[st]), (NB, len(POS[st]))) for st in WS if len(POS[st])}
def boot_net(x, ref):
    d = []
    for b in range(NB):
        ix = np.concatenate([POS[st][BIX[st][b]] for st in BIX]); w_ = ws[ix]; m_ = mS[ix]
        d.append(100 * ((x[ix] * w_)[m_].sum() - (ref[ix] * w_)[m_].sum()) / w_[m_].sum())
    return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]
def fixbrk(x, ref):
    w_ = mS & ~ref; c_ = mS & ref
    return round(100 * float((ws * (x & w_)).sum() / (ws * w_).sum()), 1), round(100 * float((ws * (~x & c_)).sum() / (ws * c_).sum()), 1)
# ---- 첫 생성 (E7) 과 '최종답과 다른 step 비율' (e11 과 같은 계산)
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
p7 = {int(r): j for j, r in enumerate(R7)}; J = np.array([p7[i] for i in S]); first = np.array([meta[int(i)]["new_type"] == "correct" for i in S])
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[J]); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[J])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True); SPs = set(tok.all_special_ids)
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
def dc(t):
    t = list(t); t = t[: next((q for q, v in enumerate(t) if v in SPs), len(t))]; return extract_answer(tok.decode(t, skip_special_tokens=True))
ANS = [[dc(r) for r in np.where(MK[:, s], IDS[:, s], committed)] for s in range(32)]
SC = np.array([np.mean([ANS[s][j] != ANS[31][j] for s in range(31)]) for j in range(len(S))])
O["base_E7_정오일치%"] = round(100 * float((base == first).mean()), 2); print("E16 base 대 E7 정오 일치", O["base_E7_정오일치%"], flush=True)
def twopass(redo):
    sel = np.zeros(len(S), bool)
    for g in range(NF):
        tr, te = gS != g, gS == g; bq, bv = 0.0, -1e9
        for q in np.linspace(0, 1, 51):
            th = np.quantile(SC[tr], 1 - q) if q > 0 else np.inf; v = wacc(np.where(SC >= th, redo, first), tr & mS)
            if v > bv: bv, bq = v, q
        th = np.quantile(SC[tr], 1 - bq) if bq > 0 else np.inf; sel[te] = SC[te] >= th
    p = np.where(sel, redo, first); f_, b_ = fixbrk(p, first)
    return {"이득": round(wacc(p, mS) - wacc(first, mS), 2), "CI": boot_net(p, first), "고침": f_, "깨짐": b_, "5/5": round(100 * float(p[sets == "five"].mean()), 1)}
res = {"base": {"1~4/5": round(wacc(base, mS), 2), "5/5": round(100 * float(base[sets == "five"].mean()), 1), "0/5": round(100 * float(base[sets == "zero"].mean()), 1)}}
for c in CONDS:
    x = np.array([E[c][i] for i in S]); f_, b_ = fixbrk(x, base); non = [NON[c][i] for i in np.array(S)[mS] if i in NON[c]]
    res[c] = {"1~4/5": round(wacc(x, mS), 2), "순이득": round(wacc(x, mS) - wacc(base, mS), 2), "CI": boot_net(x, base), "고침": f_, "깨짐": b_,
              "5/5": round(100 * float(x[sets == "five"].mean()), 1), "0/5": round(100 * float(x[sets == "zero"].mean()), 1),
              "켜진차원_평균": (round(float(np.mean(non)), 1) if non else 3584), "두번생성": twopass(x)}
    print(c, json.dumps(res[c], ensure_ascii=False), flush=True)
O["조건별"] = res
lda = [c for c in res if c.startswith("LDA")]; ref_c = {"B c0.5": res.get("[E9] B c0.5"), "Brel c1": res.get("[E9] Brel c1")}
h1 = [(c, r_) for c in lda for r_, v in ref_c.items() if v and res[c]["순이득"] - v["순이득"] >= 2 and res[c]["5/5"] >= 98]
bt = max(lda, key=lambda c: res[c]["두번생성"]["이득"]) if lda else None; br2 = res["[E9] Brel c1"]["두번생성"]["이득"] if "[E9] Brel c1" in res else None
O["판정"] = {"H19.1": dict(통과=h1, 판정=bool(h1)), "H19.2": dict(최선LDA=bt, 이득=(res[bt]["두번생성"]["이득"] if bt else None), Brel_c1_E9=br2, 판정=bool(bt and br2 is not None and res[bt]["두번생성"]["이득"] - br2 >= 1))}
print("판정", json.dumps(O["판정"], ensure_ascii=False), flush=True)
json.dump(O, open(f"{OUT}/e19_summary{'_pilot' if PIL else ''}.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("DONE", flush=True)
