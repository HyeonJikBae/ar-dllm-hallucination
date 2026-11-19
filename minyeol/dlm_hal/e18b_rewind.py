"""E18b (PREREG.md, CPU): 되감기 조향 전체 규모 계산.  step 0~4 x0hat 답 종류 수로 고른 행만 다시 생성 (E12/E18 결과) 한 것으로 간주.
비교로 생성 뒤 신호 ('최종답과 다른 step 비율', e12_sc.npy) 도 같은 틀에서.  0/5 행 (E7 의 2,000) 에 같은 문턱을 적용해 모집단 비용도 계산."""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import grp, mid, NF, OUT, MODEL, REV, extract_answer, e7_rows, set_S
D7 = f"{OUT}/e7"; _, Z7, F7 = e7_rows()
R = np.array([int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]); n = len(R); ism = mid[R]; isf = ~ism
R7 = np.load(f"{D7}/rows.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
p7 = {int(r): j for j, r in enumerate(R7)}
ALLR = np.concatenate([R, np.array([int(i) for i in Z7])]); JA = np.array([p7[i] for i in ALLR])
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[JA]); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[JA])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True); SPs = set(tok.all_special_ids)
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
def dc(t):
    t = list(t); t = t[: next((q for q, v in enumerate(t) if v in SPs), len(t))]; return extract_answer(tok.decode(t, skip_special_tokens=True))
ANS = [[dc(r) for r in np.where(MK[:, s], IDS[:, s], committed)] for s in range(5)]
EARLY_ALL = np.array([len({ANS[s][j] for s in range(5)}) for j in range(len(ALLR))], float)
EARLY = EARLY_ALL[:n]; EZ = EARLY_ALL[n:]; gZ = grp[ALLR[n:]]
SC = np.load(f"{OUT}/e12_sc.npy")
first = np.array([meta[int(i)]["new_type"] == "correct" for i in R])
E = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{OUT}/results_e12_[A-Z].jsonl")):
    for l in open(f, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        E[d["cond"]][d["row"]] = d["new_type"] == "correct"
g = grp[R]; Sset = {i for _, i in set_S()}; notS = np.array([int(i) not in Sset for i in R])
rng = np.random.default_rng(0)
def gate(score, redo, use):
    sel = np.zeros(n, bool); ths = {}
    for gg in range(NF):
        tr, te = (g != gg) & use, (g == gg) & use; bq, bv, bth = 0.0, -1e9, np.inf
        for th in np.unique(score[tr]):
            v = np.where(score >= th, redo, first)[tr & ism].mean()
            if v > bv + 1e-12: bv, bth = v, th
        v0 = first[tr & ism].mean()
        if v0 >= bv: bth = np.inf
        sel[te] = score[te] >= bth; ths[gg] = bth
    return sel, ths
def summ(p, sel, use):
    m = np.where(use & ism)[0]; d = (p.astype(float) - first.astype(float))[m]; bs = np.array([d[rng.integers(0, len(m), len(m))].mean() for _ in range(4000)]) * 100
    return {"이득": round(float(d.mean() * 100), 2), "CI": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
            "고침(오답중)": round(100 * float(p[m][~first[m]].mean()), 1), "깨짐(정답중)": round(100 * float((~p[m])[first[m]].mean()), 1), "5/5": round(100 * float(p[use & isf].mean()), 2),
            "다시생성 1~4/5": round(100 * float(sel[use & ism].mean()), 1), "다시생성 5/5": round(100 * float(sel[use & isf].mean()), 1)}
O = {"early_분포": {nm: {int(v): int(c) for v, c in zip(*np.unique(x, return_counts=True))} for nm, x in (("1~4/5", EARLY[ism]), ("5/5", EARLY[isf]), ("0/5", EZ))}}
print(O, flush=True)
W = {"zero": 44110, "mid": 5950, "five": 5165}
for c in [c for c in ("Brel c1", "B c0.5", "ACE t27", "maskgit_plus", "entropy", "Brel c0.5", "Brel c1.5", "Brel c2", "Rand c1", "BrelShuf c1") if len(E[c]) == n]:
    redo = np.array([E[c][int(i)] for i in R]); O[c] = {}
    for snm, score, zsc in (("되감기(s0~4 답 종류)", EARLY, EZ), ("두번생성(다른 step 비율)", SC, None)):
        for nm, use in (("전체", np.ones(n, bool)), ("S제외", notS)):
            sel, ths = gate(score, redo, use); O[c][f"{snm} {nm}"] = summ(np.where(sel, redo, first), sel, use)
            if zsc is not None and nm == "전체":
                zsel = np.array([zsc[j] >= ths[gZ[j]] for j in range(len(zsc))])
                cost = (W["zero"] * zsel.mean() + W["mid"] * sel[ism].mean() + W["five"] * sel[isf].mean()) / sum(W.values())
                O[c][f"{snm} {nm}"].update({"다시생성 0/5": round(100 * float(zsel.mean()), 1), "모집단 평균 추가 생성(되감기는 4/32 step 비용 포함)": round(float(4 / 32 * 0 + cost), 3)})
        print(c, json.dumps(O[c], ensure_ascii=False), flush=True)
json.dump(O, open(f"{OUT}/e18b_rewind.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("DONE", flush=True)
