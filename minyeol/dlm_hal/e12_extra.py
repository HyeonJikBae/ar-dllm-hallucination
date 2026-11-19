"""E12 보충 (사후 점검, CPU).  (1) 조향 없이 다시 생성 (E12 base) 을 다시 생성 조건으로 넣은 대조: 다시 돌리기만 해도 생기는 이득.  (2) E11 에서 쓴 S 행을 뺀 나머지에서 같은 계산 (독립 재현).
감지 점수와 겹밖 문턱은 e12_eval.py 와 같음."""
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import grp, mid, NF, OUT, MODEL, REV, extract_answer, e7_rows, set_S
D7 = f"{OUT}/e7"; _, _, F7 = e7_rows(); R = np.array([int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]); n = len(R); ism = mid[R]; isf = ~ism
R7 = np.load(f"{D7}/rows.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
p7 = {int(r): j for j, r in enumerate(R7)}; J = np.array([p7[i] for i in R]); first = np.array([meta[int(i)]["new_type"] == "correct" for i in R])
cache = f"{OUT}/e12_sc.npy"
if os.path.exists(cache): SC = np.load(cache)
else:
    IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[J]); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[J])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True); SPs = set(tok.all_special_ids)
    lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
    def dc(t):
        t = list(t); t = t[: next((q for q, v in enumerate(t) if v in SPs), len(t))]; return extract_answer(tok.decode(t, skip_special_tokens=True))
    ANS = [[dc(r) for r in np.where(MK[:, s], IDS[:, s], committed)] for s in range(32)]
    SC = np.array([np.mean([ANS[s][j] != ANS[31][j] for s in range(31)]) for j in range(n)]); np.save(cache, SC)
E = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{OUT}/results_e12_[A-Z].jsonl")):
    for l in open(f, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        E[d["cond"]][d["row"]] = d["new_type"] == "correct"
g = grp[R]
def gate(redo, use):
    sel = np.zeros(n, bool)
    for gg in range(NF):
        tr, te = (g != gg) & use, (g == gg) & use; bq, bv = 0.0, -1e9
        for q in np.linspace(0, 1, 51):
            th = np.quantile(SC[tr], 1 - q) if q > 0 else np.inf; v = np.where(SC >= th, redo, first)[tr & ism].mean()
            if v > bv + 1e-12: bv, bq = v, q
        th = np.quantile(SC[tr], 1 - bq) if bq > 0 else np.inf; sel[te] = SC[te] >= th
    return sel
rng = np.random.default_rng(0)
def summ(p, use):
    m = np.where(use & ism)[0]; d = (p.astype(float) - first.astype(float))[m]; bs = np.array([d[rng.integers(0, len(m), len(m))].mean() for _ in range(4000)]) * 100
    return {"n_1to4": int(len(m)), "이득": round(float(d.mean() * 100), 2), "CI": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
            "고침": round(100 * float((~first & p)[m].mean()), 2), "깨짐": round(100 * float((first & ~p)[m].mean()), 2), "5/5": round(100 * float(p[use & isf].mean()), 2)}
Sset = {i for _, i in set_S()}; notS = np.array([int(i) not in Sset for i in R]); allr = np.ones(n, bool); O = {"S_제외_1to4행": int((notS & ism).sum())}
for c in ("base", "B c0.5", "Brel c1", "ACE t27", "maskgit_plus", "entropy"):
    if len(E[c]) != n: O[c] = f"미완료 {len(E[c])}/{n}"; continue
    redo = np.array([E[c][int(i)] for i in R]); O[c] = {}
    for nm, use in (("전체", allr), ("S제외", notS)):
        sel = gate(redo, use); O[c][f"두번생성 {nm}"] = summ(np.where(sel, redo, first), use); O[c][f"모두다시 {nm}"] = summ(redo, use)
    if c == "base": O[c]["E7과_정오일치%"] = round(100 * float((redo == first).mean()), 2); O[c]["1~4/5 정오 다른 행%"] = round(100 * float((redo != first)[ism].mean()), 2)
    print(c, json.dumps(O[c], ensure_ascii=False), flush=True)
json.dump(O, open(f"{OUT}/e12_extra.json", "w"), ensure_ascii=False, indent=1); print("DONE", flush=True)
