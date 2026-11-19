"""E5 (PREREG.md, CPU): Qwen2.5-7B 층 20 프롬프트 끝 상태로 만든 방향과 Dream B 비교.  E1 (k별 고침 비율) 도 함께 낸다."""
import collections, json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from common import rows, grp, ty, k, fid, mid, unit, load_X, directions, NF, OUT
out = {}
# ---- E1
R = [json.loads(l) for l in open("analysis/6_steering/results_caa_all.jsonl")]; by = collections.defaultdict(dict)
for r in R: by[r["row"]][r["cond"]] = r["new_type"]
e1 = {}
for kk in (1, 2, 3, 4):
    rr = [i for i in by if k[i] == kk]; w = [i for i in rr if by[i]["base"] != "correct"]; c = [i for i in rr if by[i]["base"] == "correct"]
    e1[kk] = dict(n=len(rr), 원래정답=round(100 * len(c) / len(rr), 1), **{f"고침_{cn}": round(100 * np.mean([by[i][cn] == "correct" for i in w]), 1) for cn in ("B c0.5", "B c1")}, **{f"망침_{cn}": round(100 * np.mean([by[i][cn] != "correct" for i in c]), 1) for cn in ("B c0.5", "B c1")})
out["E1"] = e1; print("E1", json.dumps(e1, ensure_ascii=False), flush=True)
# ---- E5
X, _ = load_X(); dirs = directions(X); Bd = unit(np.mean([dirs[g]["B"] for g in range(NF)], 0))
Q = [json.loads(l) for l in open("data/3_classification/full/qwen_final_full.jsonl")]
qr = json.load(open("data/1_raw/qwen_hidden/rows.json")); Hq = np.load("data/1_raw/qwen_hidden/hid.npy").astype(np.float32)
assert len(Q) == len(rows) and all(a["sample_id"] == b["sample_id"] and a["version"] == b["version"] for a, b in zip(Q, rows)), "Qwen 라벨 순서가 Dream 과 다름"
qty = np.array([r["type3"] if not r["correct"] else "correct" for r in Q]); qk = np.array([r["k_correct"] for r in Q]); qmid = (qk >= 1) & (qk <= 4)
pos = {r: j for j, r in enumerate(qr)}
def qdir(neg):
    byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in qr:
        if not qmid[i]: continue
        t = "correct" if qty[i] == "correct" else ("neg" if (qty[i] == neg or (neg == "wrong" and qty[i] != "abstain")) else None)
        if t: byf[fid[i]][t].append(pos[i])
    return {f: Hq[v["correct"]].mean(0) - Hq[v["neg"]].mean(0) for f, v in byf.items() if v["correct"] and v["neg"]}
def half(D):
    hs = []
    for s in range(20):
        p = np.random.default_rng(s).permutation(len(D)); hs.append(float(unit(D[p[: len(p) // 2]].mean(0)) @ unit(D[p[len(p) // 2:]].mean(0))))
    return round(float(np.mean(hs)), 3)
e5 = {}
rnd = [abs(float(unit(np.random.default_rng(s).standard_normal(3584)) @ Bd)) for s in range(1000)]; e5["무작위_cos_95백분위"] = round(float(np.percentile(rnd, 95)), 3)
for neg in ("broken", "wrong"):
    d = qdir(neg); fs = sorted(d); D = np.stack([d[f] for f in fs]); Bq = unit(D.mean(0))
    bs = [float(unit(D[ix].mean(0)) @ Bd) for ix in np.random.default_rng(0).integers(0, len(fs), (2000, len(fs)))]
    e5[f"Qwen(정답-{neg})"] = dict(사실수=len(fs), cos_DreamB=[round(float(Bq @ Bd), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], Qwen반반안정성=half(D))
byf = collections.defaultdict(lambda: collections.defaultdict(list))
for i in np.where(mid & (ty != "abstain"))[0]: byf[fid[i]]["correct" if ty[i] == "correct" else "neg"].append(i)
Dd = np.stack([X[v["correct"]].mean(0) - X[v["neg"]].mean(0) for v in byf.values() if v["correct"] and v["neg"]])
e5["Dream(정답-오답)_cos_DreamB"] = round(float(unit(Dd.mean(0)) @ Bd), 3); e5["Dream(정답-오답)_반반안정성"] = half(Dd)
Bb = [];
byf2 = collections.defaultdict(lambda: collections.defaultdict(list))
for i in np.where(mid & np.isin(ty, ["correct", "broken"]))[0]: byf2[fid[i]][ty[i]].append(i)
e5["DreamB_반반안정성"] = half(np.stack([X[v["correct"]].mean(0) - X[v["broken"]].mean(0) for v in byf2.values() if v["correct"] and v["broken"]]))
Bqw = unit(np.mean(list(qdir("wrong").values()), 0)); e5["Qwen(정답-오답)_cos_Dream(정답-오답)"] = round(float(Bqw @ unit(Dd.mean(0))), 3)
def within(score_of, rows_, lab):
    g = collections.defaultdict(list)
    for i in rows_: g[fid[i]].append(i)
    a = [roc_auc_score([lab(i) for i in v], [score_of(i) for i in v]) for v in g.values() if 0 < sum(lab(i) for i in v) < len(v)]
    return round(float(np.mean(a)), 3), len(a)
qrows = [i for i in qr if qmid[i] and qty[i] != "abstain"]
e5["Qwen상태_DreamB투영_정답대오답"] = within(lambda i: Hq[pos[i]] @ Bd, qrows, lambda i: qty[i] == "correct")
e5["Qwen상태_DreamB투영_정답대붕괴"] = within(lambda i: Hq[pos[i]] @ Bd, [i for i in qrows if qty[i] in ("correct", "broken")], lambda i: qty[i] == "correct")
e5["Qwen상태_Qwen방향투영_정답대오답(제자리)"] = within(lambda i: Hq[pos[i]] @ Bqw, qrows, lambda i: qty[i] == "correct")
e5["Dream상태_DreamB투영_정답대오답"] = within(lambda i: X[i] @ dirs[grp[i]]["B"], [int(i) for i in np.where(mid & (ty != "abstain"))[0]], lambda i: ty[i] == "correct")
e5["같은행_Qwen대Dream상태_cos평균"] = round(float(np.mean([unit(Hq[pos[i]]) @ unit(X[i]) for i in qrows[:2000]])), 3)
out["E5"] = e5; print("E5", json.dumps(e5, ensure_ascii=False, indent=1), flush=True)
json.dump(out, open(f"{OUT}/e1_e5.json", "w"), ensure_ascii=False, indent=1)
