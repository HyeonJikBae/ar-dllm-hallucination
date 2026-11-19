"""E40 (서술, CPU): 같은 사실 안에서 '맞히는 문구' 와 '틀리는 문구' 의 어휘·은닉 특징 분포.  1~4/5 행 (저장 라벨)."""
import collections, json, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from common import rows, ty, k, fid, mid, grp, unit, load_X, NF, OUT, POP
X, _ = load_X(); rel = np.array([r["relation"] for r in rows]); ver = np.array([r["version"] for r in rows])
M = np.where(mid & (ty != "abstain"))[0]; corr = ty == "correct"
byf = collections.defaultdict(list)
for i in np.where(mid)[0]: byf[fid[i]].append(int(i))
orig = {f_: next((i for i in v if ver[i] == "question"), v[0]) for f_, v in byf.items()}
wset = lambda s: set(re.findall(r"[a-z0-9']+", s.lower()))
F = {}
q = [rows[i]["question"] for i in range(len(rows))]
F["단어 수"] = np.array([len(s.split()) for s in q], float)
F["글자 수"] = np.array([len(s) for s in q], float)
def subj_info(i):
    x = POP[rows[i]["sample_id"]]; low = q[i].lower()
    for nm in sorted([m for m in [x["subj"]] + json.loads(x["s_aliases"] or "[]") if m], key=len, reverse=True):
        j = low.find(nm.lower())
        if j >= 0: return 1.0, j / max(1, len(low)), float(nm == x["subj"])
    return 0.0, np.nan, 0.0
si = np.array([subj_info(i) if mid[i] else (np.nan, np.nan, np.nan) for i in range(len(rows))])
F["주어 그대로 등장 (0/1)"] = si[:, 0]; F["주어 위치 (앞 0 ~ 뒤 1)"] = si[:, 1]; F["주어 원형 이름 사용 (0/1)"] = si[:, 2]
F["원래 템플릿과 단어 겹침 (Jaccard)"] = np.array([len(wset(q[i]) & wset(q[orig[fid[i]]])) / max(1, len(wset(q[i]) | wset(q[orig[fid[i]]]))) if mid[i] else np.nan for i in range(len(rows))])
# 은닉 (층 20 프롬프트 끝 step 0), 방향은 주어 겹밖
pB = np.full(len(rows), np.nan); pK = np.full(len(rows), np.nan); cK = np.full(len(rows), np.nan)
for g in range(NF):
    tr = grp != g; te = np.where((grp == g) & mid)[0]
    bf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr & mid & (ty != "abstain"))[0]: bf[fid[i]]["c" if corr[i] else "w"].append(i)
    per = collections.defaultdict(list)
    for v in bf.values():
        if v["c"] and v["w"]: per[rel[v["c"][0]]].append(X[v["c"]].mean(0) - X[v["w"]].mean(0))
    Bg = unit(np.mean([d for L_ in per.values() for d in L_], 0)); Br = {r_: unit(np.mean(d, 0)) for r_, d in per.items() if len(d) >= 15}
    Kr = {}; C5 = {}
    for r_ in np.unique(rel):
        a = tr & (rel == r_) & (k == 5); b = tr & (rel == r_) & (k == 0)
        if a.sum() >= 20 and b.sum() >= 20: Kr[r_] = unit(X[a].mean(0) - X[b].mean(0)); C5[r_] = X[a].mean(0)
    for i in te:
        pB[i] = X[i] @ Br.get(rel[i], Bg)
        if rel[i] in Kr: pK[i] = X[i] @ Kr[rel[i]]; cK[i] = X[i] @ C5[rel[i]] / (np.linalg.norm(X[i]) * np.linalg.norm(C5[rel[i]]))
F["[은닉] Brel 투영 (겹밖)"] = pB; F["[은닉] 앎 방향 K 투영"] = pK; F["[은닉] 같은 관계 5/5 평균과 cos"] = cK
F["[은닉] 상태 크기 ‖h‖"] = np.linalg.norm(X, axis=1)
out_ = np.full(len(rows), np.nan)
for f_, v in byf.items():
    for i in v:
        oth = [j for j in v if j != i]; c = X[oth].mean(0); out_[i] = 1 - X[i] @ c / (np.linalg.norm(X[i]) * np.linalg.norm(c))
F["[은닉] 형제 문구들과의 거리 (1-cos)"] = out_
meta = {}; D7 = f"{OUT}/e7"
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
R7 = np.load(f"{D7}/rows.npy"); LP = np.load(f"{D7}/lp.npy", mmap_mode="r"); MG = np.load(f"{D7}/mg.npy", mmap_mode="r")
lp0 = np.full(len(rows), np.nan); mg0 = np.full(len(rows), np.nan)
for j, r in enumerate(R7): lp0[int(r)] = float(LP[j, 0, 0]); mg0[int(r)] = float(MG[j, 0, 0])
F["[은닉] step0 첫 칸 확신 (logprob)"] = lp0; F["[은닉] step0 첫 칸 1·2위 차 (margin)"] = mg0
# 같은 사실 안 비교
res = {}; hist = {}
mixed = [v for v in byf.values() if 0 < sum(corr[i] for i in v if ty[i] != "abstain") < len([i for i in v if ty[i] != "abstain"])]
rng = np.random.default_rng(0)
for nm, s in F.items():
    aucs = []; diffs = []; cen_c = []; cen_w = []
    for v in mixed:
        v = [i for i in v if ty[i] != "abstain" and not np.isnan(s[i])]
        y = corr[v]
        if 0 < y.sum() < len(v):
            if len(np.unique(s[v])) > 1: aucs.append(roc_auc_score(y, s[v]))
            else: aucs.append(0.5)
            diffs.append(s[v][y].mean() - s[v][~y].mean()); m_ = s[v].mean(); cen_c += list(s[v][y] - m_); cen_w += list(s[v][~y] - m_)
    diffs = np.array(diffs); bs = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)]
    allv = s[M][~np.isnan(s[M])]
    res[nm] = {"같은 사실 안 AUROC (맞힘 쪽이 클수록 >0.5)": round(float(np.mean(aucs)), 3), "사실 수": len(aucs),
               "맞힘-틀림 평균 차 [CI]": [round(float(diffs.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
               "맞힘 쪽이 큰 사실 비율 %": round(100 * float((diffs > 0).mean()), 1)}
    lo, hi = np.nanpercentile(np.r_[cen_c, cen_w], [1, 99]); bins = np.linspace(lo, hi, 31)
    hist[nm] = {"bins": bins.tolist(), "맞힘": np.histogram(np.clip(cen_c, lo, hi), bins)[0].tolist(), "틀림": np.histogram(np.clip(cen_w, lo, hi), bins)[0].tolist()}
    print(nm, res[nm], flush=True)
cat = {}
for nm, keyf in (("문구 종류 (원본 question / 변형 v1~v4)", lambda i: ver[i]), ("의문사", lambda i: (q[i].split()[0].lower().strip("?,") if q[i].split() else "") if q[i].split()[0].lower() in ("what", "who", "where", "which", "when", "in", "how", "whose") else "기타")):
    c = collections.defaultdict(list)
    for i in M: c[keyf(i)].append(corr[i])
    cat[nm] = {kx: [round(100 * float(np.mean(v)), 1), len(v)] for kx, v in sorted(c.items(), key=lambda x: -len(x[1]))}
    print(nm, cat[nm], flush=True)
json.dump({"연속": res, "범주": cat, "히스토그램": hist, "1~4/5 행 수": int(len(M)), "섞인 사실 수": len(mixed)}, open(f"{OUT}/e40_query_stats.json", "w"), ensure_ascii=False, indent=1); print("DONE")
