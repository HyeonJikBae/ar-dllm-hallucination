"""E58 (a) (PREREG.md '## E58 주 방법 확인'): TriviaQA 다섯 문구 (E56 자료) 위 Dream.  방법은 PopQA 에서 고정된 것 그대로.
--stage dirs : 겹 (질문 id 5 겹, seed 58) 과 겹밖 B 방향 (층 24 프롬프트 끝 step 0, 같은 사실 안 맞힘 평균 − 틀림 평균을 사실끼리 평균, 단위 벡터),
               크기 scale = 0.5 x 층 24 평균 ||h||.  -> e56_data/folds.json, e56_data/e58_dirs.npz  (그다음 e56_dream.py --steer 로 모든 행 재생성)
--stage eval : step 0 특징 4 개 (칸 0 top-1 logprob, 첫 4 칸 top-1 logprob 평균, 칸 0 top1−top2 margin, 가운데K S1 (층 27, 전체 K = 5/5 − 0/5 평균, 겹밖))
               -> 0/5 / 1~4/5 / 5/5 다항 로지스틱 (표준화; 모든 행을 돌렸으므로 모집단 가중 = 1) -> P(1~4/5).  SC = 32 step 중간 답 중 마지막과 다른 step 비율.
               라우팅 = P > a 그리고 SC > c.  e30c_pipeline.py 와 맞춤: 격자 = 전체 행 분위 10 개, 겹마다 학습 겹 이득 최대 마스크, 이득 <= 0 이면 '안 함'.
               연속 2 진 (서술): LP = −칸 0 logprob > t1 (5/5 거르기) 그리고 S1 > t2 (0/5 거르기) 그리고 SC > t3, 같은 겹밖 격자.
               라우팅된 행은 B24 재생성 답, 나머지는 base 답.  0/5 도 재생성 답을 그대로 씀.
usage (repo 루트, CPU): /NAS/minyeol/limit/.venv-gfmrag/bin/python followup_mk/e58a.py --stage dirs|eval
"""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from e56_common import load_items, grade, VERS

STAGE = sys.argv[sys.argv.index("--stage") + 1]; OUT = "followup_mk"; D = f"{OUT}/e56_data"; NF = 5; Bt = 2000
items = load_items(); IT = {it["qid"]: it for it in items}
dr = collections.defaultdict(dict)
for fn in sorted(glob.glob(f"{OUT}/results_e56_dream_s*.jsonl")) + sorted(glob.glob(f"{OUT}/results_e56_steer_s*.jsonl")):
    for l in open(fn, encoding="utf-8"):
        try: d = json.loads(l); dr[(d["qid"], d["ver"])][d["cond"]] = d
        except Exception: pass
H = {}
for fn in glob.glob(f"{OUT}/e56_hid/b*.npz"):
    z = np.load(fn)
    for j, (q, v) in enumerate(zip(z["qid"], z["ver"])): H[(str(q), str(v))] = (z["h24"][j], z["h27"][j])
need = ("base",) if STAGE == "dirs" else ("base", "B24")
R = [(it["qid"], v) for it in items for v in VERS if all(c in dr.get((it["qid"], v), {}) for c in need) and (it["qid"], v) in H]
cnt = collections.Counter(q for q, _ in R); R = [r for r in R if cnt[r[0]] == 5]; n = len(R)
fidr = np.array([q for q, _ in R]); print(f"행 {n}, 사실 {len(set(fidr))}")
ok = np.array([grade(dr[r]["base"]["answer"], IT[r[0]]["value"], IT[r[0]]["aliases"], IT[r[0]][r[1]]) == "correct" for r in R])
kc = collections.Counter(); [kc.__setitem__(f, kc[f] + int(o)) for f, o in zip(fidr, ok)]; kk = np.array([kc[f] for f in fidr])
X24 = np.stack([H[r][0] for r in R]).astype(np.float32); X27 = np.stack([H[r][1] for r in R]).astype(np.float32)
uq = sorted(set(fidr)); perm = np.random.default_rng(58).permutation(len(uq)); FOLD = {uq[j]: int(p % NF) for j, p in enumerate(perm)}; fo_ = np.array([FOLD[f] for f in fidr])
unit = lambda v: v / np.linalg.norm(v)

if STAGE == "dirs":
    Bd = []
    for g in range(NF):
        tr = fo_ != g; byf = collections.defaultdict(lambda: [[], []])
        for j in np.where(tr & (kk >= 1) & (kk <= 4))[0]: byf[fidr[j]][0 if ok[j] else 1].append(j)
        Bd.append(unit(np.mean([X24[a].mean(0) - X24[b].mean(0) for a, b in byf.values() if a and b], 0)))
    scale = 0.5 * float(np.linalg.norm(X24, axis=1).mean())
    np.savez(f"{D}/e58_dirs.npz", B=np.stack(Bd).astype(np.float32), scale=scale); json.dump(FOLD, open(f"{D}/folds.json", "w"))
    print("scale", scale, "겹 간 cos", np.round(np.stack(Bd) @ np.stack(Bd).T, 3).tolist()); sys.exit()

# ---------------- eval
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
assert json.load(open(f"{D}/folds.json")) == FOLD, "겹이 dirs 단계와 다름"
okS = np.array([grade(dr[r]["B24"]["answer"], IT[r[0]]["value"], IT[r[0]]["aliases"], IT[r[0]][r[1]]) == "correct" for r in R])
lp4 = np.array([dr[r]["base"]["lp4"] for r in R]); lp0 = lp4[:, 0]; lpm = lp4.mean(1); mg = np.array([dr[r]["base"]["margin0"] for r in R])
SC = np.array([np.mean([a != dr[r]["base"]["mid_ans"][31] for a in dr[r]["base"]["mid_ans"][:31]]) for r in R])
y3 = np.where(kk == 0, 0, np.where(kk == 5, 2, 1)); mid = y3 == 1; z0 = y3 == 0; f5 = y3 == 2
S1 = np.zeros(n); PM = np.zeros(n)
for g in range(NF):
    tr = fo_ != g; te = ~tr
    K = unit(X27[tr & f5].mean(0) - X27[tr & z0].mean(0)); p = X27 @ K; med = np.median(p[tr & mid]); S1[te] = -np.abs(p[te] - med)
for g in range(NF):   # e30c_pipeline.py 와 같음: 겹밖 S1 을 그대로 특징으로, 학습 겹 평균·표준편차로 표준화, 다항 로지스틱
    tr = fo_ != g; te = ~tr; F = np.c_[lp0, lpm, mg, S1]; mu, sd = F[tr].mean(0), F[tr].std(0)
    clf = LogisticRegression(max_iter=2000).fit((F[tr] - mu) / sd, y3[tr]); PM[te] = clf.predict_proba((F[te] - mu) / sd)[:, list(clf.classes_).index(1)]
from sklearn.metrics import roc_auc_score
gain = okS.astype(int) - ok.astype(int)
qs = lambda x: np.unique(np.quantile(x, np.linspace(0, 1, 11))[:-1])   # e30c 와 같은 분위 10 개 격자 (전체 행 기준)
LP = -lp0


def route(masks):
    """e30c run(): 겹마다 학습 겹 이득 합이 가장 큰 마스크, 그 이득 <= 0 이면 '안 함'."""
    sel = np.zeros(n, bool); th = []
    for g in range(NF):
        tr = fo_ != g; te = ~tr; best, bj = -1e9, None
        for j, m in enumerate(masks):
            v = gain[tr & m].sum()
            if v > best: best, bj = v, j
        th.append(int(bj) if best > 0 else None)
        if best > 0: sel[te] = masks[bj][te]
    return sel, th


def ci_boot(v, f, seed=0):
    u, inv = np.unique(f, return_inverse=True); s = np.bincount(inv, weights=v.astype(float)); c = np.bincount(inv)
    rng = np.random.default_rng(seed); W = np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(Bt)])
    bt = (W @ s) / (W @ c); return [round(float(v.mean()), 5), round(float(np.percentile(bt, 2.5)), 5), round(float(np.percentile(bt, 97.5)), 5)]


QW = {}
for fn in glob.glob(f"{OUT}/results_e56_qwen_s*.jsonl"):
    for l in open(fn, encoding="utf-8"): d = json.loads(l); QW[(d["qid"], d["ver"])] = d
qok = np.array([grade(QW[r]["answer"], IT[r[0]]["value"], IT[r[0]]["aliases"], IT[r[0]][r[1]]) == "correct" if r in QW else False for r in R])
btype = np.array([grade(dr[r]["base"]["answer"], IT[r[0]]["value"], IT[r[0]]["aliases"], IT[r[0]][r[1]]) for r in R])
wrong_r = mid & np.isin(btype, ["entity", "broken"]); dlm = wrong_r & qok; shr = wrong_r & ~qok


def stats(sel, th=None):
    fin = np.where(sel, okS, ok); dlt = fin.astype(int) - ok.astype(int)
    o = dict(routed=int(sel.sum()), recovered_mid=int((sel & mid & ~ok & okS).sum()), broken_mid=int((sel & mid & ok & ~okS).sum()),
             recover_rate_mid_wrong=round(float((sel & mid & ~ok & okS).sum() / max((mid & ~ok).sum(), 1)), 4),
             break_rate_mid_correct=round(float((sel & mid & ok & ~okS).sum() / max((mid & ok).sum(), 1)), 4),
             mid_net=int(dlt[mid].sum()), mid_change_pts=[round(100 * x, 3) for x in ci_boot(dlt[mid], fidr[mid])],
             five_acc_base=round(float(ok[f5].mean() * 100), 2), five_change_pts=[round(100 * x, 3) for x in ci_boot(dlt[f5], fidr[f5])], five_broken=int((sel & f5 & ok & ~okS).sum()),
             zero_routed_rate=round(float(sel[z0].mean()), 4), zero_gain_rows=int(dlt[z0].sum()),
             pop_net_rows=int(dlt.sum()), pop_change_pts=[round(100 * x, 3) for x in ci_boot(dlt, fidr)],
             routed_share={"mid_wrong": round(float(sel[mid & ~ok].mean()), 4), "mid_correct": round(float(sel[mid & ok].mean()), 4), "five": round(float(sel[f5].mean()), 4), "zero": round(float(sel[z0].mean()), 4)},
             recover_by_route={"dlm": dict(n=int(dlm.sum()), recovered=int((sel & dlm & okS).sum())), "shared": dict(n=int(shr.sum()), recovered=int((sel & shr & okS).sum()))})
    if th is not None: o["thresholds_by_fold"] = th
    return o


S = {"n_rows": n, "n_facts": len(uq), "k5": {int(a): int(b) for a, b in sorted(collections.Counter(kk[np.unique(fidr, return_index=True)[1]]).items())},
     "base_acc": round(float(ok.mean()), 4), "B24_all_acc": round(float(okS.mean()), 4),
     "auroc_mid_vs_rest": {"3class_P": round(float(roc_auc_score(mid, PM)), 4), "S1": round(float(roc_auc_score(mid, S1)), 4)},
     "auroc_SC_mid_wrong_vs_mid_correct": round(float(roc_auc_score(~ok[mid], SC[mid])), 4)}
S["all_B24"] = stats(np.ones(n, bool))
sel3, th3 = route([(PM > a) & (SC > c) for a in qs(PM) for c in qs(SC)]); S["three_class"] = stats(sel3, th3)
selA, thA = route([(LP > a) & (S1 > b_) & (SC > c) for a in qs(LP) for b_ in qs(S1) for c in qs(SC)]); S["sequential_AND"] = stats(selA, thA)
M = S["three_class"]
S["H58.1"] = dict(mid_net=M["mid_net"], mid_change_pts=M["mid_change_pts"], verdict=bool(M["mid_net"] > 0 and M["mid_change_pts"][1] > 0))
S["H58.2"] = dict(five_change_pts=M["five_change_pts"], verdict=bool(M["five_change_pts"][0] >= -1.0))
S["H58.3"] = dict(pop_net_rows=M["pop_net_rows"], pop_change_pts=M["pop_change_pts"], verdict=bool(M["pop_change_pts"][0] > 0 and M["pop_change_pts"][1] > 0))
json.dump(S, open(f"{OUT}/e58a_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k_: S[k_] for k_ in ("n_rows", "k5", "base_acc", "B24_all_acc", "auroc_mid_vs_rest", "auroc_SC_mid_wrong_vs_mid_correct", "H58.1", "H58.2", "H58.3")}, ensure_ascii=False))
for nm in ("all_B24", "three_class", "sequential_AND"): print(nm, {k_: S[nm][k_] for k_ in ("routed", "recovered_mid", "broken_mid", "mid_net", "five_change_pts", "zero_routed_rate", "pop_net_rows", "pop_change_pts", "recover_by_route")})
