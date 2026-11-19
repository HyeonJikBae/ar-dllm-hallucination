"""E16 사전 점검 (CPU): 층 20 프롬프트 끝 step 0 상태에서 같은 사실 안 정답-오답 신호가 일부 좌표에 몰려 있는가."""
import collections, json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from common import grp, ty, fid, mid, unit, load_X, NF, OUT
X, norm = load_X(); O = {}
mu_all = X.mean(0); sd_all = X.std(0)
big = np.argsort(-np.abs(mu_all))[:10]; O["큰평균좌표_top10_|평균|/중앙값"] = [round(float(abs(mu_all[j]) / np.median(np.abs(mu_all))), 1) for j in big]
idx = np.where(mid & (ty != "abstain"))[0]; lab = (ty == "correct")
def fact_diffs(tr):
    byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in idx[tr[idx]]: byf[fid[i]]["c" if lab[i] else "w"].append(i)
    return np.stack([X[v["c"]].mean(0) - X[v["w"]].mean(0) for v in byf.values() if v["c"] and v["w"]])
def within(score):
    g = collections.defaultdict(list)
    for i in idx: g[fid[i]].append(i)
    a = [roc_auc_score(lab[v], score[v]) for v in g.values() if 0 < lab[v].sum() < len(v)]; return round(float(np.mean(a)), 3)
D = fact_diffs(np.ones(len(X), bool)); B = D.mean(0); t = B / (D.std(0) / np.sqrt(len(D)))
e = np.sort(B ** 2)[::-1]; O["B 에너지 상위 1/5/10/25% 좌표 몫"] = [round(float(e[: int(3584 * p)].sum() / e.sum()), 3) for p in (.01, .05, .10, .25)]
O["B 와 큰평균좌표 10개 에너지 몫"] = round(float((B[big] ** 2).sum() / (B ** 2).sum()), 3)
O["|t|>3 좌표 수"] = int((np.abs(t) > 3).sum()); O["|t|>5 좌표 수"] = int((np.abs(t) > 5).sum())
# 겹밖: 상위 k 좌표 (|t| 기준) 만 쓴 B 투영의 같은 사실 안 AUROC.  무작위 k 좌표와 비교.  표준화 공간 (좌표별 sd 로 나눔) 판도.
res = collections.defaultdict(dict)
for k_ in (10, 36, 100, 358, 1000, 3584):
    for kind in ("t상위", "무작위", "표준화t상위"):
        sc = np.zeros(len(X))
        for g in range(NF):
            tr = grp != g; Dg = fact_diffs(tr); Bg = Dg.mean(0); tg = Bg / (Dg.std(0) / np.sqrt(len(Dg)) + 1e-9)
            if kind == "무작위": keep = np.random.default_rng(g).choice(3584, k_, replace=False)
            else: keep = np.argsort(-np.abs(tg))[:k_]
            v = np.zeros(3584); v[keep] = Bg[keep]
            if kind == "표준화t상위": v[keep] = Bg[keep] / (sd_all[keep] ** 2)
            te = np.where(grp == g)[0]; sc[te] = X[te] @ v
        res[kind][k_] = within(sc)
    print(k_, {kd: res[kd][k_] for kd in res}, flush=True)
O["같은사실_AUROC_k좌표"] = res
print(json.dumps(O, ensure_ascii=False)); json.dump(O, open(f"{OUT}/e16_sparse_read.json", "w"), ensure_ascii=False, indent=1)
