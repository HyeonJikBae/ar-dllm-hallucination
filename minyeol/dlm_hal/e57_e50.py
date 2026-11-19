"""E57 (b) 추가 (CPU): E50 전체 규모 순서 개입 복구 (칸0먼저 / L2R / entropy) 의 경로 차이를 E49c 와 같은 공변량으로 층화.
행 = E50 base 가 entity/broken 인 1~4/5 행 (DLM 고유 / 공유 = 같은 문구 Qwen 정오).  매개 점검: E50 base 의 step 0 금 순위 (gold_rank0) 구간."""
import collections, json, sys, time
import numpy as np
sys.path.insert(0, "followup_mk")
src = open("followup_mk/e57_routes.py", encoding="utf-8").read()
exec(src[:src.index("O = {\"공변량 정의\"")])          # covs, design, analyze, balance 정의만 가져옴 (e57_routes.py 와 같은 코드)
G = collections.defaultdict(dict)
for fn in ("results_e50_s0.jsonl", "results_e50_s1.jsonl"):
    for l in open(f"{OUT}/{fn}", encoding="utf-8"):
        x = json.loads(l)
        if x["set"] == "mid": G[x["cond"]][x["row"]] = x
R = np.array(sorted(r for r, x in G["base"].items() if x["grade"] in ("entity", "broken"))); d = q_corr[R]
O = {"n (DLM 고유 / 공유)": [int(d.sum()), int((~d).sum())]}
rk = np.array([G["base"][int(r)]["gold_rank0"] for r in R]); rkb = np.where(rk == 0, 0, np.where(rk < 5, 1, 2)); f = fid[R]
for c in ("칸0먼저", "L2R", "entropy"):
    y = np.array([G[c][int(r)]["grade"] == "correct" for r in R])
    o = analyze(R, d, y, "E50 " + c)
    o["매개: E50 base step0 금 순위 구간 (1 위 / 2~5 / 6+) 층 MH"] = boot_facts(f, lambda ix: mh_rd(y[ix], d[ix], rkb[ix])[0], B)
    o["구간별 복구 % (DLM 고유 / 공유, n)"] = {b: [round(100 * float(y[(rkb == j) & d].mean()), 1), round(100 * float(y[(rkb == j) & ~d].mean()), 1), int(((rkb == j) & d).sum()), int(((rkb == j) & ~d).sum())] for j, b in enumerate(["1 위", "2~5 위", "6 위 이하"])}
    print("매개", c, o["매개: E50 base step0 금 순위 구간 (1 위 / 2~5 / 6+) 층 MH"], o["구간별 복구 % (DLM 고유 / 공유, n)"], flush=True)
    O[c] = o
upd_summary("b_e50", O); print("DONE")
