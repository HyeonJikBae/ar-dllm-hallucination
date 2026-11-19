"""E58b 행 (CPU): DiffuLLaMA 자기 k/5 (E55 A 단계) 로 1~4/5 전부 + 0/5 행 2,000 + 5/5 행 2,000 (seed 58).  모집단 가중 = 그 범주 전체 행 수 / 뽑은 수."""
import collections, glob, json
import numpy as np
rows = [json.loads(l) for l in open("data/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
D = {}
for f in sorted(glob.glob("followup_mk/results_e55_dllm_A_s*.jsonl")):
    for l in open(f, encoding="utf-8"): o = json.loads(l); D[o["row"]] = o["base"]["type"] == "correct"
byf = collections.defaultdict(list)
for i, r in enumerate(rows): byf[r["sample_id"]].append(i)
kr = np.zeros(len(rows), int)
for f, ix in byf.items():
    for i in ix: kr[i] = sum(D[j] for j in ix)
mid = np.where((kr >= 1) & (kr <= 4))[0]; z_all = np.where(kr == 0)[0]; f_all = np.where(kr == 5)[0]
z = np.sort(np.random.default_rng(58).permutation(z_all)[:2000]); f5 = np.sort(np.random.default_rng(59).permutation(f_all)[:2000])
out = dict(rows=[int(i) for i in np.concatenate([mid, z, f5])], k={int(i): int(kr[i]) for i in np.concatenate([mid, z, f5])},
           pop_rows={"zero": int(len(z_all)), "mid": int(len(mid)), "five": int(len(f_all))}, n={"zero": len(z), "mid": len(mid), "five": len(f5)})
json.dump(out, open("followup_mk/e58b/rows.json", "w")); print(out["pop_rows"], out["n"])
