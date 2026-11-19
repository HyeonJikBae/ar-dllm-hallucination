"""E55 B 단계 행 목록 (CPU): A 단계 (DiffuLLaMA base, 55,225 행) 채점으로 DiffuLLaMA 자기 k/5 를 세고, 1~4/5 사실의 다섯 문구 행 전부를 e55_rows_B.json 에 쓴다."""
import collections, glob, json
import numpy as np
rows = [json.loads(l) for l in open("data/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
D = {}
for f in sorted(glob.glob("followup_mk/results_e55_dllm_A_s*.jsonl")):
    for l in open(f, encoding="utf-8"): o = json.loads(l); D[o["row"]] = o["base"]["type"] == "correct"
assert len(D) == len(rows), (len(D), len(rows))
byf = collections.defaultdict(list)
for i, r in enumerate(rows): byf[r["sample_id"]].append(i)
kk = {f: sum(D[i] for i in ix) for f, ix in byf.items()}
mixed = sorted(f for f, v in kk.items() if 1 <= v <= 4); R = sorted(i for f in mixed for i in byf[f])
print("정확도", round(100 * np.mean(list(D.values())), 2), "k/5:", dict(sorted(collections.Counter(kk.values()).items())), "1~4/5 사실", len(mixed), "행", len(R))
json.dump(R, open("followup_mk/e55_rows_B.json", "w"))
