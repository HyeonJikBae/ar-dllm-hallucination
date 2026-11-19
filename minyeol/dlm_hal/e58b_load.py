"""E58b 공용 적재 (CPU): base 결과와 은닉 상태를 행 순서 R 로 맞춤."""
import glob, json
import numpy as np
def load_base(pilot=False):
    sf = "_pilot" if pilot else ""; rows = [json.loads(l) for l in open("data/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
    meta = json.load(open("followup_mk/e58b/rows.json")); B = {}
    for f in sorted(glob.glob(f"followup_mk/results_e58b_base{sf}_s*.jsonl")):
        for l in open(f, encoding="utf-8"): o = json.loads(l); B[o["row"]] = o
    hs = {}
    for f in sorted(glob.glob(f"followup_mk/e58b/hid{sf}_s*.npy")):
        rr = json.load(open(f.replace(".npy", "_rows.json"))); Hm = np.load(f, mmap_mode="r")
        for q, i in enumerate(rr):
            if i in B: hs[i] = np.asarray(Hm[q])
    R = np.array(sorted(B)); H = np.stack([hs[i] for i in R])
    G = np.load("followup_mk/e58b/grp.npy")   # common.grp (주어 5 겹) 를 저장해 둔 것
    k = np.array([meta["k"][str(i)] for i in R]); grp = G[R]; fid = np.array([rows[i]["sample_id"] for i in R])
    return R, B, H, k, grp, fid
