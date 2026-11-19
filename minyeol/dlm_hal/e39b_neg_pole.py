"""E39b (서술, CPU): B/Brel 의 음의 극 (밀면 줄어드는 토큰) 이 붕괴 답의 첫 토큰과 겹치는가."""
import collections, json, sys, glob
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, "followup_mk")
from common import rows, ty, k, fid, mid, unit, load_X, MODEL, REV, OUT
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
snap = glob.glob("/home/minyeol/.cache/huggingface/hub/models--Dream-org--Dream-v0-Base-7B/snapshots/*/")[0]; idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
def W(n):
    with safe_open(snap + idx[n], framework="pt") as f: return f.get_tensor(n).float()
WU = W("lm_head.weight"); nw = W("model.norm.weight"); X, _ = load_X()
byf = collections.defaultdict(lambda: collections.defaultdict(list))
for i in np.where(mid & np.isin(ty, ["correct", "broken"]))[0]: byf[fid[i]][ty[i]].append(i)
B = unit(np.mean([X[v["correct"]].mean(0) - X[v["broken"]].mean(0) for v in byf.values() if v["correct"] and v["broken"]], 0))
s = (WU @ (nw * torch.from_numpy(B).float())).numpy()
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r"))
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); com = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
t0 = {"붕괴": collections.Counter(), "entity": collections.Counter(), "정답": collections.Counter()}
for j, r in enumerate(R7):
    r = int(r); nt = meta.get(r, {}).get("new_type")
    if r in meta and 1 <= k[r] <= 4 and nt in ("broken", "entity", "correct"): t0[{"broken": "붕괴", "entity": "entity", "correct": "정답"}[nt]][int(com[j, 0])] += 1
rk = np.argsort(np.argsort(s)) / len(s)   # 0 = 가장 음, 1 = 가장 양
for nm, c in t0.items():
    toks = list(c.elements()); print(nm, "첫 토큰 평균 순위 백분위 (B 투영, 0=음극)", round(float(np.mean(rk[toks])), 3), " 흔한 첫 토큰", [(tok.decode([t]), n) for t, n in c.most_common(8)])
