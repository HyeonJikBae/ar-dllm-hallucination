"""Split the steering results by WHEN the answer's first real slot was filled in the stored run: inside the edited window (first 8 forwards) or later.
If editing only steps 0..7 is the problem, the signal-vs-random difference should show up in the early-filled rows.   CPU only.
Usage: python3 analysis/6_steering/by_commit_time.py [results_from0of5.jsonl ...]"""
import collections, json, sys
import numpy as np
from transformers import AutoTokenizer
F = "data/1_raw/dream_hidden_steps"; T = np.load(f"{F}/tokens.npy"); M = np.load(f"{F}/conf_mask.npy", mmap_mode="r")
tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", trust_remote_code=True); SP = set(tok.all_special_ids)
WS = {int(i) for i in np.unique(T) if i >= 0 and int(i) not in SP and tok.decode([int(i)]).strip() == ""}
def first_fill(i):                         # number of forwards the first real answer slot stayed masked (<= 8 means it was decided inside forwards 0..7)
    t = T[i]; n = next((q for q, x in enumerate(t) if int(x) in SP), 32); n0 = 0
    while n0 < n and int(t[n0]) in WS: n0 += 1
    if n0 >= n: return None
    return int(np.cumprod(np.asarray(M[i])[:, n0].astype(np.int8)).sum())
for path in (sys.argv[1:] or ["analysis/6_steering/results_from0of5.jsonl", "analysis/6_steering/results_cv5.jsonl", "analysis/6_steering/results_allsignal.jsonl"]):
    R = collections.defaultdict(dict)
    for l in open(path, encoding="utf-8"): r = json.loads(l); R[r["cond"]][r["row"]] = r["new_type"]
    rows = sorted(R["base"]); ff = {i: first_fill(i) for i in rows}
    early = [i for i in rows if ff[i] is not None and ff[i] <= 8]; late = [i for i in rows if ff[i] is not None and ff[i] > 8]
    print(f"\n=== {path}: rows {len(rows)} | answer start decided inside forwards 0-7: {len(early)} | later: {len(late)} | empty answer: {len(rows) - len(early) - len(late)}")
    br = lambda c, g: np.mean([R[c][i] == "broken" for i in g]); co = lambda c, g: np.mean([R[c][i] == "correct" for i in g])
    for c in R:
        if "random" in c: continue
        print(f"  {c:<28s} EARLY broken {br(c, early):.3f} correct {co(c, early):.3f} | LATE broken {br(c, late):.3f} correct {co(c, late):.3f}")
    groups = collections.defaultdict(list)
    for c in R:
        if "random" in c: groups[c.split("#")[0].strip()].append(c)
    for gname, cs in groups.items():
        e = np.array([br(c, early) for c in cs]); la = np.array([br(c, late) for c in cs])
        print(f"  {gname:<28s} EARLY broken mean {e.mean():.3f} ({e.min():.3f}-{e.max():.3f}) | LATE broken mean {la.mean():.3f} ({la.min():.3f}-{la.max():.3f})  [{len(cs)} draws]")
