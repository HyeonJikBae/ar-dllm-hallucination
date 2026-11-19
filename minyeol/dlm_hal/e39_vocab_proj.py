"""E39a (서술, CPU): Brel / B / K 방향을 어휘로 투영 (logit lens: W_U @ (norm_w * v)).  관계별 상위 토큰과, 그 관계 정답 첫 토큰이 위로 오는지 (AUROC)."""
import collections, json, sys, glob
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, "followup_mk")
from common import rows, ty, k, fid, mid, unit, load_X, MODEL, REV
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
snap = glob.glob("/home/minyeol/.cache/huggingface/hub/models--Dream-org--Dream-v0-Base-7B/snapshots/*/")[0]
idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
def W(name):
    with safe_open(snap + idx[name], framework="pt") as f: return f.get_tensor(name).float()
WU = W("lm_head.weight"); nw = W("model.norm.weight"); V = WU.shape[0]
X, _ = load_X(); rel = np.array([r["relation"] for r in rows])
def rank_auc(s, pos):
    r = np.argsort(np.argsort(s)); p = np.zeros(len(s), bool); p[list(pos)] = True; n1 = p.sum(); n0 = len(s) - n1
    return float((r[p].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
# 방향
byf = collections.defaultdict(lambda: collections.defaultdict(list))
for i in np.where(mid & (ty != "abstain"))[0]: byf[fid[i]]["c" if ty[i] == "correct" else "w"].append(i)
per = collections.defaultdict(list); allD = []
for f_, v in byf.items():
    if v["c"] and v["w"]: d = X[v["c"]].mean(0) - X[v["w"]].mean(0); per[rel[v["c"][0]]].append(d); allD.append(d)
Brel = {r_: unit(np.mean(d, 0)) for r_, d in per.items() if len(d) >= 15}; B = unit(np.mean(allD, 0))
K = {}
for r_ in Brel:
    a = (rel == r_) & (k == 5); b = (rel == r_) & (k == 0)
    if a.sum() >= 20 and b.sum() >= 20: K[r_] = unit(X[a].mean(0) - X[b].mean(0))
def proj(v): return (WU @ (nw * torch.from_numpy(v).float())).numpy()
gfirst = collections.defaultdict(set)
for i, r in enumerate(rows):
    if r["relation"] in Brel:
        t = tok(" " + r["gold_value"], add_special_tokens=False)["input_ids"]
        if t: gfirst[r["relation"]].add(t[0])
dec = lambda ids: [tok.decode([t]).strip() for t in ids]
rng = np.random.default_rng(0); out = {}
for r_ in sorted(Brel):
    s = proj(Brel[r_]); own = gfirst[r_]; other = set().union(*[gfirst[q] for q in Brel if q != r_]) - own
    o = {"사실 수": len(per[r_]), "정답 첫 토큰 종류": len(own),
         "Brel 상위 20": dec(np.argsort(-s)[:20]), "Brel 하위 10": dec(np.argsort(s)[:10]),
         "AUROC 자기 관계 정답 토큰 대 전체 어휘": round(rank_auc(s, own), 3)}
    sub = np.array(sorted(own | other)); lab = np.isin(sub, list(own))
    o["AUROC 자기 관계 정답 토큰 대 다른 관계 정답 토큰"] = round(rank_auc(s[sub], np.where(lab)[0]), 3)
    oth = [q for q in Brel if q != r_]; o["(대조) 다른 관계 Brel 로 같은 비교 평균"] = round(float(np.mean([rank_auc(proj(Brel[q])[sub], np.where(lab)[0]) for q in oth])), 3)
    o["(대조) 무작위 방향 20 개 평균"] = round(float(np.mean([rank_auc(proj(unit(rng.standard_normal(X.shape[1])))[sub], np.where(lab)[0]) for _ in range(20)])), 3)
    if r_ in K: o["K (5/5-0/5) 로 같은 비교"] = round(rank_auc(proj(K[r_])[sub], np.where(lab)[0]), 3); o["K 상위 15"] = dec(np.argsort(-proj(K[r_]))[:15])
    o["cos(Brel, B)"] = round(float(Brel[r_] @ B), 2)
    out[r_] = o; print(r_, json.dumps({kx: v for kx, v in o.items()}, ensure_ascii=False), flush=True)
sB = proj(B); out["__B 전체"] = {"상위 25": dec(np.argsort(-sB)[:25]), "하위 15": dec(np.argsort(sB)[:15])}; print("B", out["__B 전체"])
json.dump(out, open("followup_mk/e39_vocab_proj.json", "w"), ensure_ascii=False, indent=1); print("DONE")
