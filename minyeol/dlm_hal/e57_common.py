"""E57 공용 (CPU).  경로 라벨, E7·E44·Qwen 자료, 공변량, 사실 군집 부트스트랩.  repo 루트에서 import."""
import collections, hashlib, json, re, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import rows, ty, k, fid, grp, mid, NF, OUT, POP

N = len(rows)
QF = [json.loads(l) for l in open("data/3_classification/full/qwen_final_full.jsonl", encoding="utf-8")]
assert all((a["sample_id"], a["version"]) == (b["sample_id"], b["version"]) for a, b in zip(rows, QF))
q_corr = np.array([q["correct"] for q in QF]); q_ty = np.array([q["type3"] if not q["correct"] else "correct" for q in QF])
rel = np.array([r["relation"] for r in rows]); ver = np.array([r["version"] for r in rows]); ques = [r["question"] for r in rows]

# ---- E7
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); W7 = {int(r): j for j, r in enumerate(R7)}
E7T = {}
for l in open(f"{D7}/meta.jsonl", encoding="utf-8"): d = json.loads(l); E7T[d["row"]] = d["new_type"]

# ---- E44 (step 0 칸 0 실제 로짓)
E44 = {}
for s in range(3):
    d_ = f"{OUT}/e44/shard{s}"; R = np.load(f"{d_}/rows.npy"); done = np.load(f"{d_}/done.npy")
    gr = np.load(f"{d_}/slot0_gold_rank.npy", mmap_mode="r"); am = np.load(f"{d_}/slot0_argmax.npy", mmap_mode="r")
    gl = np.load(f"{d_}/slot0_gold_logit.npy", mmap_mode="r"); tl = np.load(f"{d_}/slot0_top1_logit.npy", mmap_mode="r")
    mk = np.load(f"{d_}/slot0_mask.npy", mmap_mode="r")
    for j, l in enumerate(open(f"{d_}/meta.jsonl", encoding="utf-8")):
        x = json.loads(l); assert x["row"] == R[j]
        if done[j]:
            x.update(rank0=int(gr[j, 0]), argmax0=int(am[j, 0]), gap0=float(gl[j, 0] - tl[j, 0]), final_tok=x["slot0_final_token"],
                     lost=bool(gr[j, 0] == 0 and x["slot0_final_token"] != x["gold_id"]))
            E44[int(x["row"])] = x

# ---- Qwen forward (e57_qwen_gold.py)
DQ = f"{OUT}/e57_qwen"; RQ = np.load(f"{DQ}/rows.npy"); WQ = {int(r): j for j, r in enumerate(RQ)}
QG = {kx: np.load(f"{DQ}/{kx}.npy") for kx in ("gold_id", "gold_lp", "gold_rank", "top1_id", "top1_lp", "margin")}
def qwen(r, key): return QG[key][WQ[r]] if r in WQ else np.nan

# ---- 공변량
_tok = None
def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    return _tok

def gold_tokens():
    """사실마다 금 답 토큰 (' ' + gold_value).  반환 (첫 토큰 id [N], 답 토큰 수 [N], 첫 토큰의 사실 단위 빈도 [N])."""
    t = tok(); first = {}; ln = {}; gv = {}
    for i, r in enumerate(rows): gv.setdefault(fid[i], r["gold_value"])
    for f_, g in gv.items(): ids = t(" " + g)["input_ids"]; first[f_] = ids[0]; ln[f_] = len(ids)
    cnt = collections.Counter(first.values())
    return (np.array([first[f_] for f_ in fid]), np.array([ln[f_] for f_ in fid]), np.array([cnt[first[f_]] for f_ in fid]))

s_pop = np.array([POP[r["sample_id"]]["s_pop"] for r in rows], float); o_pop = np.array([POP[r["sample_id"]]["o_pop"] for r in rows], float)
half = np.array([int(hashlib.md5(f_.encode()).hexdigest(), 16) % 2 for f_ in fid])

def wset(s): return set(re.findall(r"[a-z0-9']+", s.lower()))
def lexical():
    """E40 과 같은 어휘 특징: 원래 템플릿 (version=question) 과 단어 Jaccard, 단어 수, 글자 수, 의문사."""
    orig = {}
    for i in range(N):
        if ver[i] == "question": orig[fid[i]] = i
    jac = np.array([len(wset(ques[i]) & wset(ques[orig[fid[i]]])) / max(1, len(wset(ques[i]) | wset(ques[orig[fid[i]]]))) for i in range(N)])
    wc = np.array([len(s.split()) for s in ques], float); cc = np.array([len(s) for s in ques], float)
    wh = np.array([(s.split()[0].lower().strip("?,") if s.split() and s.split()[0].lower().strip("?,") in ("what", "who", "where", "which", "when", "in", "how", "whose") else "other") for s in ques])
    return jac, wc, cc, wh

# ---- 통계
def qbin(x, q=5):
    """분위 구간 (동점은 순위로 나눔).  NaN 은 -1."""
    x = np.asarray(x, float); out = np.full(len(x), -1); ok = ~np.isnan(x)
    rk = np.argsort(np.argsort(x[ok], kind="stable"), kind="stable"); out[ok] = (rk * q // ok.sum()).astype(int)
    return out

def fact_index(f):
    u, inv = np.unique(f, return_inverse=True); return u, inv

def boot_facts(f, stat, B=2000, seed=0):
    """stat(idx) -> float.  사실 군집 부트스트랩: 사실을 복원 추출하고 그 사실의 행을 모두 넣음."""
    u, inv = fact_index(f); members = [np.where(inv == j)[0] for j in range(len(u))]; rng = np.random.default_rng(seed)
    est = stat(np.arange(len(f))); bs = []
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u)); idx = np.concatenate([members[j] for j in pick]); v = stat(idx)
        if v is not None and np.isfinite(v): bs.append(v)
    return [round(float(est), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

def mh_rd(y, d, s):
    """층 s 별 위험 차이 (d=1 − d=0).  Mantel-Haenszel 가중 (n1 n0 / N) 과 층 크기 가중 평균 (두 집단 모두 있는 층).  반환 (MH, 크기 가중, 덮는 행 비율)."""
    y = np.asarray(y, float); d = np.asarray(d, bool); s = np.asarray(s)
    if s.dtype.kind not in "iu" or s.min() < 0: s = np.unique(s, return_inverse=True)[1]
    m_ = int(s.max()) + 1; n1 = np.bincount(s[d], minlength=m_); n0 = np.bincount(s[~d], minlength=m_)
    y1 = np.bincount(s[d], weights=y[d], minlength=m_); y0 = np.bincount(s[~d], weights=y[~d], minlength=m_); v = (n1 > 0) & (n0 > 0)
    if not v.any(): return np.nan, np.nan, 0.0
    rd = y1[v] / n1[v] - y0[v] / n0[v]; w = n1[v] * n0[v] / (n1[v] + n0[v]); N_ = n1[v] + n0[v]
    return float((w * rd).sum() / w.sum()), float((N_ * rd).sum() / N_.sum()), float(N_.sum() / len(y))

def r4(x): return round(float(x), 4)
def upd_summary(key, obj):
    """e57_summary.json 의 key 만 갱신 (잠금 + 임시 파일 교체)."""
    import fcntl, os
    p = f"{OUT}/e57_summary.json"; conv = lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))
    with open(p + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try: S = json.load(open(p, encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError): S = {}
        S[key] = obj; json.dump(S, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=conv); os.replace(p + ".tmp", p)
