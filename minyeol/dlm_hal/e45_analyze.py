"""E45 (PREREG.md, CPU): 첫 칸 결정의 시간 동역학 분석.  usage: python followup_mk/e45_analyze.py --src e44|e7
E44 (촘촘한 궤적) 가 주 자료, E7 (step 0/4/8) 은 코드 검증용 (logit lens 없음 -> 결정 여유 대신 '첫 칸 argmax == 금 토큰' 비율을 대리로).
같은 사실 안 짝 비교 (사실 평균 제거).  AT-high = k 3,4, AT-low = k 1,2 (저장 라벨).  정오 = 그 실행의 채점."""
import collections, json, os, sys, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from common import rows, ty, k, fid, mid, grp, unit, load_X, NF, OUT, MODEL, REV
SRC = sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else "e44"
FAST = "--fast" in sys.argv
rng0 = np.random.default_rng(0)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
def gold_tok(i):
    t = tok(" " + rows[i]["gold_value"], add_special_tokens=False)["input_ids"]; return t[0] if t else -1

# ---------------------------------------------------------------- 자료 읽기 (표준 형식으로)
def load_e7():
    D = f"{OUT}/e7"; R7 = np.load(f"{D}/rows.npy"); done = np.load(f"{D}/done.npy"); cfg = json.load(open(f"{D}/config.json")); meta = {}
    for l in open(f"{D}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
    J = np.where([done[j] and int(R7[j]) in meta for j in range(len(R7))])[0]; RR = R7[J]
    IDS = np.asarray(np.load(f"{D}/ids.npy", mmap_mode="r")[J][:, :, 0]); MK = np.asarray(np.load(f"{D}/mask.npy", mmap_mode="r")[J])
    H = np.load(f"{D}/hid.npy", mmap_mode="r")
    return dict(rows=RR, J=J, nt=np.array([meta[int(i)]["new_type"] for i in RR]), steps=cfg["steps"], layers=cfg["layers"],
                getH=lambda si, li, sel: np.asarray(H[J[sel], si, li, 0], dtype=np.float32),
                s0_argmax=IDS, s0_masked=MK[:, :, 0], ncommit=(~MK).sum(2), all_steps=list(range(MK.shape[1])), LL=None,
                gold=np.array([gold_tok(int(i)) for i in RR]))
def load_e44_shards():
    """E44 가 shard0..2 로 나뉘어 저장됨 (2026-11-18 수정: 원래 load_e44 는 합친 폴더를 가정, 파일 이름도 다름)."""
    Ds = sorted(d for d in os.listdir(f"{OUT}/e44") if d.startswith("shard")); P = []
    for d in Ds:
        D = f"{OUT}/e44/{d}"; cfg = json.load(open(f"{D}/config.json")); R = np.load(f"{D}/rows.npy"); done = np.load(f"{D}/done.npy"); meta = {}
        for l in open(f"{D}/meta.jsonl"): m_ = json.loads(l); meta[m_["row"]] = m_
        J = np.where([done[j] and int(R[j]) in meta for j in range(len(R))])[0]
        P.append(dict(D=D, cfg=cfg, J=J, RR=R[J], meta=meta, H=np.load(f"{D}/hid.npy", mmap_mode="r")))
    cfg = P[0]["cfg"]; RR = np.concatenate([p["RR"] for p in P]); off = np.cumsum([0] + [len(p["J"]) for p in P])
    cat = lambda name: np.concatenate([np.asarray(np.load(f"{p['D']}/{name}.npy", mmap_mode="r")[p["J"]]) for p in P])
    def getH(si, li, sel):
        sel = np.asarray(sel); out = np.empty((len(sel), P[0]["H"].shape[-1]), np.float32)
        for q, p in enumerate(P):
            m_ = (sel >= off[q]) & (sel < off[q + 1])
            if m_.any(): out[m_] = np.asarray(p["H"][p["J"][sel[m_] - off[q]], si, li], dtype=np.float32)
        return out
    meta = {k_: v for p in P for k_, v in p["meta"].items()}
    return dict(rows=RR, J=np.arange(len(RR)), nt=np.array([meta[int(i)]["new_type"] for i in RR]), steps=cfg["steps"], layers=cfg["layers"], getH=getH,
                s0_argmax=cat("slot0_argmax"), s0_masked=cat("slot0_mask"), ncommit=cat("committed"), all_steps=list(range(32)),
                LL=dict(gold_logit=cat("ll_gold_logit").astype(np.float32), gold_rank=cat("ll_gold_rank"), top_ids=cat("ll_top_ids"), top_logits=cat("ll_top_logit").astype(np.float32)),
                gold=np.array([meta[int(i)]["gold_id"] for i in RR]), final_tok0=np.array([meta[int(i)]["slot0_final_token"] for i in RR]))
def load_e44():
    if not os.path.exists(f"{OUT}/e44/config.json"): return load_e44_shards()
    D = f"{OUT}/e44"; cfg = json.load(open(f"{D}/config.json")); R = np.load(f"{D}/rows.npy")
    done = np.load(f"{D}/done.npy") if os.path.exists(f"{D}/done.npy") else np.ones(len(R), bool)
    meta = {}
    for l in open(f"{D}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
    J = np.where([done[j] and int(R[j]) in meta for j in range(len(R))])[0]; RR = R[J]
    ld = lambda name: np.load(f"{D}/{name}.npy", mmap_mode="r")
    H = ld(cfg.get("hid_name", "hid"))
    out = dict(rows=RR, J=J, nt=np.array([meta[int(i)]["new_type"] for i in RR]), steps=cfg["steps"], layers=cfg["layers"],
               getH=lambda si, li, sel: np.asarray(H[J[sel], si, li], dtype=np.float32))
    # 첫 칸 step 별 정보 (가능한 이름 몇 가지를 허용)
    def first(*names):
        for n_ in names:
            if os.path.exists(f"{D}/{n_}.npy"): return np.asarray(ld(n_)[J])
        return None
    out["s0_argmax"] = first("slot0_argmax", "s0_argmax", "argmax0"); out["s0_masked"] = first("slot0_masked", "s0_masked", "mask0")
    out["ncommit"] = first("ncommit", "n_committed", "committed")
    out["all_steps"] = cfg.get("all_steps", list(range(out["s0_argmax"].shape[1])) if out["s0_argmax"] is not None else cfg["steps"])
    gl = first("ll_gold_logit", "gold_logit"); gr = first("ll_gold_rank", "gold_rank"); ti = first("ll_top_ids", "top_ids"); tl = first("ll_top_logits", "top_logits")
    out["LL"] = None if gl is None else dict(gold_logit=gl.astype(np.float32), gold_rank=gr, top_ids=ti, top_logits=None if tl is None else tl.astype(np.float32))
    g = first("gold_id", "gold")
    out["gold"] = g if g is not None else np.array([gold_tok(int(i)) for i in RR])
    fin = first("final_first", "final_tok0")
    out["final_tok0"] = fin
    return out
A = load_e44() if SRC == "e44" else load_e7()
RR = A["rows"]; n = len(RR); kk = k[RR]; f7 = fid[RR]; g7 = grp[RR]; nt = A["nt"]; corr = nt == "correct"
ism = (kk >= 1) & (kk <= 4); hi = ism & (kk >= 3); lo = ism & (kk <= 2)
print(f"자료 {SRC}: 행 {n} (1~4/5 {ism.sum()}, AT-high {hi.sum()}, AT-low {lo.sum()}), steps {A['steps']}, layers {A['layers']}", flush=True)
O = {"자료": SRC, "행": int(n), "AT-high 행": int(hi.sum()), "AT-low 행": int(lo.sum()), "steps": A["steps"], "layers": A["layers"]}
MI = np.where(ism)[0]
byf = collections.defaultdict(list)
for j in MI: byf[f7[j]].append(int(j))
mixed = {f: v for f, v in byf.items() if 0 < corr[v].sum() < len(v)}
fact_hi = {f: v for f, v in mixed.items() if kk[v[0]] >= 3}; fact_lo = {f: v for f, v in mixed.items() if kk[v[0]] <= 2}
print(f"섞인 사실 {len(mixed)} (high {len(fact_hi)}, low {len(fact_lo)})", flush=True)

def boot_fact(vals_by_fact, B=2000, seed=1):
    fs = [f for f, v in vals_by_fact.items() if v is not None and np.isfinite(v)]
    if not fs: return [np.nan, np.nan, np.nan]
    x = np.array([vals_by_fact[f] for f in fs]); r = np.random.default_rng(seed); bs = x[r.integers(0, len(x), (B, len(x)))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
def paired(score, facts):
    """사실마다 (맞힘 평균 - 틀림 평균).  score: 행 배열 (nan 허용)."""
    d = {}
    for f, v in facts.items():
        v = [j for j in v if np.isfinite(score[j])]; y = corr[v]
        if 0 < y.sum() < len(v): d[f] = float(score[v][y].mean() - score[v][~y].mean())
    return d

# ---------------------------------------------------------------- (1) 결정 여유 곡선
gold = np.asarray(A["gold"])
curves = {}
if A["LL"] is not None:
    LL = A["LL"]; S, L = A["steps"], A["layers"]
    top_ids, top_l, gl = LL["top_ids"], LL["top_logits"], LL["gold_logit"]
    # 금 아닌 최고 logit
    best_other = np.where(top_ids[..., 0] == gold[:, None, None], top_l[..., 1], top_l[..., 0])
    m_other = gl - best_other                                               # [n, S, L]
    fin = A.get("final_tok0")
    if fin is None and A.get("s0_argmax") is not None: fin = A["s0_argmax"][:, -1]
    m_final = np.full(gl.shape, np.nan, np.float32)
    if fin is not None:
        hit = top_ids == np.asarray(fin)[:, None, None, None]
        fl = np.where(hit.any(-1), (top_l * hit).sum(-1) / np.maximum(hit.sum(-1), 1), np.nan)
        m_final = gl - fl
        O["결정여유_최종토큰_top10_포함률"] = round(float(np.isfinite(m_final[MI]).mean()), 3)
    for li, l_ in enumerate(L):
        for nm, M_ in (("금-금아닌최고", m_other), ("금-최종첫토큰", m_final)):
            for gnm, facts in (("AT-high", fact_hi), ("AT-low", fact_lo)):
                key = f"{nm} | 층{l_} | {gnm}"; curves[key] = {}
                for si, s in enumerate(S):
                    sc = M_[:, si, li]; d = paired(sc, facts)
                    curves[key][f"s{s}"] = dict(맞힘평균=round(float(np.nanmean(sc[[j for v in facts.values() for j in v if corr[j]]])), 3),
                                                틀림평균=round(float(np.nanmean(sc[[j for v in facts.values() for j in v if not corr[j]]])), 3),
                                                짝차이=boot_fact(d))
                # 오답 유형별 (틀림 쪽만)
                for tnm in ("broken", "entity"):
                    key2 = f"{nm} | 층{l_} | {gnm} | 틀림={tnm}"; curves[key2] = {}
                    for si, s in enumerate(S):
                        sc = M_[:, si, li]; js = [j for v in facts.values() for j in v if nt[j] == tnm]
                        curves[key2][f"s{s}"] = round(float(np.nanmean(sc[js])), 3) if js else None
    # 금 순위 곡선
    gr = LL["gold_rank"]
    for li, l_ in enumerate(L):
        for gnm, sel in (("AT-high", hi), ("AT-low", lo)):
            for cn, cm in (("맞힘", corr), ("틀림", ~corr)):
                m_ = sel & cm; curves[f"금순위top1비율 | 층{l_} | {gnm} | {cn}"] = {f"s{s}": round(100 * float((gr[m_, si, li] == 0).mean()), 1) for si, s in enumerate(S)}
else:
    # E7 대리: 마지막 층 첫 칸 argmax == 금 토큰 비율, step 0..31
    am = A["s0_argmax"]; isg = am == gold[:, None]
    for gnm, sel in (("AT-high", hi), ("AT-low", lo)):
        for cn, cm in (("맞힘", corr), ("틀림", ~corr)):
            m_ = sel & cm; curves[f"[대리] 첫칸 argmax==금 비율 | {gnm} | {cn}"] = {f"s{s}": round(100 * float(isg[m_, s].mean()), 1) for s in range(am.shape[1])}
        for tnm in ("broken", "entity"):
            m_ = sel & (nt == tnm); curves[f"[대리] 첫칸 argmax==금 비율 | {gnm} | 틀림={tnm}"] = {f"s{s}": round(100 * float(isg[m_, s].mean()), 1) for s in range(am.shape[1])}
        d_by_s = {}
        for s in range(am.shape[1]):
            facts = fact_hi if gnm == "AT-high" else fact_lo; d_by_s[f"s{s}"] = boot_fact(paired(isg[:, s].astype(float), facts))
        curves[f"[대리] 짝차이 (맞힘-틀림) argmax==금 | {gnm}"] = d_by_s
O["1_결정여유_곡선"] = curves
print("(1) 끝", flush=True)

# ---------------------------------------------------------------- (2) 확정 시점, 흔들림, 확정 칸 수, 속도, 최종 상태와 cos
dyn = {}
if A.get("s0_argmax") is not None:
    am = np.asarray(A["s0_argmax"]); T = am.shape[1]
    msk = A.get("s0_masked")
    fin = np.asarray(A["final_tok0"]) if A.get("final_tok0") is not None else am[:, -1]   # 확정된 첫 토큰 (확정 뒤 argmax 는 무시)
    stable = np.ones(n, int) * (T - 1)
    for j in range(n):
        neq = np.where((am[j] != fin[j]) & (msk[j] if msk is not None else True))[0]; stable[j] = (neq.max() + 1) if len(neq) else 0
    flips = np.array([((am[j, 1:] != am[j, :-1]) & (msk[j, 1:] if msk is not None else True)).sum() for j in range(n)])
    commit = np.array([(np.where(msk[j])[0].max() + 1) if msk[j].any() else 0 for j in range(n)]) if msk is not None else np.full(n, np.nan)
    allst = A["all_steps"]
    for gnm, facts, sel in (("AT-high", fact_hi, hi), ("AT-low", fact_lo, lo)):
        dd = {}
        for nm, sc in (("argmax 안정 시점 (step)", stable.astype(float)), ("흔들림 횟수", flips.astype(float)), ("첫 칸 확정 step", commit.astype(float))):
            dd[nm] = dict(맞힘=round(float(np.nanmean(sc[sel & corr])), 2), 틀림=round(float(np.nanmean(sc[sel & ~corr])), 2),
                          붕괴=round(float(np.nanmean(sc[sel & (nt == "broken")])), 2), entity=round(float(np.nanmean(sc[sel & (nt == "entity")])), 2), 짝차이=boot_fact(paired(sc, facts)))
        if A.get("ncommit") is not None:
            nc = np.asarray(A["ncommit"]); dd["확정 칸 수 곡선"] = {f"s{allst[t]}": dict(맞힘=round(float(nc[sel & corr, t].mean()), 2), 틀림=round(float(nc[sel & ~corr, t].mean()), 2)) for t in range(nc.shape[1]) if allst[t] <= 16}
        dyn[gnm] = dd
S, L = A["steps"], A["layers"]
spd = {}
for li, l_ in enumerate(L):
    hs = [A["getH"](si, li, MI) for si in range(len(S))]
    fin_ = hs[-1]; fn = fin_ / np.linalg.norm(fin_, axis=1, keepdims=True)
    for gnm, facts, sel in (("AT-high", fact_hi, hi), ("AT-low", fact_lo, lo)):
        rowsel = sel[MI]; d = {}
        for si in range(len(S) - 1):
            v = np.linalg.norm(hs[si + 1] - hs[si], axis=1); full = np.full(n, np.nan); full[MI] = v
            d[f"속도 s{S[si]}->s{S[si + 1]}"] = dict(맞힘=round(float(np.nanmean(full[sel & corr])), 3), 틀림=round(float(np.nanmean(full[sel & ~corr])), 3), 짝차이=boot_fact(paired(full, facts)))
        for si in range(len(S)):
            c = (hs[si] / np.linalg.norm(hs[si], axis=1, keepdims=True) * fn).sum(1); full = np.full(n, np.nan); full[MI] = c
            d[f"최종상태 cos s{S[si]}"] = dict(맞힘=round(float(np.nanmean(full[sel & corr])), 4), 틀림=round(float(np.nanmean(full[sel & ~corr])), 4), 짝차이=boot_fact(paired(full, facts)))
        spd[f"층{l_} | {gnm}"] = d
    del hs
dyn["속도와 최종상태 cos"] = spd
O["2_동역학"] = dyn
print("(2) 끝", flush=True)

# ---------------------------------------------------------------- (3) (층, step) 별 같은 사실 안 정오 AUROC, MMD, sliced Wasserstein
def fact_center(Xm, js):
    Xc = Xm.copy(); pos = {j: t for t, j in enumerate(js)}
    for f, v in byf.items():
        t = [pos[j] for j in v if j in pos]
        if t: Xc[t] -= Xm[t].mean(0)
    return Xc
def within_auc(score, facts):
    a = [roc_auc_score(corr[v], score[v]) for v in facts.values() if 0 < corr[v].sum() < len(v)]
    return round(float(np.mean(a)), 4) if a else None
def mmd_perm(Z, y, groups, P=200, seed=0):
    from scipy.spatial.distance import cdist
    D2 = cdist(Z, Z, "sqeuclidean"); bw = np.median(D2[D2 > 0]); K = np.exp(-D2 / bw)
    r = np.random.default_rng(seed); Ys = [y]
    for _ in range(P):
        yy = y.copy()
        for gidx in groups: yy[gidx] = r.permutation(yy[gidx])
        Ys.append(yy)
    Y = np.stack(Ys, 1).astype(float); a = Y / Y.sum(0); b = (1 - Y) / (1 - Y).sum(0); KA = K @ a; KB = K @ b
    m = (a * KA).sum(0) + (b * KB).sum(0) - 2 * (a * KB).sum(0)
    return float(m[0]), float(m[1:].mean()), float((m[1:] >= m[0]).mean())
def swd_perm(Z, y, groups, P=200, ndir=200, seed=0):
    r = np.random.default_rng(seed); Dd = r.standard_normal((Z.shape[1], ndir)); Dd /= np.linalg.norm(Dd, axis=0); Pz = Z @ Dd; qs = np.linspace(0.02, 0.98, 49)
    def sw(yy):
        A_ = np.quantile(Pz[yy], qs, axis=0); B_ = np.quantile(Pz[~yy], qs, axis=0); return float(np.abs(A_ - B_).mean())
    obs = sw(y); nul = []
    for _ in range(P):
        yy = y.copy()
        for gidx in groups: yy[gidx] = r.permutation(yy[gidx])
        nul.append(sw(yy))
    nul = np.array(nul); return obs, float(nul.mean()), float((nul >= obs).mean())
amap = {}; eff = {}
mix_rows = np.array(sorted(j for v in mixed.values() for j in v))
for li, l_ in enumerate(L):
    for si, s in enumerate(S):
        t0 = time.time(); Xm = A["getH"](si, li, mix_rows); Xc = fact_center(Xm, list(mix_rows))
        Zp = PCA(256, random_state=0).fit_transform(Xc); Zp = Zp / Zp.std(0); sc = np.full(n, np.nan); gm = g7[mix_rows]; ym = corr[mix_rows]
        for g in range(NF):
            tr, te = gm != g, gm == g
            clf = LogisticRegression(C=0.01, max_iter=1000).fit(Zp[tr], ym[tr]); sc[mix_rows[te]] = clf.decision_function(Zp[te])
        amap[f"층{l_} s{s}"] = dict(전체=within_auc(sc, mixed), AT_high=within_auc(sc, fact_hi), AT_low=within_auc(sc, fact_lo))
        # 효과 크기: PCA 50, 표본 사실 300 개 (AT-high / AT-low 각각)
        Z50 = PCA(50, random_state=0).fit_transform(Xc); pos = {j: t for t, j in enumerate(mix_rows)}; e = {}
        for gnm, facts in (("AT_high", fact_hi), ("AT_low", fact_lo)):
            fs = list(facts); fs = [fs[q] for q in np.random.default_rng(si * 10 + li).permutation(len(fs))[:300]]
            js = [pos[j] for f in fs for j in facts[f]]; Zs = Z50[js]; ys = corr[mix_rows[js]]
            groups = []; c0 = 0
            for f in fs: groups.append(np.arange(c0, c0 + len(facts[f]))); c0 += len(facts[f])
            P_ = 50 if FAST else 200
            mo, mn, mp = mmd_perm(Zs, ys, groups, P=P_); so, sn, sp = swd_perm(Zs, ys, groups, P=P_)
            e[gnm] = dict(MMD=round(mo, 5), MMD귀무평균=round(mn, 5), MMD_p=round(mp, 3), SW=round(so, 4), SW귀무평균=round(sn, 4), SW_p=round(sp, 3), 사실=len(fs))
        eff[f"층{l_} s{s}"] = e
        print(f"  (3) 층{l_} s{s}: AUROC {amap[f'층{l_} s{s}']}, {time.time() - t0:.0f}s", flush=True)
O["3_같은사실안_정오_AUROC지도"] = amap; O["3_효과크기"] = eff
print("(3) 끝", flush=True)

# ---------------------------------------------------------------- (4) 3 차원 궤적 표본: (step, 결정 여유 (층 27), Brel 투영 (층 20, 겹밖))
X0, _ = load_X(); rel = np.array([r["relation"] for r in rows]); BREL = {}; BG = {}
for g in range(NF):
    tr = grp != g; bf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr & mid & (ty != "abstain"))[0]: bf[fid[i]]["c" if ty[i] == "correct" else "w"].append(i)
    per = collections.defaultdict(list)
    for v in bf.values():
        if v["c"] and v["w"]: per[rel[v["c"][0]]].append(X0[v["c"]].mean(0) - X0[v["w"]].mean(0))
    BREL[g] = {r_: unit(np.mean(d, 0)) for r_, d in per.items() if len(d) >= 15}; BG[g] = unit(np.mean([d for L_ in per.values() for d in L_], 0))
del X0
samp = {}; li20 = L.index(20) if 20 in L else None; li27 = L.index(27) if 27 in L else len(L) - 1
for gnm, sel in (("AT-high", hi), ("AT-low", lo)):
    for cn, cm in (("맞힘", corr), ("틀림", ~corr)):
        js = np.where(sel & cm)[0]; js = js[np.random.default_rng(7).permutation(len(js))[:100]]; items = []
        hs20 = [A["getH"](si, li20, js) for si in range(len(S))] if li20 is not None else None
        for t, j in enumerate(js):
            i = int(RR[j]); d = A_dir = BREL[grp[i]].get(rel[i], BG[grp[i]])
            it = {"row": i, "k": int(kk[j]), "type": nt[j], "fid": str(f7[j]), "steps": S,
                  "brel20": [round(float(hs20[si][t] @ A_dir), 3) for si in range(len(S))] if hs20 is not None else None}
            if A["LL"] is not None:
                it["margin_other27"] = [round(float(m_other[j, si, li27]), 3) for si in range(len(S))]
                it["margin_final27"] = [None if not np.isfinite(m_final[j, si, li27]) else round(float(m_final[j, si, li27]), 3) for si in range(len(S))]
                it["gold_rank27"] = [int(A["LL"]["gold_rank"][j, si, li27]) for si in range(len(S))]
            else:
                it["argmax_is_gold"] = [bool(A["s0_argmax"][j, s] == gold[j]) for s in range(A["s0_argmax"].shape[1])]
            items.append(it)
        samp[f"{gnm} | {cn}"] = items
json.dump(samp, open(f"{OUT}/e45_traj_samples{'_e7' if SRC == 'e7' else ''}.json", "w"), ensure_ascii=False)
print("(4) 끝", flush=True)

# ---------------------------------------------------------------- 판정
ver = {}
if A["LL"] is not None:
    def diffs(key):
        c = curves.get(key, {}); return {s: c[s]["짝차이"] for s in c}
    l27 = f"금-금아닌최고 | 층{L[li27]} | AT-high"; dh = diffs(l27); dl = diffs(f"금-금아닌최고 | 층{L[li27]} | AT-low")
    s0 = dh.get("s0", [np.nan] * 3); early = [s for s in dh if s in ("s1", "s2", "s3", "s4")]
    ver["H45.1"] = dict(s0_CI=s0, s1_4=[dh[s] for s in early], 판정=bool(s0[1] <= 0 <= s0[2] and any(dh[s][1] > 0 for s in early)))
    mh = max((v[0] for v in dh.values() if np.isfinite(v[0])), default=np.nan); ml = max((v[0] for v in dl.values() if np.isfinite(v[0])), default=np.nan)
    low_never = all(not (v[1] > 0) for v in dl.values())
    ver["H45.3"] = dict(AT_high_최대짝차이=round(float(mh), 3), AT_low_최대짝차이=round(float(ml), 3), AT_low_끝까지_안갈림=bool(low_never), 판정=bool(low_never or (ml <= 0.5 * mh)))
best = max(amap.items(), key=lambda x: (x[1]["AT_high"] or 0)); best_all = max(amap.items(), key=lambda x: (x[1]["전체"] or 0))
ver["H45.2"] = dict(전체최대=[best_all[0], best_all[1]["전체"]], AT_high최대=[best[0], best[1]["AT_high"]],
                    판정=bool(int(best_all[0].split(" s")[-1]) <= 4 and (best_all[1]["전체"] or 0) >= 0.70))
O["판정"] = ver
json.dump(O, open(f"{OUT}/e45_{'e7_dryrun' if SRC == 'e7' else 'summary'}.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(ver, ensure_ascii=False), flush=True); print("DONE", flush=True)
