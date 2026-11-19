"""E56 평가 (CPU).  repo 루트에서:  /NAS/minyeol/limit/.venv-gfmrag/bin/python followup_mk/e56_eval.py [--pilot]
채점: e56_common.grade (주 규칙), grade_contain (포함 규칙, 민감도).  Dream 정오는 이번 실행 base 의 채점 (k/5 도 base 로).
경로 (AGENT_BRIEF/E49): Dream 1~4/5 사실의 base 오답 (entity/broken; abstain 제외) 행 중, 같은 문구 Qwen2.5-7B 가 맞혔으면 'DLM 고유', 아니면 '공유'.
H56.1 (= H55.1 을 Dream/Qwen 에): DLM 고유의 step 0 칸 0 금 첫 토큰 1 위 비율 > 공유, 사실 군집 부트스트랩 차이 하한 > 0.
H56.2 (= H55.2): 칸0먼저의 오답 복구율 DLM 고유 > 공유, 하한 > 0.
CI: 사실 (qid) 군집 부트스트랩 B = 4000, 2.5/97.5 백분위.  같은 사실 안 비교: 두 집단이 모두 있는 사실만, 사실별 차이의 평균, 사실 재표집.
"""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from e56_common import load_items, grade, grade_contain, VERS, D56

PILOT = "--pilot" in sys.argv; OUT = "followup_mk"; B = 4000
items = load_items(); IT = {it["qid"]: it for it in items}
dr = collections.defaultdict(dict)
for fn in ([f"{OUT}/results_e56_dream_pilot.jsonl"] if PILOT else sorted(glob.glob(f"{OUT}/results_e56_dream_s*.jsonl"))):
    for l in open(fn, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        dr[(d["qid"], d["ver"])][d["cond"]] = d
qr = {}
for fn in sorted(glob.glob(f"{OUT}/results_e56_qwen{'_pilot' if PILOT else ''}_s*.jsonl")):
    for l in open(fn, encoding="utf-8"):
        try: d = json.loads(l); qr[(d["qid"], d["ver"])] = d
        except Exception: pass
R = [(it["qid"], v) for it in items for v in VERS if (it["qid"], v) in qr and len(dr.get((it["qid"], v), {})) == 2]
if not PILOT:   # 다섯 문구가 모두 있는 사실만
    cnt = collections.Counter(q for q, _ in R); R = [r for r in R if cnt[r[0]] == 5]
print(f"행 {len(R)} (Dream 기록 {len(dr)}, Qwen 기록 {len(qr)})")
n = len(R); fidr = np.array([q for q, _ in R]); verr = np.array([v for _, v in R])
qt = lambda r: IT[r[0]][r[1]]
G = {c: np.array([grade(dr[r][c]["answer"], IT[r[0]]["value"], IT[r[0]]["aliases"], qt(r)) for r in R]) for c in ("base", "칸0먼저")}
GQ = np.array([grade(qr[r]["answer"], IT[r[0]]["value"], IT[r[0]]["aliases"], qt(r)) for r in R])
GC = {c: np.array([grade_contain(dr[r][c]["answer"], IT[r[0]]["value"], IT[r[0]]["aliases"], qt(r)) for r in R]) for c in ("base", "칸0먼저")}
GQC = np.array([grade_contain(qr[r]["answer"], IT[r[0]]["value"], IT[r[0]]["aliases"], qt(r)) for r in R])
rank0 = np.array([dr[r]["base"]["rank0"] for r in R]); rank0a = np.array([dr[r]["base"]["rank0_any"] for r in R])
tok0 = np.array([dr[r]["base"]["tok0"] for r in R]); gtok = np.array([dr[r]["base"]["gold_tok"] for r in R])
gset = [set(dr[r]["base"]["gold_set"]) for r in R]; arg0 = np.array([dr[r]["base"]["arg0"] for r in R])
s0step = np.array([dr[r]["base"]["slot0_step"] for r in R]); qrank = np.array([qr[r]["s0_rank"] for r in R])
qlp = np.array([qr[r].get("s0_lp", np.nan) for r in R]); alen_b = np.minimum(np.array([qr[r].get("gold_len", 1) for r in R]), 4)


def qbin(x, q=5):
    """E57 와 같은 분위 (동점은 순위로)."""
    x = np.asarray(x, float); out = np.full(len(x), -1); ok = ~np.isnan(x)
    rk = np.argsort(np.argsort(x[ok], kind="stable"), kind="stable"); out[ok] = (rk * q // ok.sum()).astype(int)
    return out


def mh_rd(y, d, s):
    """E57 mh_rd 와 같음: 층별 위험 차이 (d − ~d) 의 Mantel-Haenszel 가중 평균 (두 집단 모두 있는 층)."""
    y = np.asarray(y, float); d = np.asarray(d, bool); s = np.unique(np.asarray(s), return_inverse=True)[1]
    m_ = int(s.max()) + 1; n1 = np.bincount(s[d], minlength=m_); n0 = np.bincount(s[~d], minlength=m_)
    y1 = np.bincount(s[d], weights=y[d], minlength=m_); y0 = np.bincount(s[~d], weights=y[~d], minlength=m_); v = (n1 > 0) & (n0 > 0)
    if not v.any(): return np.nan, 0.0
    w = n1[v] * n0[v] / (n1[v] + n0[v]); return float((w * (y1[v] / n1[v] - y0[v] / n0[v])).sum() / w.sum()), float((n1[v] + n0[v]).sum() / len(y))


def strat(y, d, f, s, Bs=2000):
    """MH 차이와 사실 군집 부트스트랩 CI."""
    u, inv = np.unique(f, return_inverse=True); mem = [np.where(inv == j)[0] for j in range(len(u))]; rng = np.random.default_rng(3)
    est, cov = mh_rd(y, d, s); bs = []
    for _ in range(Bs):
        ix = np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))]); v, _ = mh_rd(y[ix], d[ix], s[ix])
        if np.isfinite(v): bs.append(v)
    return dict(mh_diff=round(est, 4), ci=ci(np.array(bs)), coverage=round(cov, 4))


def kvec(ok):
    c = collections.Counter(); [c.__setitem__(f, c[f] + int(o)) for f, o in zip(fidr, ok)]
    return np.array([c[f] for f in fidr])


def wmat(f_sel, seed=0):
    uf, inv = np.unique(f_sel, return_inverse=True); rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(uf), len(uf)), minlength=len(uf)) for _ in range(B)]).astype(np.float32)
    return uf, inv, W


agg = lambda inv, nf, v: np.bincount(inv, weights=v.astype(float), minlength=nf)
ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def rate_ci(sel, y, W, inv, nf):
    num = agg(inv, nf, (y & sel) if y.dtype == bool else y * sel); den = agg(inv, nf, sel)
    bt = (W @ num) / np.maximum(W @ den, 1e-9)
    return dict(n=int(sel.sum()), rate=round(float(num.sum() / max(den.sum(), 1e-9)), 4), ci=ci(bt)), bt


def diff_ci(selA, selB, y, W, inv, nf):
    a, ba = rate_ci(selA, y, W, inv, nf); b, bb = rate_ci(selB, y, W, inv, nf); d = ba - bb
    return dict(A=a, B=b, diff=round(a["rate"] - b["rate"], 4), diff_ci=ci(d), p_le0=round(float((d <= 0).mean()), 4))


def within_fact(selA, selB, y, f_sel):
    byf = collections.defaultdict(lambda: [[], []])
    for j in np.where(selA | selB)[0]: byf[f_sel[j]][0 if selA[j] else 1].append(float(y[j]))
    ds = np.array([np.mean(a) - np.mean(b) for a, b in byf.values() if a and b])
    if len(ds) == 0: return None
    bt = ds[np.random.default_rng(1).integers(0, len(ds), (B, len(ds)))].mean(1)
    return dict(n_fact=len(ds), n_rows_A=int(sum(len(a) for a, b in byf.values() if a and b)), n_rows_B=int(sum(len(b) for a, b in byf.values() if a and b)), diff=round(float(ds.mean()), 4), ci=ci(bt))


def analysis(Gb, Gp, Qok, tag):
    """Gb/Gp: Dream base/칸0먼저 유형 배열 (correct/entity/broken/abstain), Qok: Qwen 정답 bool."""
    S = {}
    okb = Gb == "correct"; kd = kvec(okb); kq = kvec(Qok); first = np.unique(fidr, return_index=True)[1]
    S["acc"] = dict(dream_base=round(float(okb.mean()), 4), dream_kan0=round(float((Gp == "correct").mean()), 4), qwen=round(float(Qok.mean()), 4),
                    dream_base_by_ver={v: round(float(okb[verr == v].mean()), 4) for v in VERS}, qwen_by_ver={v: round(float(Qok[verr == v].mean()), 4) for v in VERS})
    S["k5"] = dict(dream={int(x): int(c) for x, c in sorted(collections.Counter(kd[first]).items())}, qwen={int(x): int(c) for x, c in sorted(collections.Counter(kq[first]).items())})
    ct = np.zeros((6, 6), int)
    for j in first: ct[kd[j], kq[j]] += 1
    S["k5_cross_dream_rows_qwen_cols"] = ct.tolist()
    S["types"] = {nm: dict(collections.Counter(x.tolist())) for nm, x in [("dream_base", Gb), ("dream_kan0", Gp)]}
    mid = (kd >= 1) & (kd <= 4); S["n_mid_facts"] = int(len(np.unique(fidr[mid])))
    # 같은 사실 안 문구 정오 일치 (E49 (2) 대응, 서술): Dream 1~4/5 이고 Qwen 도 섞인 사실
    mixed = mid & (kq >= 1) & (kq <= 4); per = []
    for f in np.unique(fidr[mixed]):
        j = fidr == f; per.append(okb[j & Qok].mean() - okb[j & ~Qok].mean())
    S["concordance_mixed_facts"] = dict(n_fact=len(per), mean=round(float(np.mean(per)), 4) if per else None)
    wrong = mid & np.isin(Gb, ["entity", "broken"]); dlm = wrong & Qok; shr = wrong & ~Qok
    S["routes"] = dict(n_wrong=int(wrong.sum()), n_dlm=int(dlm.sum()), n_shared=int(shr.sum()), n_abstain_excluded=int((mid & (Gb == "abstain")).sum()),
                       types={"dlm": dict(collections.Counter(Gb[dlm].tolist())), "shared": dict(collections.Counter(Gb[shr].tolist()))},
                       ver_share_original={"dlm": round(float((verr[dlm] == "question").mean()), 4) if dlm.any() else None, "shared": round(float((verr[shr] == "question").mean()), 4) if shr.any() else None},
                       facts={"dlm": int(len(np.unique(fidr[dlm]))), "shared": int(len(np.unique(fidr[shr])))})
    sub = lambda v: v[mid]; um, inv, W = wmat(fidr[mid]); nf = len(um)
    top1 = rank0 == 0; top1a = rank0a == 0; lost = top1 & (tok0 != gtok); top5 = rank0 < 5
    H1 = {}
    for nm, y in [("gold_top1", top1), ("any_alias_top1", top1a), ("gold_top5", top5), ("lost_top1_then_slot0_not_gold", lost)]:
        d = diff_ci(sub(dlm), sub(shr), sub(y), W, inv, nf); d["within_fact"] = within_fact(sub(dlm), sub(shr), sub(y), fidr[mid]); H1[nm] = d
    l2 = np.log2(rank0 + 1.0); d = diff_ci(sub(dlm), sub(shr), sub(l2), W, inv, nf); d["within_fact"] = within_fact(sub(dlm), sub(shr), sub(l2), fidr[mid]); H1["log2_rank_plus1"] = d
    H1["median_rank"] = {"dlm": float(np.median(rank0[dlm])) if dlm.any() else None, "shared": float(np.median(rank0[shr])) if shr.any() else None,
                         "mid_correct": float(np.median(rank0[mid & okb])) if (mid & okb).any() else None}
    H1["top1_reference"] = {"mid_correct": round(float(top1[mid & okb].mean()), 4) if (mid & okb).any() else None, "zero": round(float(top1[kd == 0].mean()), 4), "five": round(float(top1[kd == 5].mean()), 4)}
    H1["qwen_first_step_top1"] = {"dlm": round(float((qrank[dlm] == 0).mean()), 4) if dlm.any() else None, "shared": round(float((qrank[shr] == 0).mean()), 4) if shr.any() else None}
    H1["by_type"] = {t: diff_ci(sub(dlm & (Gb == t)), sub(shr & (Gb == t)), sub(top1), W, inv, nf) for t in ["entity", "broken"]}
    H1["original_question_only"] = diff_ci(sub(dlm & (verr == "question")), sub(shr & (verr == "question")), sub(top1), W, inv, nf)
    S["H56.1_data"] = H1
    S["H56.1"] = dict(dlm=H1["gold_top1"]["A"]["rate"], shared=H1["gold_top1"]["B"]["rate"], diff=H1["gold_top1"]["diff"], ci=H1["gold_top1"]["diff_ci"],
                      verdict=bool(H1["gold_top1"]["diff_ci"][0] > 0), within_fact=H1["gold_top1"]["within_fact"],
                      any_alias=dict(diff=H1["any_alias_top1"]["diff"], ci=H1["any_alias_top1"]["diff_ci"]))
    yr = Gp == "correct"; H2 = diff_ci(sub(dlm), sub(shr), sub(yr), W, inv, nf); H2["within_fact"] = within_fact(sub(dlm), sub(shr), sub(yr), fidr[mid])
    H2["by_type"] = {t: diff_ci(sub(dlm & (Gb == t)), sub(shr & (Gb == t)), sub(yr), W, inv, nf) for t in ["entity", "broken"]}
    H2["given_prefill_gold"] = {g: dict(n=int((sel & (arg0 == gtok)).sum()), rate=round(float(yr[sel & (arg0 == gtok)].mean()), 4) if (sel & (arg0 == gtok)).any() else None,
                                        n_not=int((sel & (arg0 != gtok)).sum()), rate_not=round(float(yr[sel & (arg0 != gtok)].mean()), 4) if (sel & (arg0 != gtok)).any() else None) for g, sel in [("dlm", dlm), ("shared", shr)]}
    S["H56.2_data"] = H2
    S["H56.2"] = dict(dlm=H2["A"]["rate"], shared=H2["B"]["rate"], diff=H2["diff"], ci=H2["diff_ci"], verdict=bool(H2["diff_ci"][0] > 0), within_fact=H2["within_fact"])
    # E57 식 층화: 경로 행 (DLM 고유 + 공유) 안에서 Qwen 금 첫 토큰 log 확률 5 분위 x 금 답 토큰 수 (1,2,3,4+)
    rr = wrong & ~np.isnan(qlp)
    if rr.sum() > 10 and dlm[rr].any() and shr[rr].any():
        sq = np.array([f"{a}_{b}" for a, b in zip(qbin(qlp[rr]), alen_b[rr])]); dd = dlm[rr]; ff = fidr[rr]
        S["stratified_qwen_lp_q5_x_len"] = {"H56.1_top1": strat(top1[rr].astype(float), dd, ff, sq), "H56.1_any_alias_top1": strat(top1a[rr].astype(float), dd, ff, sq),
                                            "H56.2_kan0_recovery": strat(yr[rr].astype(float), dd, ff, sq),
                                            "qwen_lp_median": {"dlm": round(float(np.median(qlp[dlm])), 3), "shared": round(float(np.median(qlp[shr])), 3)}}
    # 칸0먼저 전체 효과 (서술)
    eff = {}
    for nm, sel in [("mid", mid), ("zero", kd == 0), ("five", kd == 5)]:
        if not sel.any(): continue
        u_, i_, W_ = wmat(fidr[sel]); dd = (Gp[sel] == "correct").astype(float) - okb[sel]
        r_, _ = rate_ci(np.ones(sel.sum(), bool), dd, W_, i_, len(u_))
        eff[nm] = dict(n=int(sel.sum()), base=round(float(okb[sel].mean()), 4), kan0=round(float((Gp[sel] == "correct").mean()), 4), diff=r_["rate"], ci=r_["ci"],
                       recover_of_wrong=round(float((Gp[sel] == "correct")[~okb[sel]].mean()), 4) if (~okb[sel]).any() else None,
                       break_of_correct=round(float((Gp[sel] != "correct")[okb[sel]].mean()), 4) if okb[sel].any() else None)
    S["kan0_effect"] = eff
    bc = mid & okb; brk = Gp != "correct"
    d = diff_ci(sub(bc & ~Qok), sub(bc & Qok), sub(brk), W, inv, nf); d["within_fact"] = within_fact(sub(bc & ~Qok), sub(bc & Qok), sub(brk), fidr[mid])
    S["break_mid_correct_by_qwen"] = d
    S["slot0_step_base_mean"] = {"dlm": round(float(s0step[dlm].mean()), 3) if dlm.any() else None, "shared": round(float(s0step[shr].mean()), 3) if shr.any() else None,
                                 "mid_correct": round(float(s0step[mid & okb].mean()), 3) if (mid & okb).any() else None}
    return S


S = {"tag": "pilot" if PILOT else "full", "n_rows": n, "n_facts": int(len(np.unique(fidr)))}
S["sanity"] = dict(prefill_eq_arg0=float(np.mean([dr[r]["칸0먼저"]["prefill_tok"] == dr[r]["base"]["arg0"] for r in R])),
                   kan0_tok0_eq_prefill=float(np.mean([dr[r]["칸0먼저"]["tok0"] == dr[r]["칸0먼저"]["prefill_tok"] for r in R])),
                   kan0_slot0_step0=float(np.mean([dr[r]["칸0먼저"]["slot0_step"] == 0 for r in R])),
                   base_rank_traj0_eq_rank0=float(np.mean([dr[r]["base"]["rank_traj"][0] == dr[r]["base"]["rank0"] for r in R])),
                   gold_in_gold_set=float(np.mean([g in s for g, s in zip(gtok, gset)])),
                   prefill_special=float(np.mean([dr[r]["칸0먼저"]["prefill_special"] for r in R])))
S["main"] = analysis(G["base"], G["칸0먼저"], GQ == "correct", "main")
Gbc = np.where(GC["base"], "correct", np.where(G["base"] == "correct", "entity", G["base"])); Gpc = np.where(GC["칸0먼저"], "correct", np.where(G["칸0먼저"] == "correct", "entity", G["칸0먼저"]))
S["contain_rule"] = analysis(Gbc, Gpc, GQC, "contain")
S["grading_agreement"] = dict(dream_base_main_vs_contain=round(float(((G["base"] == "correct") == GC["base"]).mean()), 4), qwen=round(float(((GQ == "correct") == GQC).mean()), 4),
                              dream_contain_only=int((GC["base"] & (G["base"] != "correct")).sum()), qwen_contain_only=int((GQC & (GQ != "correct")).sum()))
fo = f"{OUT}/e56_summary{'_pilot' if PILOT else ''}.json"; json.dump(S, open(fo, "w"), ensure_ascii=False, indent=1)
M = S["main"]; pct = lambda x: f"{100 * x:.1f}"
print("sanity", S["sanity"]); print("acc", M["acc"]); print("k5", M["k5"]); print("routes", M["routes"])
print("H56.1", M["H56.1"]); print("H56.2", M["H56.2"]); print("contain H56.1", S["contain_rule"]["H56.1"]); print("contain H56.2", S["contain_rule"]["H56.2"])
print("층화", M.get("stratified_qwen_lp_q5_x_len")); print("kan0 effect", M["kan0_effect"]); print("저장:", fo)
