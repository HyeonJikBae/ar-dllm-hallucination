"""Is "knows but wrong" a separate state, or the middle of a continuous know axis?

Know axis: difference-in-means direction K5 - U of binary SAE activity (relation-stratified, learned on train
facts). Every held-out row gets z = (score - U mean) / (K5 mean - U mean), normalised INSIDE its relation with
the held-out U and K5 rows of that relation, so 0 = doesn't know, 1 = knows and relation offsets cancel.

  1. position by k (1/5 .. 4/5) of the WRONG rows (all types, type 1, type 2) and of the CORRECT rows of the
     same partially-known facts (these rows were not in any group before)
  2. variance decomposition over 1-4/5 facts: share of z variance between facts vs within facts, and the
     within-fact gap correct - wrong (in K5-U units) with a within-fact AUROC.
     If the gap is ~0 while between-fact variance dominates, q_last reflects how familiar the FACT is,
     not whether this paraphrase will be answered correctly.
  3. sensitivity: k recomputed with audit-flagged "correct" rows counted as wrong (audit_flag in 7. final)
  4. wrong-answer type mix of 0/5 vs each k, with the 0/5 rows re-weighted to the same relation mix

Rows: analysis_use rows (rule_unsure == False, no gold_issue). Resampling unit: fact.

Usage:
    FEAT=sae_analysis/features_refined python3 sae_analysis/know_axis_by_k.py --position q_last
"""

import argparse
import collections
import json
import os

import numpy as np

from twelve_cells import FEAT, auroc, strat_diff

FINAL = "analysis/graded/7. final/qwen2.5-7b_popqa_final.jsonl"
TYPES = (1, 2, 3, 6)


def load(position, k_mode):
    meta = [json.loads(l) for l in open(f"{FEAT}/qwen_popqa_meta.jsonl", encoding="utf-8")]
    audit = {}
    for l in open(FINAL, encoding="utf-8"):
        r = json.loads(l)
        audit[(r["sample_id"], r["version"])] = bool(r.get("audit_flag"))
    ok = np.array([not m["gold_issue"] and m["verdict"] != "uncertain" for m in meta])
    corr = np.array([m["verdict"] == "correct" or m["verdict_suggested"] == "correct" for m in meta])
    flagged = np.array([audit[(m["sample_id"], m["version"])] for m in meta])
    use = ok & ~np.array([bool(m.get("rule_unsure")) for m in meta])
    reason = np.array([m["reason"] if m["reason"] is not None else -1 for m in meta])
    sid = np.array([m["sample_id"] for m in meta])
    rel = np.array([m["relation"] for m in meta])
    counted = corr & ~flagged if k_mode == "audit" else corr
    kf = collections.Counter()
    for s, c, o in zip(sid, counted, ok):
        if o:
            kf[s] += int(c)
    k = np.array([kf.get(s, -1) for s in sid])
    keep = np.where(use & (k >= 0))[0]
    acts = np.load(f"{FEAT}/qwen_popqa_{position}.npy", mmap_mode="r")
    A = np.asarray(acts[keep]) > 0
    return A, corr[keep], reason[keep], sid[keep], rel[keep], k[keep]


def fact_matrix(A, sid, rel):
    facts, inv = np.unique(sid, return_inverse=True)
    X = np.zeros((len(facts), A.shape[1]), dtype=np.float32)
    np.add.at(X, inv, A.astype(np.float32))
    X /= np.bincount(inv)[:, None]
    fr = np.empty(len(facts), dtype=object)
    fr[inv] = rel
    return facts, X, fr


def analyse(position, k_mode, args):
    A, corr, reason, sid, rel, k = load(position, k_mode)
    elig = A.mean(0) >= args.min_freq
    A = A[:, elig]
    Af = A.astype(np.float32)
    isK5 = corr & (k == 5)
    isU = (k == 0) & ~corr
    part = (k >= 1) & (k <= 4)
    facts_all = np.array(sorted(set(sid)))
    groups = {}
    for kk in (1, 2, 3, 4):
        groups[f"{kk}/5 오답(전체)"] = part & (k == kk) & ~corr
        groups[f"{kk}/5 오답 유형1"] = part & (k == kk) & ~corr & (reason == 1)
        groups[f"{kk}/5 오답 유형2"] = part & (k == kk) & ~corr & (reason == 2)
        groups[f"{kk}/5 정답"] = part & (k == kk) & corr
    pos = collections.defaultdict(list)
    vd = collections.defaultdict(list)
    for seed in range(args.splits):
        rng = np.random.default_rng(seed)
        tr = np.isin(sid, rng.permutation(facts_all)[: int(0.6 * len(facts_all))])
        te = ~tr
        fK, XK, rK = fact_matrix(A[tr & isK5], sid[tr & isK5], rel[tr & isK5])
        fU, XU, rU = fact_matrix(A[tr & isU], sid[tr & isU], rel[tr & isU])
        d, _ = strat_diff(XK, rK, XU, rU, args.min_rel_facts)
        s = Af @ d.astype(np.float32)
        # normalise inside relation with held-out U / K5 means
        z = np.full(len(s), np.nan)
        for r in set(rel):
            mr = te & (rel == r)
            nk, nu = len(set(sid[mr & isK5])), len(set(sid[mr & isU]))
            if nk < args.min_rel_facts or nu < args.min_rel_facts:
                continue
            mu, mk = s[mr & isU].mean(), s[mr & isK5].mean()
            if mk - mu <= 0:
                continue
            z[mr] = (s[mr] - mu) / (mk - mu)
        valid = te & ~np.isnan(z)
        pos["모두 오답 U (0/5)"].append(np.mean(z[valid & isU]))
        pos["모두 정답 K5 (5/5)"].append(np.mean(z[valid & isK5]))
        for name, m in groups.items():
            if (valid & m).sum():
                pos[name].append(np.mean(z[valid & m]))
        # variance decomposition over 1-4/5 facts (correct + wrong rows)
        m = valid & part
        fz = collections.defaultdict(list)
        for i in np.where(m)[0]:
            fz[sid[i]].append((z[i], corr[i]))
        means = np.array([np.mean([v for v, _ in rows]) for rows in fz.values()])
        within = np.array([np.var([v for v, _ in rows]) for rows in fz.values() if len(rows) > 1])
        between = np.var(means)
        gaps, aucs = [], []
        for rows in fz.values():
            zc = np.array([v for v, c in rows if c])
            zw = np.array([v for v, c in rows if not c])
            if len(zc) and len(zw):
                gaps.append(zc.mean() - zw.mean())
                aucs.append(auroc(zc, zw))
        vd["fact 간 분산 비율"].append(between / (between + within.mean()))
        vd["같은 fact 안: 정답 - 오답 (K5-U 단위)"].append(np.mean(gaps))
        vd["같은 fact 안 AUROC (정답 > 오답)"].append(np.mean(aucs))
        vd["짝지은 fact 수"].append(len(gaps))
        # same contrast between k groups for scale: 4/5 wrong vs 1/5 wrong
    return pos, vd, (A, corr, reason, sid, rel, k)


def type_mix(corr, reason, rel, k):
    """Type shares of wrong rows: U re-weighted to each k's relation mix."""
    out = {}
    wrong = ~corr
    lab = np.where(np.isin(reason, TYPES), reason, 0)  # 0 = 기권/4/5
    for kk in (1, 2, 3, 4):
        wk = wrong & (k == kk)
        u = wrong & (k == 0)
        num_u, num_w, tot = np.zeros(5), np.zeros(5), 0.0
        for r in set(rel[wk]):
            wr = wk & (rel == r)
            ur = u & (rel == r)
            if ur.sum() < 10:
                continue
            n = wr.sum()
            for j, t in enumerate((1, 2, 3, 6, 0)):
                num_w[j] += n * (lab[wr] == t).mean()
                num_u[j] += n * (lab[ur] == t).mean()
            tot += n
        out[kk] = (num_u / tot, num_w / tot, int(tot), int(wk.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--min-freq", type=float, default=0.005)
    ap.add_argument("--min-rel-facts", type=int, default=3)
    ap.add_argument("--out", default="sae_analysis/results_twelve")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {}
    for k_mode in ("final", "audit"):
        pos, vd, (A, corr, reason, sid, rel, k) = analyse(args.position, k_mode, args)
        label = "k = 7. final 그대로" if k_mode == "final" else "k = 점검 플래그 정답을 오답으로 다시 셈"
        nf = {kk: len(set(sid[k == kk])) for kk in range(6)}
        print(f"\n##### {args.position} | {label} | fact 수 0/5..5/5: {nf}")
        print("\n[1] 앎 축 위치 (0 = 모두 오답, 1 = 모두 정답; 관계 안에서 정규화, test fact, 평균±sd)")
        print(f"{'':<14}{'오답(전체)':>14}{'오답 유형1':>14}{'오답 유형2':>14}{'정답':>14}")
        for kk in (1, 2, 3, 4):
            cells = [pos.get(f"{kk}/5 {n}", []) for n in ("오답(전체)", "오답 유형1", "오답 유형2", "정답")]
            print(f"  {kk}/5 fact    " + "".join(f"{np.mean(c):>+9.2f}±{np.std(c):.2f}" if c else f"{'-':>14}" for c in cells))
        print(f"  기준: 0/5 {np.mean(pos['모두 오답 U (0/5)']):+.2f}, 5/5 {np.mean(pos['모두 정답 K5 (5/5)']):+.2f}")
        print("\n[2] 1~4/5 fact의 분산 분해 (맞힌 행 + 틀린 행)")
        for n, v in vd.items():
            print(f"  {n}: {np.mean(v):.3f}±{np.std(v):.3f}")
        report[k_mode] = {"facts_per_k": nf, "position": {n: [float(np.mean(v)), float(np.std(v))] for n, v in pos.items()},
                          "variance": {n: [float(np.mean(v)), float(np.std(v))] for n, v in vd.items()}}
        if k_mode == "final":
            print("\n[4] 오답 유형 비율: 0/5 오답(각 k와 같은 관계 구성으로 재가중) vs k별 오답")
            print(f"{'':<10}{'유형1':>13}{'유형2':>13}{'유형3':>13}{'유형6':>13}{'기권 등':>13}   행")
            mix = type_mix(corr, reason, rel, k)
            for kk, (u, w, n, nw) in mix.items():
                print(f"  {kk}/5     " + "".join(f"{uu:>6.1%}→{ww:<6.1%}" for uu, ww in zip(u, w)) + f"   {nw}")
            report["type_mix"] = {kk: {"U_reweighted": u.tolist(), "W": w.tolist(), "rows": nw} for kk, (u, w, n, nw) in mix.items()}
    out = os.path.join(args.out, f"know_axis_by_k_{args.position}.json")
    json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
