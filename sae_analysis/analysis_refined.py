"""Knows / knows-but-wrong / doesn't-know and within-fact paired analysis on the refined PopQA labels.

Same design as three_group.py and paired_within_fact.py, on features_refined (11,045 facts):
  K    effective-correct rows (verdict correct or suggested correct) in facts with >=1 correct
  W_t  wrong rows of type t (1/2/3/6) in facts with 1-4 correct rows ("knows but wrong")
  U    all rows of 0-of-5 facts ("doesn't know")
Paired: per fact with both correct and type-t rows, d = rate(wrong) - rate(correct); averaged over facts,
sign-flip permutation test, BH-FDR. "common" = wrong rows of any of the four types vs correct.

Usage:
    python3 sae_analysis/analysis_refined.py --position q_last
"""

import argparse
import json

import numpy as np
from scipy.stats import rankdata

FEAT = __import__("os").environ.get("FEAT", "sae_analysis/features_refined")
TYPES = {1: "related", 2: "unrelated", 3: "copy", 6: "collapse"}


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def bh(p):
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1].clip(max=1)
    return q


def sign_flip_p(d, n_perm, rng):
    obs = np.abs(d.mean(0))
    cnt = np.zeros(d.shape[1])
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=(d.shape[0], 1)).astype(np.float32)
        cnt += np.abs((d * s).mean(0)) >= obs
    return (cnt + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--min-freq", type=float, default=0.05)
    ap.add_argument("--confident", action="store_true", help="drop ambiguous correct (hypernym etc.) and unsure/assumed wrong rows")
    args = ap.parse_args()

    meta = [json.loads(l) for l in open(f"{FEAT}/qwen_popqa_meta.jsonl", encoding="utf-8")]
    B = np.load(f"{FEAT}/qwen_popqa_{args.position}.npy") > 0
    ok = [not m["gold_issue"] and m["verdict"] != "uncertain" for m in meta]
    corr = np.array([m["verdict"] == "correct" or m["verdict_suggested"] == "correct" for m in meta])
    reason = np.array([m["reason"] if m["reason"] is not None else -1 for m in meta])
    sid = np.array([m["sample_id"] for m in meta])
    ok = np.array(ok)
    kfact = {}
    for s, c, o in zip(sid, corr, ok):
        if o:
            kfact[s] = kfact.get(s, 0) + int(c)
    k = np.array([kfact.get(s, -1) for s in sid])
    grp = np.full(len(meta), "", dtype=object)
    for i in range(len(meta)):
        if not ok[i]:
            continue
        if k[i] == 0:
            grp[i] = "U"
        elif corr[i]:
            grp[i] = "K"
        elif reason[i] in TYPES and 1 <= k[i] <= 4:
            grp[i] = f"W{reason[i]}"
    if args.confident:
        rule = np.array([m.get("rule") or "" for m in meta])
        amb = corr & np.array([r.startswith("C-") and r != "C-alias" for r in rule])
        uns = np.array([bool(m.get("rule_unsure")) or m.get("rule") == "1-role-assumed" for m in meta])
        use = np.array(ok) & ~amb & ~uns
        grp[~use] = ""
        corr = corr & use
        ok = use
        print(f"confident subset: dropped ambiguous-correct {int(amb.sum())}, unsure/assumed-wrong {int(uns.sum())}")
    names = ["K", "W1", "W2", "W3", "W6", "U"]
    print(f"[{args.position}] facts {len(set(sid[ok]))} | rows per group:", {n: int((grp == n).sum()) for n in names})
    freq_all = B[ok].mean(0)
    elig = freq_all >= args.min_freq
    print(f"eligible latents (active in >={args.min_freq:.0%} of rows): {elig.sum()}")

    facts = np.array(sorted(set(sid[ok])))
    # ---- 3-state know axis ----
    res = {n: [] for n in names}
    au = {k_: [] for k_ in ("K_vs_U", "K_vs_W1", "K_vs_W2", "K_vs_W3", "K_vs_W6", "W1_vs_U", "W2_vs_U", "W3_vs_U", "W6_vs_U")}
    Bf = B.astype(np.float32)
    for seed in range(args.splits):
        rng = np.random.default_rng(seed)
        tr = np.isin(sid, rng.permutation(facts)[: int(0.6 * len(facts))])
        te = ~tr
        d = Bf[tr & (grp == "K")].mean(0) - Bf[tr & (grp == "U")].mean(0)
        sc = Bf @ d
        m = {n: sc[te & (grp == n)].mean() if (te & (grp == n)).any() else np.nan for n in names}
        den = m["K"] - m["U"]
        for n in names:
            res[n].append((m[n] - m["U"]) / den)
        g = lambda n: sc[te & (grp == n)]
        for t in (1, 2, 3, 6):
            au[f"K_vs_W{t}"].append(auroc(g("K"), g(f"W{t}")))
            au[f"W{t}_vs_U"].append(auroc(g(f"W{t}"), g("U")))
        au["K_vs_U"].append(auroc(g("K"), g("U")))
    print("know-axis position (0 = doesn't know, 1 = knows), mean +- sd over splits:")
    print("  " + "  ".join(f"{n} {np.nanmean(res[n]):+.2f}+-{np.nanstd(res[n]):.2f}" for n in names))
    print("AUROC (test rows):", {k_: round(float(np.nanmean(v)), 3) for k_, v in au.items()})

    # ---- within-fact paired ----
    by_fact = {}
    for i in np.where(ok)[0]:
        by_fact.setdefault(sid[i], []).append(i)
    per = {t: [] for t in list(TYPES) + ["common"]}
    for rows in by_fact.values():
        c = [i for i in rows if corr[i]]
        if not c:
            continue
        cm = Bf[c].mean(0)
        wrong_all = [i for i in rows if grp[i].startswith("W")]
        for t in TYPES:
            w = [i for i in rows if grp[i] == f"W{t}"]
            if w:
                per[t].append(Bf[w].mean(0) - cm)
        if wrong_all:
            per["common"].append(Bf[wrong_all].mean(0) - cm)
    rng = np.random.default_rng(0)
    summ = {}
    for t, lst in per.items():
        d = np.array(lst, dtype=np.float32)[:, elig]
        if len(d) < 15:
            print(f"\npaired {t}: facts {len(d)} (too few)")
            continue
        mean = d.mean(0)
        p = sign_flip_p(d, args.n_perm, rng)
        q = bh(p)
        summ[t] = (np.where(elig)[0], mean, q)
        name = TYPES.get(t, "common (any wrong)")
        print(f"\npaired {t} {name}: facts {len(d)}, FDR<0.1 latents {(q < .1).sum()}, FDR<0.05 {(q < .05).sum()}")
        for dirn, sg in (("wrong_up", 1), ("correct_up", -1)):
            o = np.argsort(-sg * mean)[:5]
            print(f"  {dirn}: " + "  ".join(f"{summ[t][0][j]}({sg*mean[j]:+.3f},q={q[j]:.3f})" for j in o))
    # 4개 유형 모두에서 같은 방향으로 유의한 latent (공통 신호)
    if all(t in summ for t in TYPES):
        idx = summ[1][0]
        mat = np.stack([summ[t][1] for t in TYPES])
        qm = np.stack([summ[t][2] for t in TYPES])
        for dirn, sg in (("wrong_up", 1), ("correct_up", -1)):
            mn = (sg * mat).min(0)
            sig = ((sg * mat) > 0).all(0) & (qm < 0.1).all(0)
            o = np.argsort(-mn)[:8]
            print(f"\n공통(4유형 모두 {dirn}) 상위: " + "  ".join(f"{idx[j]}(min {mn[j]:+.3f}, maxq={qm[:, j].max():.2f})" for j in o))
            print(f"  4유형 모두 같은 방향 + 모두 FDR<0.1 인 latent 수: {int(sig.sum())}")


if __name__ == "__main__":
    main()
