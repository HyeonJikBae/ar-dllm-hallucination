"""Two confounds of the know axis: lucky guesses and popularity.

(a) Guess baseline per relation (outputs only). In relations with a small answer space (religion, country, sport,
    genre ...) a 1/5 or 2/5 fact can be a lucky default answer rather than access to the fact.
      modal baseline   accuracy if the model's most frequent answer in the relation were given to every fact
      best constant    accuracy of the best single answer for the relation (upper bound of any constant guess)
      lucky facts      1-2/5 facts whose correct answers are all among the model's top-3 answers in the relation
    The know-axis position by k is then recomputed without lucky facts and without high-baseline relations.
(b) Popularity (PopQA s_pop). Positions of U / K5 / partial-fact rows inside s_pop tertiles, the K5-vs-U AUROC
    inside each tertile, and the within-state correlation of the axis score with log s_pop. If the axis were
    "famous entity" it would collapse inside a tertile and correlate with s_pop within U and within K5.

Know axis and z are as in know_axis_by_k.py (K5 - U direction on train facts, z normalised inside relation with
held-out U / K5 rows: 0 = doesn't know, 1 = knows).

Usage:
    PYTHONPATH=sae_analysis python3 sae_analysis/guess_pop_check.py --position q_last
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

from know_axis_by_k import fact_matrix
from twelve_cells import FEAT, auroc, strat_auroc, strat_diff

sys.path.insert(0, "TrivaQA_ex")
from scoring import normalize_answer, strip_answer_wrapper  # noqa: E402

FINAL = "analysis/graded/7. final/qwen2.5-7b_popqa_final.jsonl"
RAW = "data/popqa/1. popqa_raw.jsonl"


def norm(x):
    return normalize_answer(strip_answer_wrapper(x or ""))


def load(position):
    meta = [json.loads(l) for l in open(f"{FEAT}/qwen_popqa_meta.jsonl", encoding="utf-8")]
    final = {(r["sample_id"], r["version"]): r for r in map(json.loads, open(FINAL, encoding="utf-8"))}
    pop = {str(x["id"]): x["s_pop"] for x in map(json.loads, open(RAW, encoding="utf-8"))}
    ok = np.array([not m["gold_issue"] for m in meta])
    corr = np.array([m["verdict"] == "correct" or m["verdict_suggested"] == "correct" for m in meta])
    use = ok & ~np.array([bool(m.get("rule_unsure")) for m in meta])
    sid = np.array([m["sample_id"] for m in meta])
    kf = collections.Counter()
    for s, c, o in zip(sid, corr, ok):
        if o:
            kf[s] += int(c)
    k = np.array([kf.get(s, -1) for s in sid])
    keep = np.where(use & (k >= 0))[0]
    acts = np.load(f"{FEAT}/qwen_popqa_{position}.npy", mmap_mode="r")
    A = np.asarray(acts[keep]) > 0
    rows = [final[(meta[i]["sample_id"], meta[i]["version"])] for i in keep]
    ans = np.array([norm(r["model_answer"]) for r in rows], dtype=object)
    spop = np.array([pop.get(str(s), np.nan) for s in sid[keep]], dtype=float)
    rel = np.array([meta[i]["relation"] for i in keep])
    return A, corr[keep], sid[keep], rel, k[keep], ans, spop


def guess_table():
    """Per relation: modal baseline, best-constant baseline, model's top-3 answers (all final rows)."""
    rows = [json.loads(l) for l in open(FINAL, encoding="utf-8")]
    golds = {}
    for r in rows:
        if not r["gold_issue"]:
            golds[r["sample_id"]] = {norm(g) for g in [r["gold_value"], *(r.get("gold_aliases") or [])] if norm(g)}
    rel_of = {r["sample_id"]: r["relation"] for r in rows}
    out = {}
    for rel in sorted(set(rel_of.values())):
        facts = [f for f in golds if rel_of[f] == rel]
        cnt = collections.Counter(norm(r["model_answer"]) for r in rows if r["relation"] == rel and norm(r["model_answer"]))
        top3 = [a for a, _ in cnt.most_common(3)]
        modal = top3[0]
        gcnt = collections.Counter(g for f in facts for g in golds[f])
        out[rel] = {"facts": len(facts), "modal_answer": modal, "modal_share_of_outputs": cnt[modal] / sum(cnt.values()),
                    "modal_baseline": sum(modal in golds[f] for f in facts) / len(facts),
                    "best_constant": gcnt.most_common(1)[0][1] / len(facts), "best_constant_answer": gcnt.most_common(1)[0][0],
                    "top3": top3}
    return out


def axis_z(A, corr, sid, rel, k, splits, min_rel):
    """Yields (z, test mask) per split."""
    Af = A.astype(np.float32)
    isK5, isU = corr & (k == 5), (k == 0) & ~corr
    facts = np.array(sorted(set(sid)))
    for seed in range(splits):
        rng = np.random.default_rng(seed)
        tr = np.isin(sid, rng.permutation(facts)[: int(0.6 * len(facts))])
        te = ~tr
        _, XK, rK = fact_matrix(A[tr & isK5], sid[tr & isK5], rel[tr & isK5])
        _, XU, rU = fact_matrix(A[tr & isU], sid[tr & isU], rel[tr & isU])
        d, _ = strat_diff(XK, rK, XU, rU, min_rel)
        s = Af @ d.astype(np.float32)
        z = np.full(len(s), np.nan)
        for r in set(rel):
            mr = te & (rel == r)
            if len(set(sid[mr & isK5])) < min_rel or len(set(sid[mr & isU])) < min_rel:
                continue
            mu, mk = s[mr & isU].mean(), s[mr & isK5].mean()
            if mk > mu:
                z[mr] = (s[mr] - mu) / (mk - mu)
        yield z, te & ~np.isnan(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--min-freq", type=float, default=0.005)
    ap.add_argument("--min-rel-facts", type=int, default=3)
    ap.add_argument("--high-baseline", type=float, default=0.2)
    ap.add_argument("--out", default="sae_analysis/results_twelve")
    args = ap.parse_args()
    report = {}

    # ---------- (a) guess baseline
    gt = guess_table()
    report["guess_table"] = gt
    A, corr, sid, rel, k, ans, spop = load(args.position)
    A = A[:, A.mean(0) >= args.min_freq]
    lucky = set()
    per_rel_lucky = collections.Counter()
    per_rel_low = collections.Counter()
    for f in set(sid[(k >= 1) & (k <= 2)]):
        m = (sid == f) & corr
        r = rel[sid == f][0]
        per_rel_low[r] += 1
        if m.any() and all(a in gt[r]["top3"] for a in ans[m]):
            lucky.add(f)
            per_rel_lucky[r] += 1
    print(f"##### {args.position}")
    print("\n[a] 관계별 추측 기준선 (출력만 사용; 기준선 높은 순)")
    print(f"{'관계':<15}{'fact':>6}{'최빈 답':>22}{'최빈답 기준선':>12}{'최적 상수':>10}{'1~2/5 fact':>11}{'운 좋은 추측':>12}")
    for r, g in sorted(gt.items(), key=lambda x: -x[1]["modal_baseline"]):
        lk = f"{per_rel_lucky[r]}/{per_rel_low[r]}" if per_rel_low[r] else "-"
        print(f"{r:<15}{g['facts']:>6}{g['modal_answer'][:20]:>22}{g['modal_baseline']:>11.1%}{g['best_constant']:>10.1%}{per_rel_low[r]:>11}{lk:>12}")
    hi_rel = {r for r, g in gt.items() if g["modal_baseline"] > args.high_baseline}
    print(f"  기준선 > {args.high_baseline:.0%} 관계: {sorted(hi_rel)} | 운 좋은 추측 fact(1~2/5): {len(lucky)} / {len(set(sid[(k >= 1) & (k <= 2)]))}")
    report["lucky_facts"] = len(lucky)
    report["high_baseline_relations"] = sorted(hi_rel)

    conds = {"전체": np.ones(len(sid), bool),
             "운 좋은 추측 fact 제외": ~np.isin(sid, list(lucky)),
             f"기준선 > {args.high_baseline:.0%} 관계 제외": ~np.isin(rel, list(hi_rel))}
    pos = collections.defaultdict(list)
    pop_res = collections.defaultdict(list)
    tert = np.nanquantile(np.log10(spop[~np.isnan(spop)]), [1 / 3, 2 / 3])
    pbin = np.digitize(np.log10(spop), tert)  # 0 low, 1 mid, 2 high; nan -> 3
    pbin[np.isnan(spop)] = -1
    isK5, isU = corr & (k == 5), (k == 0) & ~corr
    for z, v in axis_z(A, corr, sid, rel, k, args.splits, args.min_rel_facts):
        for cname, cm in conds.items():
            for kk in (1, 2, 3, 4):
                for lab, cmask in (("오답", ~corr), ("정답", corr)):
                    m = v & cm & (k == kk) & cmask
                    if m.any():
                        pos[(cname, kk, lab)].append(z[m].mean())
        # ---------- (b) popularity
        for b, bname in enumerate(("하", "중", "상")):
            inb = v & (pbin == b)
            pop_res[(bname, "U")].append(z[inb & isU].mean())
            pop_res[(bname, "K5")].append(z[inb & isK5].mean())
            for lab, km, cmask in (("1~2/5 오답", (k >= 1) & (k <= 2), ~corr), ("3~4/5 오답", (k >= 3) & (k <= 4), ~corr),
                                   ("1~2/5 정답", (k >= 1) & (k <= 2), corr), ("3~4/5 정답", (k >= 3) & (k <= 4), corr)):
                m = inb & km & cmask
                if m.any():
                    pop_res[(bname, lab)].append(z[m].mean())
            pop_res[(bname, "AUROC K5>U")].append(strat_auroc(z[inb & isK5], rel[inb & isK5], z[inb & isU], rel[inb & isU], 5))
            pop_res[(bname, "n_K5_U")].append(((inb & isK5).sum(), (inb & isU).sum()))
        for st, sm in (("U", isU), ("K5", isK5)):
            rs, ws = [], []
            for r in set(rel):
                m = v & sm & (rel == r) & ~np.isnan(spop)
                if m.sum() >= 30:
                    rs.append(spearmanr(z[m], np.log10(spop[m])).correlation)
                    ws.append(m.sum())
            pop_res[("corr", st)].append(np.average(rs, weights=ws))
        pop_res[("corr", "all(K5+U)")].append(spearmanr(z[v & (isU | isK5)], np.log10(spop[v & (isU | isK5)]), nan_policy="omit").correlation)

    print("\n[a'] k별 앎 축 위치 (0 = 모름, 1 = 앎): 조건별 비교")
    print(f"{'':<22}" + "".join(f"{f'{kk}/5 오답':>11}{f'{kk}/5 정답':>11}" for kk in (1, 2, 3, 4)))
    for cname in conds:
        print(f"  {cname:<20}" + "".join(
            f"{np.mean(pos[(cname, kk, '오답')]):>+11.2f}{np.mean(pos[(cname, kk, '정답')]):>+11.2f}" for kk in (1, 2, 3, 4)))
    report["position_by_condition"] = {f"{c}|{kk}|{l}": [float(np.mean(v)), float(np.std(v))] for (c, kk, l), v in pos.items()}

    print("\n[b] 유명도(s_pop) 3분위 안에서의 앎 축 위치 (관계 안 정규화는 전체 기준)")
    print(f"{'s_pop':<6}{'U':>8}{'K5':>8}{'1~2/5 오답':>12}{'1~2/5 정답':>12}{'3~4/5 오답':>12}{'3~4/5 정답':>12}{'AUROC K5>U':>12}{'행 K5/U':>16}")
    for bname in ("하", "중", "상"):
        g = lambda key: np.mean(pop_res[(bname, key)]) if pop_res[(bname, key)] else np.nan
        nk, nu = np.mean([x[0] for x in pop_res[(bname, "n_K5_U")]]), np.mean([x[1] for x in pop_res[(bname, "n_K5_U")]])
        print(f"  {bname:<4}{g('U'):>+8.2f}{g('K5'):>+8.2f}{g('1~2/5 오답'):>+12.2f}{g('1~2/5 정답'):>+12.2f}"
              f"{g('3~4/5 오답'):>+12.2f}{g('3~4/5 정답'):>+12.2f}{g('AUROC K5>U'):>12.3f}{nk:>8.0f}/{nu:<7.0f}")
    print(f"  s_pop 3분위 경계(위키 조회수): {10 ** tert[0]:.0f}, {10 ** tert[1]:.0f}")
    print("  앎 축 점수와 log s_pop의 Spearman 상관 (관계 안 평균): "
          f"U 안 {np.mean(pop_res[('corr', 'U')]):+.3f}, K5 안 {np.mean(pop_res[('corr', 'K5')]):+.3f}, "
          f"K5+U 전체 {np.mean(pop_res[('corr', 'all(K5+U)')]):+.3f}")
    report["popularity"] = {f"{a}|{b}": (np.mean(v, axis=0).tolist() if b == "n_K5_U" else float(np.nanmean(v))) for (a, b), v in pop_res.items()}
    os.makedirs(args.out, exist_ok=True)
    json.dump(report, open(os.path.join(args.out, f"guess_pop_check_{args.position}.json"), "w"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
