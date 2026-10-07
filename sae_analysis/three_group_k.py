"""three_group.py / three_state_summary.py with a stricter "knows" definition.

  K   correct rows from facts with k >= kmin correct (k = #correct of 5)
  W_t wrong rows of type t from facts with kmin <= k <= kmax    ("knows but wrong")
  U   all rows of 0-of-5 facts                                   ("doesn't know")
Facts with 1 <= k < kmin are dropped entirely.

Usage:
    PYTHONPATH=sae_analysis python3 sae_analysis/three_group_k.py --kmin 2 --kmax 4
"""

import argparse
import collections
import json

import numpy as np

from three_group import DATASETS, FEAT, PREFIX, auroc

KINDS = {
    "안다": lambda k, w, u: np.minimum(k - w, k - u),
    "모른다": lambda k, w, u: np.minimum(u - k, u - w),
    "아는데 틀림(오답에서 높음)": lambda k, w, u: np.minimum(w - k, w - u),
    "아는데 틀림(오답에서 낮음)": lambda k, w, u: np.minimum(k - w, u - w),
}


def load_k(ds, position, kmin, kmax):
    rows = []
    meta = [json.loads(l) for l in open(f"{FEAT}/{PREFIX}_{ds}_meta.jsonl", encoding="utf-8")]
    acts = np.load(f"{FEAT}/{PREFIX}_{ds}_{position}.npy") > 0
    k = collections.Counter(m["sample_id"] for m in meta if m["verdict"] == "correct")
    for m, a in zip(meta, acts):
        if m["gold_issue"]:
            continue
        kk = k[m["sample_id"]]
        if m["verdict"] == "correct" and kk >= kmin:
            rows.append((m["sample_id"], "K", a))
        elif m["verdict"] != "correct" and kmin <= kk <= kmax:
            rows.append((m["sample_id"], f"W{m['reason']}", a))
    zmeta = [json.loads(l) for l in open(f"{FEAT}/{PREFIX}_{ds}_zero_meta.jsonl", encoding="utf-8")]
    zacts = np.load(f"{FEAT}/{PREFIX}_{ds}_zero_{position}.npy") > 0
    for m, a in zip(zmeta, zacts):
        if not m["gold_issue"]:
            rows.append((m["sample_id"], f"U{m['reason']}", a))
    sid = np.array([r[0] for r in rows])
    grp = np.array([r[1] for r in rows])
    B = np.stack([r[2] for r in rows]).astype(np.float32)
    return sid, grp, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    data = {ds: load_k(ds, args.position, args.kmin, args.kmax) for ds in DATASETS}
    print(f"##### {args.position}, W = facts with {args.kmin}<=k<={args.kmax}, K = correct rows with k>={args.kmin}")
    for ds, (sid, grp, B) in data.items():
        isU = np.char.startswith(grp, "U")
        print(f"{ds}: rows K={int((grp=='K').sum())} W1={int((grp=='W1').sum())} W2={int((grp=='W2').sum())} U={int(isU.sum())}; facts={len(set(sid))}")

    # --- position on know-axis
    for ds, (sid, grp, B) in data.items():
        isU = np.char.startswith(grp, "U")
        facts = np.array(sorted(set(sid)))
        pos = {n: [] for n in ("W1", "W2")}
        au = collections.defaultdict(list)
        for seed in range(args.splits):
            rng = np.random.default_rng(seed)
            tr = np.isin(sid, rng.permutation(facts)[: int(0.6 * len(facts))])
            te = ~tr
            d = B[tr & (grp == "K")].mean(0) - B[tr & isU].mean(0)
            sc = B @ d
            mK, mU = sc[te & (grp == "K")].mean(), sc[te & isU].mean()
            for n in pos:
                if (te & (grp == n)).any():
                    pos[n].append((sc[te & (grp == n)].mean() - mU) / (mK - mU))
            s = lambda g: sc[te & g]
            au["K_vs_U"].append(auroc(s(grp == "K"), s(isU)))
            for n in ("W1", "W2"):
                au[f"K_vs_{n}"].append(auroc(s(grp == "K"), s(grp == n)))
        print(f"\n[{ds}] know-axis position (0=U, 1=K): " + ", ".join(f"{n} {np.mean(v):+.2f}±{np.std(v):.2f}" for n, v in pos.items()))
        print("   AUROC:", {k: f"{np.nanmean(v):.3f}" for k, v in au.items()})

    # --- state-specific latents (selection on train, val/test sign criterion)
    split = {}
    for ds, (sid, grp, B) in data.items():
        rng = np.random.default_rng(0)
        facts = rng.permutation(sorted(set(sid)))
        a, b = int(0.5 * len(facts)), int(0.6 * len(facts))
        split[ds] = {"train": np.isin(sid, facts[:a]), "val": np.isin(sid, facts[a:b]), "test": np.isin(sid, facts[b:])}
    for t in (1, 2):
        f = {}
        for ds, (sid, grp, B) in data.items():
            for s in ("train", "val", "test"):
                m = split[ds][s]
                f[(ds, s)] = tuple(B[m & mask].mean(0) for mask in (grp == "K", grp == f"W{t}", np.char.startswith(grp, "U")))
        print(f"\n=== type{t} (W{t} test rows: " + ", ".join(f"{ds} {int((split[ds]['test'] & (data[ds][1] == f'W{t}')).sum())}" for ds in DATASETS) + ")")
        for kind, fn in KINDS.items():
            sc = {k: fn(*v) for k, v in f.items()}
            tr = np.minimum(sc[("popqa", "train")], sc[("triviaqa", "train")])
            ok = np.ones_like(tr, bool)
            for ds in DATASETS:
                for s in ("val", "test"):
                    ok &= sc[(ds, s)] > 0
            top = [j for j in np.argsort(-tr) if ok[j]]
            print(f"  {kind}: {len(top[:8]) if False else int(ok[np.argsort(-tr)[:8]].sum())}/8 of top-8 pass; top passing:")
            for j in top[: args.top]:
                kw = lambda ds: "/".join(f"{x[j]:.2f}" for x in f[(ds, 'test')])
                print(f"     {j:6d} train {tr[j]:+.3f} test {sc[('popqa','test')][j]:+.3f}/{sc[('triviaqa','test')][j]:+.3f}  K/W/U popqa {kw('popqa')} trivia {kw('triviaqa')}")


if __name__ == "__main__":
    main()
