"""Two summary tables for the knows (K) / knows-but-wrong (W) / doesn't know (U) experiment.

Table 1: most prominent latents per state (selected on train, min over datasets; kept only if val and
         test specificity are > 0 in both datasets).
Table 2: activation frequency K / W / U (test split, per dataset) for those latents + one example question.

Usage:
    PYTHONPATH=sae_analysis python3 sae_analysis/three_state_summary.py --position q_last
"""

import argparse
import json

import numpy as np

from three_group import DATASETS, FEAT, PREFIX, TAG, load

TYPES = {1: "관련 엔티티", 2: "무관 엔티티"}
KINDS = {
    "안다": lambda k, w, u: np.minimum(k - w, k - u),
    "모른다": lambda k, w, u: np.minimum(u - k, u - w),
    "아는데 틀림(오답에서 높음)": lambda k, w, u: np.minimum(w - k, w - u),
    "아는데 틀림(오답에서 낮음)": lambda k, w, u: np.minimum(k - w, u - w),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, questions = {}, {}
    for ds in DATASETS:
        sid, grp, B = load(ds, args.position)
        rng = np.random.default_rng(args.seed)
        facts = rng.permutation(sorted(set(sid)))
        a, b = int(0.5 * len(facts)), int(0.6 * len(facts))
        sp = {"train": facts[:a], "val": facts[a:b], "test": facts[b:]}
        data[ds] = (sid, grp, B, {k: np.isin(sid, v) for k, v in sp.items()})

    # example questions: highest activation over all rows (kept + zero), both datasets
    ex_rows, ex_acts = [], []
    for ds in DATASETS:
        for suffix in ("", "_zero"):
            meta = [json.loads(l) for l in open(f"{FEAT}/{PREFIX}_{ds}{suffix}_meta.jsonl", encoding="utf-8")]
            ex_rows += [(ds, m["question"]) for m in meta]
            ex_acts.append(np.load(f"{FEAT}/{PREFIX}_{ds}{suffix}_{args.position}.npy"))
    ex_acts = np.concatenate(ex_acts)

    def freq(ds, split, kind_mask):
        sid, grp, B, masks = data[ds]
        return B[masks[split] & kind_mask(grp)].mean(0)

    sel = {}  # (type, kind) -> list of (latent, train_score, specificity dict)
    for t in TYPES:
        isK = lambda g: g == "K"
        isW = lambda g, t=t: g == f"W{t}"
        isU = lambda g: np.char.startswith(g, "U")
        f = {(ds, s): (freq(ds, s, isK), freq(ds, s, isW), freq(ds, s, isU)) for ds in DATASETS for s in ("train", "val", "test")}
        for kind, fn in KINDS.items():
            sc = {key: fn(*v) for key, v in f.items()}
            train_min = np.minimum(*[sc[(ds, "train")] for ds in DATASETS])
            ok = np.ones(train_min.shape, bool)
            for ds in DATASETS:
                for s in ("val", "test"):
                    ok &= sc[(ds, s)] > 0
            cand = [j for j in np.argsort(-train_min) if ok[j]][: args.top]
            sel[(t, kind)] = [(int(j), float(train_min[j]), {k: sc[k][j] for k in sc},
                               {ds: tuple(float(x[j]) for x in f[(ds, "test")]) for ds in DATASETS}) for j in cand]

    def example(j):
        i = int(np.argmax(ex_acts[:, j].astype(np.float32)))
        return ex_rows[i][1][:70].replace("|", "/")

    out = [f"# 안다 / 아는데 틀림 / 모른다 — 두드러지는 feature ({args.position})\n"]
    out.append("선별 규칙: train(PopQA, TriviaQA 중 min)에서 해당 상태에만 특이한 점수 순, "
               "val·test 4칸 모두 > 0인 feature만 남김. 점수 = 해당 상태의 켜진 비율이 다른 두 상태보다 높은(낮은) 정도의 최솟값.\n")
    out.append("\n## 표 1. 상태별 두드러지는 feature\n")
    out.append("| 상태 | 오답 유형 | latent | train 점수 | test PopQA | test TriviaQA |")
    out.append("|---|---|---|---|---|---|")
    for kind in KINDS:
        for t in TYPES:
            for j, tr, sc, _ in sel[(t, kind)]:
                out.append(f"| {kind} | 유형 {t} ({TYPES[t]}) | {j} | {tr:+.3f} | {sc[('popqa','test')]:+.3f} | {sc[('triviaqa','test')]:+.3f} |")
    out.append("\n## 표 2. 켜진 비율 K / W / U (test)와 읽는 법\n")
    out.append("| feature | 상태 (오답 유형) | K / W / U (PopQA) | K / W / U (TriviaQA) | 가장 크게 켜지는 질문 예시 |")
    out.append("|---|---|---|---|---|")
    for kind in KINDS:
        for t in TYPES:
            for j, tr, sc, fr in sel[(t, kind)]:
                fmt = lambda ds: " / ".join(f"{x:.2f}" for x in fr[ds])
                out.append(f"| {j} | {kind} (유형 {t}) | {fmt('popqa')} | {fmt('triviaqa')} | {example(j)} |")
    path = f"sae_analysis/three_state_summary{TAG}_{args.position}.md"
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(path)


if __name__ == "__main__":
    main()
