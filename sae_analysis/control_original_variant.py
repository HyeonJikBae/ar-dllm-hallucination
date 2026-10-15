"""Control: is the gain from the signal just "prefer the original wording"?

Uses the cross-validated per-row scores of combine_signals.py.
  [1] correct rate by question version in the paired facts
  [2] share of top-scored picks that are the original question, per signal
  [3] pick quality among the four paraphrases only (v1-v4): facts that still have >=1 correct and >=1 wrong paraphrase;
      top-scored paraphrase vs random paraphrase (= share correct), 95% bootstrap CI over facts
Usage:  PYTHONPATH=sae_analysis python3 sae_analysis/control_original_variant.py
"""
import collections
import json

import numpy as np

D = "sae_analysis/features_lens_v2/"
NAMES = ["conf", "sae5", "sae20", "sae_all", "conf_sae5", "conf_sae20", "conf_all"]


def main():
    meta = [json.loads(l) for l in open(D + "meta.jsonl", encoding="utf-8")]
    bs_rng = np.random.default_rng(1)
    for position in ("prompt_last", "q_last"):
        Z = np.load(f"sae_analysis/results_twelve/combine_scores_{position}.npz")
        rows = Z["rows"]
        sid = np.array([meta[i]["sample_id"] for i in rows]); ver = np.array([meta[i]["version"] for i in rows])
        y = np.char.startswith(np.array([meta[i]["final_label"] for i in rows]).astype(str), "correct").astype(int)
        facts = np.unique(sid)
        print(f"\n##### {position} | 짝지은 사실 {len(facts)}개, 행 {len(rows)}개")
        if position == "prompt_last":
            print("[1] 질문 버전별 맞힌 비율 (이 행 집합): " + "  ".join(f"{v} {y[ver == v].mean():.1%}({(ver == v).sum()})" for v in ("question", "v1", "v2", "v3", "v4")))
        print("[2] 신호별로 점수 1등으로 뽑힌 변형 중 원래 질문의 비율 (무작위로 뽑으면 약 20%)")
        print("   " + "  ".join(f"{n} {np.mean([ver[np.where(sid == f)[0][np.argmax(Z[n][np.where(sid == f)[0]])]] == 'question' for f in facts]):.0%}" for n in NAMES))
        # paraphrases only
        idx_by = {f: np.where((sid == f) & (ver != "question"))[0] for f in facts}
        keep = [f for f in facts if len(idx_by[f]) >= 2 and y[idx_by[f]].min() == 0 and y[idx_by[f]].max() == 1]
        base = np.array([y[idx_by[f]].mean() for f in keep])
        bs = bs_rng.integers(0, len(keep), (2000, len(keep)))
        print(f"[3] 원래 질문을 빼고 변형 v1~v4 안에서만 고르기: 맞힘/틀림이 모두 있는 사실 {len(keep)}개, 무작위 기준 {base.mean():.1%}")
        print(f"   {'신호':<12}{'점수 1등 맞힘':>14}{'차이 [95% 신뢰구간]':>26}")
        for n in NAMES:
            top = np.array([y[idx_by[f]][np.argmax(Z[n][idx_by[f]])] for f in keep])
            d = top - base
            lo, hi = np.percentile(d[bs].mean(1), [2.5, 97.5])
            print(f"   {n:<12}{top.mean():>13.1%}{d.mean():>+14.1%} [{lo:+.1%}, {hi:+.1%}]")
        for a, b in (("conf_sae20", "conf"), ("conf_sae5", "conf")):
            ta = np.array([y[idx_by[f]][np.argmax(Z[a][idx_by[f]])] for f in keep]); tb = np.array([y[idx_by[f]][np.argmax(Z[b][idx_by[f]])] for f in keep])
            d = ta - tb
            lo, hi = np.percentile(d[bs].mean(1), [2.5, 97.5])
            print(f"   {a} − {b}: {d.mean():+.1%} [{lo:+.1%}, {hi:+.1%}]")


if __name__ == "__main__":
    main()
