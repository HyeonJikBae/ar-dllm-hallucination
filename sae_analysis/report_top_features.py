"""Markdown report: top separating latents for correct vs wrong type 1 / 2, with val/test scores
(from separation_scores.py CSVs) and the questions that activate each latent most strongly.

Usage:
    python3 sae_analysis/report_top_features.py --position q_last --top 8
"""

import argparse
import json

import numpy as np
import pandas as pd

FEAT = "sae_analysis/features"
RES = "sae_analysis/results"
DATASETS = ["popqa", "triviaqa"]
TYPES = {1: "related_entity", 2: "unrelated_entity"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    meta, acts = [], []
    for ds in DATASETS:
        for l in open(f"{FEAT}/qwen_{ds}_meta.jsonl", encoding="utf-8"):
            m = json.loads(l)
            m["ds"] = ds
            meta.append(m)
        acts.append(np.load(f"{FEAT}/qwen_{ds}_{args.position}.npy"))
    acts = np.concatenate(acts)
    label = np.array(["correct" if m["verdict"] == "correct" else f"type{m['reason']}" for m in meta])
    usable = np.array([not m["gold_issue"] for m in meta])

    out = [f"# Top separating latents ({args.position}), correct vs wrong type\n"]
    for t, name in TYPES.items():
        for direction in ("wrong_up", "correct_up"):
            df = pd.read_csv(f"{RES}/{args.position}_type{t}_{name}_{direction}.csv").head(args.top)
            out.append(f"\n## type{t} {name} - {direction}\n")
            out.append("| latent | min train | val popqa | val trivia | test popqa | test trivia | freq correct | freq type | freq all |")
            out.append("|---|---|---|---|---|---|---|---|---|")
            for _, r in df.iterrows():
                j = int(r.latent)
                on = acts[:, j] > 0
                fc = on[(label == "correct") & usable].mean()
                ft = on[(label == f"type{t}") & usable].mean()
                out.append(f"| {j} | {r.min_train_score:+.3f} | {r.val_popqa:+.3f} | {r.val_triviaqa:+.3f} | "
                           f"{r.test_popqa:+.3f} | {r.test_triviaqa:+.3f} | {fc:.2f} | {ft:.2f} | {on.mean():.2f} |")
            for _, r in df.head(5).iterrows():
                j = int(r.latent)
                order = np.argsort(-acts[:, j].astype(np.float32))[: args.examples]
                out.append(f"\n**latent {j}** top-activating rows:")
                for i in order:
                    m = meta[i]
                    out.append(f"- [{m['ds']}|{label[i]}|act={float(acts[i, j]):.1f}] {m['question']} -> {m['model_answer'][:50]!r}")
    path = f"sae_analysis/report_{args.position}.md"
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(path)


if __name__ == "__main__":
    main()
