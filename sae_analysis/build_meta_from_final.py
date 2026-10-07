"""Build the per-row meta file the SAE analysis scripts read, from the final classification file (7. final).

The analysis scripts (state_signals_refined.py, analysis_refined.py) expect: sample_id, version, question, model_answer, gold_value,
verdict, verdict_suggested, reason, reason_orig, rule, rule_unsure, relation, gold_issue.
Here  verdict = verdict_original, verdict_suggested = 'correct' for correct_lenient rows, reason = 1..6 (None for abstain),
rule_unsure = not analysis_use  (so the scripts' own 'confident' filter reproduces analysis_use exactly).
Rows with verdict_original == 'uncertain' are dropped (same row order as the feature files).

Usage: python3 sae_analysis/build_meta_from_final.py [out_dir]"""
import json, os, sys

FINAL = "analysis/graded/7. final/qwen2.5-7b_popqa_final.jsonl"
KEYS = ("sample_id", "version", "question", "model_answer", "gold_value", "verdict", "verdict_suggested", "reason", "reason_orig", "rule", "rule_unsure", "relation", "gold_issue")


def legacy_rows(path=FINAL):
    rows = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r["verdict_original"] == "uncertain":
            continue
        lab = r["final_label"]
        rows.append({
            "sample_id": r["sample_id"], "version": r["version"], "question": r["question"], "model_answer": r["model_answer"],
            "gold_value": r["gold_value"], "verdict": r["verdict_original"],
            "verdict_suggested": "correct" if lab == "correct_lenient" else None,
            "reason": int(lab) if lab in ("1", "2", "3", "4", "5", "6") else None, "reason_orig": None, "rule": None,
            "rule_unsure": not r["analysis_use"], "relation": r["relation"], "gold_issue": r["gold_issue"]})
    return rows


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sae_analysis/features_g"
    os.makedirs(out, exist_ok=True)
    rows = legacy_rows()
    with open(os.path.join(out, "qwen_popqa_meta.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in KEYS}, ensure_ascii=False) + "\n")
    print("meta rows", len(rows), "->", out)
