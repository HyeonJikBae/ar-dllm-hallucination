"""Step 4: classify same-fact (correct, wrong) paraphrase pairs by WHERE their prompts differ relative to the subject.

For an AR model the activations up to the subject's last token are identical when the token prefix up to there is
identical, so patching at the subject only makes sense for pairs whose prefix differs. Categories, on the full prompt
tokens (run_inference.build_prompt, Qwen tokenizer):
  same_prefix   tokens up to and including the subject are identical; prompts differ only after the subject
  same_suffix   tokens after the subject are identical; prompts differ only before / inside the subject
  both          differ before and after the subject
  no_subject    subject string (PopQA subj or any s_alias) not found in one of the questions
Rows: logit_lens_extract.py selection, facts with both correct and wrong rows (1-4/5).

Usage:
    python3 sae_analysis/pair_prefix_types.py
"""

import collections
import itertools
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_inference import build_prompt  # noqa: E402

D = "sae_analysis/features_lens"


def subject_span(tok, prompt, names):
    low = prompt.lower()
    for n in sorted(names, key=len, reverse=True):
        i = low.find(n.lower())
        if i >= 0:
            enc = tok(prompt, return_offsets_mapping=True)
            toks = [j for j, (s, e) in enumerate(enc["offset_mapping"]) if s < i + len(n) and e > i]
            return enc["input_ids"], toks[0], toks[-1]
    return tok(prompt)["input_ids"], None, None


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", revision="d149729398750b98c0af14eb82c78cfe92750796")
    raw = {}
    for l in open("data/popqa/1. popqa_raw.jsonl"):
        x = json.loads(l)
        raw[str(x["id"])] = [x["subj"]] + json.loads(x["s_aliases"] or "[]")
    meta = [json.loads(l) for l in open(f"{D}/meta.jsonl", encoding="utf-8")]
    by = collections.defaultdict(list)
    for i, m in enumerate(meta):
        if m["fact_group"] not in ("0/5", "5/5"):
            by[m["sample_id"]].append((i, m))
    cat = collections.Counter()
    cat_type = collections.Counter()
    pairs = []
    for f, rows in by.items():
        info = {i: subject_span(tok, build_prompt(m["question"]), raw[f]) for i, m in rows}
        for (i, a), (j, b) in itertools.product(rows, rows):
            if not a["final_label"].startswith("correct") or b["final_label"].startswith("correct"):
                continue
            (ta, sa, ea), (tb, sb, eb) = info[i], info[j]
            if sa is None or sb is None:
                c = "no_subject"
            else:
                pre = ta[: ea + 1] == tb[: eb + 1]
                suf = ta[ea + 1:] == tb[eb + 1:]
                c = "same_prefix" if pre and not suf else "same_suffix" if suf and not pre else "identical" if pre and suf else "both"
            cat[c] += 1
            cat_type[(c, b["final_label"])] += 1
            pairs.append({"fact": f, "correct_row": i, "wrong_row": j, "cat": c, "wrong_type": b["final_label"],
                          "subj_end": [ea, eb], "len": [len(ta), len(tb)]})
    print(f"같은 fact의 (정답 문구, 오답 문구) 쌍: {len(pairs):,}  (fact {len(by):,})")
    for c in ("same_prefix", "same_suffix", "both", "identical", "no_subject"):
        t = {k[1]: v for k, v in cat_type.items() if k[0] == c}
        print(f"  {c:<12} {cat[c]:>6,}  ({cat[c] / len(pairs):.1%})   오답 유형별: {dict(sorted(t.items()))}")
    json.dump(pairs, open("sae_analysis/results_twelve/pair_prefix_types.json", "w"))


if __name__ == "__main__":
    main()
