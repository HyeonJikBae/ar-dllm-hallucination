"""Step 2: correct / wrong for every row of both models, from 1_raw.

Rule (one rule for both models, no row excluded; in the spirit of Ferrando et al. 2025, fuzzy string matching):
  correct  <=>  the cleaned answer (web boilerplate and "X was directed by Y" frames removed) matches the gold value or one of its aliases with
                difflib similarity >= 90 (ratio or token-sort ratio, on lower-cased, accent-free, punctuation-free strings),
                and the regnal numerals (I ... XII) of answer and match agree.
Also writes k_correct / fact_group: how many of the 5 paraphrases of one fact are correct (0/5 ... 5/5).

Usage (from this folder):  python3 run_grading.py            -> writes qwen_grading.jsonl, dream_grading.jsonl
                           python3 run_grading.py --check    -> also compares with the saved files and prints the number of differences
"""
import collections
import difflib
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "1_raw")
TH = 90
ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}
JUNK = re.compile(r"\s*(Take the 2-minute tour\s*×?|You are here|Skip to main content|The Interview Process|The Executive Committee)\s*$")
TMPL = re.compile(r"^(?:.{0,80}? )?(?:was|were|is|are) (?:directed|written|produced|composed|scripted|created|authored) by (.+)$")
STOP = re.compile(r"(?:[.;:!?]| and | in | who | which | for | with | at | on | from |, | \(|\n).*$")


def norm(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def core(a):
    b = JUNK.sub("", a).strip(" .")
    m = TMPL.match(b)
    if m:
        c = STOP.sub("", m.group(1)).strip(" .\"'")
        if 0 < len(c.split()) <= 6:
            return c
    return b


def _ratio(a, b):
    return 100 * difflib.SequenceMatcher(None, a, b).ratio()


def _tsr(a, b):
    return _ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def is_correct(r):
    a = r["model_answer"] or ""
    cands = []
    for s in (a, core(a)):
        n = norm(s)
        if n and n not in cands:
            cands.append(n)
    golds = [norm(r["gold_value"])] + [norm(x) for x in r["gold_aliases"]]
    for c in cands:
        for g in golds:
            if not g:
                continue
            if set(c.split()) & ROMAN != set(g.split()) & ROMAN:
                continue
            if c == g or max(_ratio(c, g), _tsr(c, g)) >= TH:
                return True
    return False


def main():
    check = "--check" in sys.argv
    for model in ("qwen", "dream"):
        rows = [json.loads(l) for l in open(f"{RAW}/{model}_raw.jsonl", encoding="utf-8")]
        for r in rows:
            r["correct"] = is_correct(r)
        k = collections.Counter()
        for r in rows:
            k[r["sample_id"]] += r["correct"]
        for r in rows:
            r["k_correct"] = k[r["sample_id"]]
            r["fact_group"] = f"{r['k_correct']}/5"
        out = f"{HERE}/{model}_grading.jsonl"
        if check:
            old = [json.loads(l) for l in open(out, encoding="utf-8")]
            diff = sum(a["correct"] != b["correct"] or a["fact_group"] != b["fact_group"] for a, b in zip(old, rows))
            print(model, "rows", len(rows), "correct", sum(r["correct"] for r in rows), "differences vs saved file:", diff)
            continue
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(model, "rows", len(rows), "correct", sum(r["correct"] for r in rows))


if __name__ == "__main__":
    main()
