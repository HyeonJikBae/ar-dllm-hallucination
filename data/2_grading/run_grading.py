"""Step 2: correct / wrong for every row of both models.  One rule for both models, no row excluded.

The answer is the first line of the generation (leading spaces and line breaks removed first; see ../1_raw).
  1. unrelated text inside that line -> WRONG  (web leftovers, a math problem, a new sentence glued without a space,
     prompt / chat leak such as "Question:" or "user:", or a repetition loop)
  2. otherwise the normalised answer (lower case, no accents, no punctuation) is compared with the gold value and every alias:
     CORRECT if equal, or if difflib similarity (ratio or token-sort ratio) >= 90, and the regnal numerals (I ... XII) agree
  3. otherwise WRONG
Also writes k_correct / fact_group: how many of the 5 paraphrases of one fact are correct (0/5 ... 5/5).

Usage (from this folder):  python3 run_grading.py [--raw-dir ../1_raw] [--out-dir .] [--check]
"""
import argparse
import collections
import difflib
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
TH = 90
ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}
STOP = set("the a an of in on at to for and or is was are were be by with as that this it its from his her their who which what".split())
LEAK = re.compile(r"(?i)question:|answer:|direct answer|\buser:|\bassistant:")
WEB = re.compile(r"Take the 2-minute tour|You are here|Skip to main content|The Interview Process|The Executive Committee")
MATH = re.compile(r"What is the (?:sum|value|product)|equation|\$|\bSolve\b|\bFind the\b")
GLUED = re.compile(r"[a-z0-9)][.!?](?=(?:What|Who|Which|When|Where|How|The|This|It|In|You|Take|Skip|Find|Let|Solve|Question|Answer)\b)")


def first_line(text):
    return (text or "").lstrip().split("\n")[0].strip()


def norm(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def loopy(t):
    w = t.split()
    if len(w) < 4:
        return False
    if collections.Counter(zip(w, w[1:])).most_common(1)[0][1] >= 3:
        return True
    c = collections.Counter(x for x in w if x not in STOP and len(x) >= 3)
    return bool(c) and c.most_common(1)[0][1] >= 4


def continuation_start(a):
    """index where unrelated text (web leftovers, math, glued new sentence) starts, or None"""
    hits = [m.start() for m in (WEB.search(a), MATH.search(a)) if m] + [m.end() for m in [GLUED.search(a)] if m]
    return min(hits) if hits else None


def junk(a):
    """which kind of unrelated text the answer line carries, or None"""
    if LEAK.search(a):
        return "leak"
    if loopy(norm(a)):
        return "loop"
    if continuation_start(a) is not None:
        return "continuation"
    return None


def _ratio(a, b):
    return 100 * difflib.SequenceMatcher(None, a, b).ratio()


def _tsr(a, b):
    return _ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def is_correct(r):
    a = r["model_answer"] or ""
    if junk(a):
        return False
    c = norm(a)
    if not c:
        return False
    for g in [norm(r["gold_value"])] + [norm(x) for x in r["gold_aliases"]]:
        if not g or set(c.split()) & ROMAN != set(g.split()) & ROMAN:
            continue
        if c == g or max(_ratio(c, g), _tsr(c, g)) >= TH:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(HERE, "..", "1_raw"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--check", action="store_true", help="compare with the files already in --out-dir instead of writing")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for model in ("qwen", "dream"):
        rows = [json.loads(l) for l in open(os.path.join(args.raw_dir, f"{model}_raw.jsonl"), encoding="utf-8")]
        for r in rows:
            r["correct"] = is_correct(r)
        k = collections.Counter()
        for r in rows:
            k[r["sample_id"]] += r["correct"]
        for r in rows:
            r["k_correct"] = k[r["sample_id"]]
            r["fact_group"] = f"{r['k_correct']}/5"
        out = os.path.join(args.out_dir, f"{model}_grading.jsonl")
        if args.check:
            old = [json.loads(l) for l in open(out, encoding="utf-8")]
            diff = sum(a["correct"] != b["correct"] or a["fact_group"] != b["fact_group"] for a, b in zip(old, rows))
            print(model, "rows", len(rows), "correct", sum(r["correct"] for r in rows), "differences vs saved file:", diff)
            continue
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(model, "rows", len(rows), "correct", sum(r["correct"] for r in rows), "->", out)


if __name__ == "__main__":
    main()
