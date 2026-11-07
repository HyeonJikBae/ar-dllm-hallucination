"""Rule-only wrong-answer types (entity / broken / abstain), same rules for every model.  No agent step.

Input : ../2_grading/<model>_grading.jsonl   (wrong rows only are typed; `correct` comes from run_grading.py)
Output: <model>_classification.jsonl (wrong rows) and full/<model>_final_full.jsonl (all rows, all columns)
Rules (first match wins):
  1 broken-empty          normalised answer is empty
  2 abstain               head (text before any unrelated continuation) has <= 14 words and a refusal phrase, or is just "no" / "none" / "nobody" / "nothing"
  3 broken-leak           "question:", "answer:", "direct answer", "user:", "assistant:" in the answer
  4 broken-loop           >= 4 words and a word pair 3+ times, or a non-stopword (>= 3 letters) 4+ times
  5 broken-continuation   web leftovers, a math problem, or a new sentence glued without a space ("Paris.What")
  6 broken-invalid        an answer that cannot name a person / place / field / colour: digits only ("1", "10"),
                          a single letter ("X", "B."), or a single function word ("Me", "It", "Yes")
  7 broken-copy           every content word of the answer is a word of the question
  8 broken-fabricated     (not for father / mother) 1-6 word answer, content words partly from the question and partly not, and no Wikidata candidate
                          (names missing from wd_cache.json: looked up with --online, otherwise left as entity with unverified=True)
  9 entity                everything else
Usage:  python3 classify_rules.py qwen|dream [--online] [--check]   (from this folder)
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["WD_CACHE"] = os.path.join(HERE, "wd_cache.json")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "2_grading"))
from run_grading import LEAK, continuation_start, loopy, norm  # noqa: E402  (same unrelated-text rules as grading)

STOP = set("the a an of in on at to for and or is was are were be by with as that this it its from his her their who which what".split())
KIN = {"father", "mother"}          # sharing the subject's family name is normal there, so no "fabricated" rule
TITLES = {"sir", "dr", "mr", "mrs", "ms", "prof", "professor", "lord", "lady", "dame", "the", "a", "an", "saint", "st", "jr", "sr"}
REFUSAL = re.compile(r"\b(not specified|no information|not provided|not mentioned|don t know|do not know|unknown|cannot|can t|unclear|no answer|"
                     r"not available|not given|not known|not sure|there is no|n a|not explicitly|not stated|not named|not recognized|not a real|not identified)\b")




FUNCTION = set("""i me my mine you your yours he him his she her hers it its we us our ours they them their theirs myself yourself himself herself itself
ourselves themselves this that these those here there the a an and or but nor of to in on at for from by with as into onto than
is am are was were be been being do does did has have had will would shall should can could may might must not no yes ok okay
what who whom whose which when where why how""".split())


def invalid_shape(a):
    """an answer that cannot name a person, place, field or colour: digits only, a single letter, or a single function word"""
    n = norm(a)
    if not n:
        return False
    if n.replace(" ", "").isdigit():
        return True
    if len(n.replace(" ", "")) == 1:
        return True
    return len(n.split()) == 1 and n in FUNCTION          # one function word ("Me", "Yes"); "The Who", "I Am" can be real titles


def content(s):
    return [w for w in norm(s).split() if w not in STOP and w not in TITLES]


def classify(answer, question, relation, has_candidate):
    """has_candidate(name) -> True / False / None (unknown).  Returns (type3, rule, unverified)."""
    a = answer or ""
    if not norm(a):
        return "broken", "broken-empty", False
    cut = continuation_start(a)
    head = a[:cut] if cut is not None else a
    nh = norm(head)
    if nh and len(nh.split()) <= 14 and (REFUSAL.search(nh) or nh in ("no", "none", "nobody", "nothing")):
        return "abstain", "abstain", False
    if LEAK.search(a):
        return "broken", "broken-leak", False
    if loopy(norm(a)):
        return "broken", "broken-loop", False
    if cut is not None:
        return "broken", "broken-continuation", False
    if invalid_shape(a):
        return "broken", "broken-invalid", False
    q = set(norm(question).split())
    cw = content(a)
    if cw and all(w in q for w in cw):
        return "broken", "broken-copy", False
    name = a.strip(" .\"'")
    cn = content(name)
    if relation not in KIN and 1 <= len(name.split()) <= 6 and any(w in q for w in cn) and any(w not in q for w in cn):
        found = has_candidate(name)
        if found is False:
            return "broken", "broken-fabricated", False
        if found is None:
            return "entity", "entity", True
    return "entity", "entity", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["qwen", "dream"])
    ap.add_argument("--online", action="store_true", help="look up names missing from wd_cache.json")
    ap.add_argument("--check", action="store_true", help="only compare with the saved <model>_classification.jsonl, write nothing")
    args = ap.parse_args()
    import wd  # loads wd_cache.json
    names = wd._cache.get("names", {})

    def has_candidate(name):
        if name not in names:
            if not args.online:
                return None
            try:
                wd.prefetch_names([name], batch=1)
            except RuntimeError:
                return None
        return bool(names.get(name))

    rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "2_grading", f"{args.model}_grading.jsonl"), encoding="utf-8")]
    full = []
    for r in rows:
        if r["correct"]:
            t, rule, unv = "correct", "correct", False
        else:
            t, rule, unv = classify(r["model_answer"], r["question"], r["relation"], has_candidate)
        full.append({"sample_id": r["sample_id"], "version": r["version"], "relation": r["relation"], "question": r["question"],
                     "model_answer": r["model_answer"], "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"],
                     "correct": r["correct"], "k_correct": r["k_correct"], "fact_group": r["fact_group"],
                     "type3": t, "rule": rule, "unverified": unv})
    if args.online:
        wd.save()
    wrong = [o for o in full if not o["correct"]]
    print(args.model, "rows", len(full), "| wrong", len(wrong))
    print(" type3:", dict(collections.Counter(o["type3"] for o in wrong)))
    print(" rule :", dict(collections.Counter(o["rule"] for o in wrong).most_common()))
    print(" unverified (name not in cache):", sum(o["unverified"] for o in wrong))
    keys = ("sample_id", "version", "relation", "question", "model_answer", "gold_value", "fact_group", "type3", "rule", "unverified")
    if args.check:
        ref = [json.loads(l) for l in open(os.path.join(HERE, f"{args.model}_classification.jsonl"), encoding="utf-8")]
        diff = sum(a != {k: b[k] for k in keys} for a, b in zip(ref, wrong)) + abs(len(ref) - len(wrong))
        print(" differences vs saved file:", diff)
        return
    with open(os.path.join(HERE, f"{args.model}_classification.jsonl"), "w", encoding="utf-8") as f:   # wrong rows only
        for o in wrong:
            f.write(json.dumps({k: o[k] for k in keys}, ensure_ascii=False) + "\n")
    os.makedirs(os.path.join(HERE, "full"), exist_ok=True)
    with open(os.path.join(HERE, "full", f"{args.model}_final_full.jsonl"), "w", encoding="utf-8") as f:   # all rows, all columns
        for o in full:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(" ->", f"{args.model}_classification.jsonl", "and", f"full/{args.model}_final_full.jsonl")


if __name__ == "__main__":
    main()
