"""Step 3: wrong-answer types for one model (same rules for qwen and dream).  Rule order: see RULES.md.

  python3 classify.py qwen|dream               rules only: writes work/<model>_classified.jsonl and work/<model>_agent_queue.tsv
  python3 classify.py qwen|dream --agents DIR  also reads the agent codes (DIR/*.txt, lines "queue_id<TAB>code") for the rows the rules could not decide
  python3 classify.py qwen|dream --offline     Wikidata from wd_cache.json only (no network)
  python3 classify.py qwen|dream --check       compares with <model>_classification.jsonl (rows decided by rules only)

Pipeline (per wrong row; the agent step works on (relation, gold, answer) pairs):
  1. string rules: refusal, collapse (empty / loop), copy of the question
  2. Wikidata rules on the cleaned answer: 1 related / 2 unrelated
  3. group rules (genre, religion, sport, occupation, ...)
  4. agent (classification_policy.md, agent_prompt_en.md) for what the rules cannot decide
Final type3: entity = 1, 2, 4 | broken = 3, 5, 6 | abstain.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["WD_CACHE"] = os.path.join(ROOT, "wd_cache.json")
WORK = os.path.join(ROOT, "work")
GRADING = os.path.join(ROOT, "..", "2_grading")
POPQA = os.path.join(ROOT, "..", "1_raw", "popqa_raw.jsonl")
sys.path.insert(0, ROOT)
import groups  # noqa: E402
import rules  # noqa: E402
import wd  # noqa: E402

TYPE = {**{r: "A" for r in ["director", "producer", "screenwriter", "composer", "author"]}, **{r: "B" for r in ["father", "mother"]},
        **{r: "C" for r in ["genre", "religion", "sport", "occupation", "color"]}, **{r: "D" for r in ["place of birth", "country", "capital", "capital of"]}}
TITLES = {"sir", "dr", "mr", "mrs", "ms", "prof", "professor", "lord", "lady", "dame", "the", "a", "an", "saint", "st", "jr", "sr"}
STOP = set("the a an of in on at to for and or is was are were be by with as that this it its from his her their who which what".split())
ABST = re.compile(r"\b(not specified|no information|not provided|not mentioned|don t know|do not know|unknown|cannot|can t|unclear|no answer|"
                  r"not available|not given|not known|not sure|there is no|n a|not explicitly|not stated|not named|not recognized|not a real|not identified)\b")
JUNK = re.compile(r"\s*(Take the 2-minute tour\s*×?|You are here|Skip to main content|The Interview Process|The Executive Committee)\s*$")
TMPL = re.compile(r"^(?:.{0,80}? )?(?:was|were|is|are) (?:directed|written|produced|composed|scripted|created|authored|fathered|mothered|screenwritten) by (.+)$")
CUT = re.compile(r"(?:[.;:!?]| and | in | who | which | for | with | at | on | from |, | \(|\n).*$")
TYPE3 = {"1": "entity", "2": "entity", "4": "entity", "3": "broken", "5": "broken", "6": "broken", "abstain": "abstain"}
IMPL = set(rules.ROLE) | {"genre", "father", "mother"} | set(rules.PLACE_REL)


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


def phrase_in(p, text):
    return bool(p) and bool(re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", text))


def string_rule(answer, question):
    """6 / abstain / 3 from the raw answer alone, else None."""
    na, nq = norm(answer), norm(question)
    if not na:
        return "6"
    if re.search(r"(?i)question:|answer:|direct answer", answer or ""):
        return "6"
    if loopy(na):
        return "6"
    if len(na.split()) <= 14 and ABST.search(na):
        return "abstain"
    wa = [w for w in na.split() if w not in TITLES]
    if wa and len(" ".join(wa)) >= 3 and phrase_in(" ".join(wa), nq):
        return "3"
    return None


def clean(a):
    b = JUNK.sub("", a).strip(" .")
    m = TMPL.match(b)
    if m:
        c = CUT.sub("", m.group(1)).strip(" .\"'")
        if 0 < len(c.split()) <= 6 and len(c) < 60:
            return c
    return b


def only_question_words(answer, question):
    aw = [w for w in norm(answer).split() if w not in STOP and w not in TITLES]
    return bool(aw) and all(w in set(norm(question).split()) for w in aw)


def prefetch(pairs, offline):
    """Fill the Wikidata cache for the pairs' answers, golds, countries and admin regions."""
    if offline:
        return
    todo = [k for k in pairs if 0 < len(k[3].split()) <= 6 and len(k[3]) < 60 and k[0] in IMPL]
    names = [k[3] for k in todo if k[3] not in wd._cache["names"]]
    for b in range(0, len(names), 8):
        chunk = names[b:b + 8]
        try:
            wd.prefetch_names(chunk, batch=8)
        except RuntimeError:
            for n in chunk:
                try:
                    wd.prefetch_names([n], batch=1)
                except RuntimeError:
                    wd._cache["names"][n] = []
    cand = [q for k in todo for q in wd._cache["names"].get(k[3], [])]
    wd.prefetch_ents(list(dict.fromkeys(cand + [k[1] for k in todo if k[1]])))
    hop = lambda ps: [v for q in list(wd._cache["ent"]) for p in ps for v in (wd._cache["ent"][q] or {"claims": {p: []}})["claims"].get(p, [])]
    wd.prefetch_ents(list(dict.fromkeys(hop(("P17", "P27", "P495", "P47")))))
    for _ in range(4):
        wd.prefetch_ents(list(dict.fromkeys(hop(("P131",)))))
    wd.save()


def rule_label(rel, gq, gn, ans, subj):
    if rel in groups.TABLES:
        return groups.label(rel, gn, ans)
    if not (0 < len(ans.split()) <= 6 and len(ans) < 60) or rel not in IMPL:
        return "?", "lo", "no rule for this relation or answer length"
    try:
        return rules.label_pair(rel, gq, gn, ans, subj)
    except Exception as e:  # noqa: BLE001
        return "?", "lo", f"error {type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["qwen", "dream"])
    ap.add_argument("--agents")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    rows = [r for r in (json.loads(l) for l in open(f"{GRADING}/{args.model}_grading.jsonl", encoding="utf-8")) if not r["correct"]]
    gold_q, subj_of = {}, {}
    for l in open(POPQA, encoding="utf-8"):
        d = json.loads(l)
        gold_q[str(d["id"])] = d["o_uri"].rsplit("/", 1)[-1]
        subj_of[str(d["id"])] = d["s_uri"].rsplit("/", 1)[-1]
    print(args.model, "wrong rows", len(rows))

    # 1. string rules, then group the rest into (relation, gold, cleaned answer) pairs
    out = [None] * len(rows)
    pairs = collections.OrderedDict()
    for i, r in enumerate(rows):
        s = string_rule(r["model_answer"] or "", r["question"])
        if s:
            out[i] = (s, True, "string-rule")
        else:
            pairs.setdefault((r["relation"], gold_q.get(r["sample_id"]), r["gold_value"], clean((r["model_answer"] or "").strip())), []).append(i)
    print("string rules:", collections.Counter(o[0] for o in out if o), "| pairs left", len(pairs))

    # 2. Wikidata rules and group tables
    prefetch(list(pairs), args.offline)
    queue = []
    for k, ix in pairs.items():
        rel, gq, gn, ans = k
        lab, conf, ev = rule_label(rel, gq, gn, ans, subj_of.get(rows[ix[0]]["sample_id"]))
        if conf == "hi" and lab in ("1", "2"):
            for i in ix:
                out[i] = (lab, True, "group-rule" if rel in groups.TABLES else "wikidata-rule")
        else:
            queue.append((k, ix, lab, conf, ev))
    print("rules decided", sum(o is not None for o in out), "rows | agent queue", len(queue), "pairs")

    # 3. agent queue / agent codes
    with open(f"{WORK}/{args.model}_agent_queue.tsv", "w", encoding="utf-8") as f:
        for j, (k, ix, lab, conf, ev) in enumerate(queue):
            f.write(f"{j}\t{k[0]}\tgold={k[2]}\tans={k[3][:200]}\tq={rows[ix[0]]['question']}\thint={lab}({conf}) {ev[:120]}".replace("\n", " ") + "\n")
    codes = {}
    for fn in glob.glob(f"{args.agents}/*.txt") if args.agents else []:
        for l in open(fn, encoding="utf-8"):
            p = l.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0].strip().isdigit():
                codes[int(p[0])] = p[1].strip()
    for j, (k, ix, lab, conf, ev) in enumerate(queue):
        c = codes.get(j)
        for i in ix:
            if c is None:
                out[i] = ("pending", False, "agent-pending")
            else:
                c2 = c.rstrip("?")
                out[i] = ("1" if c2 == "C" else "abstain" if c2 == "N" else c2, not c.endswith("?"), "agent")

    # 4. a 5 made only of question words is a 3
    for i, r in enumerate(rows):
        if out[i][0] == "5" and only_question_words(r["model_answer"] or "", r["question"]):
            out[i] = ("3", out[i][1], out[i][2] + "+copy")

    path = f"{WORK}/{args.model}_classified.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r, (lab, conf, src) in zip(rows, out):
            f.write(json.dumps({"sample_id": r["sample_id"], "version": r["version"], "relation": r["relation"], "question": r["question"],
                                "model_answer": r["model_answer"], "gold_value": r["gold_value"], "final_label": lab,
                                "type3": TYPE3.get(lab, "pending"), "label_source": src, "type_confident": conf, "unverified": False,
                                "final_conf": conf, "note": ""}, ensure_ascii=False) + "\n")
    print("type3:", collections.Counter(TYPE3.get(o[0], "pending") for o in out), "->", path)

    if args.check:
        ref = {(r["sample_id"], r["version"]): r for r in (json.loads(l) for l in open(f"{ROOT}/{args.model}_classification.jsonl", encoding="utf-8"))}
        dec = [(r, o) for r, o in zip(rows, out) if o[2] != "agent-pending"]
        same3 = sum(TYPE3[o[0]] == ref[(r["sample_id"], r["version"])]["type3"] for r, o in dec)
        e = [(r, o) for r, o in dec if TYPE3[o[0]] == "entity"]
        same12 = sum(o[0] == ref[(r["sample_id"], r["version"])]["final_label"] for r, o in e)
        print(f"check: {len(dec)} rows decided by rules/agents | type3 equal to saved {same3} ({same3 / len(dec):.1%}) | entity rows with the same 1/2 label {same12}/{len(e)}")


if __name__ == "__main__":
    main()
