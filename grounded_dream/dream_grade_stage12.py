"""Dream-v0-Base-7B PopQA wrong-answer labels, stages 1-2 of the 7. final policy (Qwen pipeline, same rules).

Input : analysis/graded/1. reviewed 1/dream-7b_popqa_graded.jsonl (55,225 rows, first-pass verdict)
Stage 1 string rules (same logic as 7. final/code/string_rules_prep.py), on incorrect rows:
        6-empty / 6-leak / 6-loop -> 6, N-abstain -> abstain, C-alias / C-short -> correct_lenient, 3-copy -> 3
Stage 2 remaining incorrect rows, pair unit (relation, gold qid, gold, answer):
        sport / religion / occupation -> groups.py ; director..author, genre, places, parents -> rules.label_pair
        (Wikidata, short answers only, as in grounded/full.py). Confident (hi) -> 1/2. Everything else -> agent queue.
Outputs (grounded_dream/):
  stage12_rows.jsonl   one line per row: provisional label, confidence, source, evidence
  agent_queue.tsv      pairs that need the agent (agent_prompt_en.md input format, hint column)
Wikidata cache: grounded_dream/wd_cache.json (copied from grounded/wd_cache.json; the tracked file is not touched).

Usage:  python3 grounded_dream/dream_grade_stage12.py
"""
import collections
import json
import os
import re
import shutil
import sys
import time
import unicodedata

HERE = "grounded_dream"
os.makedirs(HERE, exist_ok=True)
if not os.path.exists(f"{HERE}/wd_cache.json"):
    shutil.copy("grounded/wd_cache.json", f"{HERE}/wd_cache.json")
os.environ["WD_CACHE"] = f"{HERE}/wd_cache.json"
sys.path.insert(0, "grounded")
import groups  # noqa: E402
import rules  # noqa: E402
import wd  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f}m]", *a, flush=True)
SRC = "analysis/graded/1. reviewed 1/dream-7b_popqa_graded.jsonl"

# ---------------- stage 1: string rules (verbatim logic of string_rules_prep.py)
TYPE = {**{r: "A" for r in ["director", "producer", "screenwriter", "composer", "author"]}, **{r: "B" for r in ["father", "mother"]},
        **{r: "C" for r in ["genre", "religion", "sport", "occupation", "color"]}, **{r: "D" for r in ["place of birth", "country", "capital", "capital of"]}}


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower()); s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


TITLES = {"sir", "dr", "mr", "mrs", "ms", "prof", "professor", "lord", "lady", "dame", "the", "a", "an", "saint", "st", "jr", "sr", "ii", "iii", "iv"}
ABST = re.compile(r"\b(not specified|no information|not provided|not mentioned|don t know|do not know|unknown|cannot|can t|unclear|no answer|not available|not given|not known|not sure|there is no|n a)\b")


def loopy(t):
    w = t.split()
    if len(w) >= 4:
        c = collections.Counter(w)
        if c.most_common(1)[0][1] >= 3 and len(w) >= 5:
            return True
        bg = collections.Counter(zip(w, w[1:]))
        if bg.most_common(1)[0][1] >= 3:
            return True
    return False


def string_rule(r, rel):
    a = r["model_answer"] or ""; na = norm(a); ng = norm(r["gold_value"])
    nal = {norm(x) for x in r["gold_aliases"]} | {ng}; nq = norm(r["question"])
    if not na:
        return "6-empty"
    if "question" in na.split() and re.search(r"\b(question|answer)\b ?:?", a.lower()) and re.search(r"(?i)question:|answer:|direct answer", a):
        return "6-leak"
    if loopy(na):
        return "6-loop"
    if len(na.split()) <= 14 and ABST.search(na):
        return "N-abstain"
    if na in nal:
        return "C-alias"
    wa = [w for w in na.split() if w not in TITLES]; wg = [w for w in ng.split() if w not in TITLES]
    if wa and wg and set(wa) == set(wg):
        return "C-alias"
    if wa and wg and TYPE.get(rel) in ("A", "B") and len(wa) >= 2 and set(wa) < set(wg):
        return "C-short"
    if wa and wg and len(wg) >= 2 and wa == [wg[-1]] and TYPE.get(rel) in ("A", "B"):
        return "C-short"
    if wa and wg and set(wg) < set(wa) and len(wa) - len(wg) <= 2 and len(wa) <= 6:
        return "C-alias"
    if wa and len(" ".join(wa)) >= 3 and re.search(r"(?<![a-z0-9])" + re.escape(" ".join(wa)) + r"(?![a-z0-9])", nq):
        return "3-copy"
    return None


CODE2LABEL = {"6-empty": "6", "6-leak": "6", "6-loop": "6", "N-abstain": "abstain", "C-alias": "correct_lenient", "C-short": "correct_lenient", "3-copy": "3"}

raw = {}
for l in open("data/popqa/1. popqa_raw.jsonl"):
    x = json.loads(l)
    raw[str(x["id"])] = (x["prop"], x["o_uri"].rsplit("/", 1)[-1], x["s_uri"].rsplit("/", 1)[-1])
rows = [json.loads(l) for l in open(SRC)]
out = []
pairs, pair_subj = collections.OrderedDict(), {}
for i, r in enumerate(rows):
    rel, gq, sq = raw[r["sample_id"]]
    rec = {"sample_id": r["sample_id"], "version": r["version"], "relation": rel, "question": r["question"], "model_answer": r["model_answer"],
           "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"], "verdict_original": r["verdict"], "gold_issue": r.get("gold_issue")}
    if r["verdict"] != "incorrect":
        rec.update(label=r["verdict"], conf=r["verdict"] == "correct", source="original-" + r["verdict"], ev=None)
    else:
        code = string_rule(r, rel)
        if code:
            rec.update(label=CODE2LABEL[code], conf=True, source="string-rule", ev=code)
        else:
            a = (r["model_answer"] or "").strip()
            k = (rel, gq, r["gold_value"], a)
            pairs.setdefault(k, []).append(i)
            pair_subj.setdefault(k, sq)
            rec.update(label=None, conf=None, source="pair", ev=None, pair=list(k))
    out.append(rec)
log("rows", len(rows), "| first-pass:", dict(collections.Counter(r["verdict"] for r in rows)))
log("stage1 string rules:", dict(collections.Counter(o["ev"] for o in out if o["source"] == "string-rule")))
log("stage2 pairs:", len(pairs), "rows", sum(map(len, pairs.values())))

# ---------------- stage 2: groups + Wikidata rules (same selection as grounded/full.py)
IMPL = set(rules.ROLE) | {"genre", "father", "mother"} | set(rules.PLACE_REL)
todo = [k for k in pairs if k[0] in IMPL and 0 < len(k[3].split()) <= 6 and len(k[3]) < 60]
wd.prefetch_names([k[3] for k in todo]); log("names done")
cand_q = [q for k in todo for q in wd._cache["names"].get(k[3], [])]
wd.prefetch_ents(list(dict.fromkeys(cand_q + [k[1] for k in todo if k[1]] + [pair_subj[k] for k in todo if k[0] == "producer"]))); log("entities done", len(wd._cache["ent"]))
hop = lambda ps: [v for q in list(wd._cache["ent"]) for p in ps for v in (wd._cache["ent"][q] or {"claims": {p: []}})["claims"].get(p, [])]
wd.prefetch_ents(list(dict.fromkeys(hop(("P17", "P27", "P495", "P47"))))); log("countries done")
for h in range(4):
    wd.prefetch_ents(list(dict.fromkeys(hop(("P131",))))); log("admin hop", h + 1, len(wd._cache["ent"]))
wd.save()
todo_set = set(todo)
dec = {}
cnt = collections.Counter()
for n, k in enumerate(pairs):
    rel, gq, gn, ans = k
    if rel in groups.TABLES:
        lab, conf, ev = groups.label(rel, gn, ans)
    elif k in todo_set:
        try:
            lab, conf, ev = rules.label_pair(rel, gq, gn, ans, pair_subj.get(k))
        except Exception as e:
            lab, conf, ev = "?", "lo", f"error {type(e).__name__}: {e}"
    else:
        lab, conf, ev = "?", "lo", "relation not implemented" if rel not in IMPL else "long or empty answer"
    dec[k] = (lab, conf, ev)
    cnt[(lab, conf)] += 1
log("stage2 decisions", dict(cnt))

queue = []
for o in out:
    if o["source"] != "pair":
        continue
    k = tuple(o.pop("pair"))
    lab, conf, ev = dec[k]
    if conf == "hi" and lab in ("1", "2"):
        o.update(label=lab, conf=True, source="group-rule" if k[0] in groups.TABLES else "wikidata-rule", ev=ev[:80])
    else:
        o.update(label=None, conf=None, source="agent-queue", ev=f"{lab}({conf}) {ev}"[:200], pair=list(k))
with open(f"{HERE}/stage12_rows.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
qp = collections.OrderedDict()
for o in out:
    if o["source"] == "agent-queue":
        k = tuple(o["pair"])
        qp.setdefault(k, {"q": o["question"], "n": 0, "hint": o["ev"]})["n"] += 1
with open(f"{HERE}/agent_queue.tsv", "w") as f:
    for j, (k, v) in enumerate(qp.items()):
        f.write(f"{j}\t{k[0]}\tgold={k[2]}\tans={k[3][:140]}\tq={v['q'][:90]}\thint={v['hint']}\n")
json.dump([list(k) for k in qp], open(f"{HERE}/agent_queue_keys.json", "w"), ensure_ascii=False)
log("DONE | labels:", dict(collections.Counter(o["label"] for o in out)), "| agent queue pairs", len(qp), "rows",
    sum(v["n"] for v in qp.values()))
