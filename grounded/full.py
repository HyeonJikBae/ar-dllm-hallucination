"""Full grounded run: every wrong-answer pair currently labelled 1/2 -> grounded label + confidence + evidence.
Resumable (wd cache). Input: 7. final (rows labelled 1/2). Output: argv[1] (default grounded/full_out.jsonl), one line per (relation, gold qid, answer)."""
import collections, json, sys, time
sys.path.insert(0, "grounded")
import wd, rules

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f}m]", *a, flush=True)
raw = {}
for l in open("data/popqa/1. popqa_raw.jsonl"):
    r = json.loads(l); raw[str(r["id"])] = r["o_uri"].rsplit("/", 1)[-1]
FINAL = "analysis/graded/7. final/qwen2.5-7b_popqa_final.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "grounded/full_out.jsonl"
rows = [json.loads(l) for l in open(FINAL)]
pairs = {}
for r in rows:
    if r["final_label"] in ("1", "2"):          # 1/2 로 분류된 오답 행의 (관계, 정답ID, 정답, 답) 쌍
        a = (r.get("model_answer") or "").strip()
        pairs.setdefault((r["relation"], raw.get(r["sample_id"]), r["gold_value"], a), []).append(int(r["final_label"]))
log("unique pairs", len(pairs), collections.Counter(k[0] for k in pairs).most_common(6))
short = [k for k in pairs if 0 < len(k[3].split()) <= 6 and len(k[3]) < 60]
IMPL = set(rules.ROLE) | {"genre", "father", "mother"} | set(rules.PLACE_REL)
todo = [k for k in short if k[0] in IMPL]
wd.prefetch_names([k[3] for k in todo]); log("names done")
cand_q = [q for k in todo for q in wd._cache["names"].get(k[3], [])]
wd.prefetch_ents(list(dict.fromkeys(cand_q + [k[1] for k in todo if k[1]]))); log("entities done", len(wd._cache["ent"]))
# second-hop entities: countries / neighbours / relatives, then admin chains for places
hop = lambda ps: [v for q in list(wd._cache["ent"]) for p in ps for v in (wd._cache["ent"][q] or {"claims": {p: []}})["claims"].get(p, [])]
wd.prefetch_ents(list(dict.fromkeys(hop(("P17", "P27", "P495", "P47"))))); log("countries done")
for i in range(4):
    wd.prefetch_ents(list(dict.fromkeys(hop(("P131",))))); log("admin hop", i + 1, len(wd._cache["ent"]))
wd.save()
todo_set = set(todo)
out = open(OUT, "w")
cnt = collections.Counter()
for n, k in enumerate(pairs):
    rel, gq, gn, ans = k
    if k not in todo_set:
        lab, conf, ev = "?", "lo", "relation not implemented" if rel not in IMPL else "long or empty answer"
    else:
        try:
            lab, conf, ev = rules.label_pair(rel, gq, gn, ans)
        except Exception as e:
            lab, conf, ev = "?", "lo", f"error {type(e).__name__}: {e}"
    cnt[(lab, conf)] += 1
    out.write(json.dumps({"rel": rel, "gold_qid": gq, "gold": gn, "ans": ans, "cur": sorted(set(pairs[k])), "n": len(pairs[k]), "rule": lab, "conf": conf, "ev": ev}, ensure_ascii=False) + "\n")
    if n % 2000 == 0:
        log("labeled", n, dict(cnt))
out.close(); wd.save(); log("DONE", dict(cnt))
