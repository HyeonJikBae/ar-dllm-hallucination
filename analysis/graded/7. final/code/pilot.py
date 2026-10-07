import json, random, collections, sys
sys.path.insert(0, "grounded")
import wd, rules
random.seed(11)
raw = {}
for l in open("data/popqa/1. popqa_raw.jsonl"):
    r = json.loads(l); raw[str(r["id"])] = r["o_uri"].rsplit("/", 1)[-1]
rows = [json.loads(l) for l in open("analysis/graded/7. final/qwen2.5-7b_popqa_final.jsonl")]
pairs = {}
for r in rows:
    if r["final_label"] in ("1", "2") and r["final_conf"]:
        a = (r.get("model_answer") or "").strip()
        if 0 < len(a.split()) <= 5 and len(a) < 50:
            pairs.setdefault((r["relation"], raw.get(r["sample_id"]), r["gold_value"], a), int(r["final_label"]))
byrel = collections.defaultdict(list)
for k, v in pairs.items():
    byrel[k[0]].append((k, v))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
sel = []
for rel in ["director", "producer", "screenwriter", "composer", "author", "genre", "place of birth", "capital", "country", "capital of", "father", "mother"]:
    sel += [(rel, k, cur) for k, cur in random.sample(byrel[rel], min(N, len(byrel[rel])))]
wd.prefetch_names([k[3] for _, k, _ in sel])
wd.prefetch_ents([q for _, k, _ in sel for q in [k[1]] + wd._cache["names"].get(k[3], [])])
wd.prefetch_ents([v for q in list(wd._cache["ent"]) for p in ("P17", "P47", "P22", "P25", "P40", "P26", "P3373", "P1038") for v in wd._cache["ent"][q]["claims"][p]][:3000])
out = []
for rel in ["director", "producer", "screenwriter", "composer", "author", "genre", "place of birth", "capital", "country", "capital of", "father", "mother"]:
    for (r_, k, cur) in [x for x in sel if x[0] == rel]:
        try:
            lab, conf, ev = rules.label_pair(*k[:1], k[1], k[2], k[3])
        except Exception as e:
            lab, conf, ev = "?", "lo", f"error {type(e).__name__}"
        out.append({"rel": rel, "gold": k[2], "ans": k[3], "cur": str(cur), "rule": lab, "conf": conf, "ev": ev})
    wd.save()
json.dump(out, open("grounded/pilot_out.json", "w"), ensure_ascii=False, indent=0)
print("saved", len(out))
