"""full_out.jsonl (Wikidata rules) + groups.py (sport/religion/occupation) -> merged_v3.json  (one entry per (relation, gold qid, answer)).
Usage: python3 grounded/merge_groups.py [full_out.jsonl] [merged.json]"""
import json, sys
sys.path.insert(0, "grounded")
import groups
src = sys.argv[1] if len(sys.argv) > 1 else "grounded/full_out.jsonl"
dst = sys.argv[2] if len(sys.argv) > 2 else "grounded/merged_v3.json"
out = [json.loads(l) for l in open(src)]
for x in out:
    if x["rel"] in groups.TABLES:
        x["rule"], x["conf"], x["ev"] = groups.label(x["rel"], x["gold"], x["ans"])
json.dump(out, open(dst, "w"), ensure_ascii=False)
print("pairs", len(out), "confident", sum(x["conf"] == "hi" for x in out), "->", dst)
