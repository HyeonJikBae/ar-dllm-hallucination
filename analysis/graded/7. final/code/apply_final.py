"""Re-apply the grounded decisions to the final file.

rule decisions (merged_v3.json, conf 'hi') + agent decisions (agent_v3.json, for the rest) -> final_label / final_conf / label_source
of every row currently labelled 1 or 2. Rows are matched to a decision by (relation, gold qid, gold, answer).
Usage: python3 grounded/apply_final.py [out.jsonl]      (default: writes next to the final file as *_rerun.jsonl; the final file is never overwritten)"""
import collections, json, sys
sys.path.insert(0, "grounded")
import groups
FINAL = "analysis/graded/7. final/qwen2.5-7b_popqa_final.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else FINAL.replace(".jsonl", "_rerun.jsonl")
key = lambda x: (x["rel"], x["gold_qid"], x["gold"], x["ans"])
agent = {tuple(k): v for k, v in json.load(open("grounded/agent_v3.json"))}
dec = {}
for x in json.load(open("grounded/merged_v3.json")):
    if x["conf"] == "hi":
        dec[key(x)] = (x["rule"], "rule", True, x["ev"][:80], x["rel"])
    else:
        c = agent[key(x)]; unsure = c.endswith("?"); c = c.rstrip("?")
        if unsure and x["rule"] in ("1", "2"): c = x["rule"]
        elif unsure: c = "KEEP"
        dec[key(x)] = (c, "agent", not unsure, (x["rule"] + " " + x["ev"])[:80], x["rel"])
raw = {}
for l in open("data/popqa/1. popqa_raw.jsonl"):
    r = json.loads(l); raw[str(r["id"])] = r["o_uri"].rsplit("/", 1)[-1]
LABEL = {"C": "correct_lenient", "N": "abstain"}
n = 0
with open(OUT, "w") as f:
    for l in open(FINAL):
        r = json.loads(l)
        if r["final_label"] in ("1", "2"):
            k = (r["relation"], raw.get(r["sample_id"]), r["gold_value"], (r.get("model_answer") or "").strip())
            if k in dec and dec[k][0] != "KEEP":
                code, src, conf, ev, rel = dec[k]
                r["final_label"] = LABEL.get(code, code); r["final_conf"] = conf; r["note"] = ev
                if code == "C": r["label_source"] = "lenient-correct-grounded"
                elif rel in groups.TABLES: r["label_source"] = "group-rule" if src == "rule" else "agent+hint"
                else: r["label_source"] = "wikidata-rule" if src == "rule" else "agent+wikidata-hint"
                n += 1
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("decisions applied to", n, "rows ->", OUT)
