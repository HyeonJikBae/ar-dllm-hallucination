"""E56 2 단계 마무리 (CPU): 생성 + 검사 결과를 합쳐 다섯 문구 사실 목록을 만든다.
문구 통과 = (a) JSON 정상 생성, (b) 비지 않음, (c) 원문·다른 문구와 정규화 후 같지 않음, (d) 규칙 노출 없음 (채점 별칭 중 하나가 문구에 단어 단위로 나오는데 원문에는 없음),
          (e) Qwen3-14B 검사 same == yes, (f) leak == no.  네 문구 모두 통과한 질문만 남김 (원문 + 4 = 5 문구).
출력: e56_data/items.jsonl, e56_data/para_counts.json, e56_data/spotcheck.txt (seed 0 무작위 30 사실).
usage (repo 루트): python followup_mk/e56_items.py [--pilot]
"""
import collections, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from e56_common import D56, norm, word_in

sfx = "_pilot" if "--pilot" in sys.argv else ""
Q = [json.loads(l) for l in open(f"{D56}/questions.jsonl", encoding="utf-8")]
gen = {}; tried = set()
for l in open(f"{D56}/para_gen{sfx}.jsonl", encoding="utf-8"):
    d = json.loads(l); tried.add(d["qid"])
    if d.get("ok") and d["qid"] not in gen: gen[d["qid"]] = d
val = {}
for l in open(f"{D56}/para_val{sfx}.jsonl", encoding="utf-8"):
    d = json.loads(l); val[(d["qid"], d["ver"])] = d
if sfx: Q = [x for x in Q if x["qid"] in tried]
C = collections.Counter(); C["questions"] = len(Q); keep = []; fails = collections.Counter(); per = []
for x in Q:
    g = gen.get(x["qid"])
    if g is None: C["gen_failed"] += 1; continue
    C["gen_ok"] += 1; vs = [g[v] for v in ("v1", "v2", "v3", "v4")]; npass = 0; why_all = []
    for j, v in enumerate(("v1", "v2", "v3", "v4")):
        p = g[v]; why = []
        if not norm(p): why.append("empty")
        if norm(p) == norm(x["question"]) or any(norm(p) == norm(o) for o in vs[:j]): why.append("identical")
        if any(word_in(a, p) and not word_in(a, x["question"]) for a in x["aliases"] if len(norm(a)) >= 2): why.append("rule_leak")
        vd = val.get((x["qid"], v))
        if vd is None: why.append("val_missing")
        else:
            if vd["same"] != "yes": why.append("llm_not_same")
            if vd["leak"] != "no": why.append("llm_leak")
        C["paraphrases"] += 1; C["paraphrase_pass"] += not why
        for w in why: fails[w] += 1
        npass += not why; why_all.append(why)
    per.append(npass)
    if npass == 4:
        keep.append(dict(qid=x["qid"], question=x["question"], v1=vs[0], v2=vs[1], v3=vs[2], v4=vs[3], value=x["value"], aliases=x["aliases"],
                         aliases_exposed=x["aliases_exposed"], source=x["source"], answer_type=g.get("answer_type")))
C["facts_4of4_pass"] = len(keep)
out = dict(counts=dict(C), fail_reasons=dict(fails), n_pass_per_question=dict(sorted(collections.Counter(per).items())),
           paraphrase_pass_rate=round(C["paraphrase_pass"] / max(C["paraphrases"], 1), 4), fact_keep_rate=round(len(keep) / max(C["gen_ok"], 1), 4))
json.dump(out, open(f"{D56}/para_counts{sfx}.json", "w"), indent=1, ensure_ascii=False)
with open(f"{D56}/items{sfx}.jsonl", "w", encoding="utf-8") as f:
    for x in keep: f.write(json.dumps(x, ensure_ascii=False) + "\n")
rng = np.random.default_rng(0); pick = sorted(rng.choice(len(keep), min(30, len(keep)), replace=False)) if keep else []
with open(f"{D56}/spotcheck{sfx}.txt", "w", encoding="utf-8") as f:
    for j in pick:
        x = keep[j]; f.write(f"[{x['qid']}] 답: {x['value']}\n  원문: {x['question']}\n" + "".join(f"  {v}: {x[v]}\n" for v in ("v1", "v2", "v3", "v4")) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=1))
