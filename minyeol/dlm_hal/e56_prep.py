"""E56 1 단계: TriviaQA rc.nocontext validation (HF mandarjoshi/trivia_qa, parquet sha256 48a5005c...) 에서 질문 표본.
걸러내기: question_id 중복 제거 -> answer.type == WikipediaEntity -> answer.value <= 5 단어 -> 별칭 >= 1 (value 포함) ->
  value 가 질문에 그대로 나오지 않음 (word_in) -> 질문 정리 (겉 따옴표, "" -> ") 후 4~60 단어.  남은 것에서 seed 56 으로 4,000 개.
출력: followup_mk/e56_data/questions.jsonl, e56_data/prep_counts.json
usage (repo 루트): /home/minyeol/.conda/envs/sae-dream/bin/python followup_mk/e56_prep.py
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "followup_mk")
from e56_common import D56, alias_forms

SEED, N = 56, 4000
d = pd.read_parquet(f"{D56}/raw/rc.nocontext_validation.parquet")
C = {"rows_parquet": len(d)}
d = d.drop_duplicates("question_id").reset_index(drop=True); C["unique_question_id"] = len(d)

def clean(q):
    q = q.strip()
    if len(q) >= 2 and q[0] == '"' and q[-1] == '"': q = q[1:-1].strip()
    return q.replace('""', '"').strip()

keep = []
c_ent = c_len = c_alias = c_exp = c_qlen = 0
for q, qid, src, a in zip(d.question, d.question_id, d.question_source, d.answer):
    if a["type"] != "WikipediaEntity": continue
    c_ent += 1
    if len(a["value"].split()) > 5: continue
    c_len += 1
    if len(a["aliases"]) < 1: continue
    c_alias += 1
    qc = clean(q)
    al, ex = alias_forms(a["value"], list(a["aliases"]), qc)
    if al is None: continue
    c_exp += 1
    if not (4 <= len(qc.split()) <= 60): continue
    c_qlen += 1
    keep.append(dict(qid=qid, question=qc, question_raw=q, source=src, value=a["value"], aliases_raw=list(a["aliases"]), aliases=al, aliases_exposed=ex))
C.update(type_WikipediaEntity=c_ent, value_le5_words=c_len, has_alias=c_alias, value_not_in_question=c_exp, question_4_60_words=c_qlen)
idx = np.random.default_rng(SEED).permutation(len(keep))[:N]
sample = [keep[j] for j in sorted(idx)]
C["sampled"] = len(sample); C["seed"] = SEED
C["exposed_alias_removed_questions"] = sum(bool(x["aliases_exposed"]) for x in sample)
with open(f"{D56}/questions.jsonl", "w", encoding="utf-8") as f:
    for x in sample: f.write(json.dumps(x, ensure_ascii=False) + "\n")
json.dump(C, open(f"{D56}/prep_counts.json", "w"), indent=1)
print(C)
