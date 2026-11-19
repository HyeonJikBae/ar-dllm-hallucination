"""E56 2 단계: Qwen/Qwen3-14B (bf16, 생각 끔 enable_thinking=False) 로 다른 문구 4 개 생성 -> 같은 모델로 따로 검사.
gen : 질문마다 따로 한 대화 (PopQA 의 generate_paraphrases_openai.py 와 같은 규칙을 TriviaQA 에 맞게 고친 시스템 지시문).  답은 보여주지 않음 (PopQA 와 같음).
      표집 temperature 0.7, top_p 0.8, top_k 20 (Qwen3 생각 끔 권장값), 묶음마다 seed = 56 + 시도*100000 + 묶음 번호.  JSON 이 깨지면 최대 2 번 더 (다른 seed).
val : 문구마다 따로 한 대화.  원문, 정답 (answer.value), 문구를 보여 주고 두 질문: same (같은 것을 물어 정답이 그대로인가), leak (정답이나 그 일부를 담거나 원문보다 더 직접 드러내는가).
      greedy.  통과 = same yes 이고 leak no.  (규칙 검사: 별칭 노출, 원문과 같음 은 e56_items.py 에서)
출력: e56_data/para_gen.jsonl, e56_data/para_val.jsonl (이어 돌리기 가능).
usage (repo 루트): python followup_mk/e56_para.py --phase gen|val|both [--limit N] [--bs 48]
"""
import argparse, json, os, re, sys, time
import torch
sys.path.insert(0, "followup_mk")
from e56_common import D56

MODEL = "Qwen/Qwen3-14B"
SYS_GEN = """You paraphrase a single trivia question into 4 distinct versions (v1-v4).
First identify:
- ANSWER_TYPE: what kind of thing the answer must be (e.g. "which country", "which film", "which composer", "what word")
- CLUES: every piece of information in the question that is needed to identify the answer (names, dates, numbers, places, quotations, descriptions)
Then generate v1-v4 following these rules:
MUST KEEP:
1. Keep the ANSWER_TYPE's category and its level of specificity (film -> movie OK; film -> actor NOT OK; surname -> full name NOT OK).
2. Keep ALL the CLUES, with the same meaning. Do not drop, generalise, or change any name, date, number or quotation. The paraphrase must have exactly the same single correct answer as the original.
3. Do not flip what is asked (e.g. asking for the subject instead of the object).
MUST CHANGE:
4. Substantially vary word order, verb choice, and sentence structure across v1-v4. Do not just swap one word for a synonym.
5. Vary the opening across v1-v4 (Which/What/Who/Name.../Can you name.../statement form ending with a question mark).
6. Every one of v1-v4 must be worded differently from the original question and from each other.
FORBIDDEN:
7. Do not add any information that is not in the original question, especially hints about the answer.
8. Do not include the answer or any part of it.
9. Do not produce grammatically awkward or unnatural English.
Respond with ONLY a JSON object with exactly these keys: answer_type, clues, v1, v2, v3, v4. No other text."""
SYS_VAL = "You are a strict checker of paraphrased trivia questions. You answer only with the requested JSON."
USR_VAL = """Original question: {q}
Correct answer to the original question: {a}
Paraphrased question: {p}

Answer two checks:
1. same: Does the paraphrased question ask for exactly the same information as the original (no clue dropped, changed or added), so that its single correct answer is still "{a}" at the same level of specificity? (yes/no)
2. leak: Does the paraphrased question contain the answer or a part of it, or reveal the answer more directly than the original question does? (yes/no)
Respond with ONLY a JSON object: {{"same": "yes" or "no", "leak": "yes" or "no"}}"""


def parse_json(s):
    s = s.strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a: return None
    try: return json.loads(s[a:b + 1])
    except Exception:
        try: return json.loads(re.sub(r",\s*}", "}", s[a:b + 1]))
        except Exception: return None


def done_set(fn, key):
    out = set()
    if os.path.exists(fn):
        for l in open(fn, encoding="utf-8"):
            try: out.add(key(json.loads(l)))
            except Exception: pass
    return out


@torch.inference_mode()
def run_chat(model, tok, convs, bs, max_new, sample, seed0, log_every=20):
    """convs: [(key, system, user)] -> {key: text}.  길이순 묶음, 왼쪽 패딩."""
    texts = [(k_, tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)) for k_, s, u in convs]
    texts.sort(key=lambda x: len(tok(x[1])["input_ids"]))
    out = {}; t0 = time.time()
    for bi, b0 in enumerate(range(0, len(texts), bs)):
        ch = texts[b0:b0 + bs]
        enc = tok([t for _, t in ch], return_tensors="pt", padding=True).to("cuda")
        kw = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20) if sample else dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        if sample: torch.manual_seed(seed0 + bi)
        o = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **kw)
        for (k_, _), seq in zip(ch, o[:, enc["input_ids"].shape[1]:]):
            out[k_] = tok.decode(seq, skip_special_tokens=True)
        yield out; out = {}
        if (bi + 1) % log_every == 0: print(f"    묶음 {bi + 1}/{(len(texts) + bs - 1) // bs}, {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="both"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=48)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    Q = [json.loads(l) for l in open(f"{D56}/questions.jsonl", encoding="utf-8")]
    if args.limit: Q = Q[:args.limit]
    sfx = "_pilot" if args.limit else ""
    fg, fv = f"{D56}/para_gen{sfx}.jsonl", f"{D56}/para_val{sfx}.jsonl"
    if args.phase in ("gen", "both"):
        for att in range(3):
            done = done_set(fg, lambda d: d["qid"] if d.get("ok") else None) - {None}
            tried = done_set(fg, lambda d: (d["qid"], d["attempt"]))
            todo = [x for x in Q if x["qid"] not in done and (x["qid"], att) not in tried]
            print(f"gen 시도 {att}: 남은 질문 {len(todo)}", flush=True)
            if not todo: continue
            fo = open(fg, "a", encoding="utf-8")
            for out in run_chat(model, tok, [(x["qid"], SYS_GEN, f"Original question: {x['question']}") for x in todo], args.bs, 400, True, 56 + att * 100000):
                for qid, txt in out.items():
                    j = parse_json(txt); ok = bool(j) and all(isinstance(j.get(v), str) and j[v].strip() for v in ("v1", "v2", "v3", "v4"))
                    rec = dict(qid=qid, attempt=att, ok=ok, raw=txt)
                    if ok: rec.update({v: j[v].strip() for v in ("v1", "v2", "v3", "v4")}, answer_type=j.get("answer_type"), clues=j.get("clues"))
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fo.flush()
            fo.close()
    if args.phase in ("val", "both"):
        gen = {}
        for l in open(fg, encoding="utf-8"):
            d = json.loads(l)
            if d.get("ok") and d["qid"] not in gen: gen[d["qid"]] = d
        qd = {x["qid"]: x for x in Q}
        done = done_set(fv, lambda d: (d["qid"], d["ver"]))
        convs = [((qid, v), SYS_VAL, USR_VAL.format(q=qd[qid]["question"], a=qd[qid]["value"], p=g[v])) for qid, g in gen.items() if qid in qd for v in ("v1", "v2", "v3", "v4") if (qid, v) not in done]
        print(f"val: 문구 {len(convs)}", flush=True)
        fo = open(fv, "a", encoding="utf-8")
        for out in run_chat(model, tok, convs, args.bs * 2, 40, False, 0):
            for (qid, v), txt in out.items():
                j = parse_json(txt) or {}
                fo.write(json.dumps(dict(qid=qid, ver=v, same=str(j.get("same", "")).lower(), leak=str(j.get("leak", "")).lower(), raw=txt), ensure_ascii=False) + "\n")
            fo.flush()
        fo.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
