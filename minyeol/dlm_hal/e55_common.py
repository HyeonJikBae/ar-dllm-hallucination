"""E55 공용 (DiffuLLaMA-7B / LLaMA-2-7B).  repo 루트에서 실행.  common.py 는 고치지 않고 가져다 쓴다.
- grade(i, ans): common.Dream.grade 와 같은 채점 (is_correct -> classify, Wikidata 이름 캐시만 사용) 을 Dream 없이.
- gold_first(tok, i): 금 첫 토큰.  LLaMA sentencepiece 는 tok(' ' + gold) 에 따로 떨어진 '▁' (29871) 를 앞에 붙이므로
  Dream 규칙 tok(' ' + gold)[0] 을 그대로 쓰면 모든 행의 금 토큰이 '▁' 가 된다.  그래서 '프롬프트 + " " + gold' 를 통째로 토큰화해
  프롬프트 토큰 바로 다음 토큰을 쓴다 (= 문맥 안에서 답의 첫 토큰, 예 '▁politician').  tok(gold, add_special_tokens=False)[0] 와 같다.
- groups(items, idf, bs): 같은 길이 프롬프트끼리 bs 씩 (패딩 없음; common.by_length 와 같은 규칙, 묶음 크기만 인자).
"""
import collections, json, sys
sys.path.insert(0, "followup_mk")
from common import rows, build_prompt, extract_answer, is_correct, classify, wd, k as dream_k, fid, mid as dream_mid

_names = wd._cache.get("names", {})
_has_cand = lambda n: (bool(_names.get(n)) if n in _names else None)

def grade(i, ans):
    r = rows[i]; ok = is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]})
    return "correct" if ok else classify(ans, r["question"], r["relation"], _has_cand)[0]

def prompt_ids(tok, i): return tok(build_prompt(rows[i]["question"]))["input_ids"]

def gold_first(tok, i, pids=None):
    p = build_prompt(rows[i]["question"]); pids = pids or tok(p)["input_ids"]; full = tok(p + " " + rows[i]["gold_value"])["input_ids"]
    if full[:len(pids)] != pids or len(full) <= len(pids): return -1
    return full[len(pids)]

def groups(items, idf, bs):
    g = collections.defaultdict(list)
    for it in items: g[len(idf(it))].append(it)
    for L, xs in sorted(g.items()):
        for b0 in range(0, len(xs), bs): yield L, xs[b0:b0 + bs]

def decode_answer(tok, gen, SP):
    gen = gen[: next((q for q, t in enumerate(gen) if t in SP), len(gen))]
    return extract_answer(tok.decode(gen, skip_special_tokens=True))

def oom_retry(fn, tries=60, wait=60):
    """GPU 를 같이 쓰므로 묶음 하나가 메모리 부족으로 죽으면 캐시를 비우고 기다렸다가 같은 묶음을 다시 돌린다 (결과는 같음)."""
    import time, torch
    for t in range(tries):
        try: return fn()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  OOM, {wait}s 뒤 다시 ({t + 1}/{tries})", flush=True); time.sleep(wait)
    raise RuntimeError("OOM 반복")
