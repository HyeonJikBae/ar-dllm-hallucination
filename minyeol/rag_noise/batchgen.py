"""토큰 길이가 정확히 같은 프롬프트끼리만 묶어 생성한다 (padding 없음).  gen.py 의 생성 설정을 그대로 쓴다.
Dream 과 Qwen 은 배치 안 padding 에 답이 흔들리므로 (PREREG.md), 길이가 다른 프롬프트는 절대 한 배치에 넣지 않는다.
usage (같은 결과가 나오는지 확인): CUDA_VISIBLE_DEVICES=0 python batchgen.py --check qwen|dream [--n 400] [--bs 16]
"""
import argparse, collections, json, os, time
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from gen import MODELS, prompt, cond_docs, first_line, generate

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    dream = name == "dream"
    tok = AutoTokenizer.from_pretrained(MODELS[name], trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cls = AutoModel if dream else AutoModelForCausalLM
    kw = dict(trust_remote_code=True) if dream else {}
    return tok, cls.from_pretrained(MODELS[name], torch_dtype=torch.bfloat16, **kw).cuda().eval(), dream


def gen_all(model, tok, dream, prompts, bs=16, max_new=16, log=None, on_batch=None, fn=None):
    """prompts 순서대로 raw 생성문을 돌려준다.  on_batch(인덱스 목록, 생성문 목록) 로 중간 저장 가능."""
    by = collections.defaultdict(list)
    for i, p in enumerate(prompts): by[len(tok(p)["input_ids"])].append(i)
    out = [None] * len(prompts); done = 0; t0 = time.perf_counter(); n_b = 0
    for n in sorted(by):
        ix = by[n]
        for b in range(0, len(ix), bs):
            sub = ix[b:b + bs]; texts = fn([prompts[i] for i in sub]) if fn else generate(model, tok, [prompts[i] for i in sub], max_new, dream)[0]
            for i, t in zip(sub, texts): out[i] = t
            if on_batch: on_batch(sub, texts)
            done += len(sub); n_b += 1
            if log and n_b % 25 == 0: print(f"  {log} {done}/{len(prompts)}  {time.perf_counter() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()   # 길이가 바뀔 때마다 예약 메모리를 돌려준다 (길이별 묶음이 많아 조각이 쌓이면 한 프로세스가 57 GB 까지 불어난다.  GPU 를 나눠 쓰므로)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", required=True); ap.add_argument("--n", type=int, default=400); ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{HERE}/data/musique_bindswap.jsonl")]
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]
    ref = {(r["qid"], r["cond"]): r["pred"] for r in map(json.loads, open(f"{HERE}/results/{a.check}_bindswap.jsonl"))}
    jobs = [(r["qid"], c, prompt(r["question"], cond_docs(r, c), shots)) for r in rows[::4] for c in r["ctx"]][: a.n]
    tok, model, dream = load(a.check)
    lens = collections.Counter(len(tok(p)["input_ids"]) for _, _, p in jobs)
    print(f"프롬프트 {len(jobs)}개, 길이 종류 {len(lens)}, 평균 묶음 {len(jobs) / len(lens):.1f}", flush=True)
    t0 = time.perf_counter(); out = gen_all(model, tok, dream, [p for _, _, p in jobs], bs=a.bs); dt = time.perf_counter() - t0
    same = sum(first_line(t) == ref[(q, c)] for (q, c, _), t in zip(jobs, out))
    print(f"[{a.check}] 기존 결과와 같은 답 {same}/{len(jobs)} ({100 * same / len(jobs):.1f}%), {dt:.0f}s, 프롬프트당 {dt / len(jobs):.2f}s", flush=True)
    for (q, c, _), t in zip(jobs, out):
        if first_line(t) != ref[(q, c)]: print("  다름:", q, c, repr(first_line(t)), "| 기존", repr(ref[(q, c)]))


if __name__ == "__main__":
    main()
