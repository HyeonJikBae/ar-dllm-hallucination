"""AR (Qwen2.5-7B) vs dLLM (Dream-v0-Base-7B) -- 근거가 모두 들어 있는 검색 문맥에서 노이즈가 생성을 망치는가.

--data full5 (기본): data/musique_full5.jsonl (LACE top-5 에 근거 전부, 460 질의).  조건 4개, 질의마다 같은 문서·같은 few-shot:
  closed     문서 없음 (모델 자체 지식)
  gold       근거 문서만 (top-5 에서 방해 문서를 뺀 것, 검색 순서 유지)
  top5       top-5 그대로 (근거 + 방해 1-3개, 검색 순서)
  top5_shuf  top-5 순서만 무작위 (질의 번호로 고정한 seed)
--data anchor: data/musique_anchor.jsonl (1,000 질의, 근거 전부; build_anchor.py).  조건 = 행의 ctx: closed / gold / hard3 / rand3
--data poison_pilot / poison_mainD / poison_star: build_poison.py 출력 (ctx gold / R / D / *_sw ...), 생성 방식은 anchor 와 같음.
  Dream 은 배치 안 왼쪽 padding 길이에 따라 답이 바뀔 수 있어 (2hop__689472_52835: padding 20 토큰에서 다른 답) 본 실행은 --bs 1
짧은 답 지시문 + few-shot 2개 (data/fewshot.json; 20 질의 파일럿에서 0-shot 은 Dream 형식 오류가 커서 2-shot).  답 = 생성문의 첫 줄.
AR: greedy.  Dream: temperature 0, alg=entropy, steps = --steps (기본 = max_new_tokens, 한 걸음에 한 토큰; 작을수록 병렬).
--trace (Dream): 단계마다 생성 칸의 argmax 토큰과 mask 여부를 results/<name>_trace.npz 에 저장 (argmax [N, steps, L], masked [N, steps, L],
  final [N, L]; N 순서 = jsonl 줄 순서).  단계 i 의 현재 답 = masked 칸은 argmax, 나머지는 final.  Dream 샘플러는 확정한 토큰을 다시 바꾸지 않으므로 '중간에 맞았다가 바뀜' 은 아직 mask 인 칸의 예측으로 본다.
usage: CUDA_VISIBLE_DEVICES=0 python gen.py --model qwen|dream [--data full5|anchor] [--limit N] [--shots K] [--tag T]
출력: results/<model>[_<tag>].jsonl  (질의 x 조건 한 줄, raw 생성문 포함)
"""
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"; os.makedirs(RES, exist_ok=True)
MODELS = {"qwen": "Qwen/Qwen2.5-7B", "dream": "Dream-org/Dream-v0-Base-7B"}
CONDS = ["closed", "gold", "top5", "top5_shuf"]
HEAD_DOC = "Answer each question using the given passages. Reply with only the short answer.\n\n"
HEAD_CB = "Answer each question. Reply with only the short answer.\n\n"
SEP = "\n\n---\n\n"


def passages(docs):
    return "\n\n".join(f"Passage {i + 1}: {d['title']}\n{d['text']}" for i, d in enumerate(docs))


def prompt(question, docs, shots):
    if docs is None:
        blocks = [f"Question: {s['question']}\nAnswer: {s['answer']}" for s in shots] + [f"Question: {question}\nAnswer:"]
        return HEAD_CB + SEP.join(blocks)
    blocks = [f"{passages(s['docs'])}\n\nQuestion: {s['question']}\nAnswer: {s['answer']}" for s in shots]
    blocks.append(f"{passages(docs)}\n\nQuestion: {question}\nAnswer:")
    return HEAD_DOC + SEP.join(blocks)


def cond_docs(r, c):
    D = lambda ids: [r["docs"][str(i)] for i in ids]
    if "ctx" in r: return None if r["ctx"][c] is None else D(r["ctx"][c])
    if c == "closed": return None
    if c == "gold": return D(r["gold"])
    if c == "top5": return D(r["top5"])
    ids = list(r["top5"]); np.random.default_rng(r["qi"]).shuffle(ids); return D(ids)


def first_line(s):
    s = s.split("\n")[0]
    for cut in ("---", "Question:", "Passage", "user:", "assistant:"):
        s = s.split(cut)[0]
    return s.strip()


@torch.inference_mode()
def generate(model, tok, prompts, max_new, dream, steps=None, trace=False):
    """생성문 리스트와 (trace 면) 샘플별 (argmax [steps, L], masked [steps, L])."""
    x = tok(prompts, return_tensors="pt", padding=True).to(model.device); tr = []
    if dream:
        def hook(i, xx, logits):                                   # logits 는 위치 정렬된 상태 (generation_utils._sample)
            if trace: tr.append(((logits[:, -max_new:].argmax(-1)).int().cpu(), (xx[:, -max_new:] == tok.mask_token_id).cpu()))
            return logits
        out = model.diffusion_generate(inputs=x["input_ids"], attention_mask=x["attention_mask"], max_new_tokens=max_new, steps=steps or max_new,
                                       temperature=0.0, alg="entropy", mask_token_id=tok.mask_token_id,
                                       pad_token_id=tok.pad_token_id or tok.eos_token_id, generation_logits_hook_func=hook)
        out = out.sequences if hasattr(out, "sequences") else out
    else:
        out = model.generate(**x, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    texts = [tok.decode(o[x["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]
    if not trace: return texts, None
    A, Mk = torch.stack([a for a, _ in tr], 1).numpy(), torch.stack([m for _, m in tr], 1).numpy()
    fin = out[:, -max_new:].int().cpu().numpy()                       # 확정 토큰 (한 번 확정되면 바뀌지 않음)
    return texts, [(A[b], Mk[b], fin[b]) for b in range(len(prompts))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS)); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--data", default="full5", choices=["full5", "anchor", "poison_pilot", "poison_mainD", "poison_star", "bindswap_pilot", "bindswap"])
    ap.add_argument("--steps", type=int, default=0, help="Dream 디노이징 단계 수 (0 = max_new, 한 걸음에 한 토큰)")
    ap.add_argument("--trace", action="store_true", help="Dream 단계별 argmax / mask 기록")
    ap.add_argument("--shots", type=int, default=2, help="few-shot 예시 수 (0 이면 Dream 이 답 뒤에 대화 형식 글을 붙여 형식 오류가 큼)"); ap.add_argument("--tag", default="")
    ap.add_argument("--conds", default="", help="쉼표로 구분한 조건만 생성 (기본: 데이터의 모든 조건)")
    a = ap.parse_args(); dream = a.model == "dream"
    rows = [json.loads(l) for l in open(f"{HERE}/data/musique_{a.data}.jsonl")][: a.limit or None]
    conds = list(rows[0]["ctx"]) if "ctx" in rows[0] else CONDS
    if a.conds: conds = [c for c in conds if c in a.conds.split(",")]
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[: a.shots]
    tok = AutoTokenizer.from_pretrained(MODELS[a.model], trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if dream: model = AutoModel.from_pretrained(MODELS[a.model], torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    else: model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], torch_dtype=torch.bfloat16).cuda().eval()
    jobs = [(r, c, prompt(r["question"], cond_docs(r, c), shots)) for r in rows for c in conds]
    lens = [len(tok(p)["input_ids"]) for _, _, p in jobs]
    print(f"[{a.model}] {len(rows)} 질의 x {len(conds)} 조건 = {len(jobs)} 생성.  프롬프트 토큰 max {max(lens)} / mean {np.mean(lens):.0f}"
          f"  (2048 초과 {sum(l > 2048 for l in lens)})", flush=True)
    order = np.argsort(lens); out = [None] * len(jobs); trs = [None] * len(jobs); t0 = time.perf_counter()
    for b in range(0, len(jobs), a.bs):
        ix = order[b:b + a.bs]
        texts, tr = generate(model, tok, [jobs[i][2] for i in ix], a.max_new, dream, a.steps or None, a.trace and dream)
        for k, i in enumerate(ix):
            out[i] = texts[k]
            if tr: trs[i] = tr[k]
        if (b // a.bs) % 25 == 0: print(f"  {b + len(ix)}/{len(jobs)}  {time.perf_counter() - t0:.0f}s", flush=True)
    name = a.model + (f"_{a.tag}" if a.tag else "") + (f"_s{a.steps or a.max_new}" if dream and a.data == "anchor" else "")
    if trs[0] is not None:
        np.savez_compressed(f"{RES}/{name}_trace.npz", argmax=np.stack([t[0] for t in trs]), masked=np.stack([t[1] for t in trs]),
                            final=np.stack([t[2] for t in trs]))
    with open(f"{RES}/{name}.jsonl", "w") as f:
        for (r, c, p), g, n in zip(jobs, out, lens):
            f.write(json.dumps(dict(qid=r["qid"], qi=r["qi"], cond=c, n_gold=len(r["gold"]), n_prompt_tok=n, raw=g, pred=first_line(g),
                                    answers=r["answers"]), ensure_ascii=False) + "\n")
    print(f"[{name}] done {time.perf_counter() - t0:.0f}s -> {RES}/{name}.jsonl", flush=True)


if __name__ == "__main__":
    main()
