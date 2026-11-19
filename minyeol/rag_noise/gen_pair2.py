"""다른 모델 쌍 생성 (PREREG_poison.md v14).  model: llama2, diffullama, lladamoe, dream (샘플러 확인용).
usage: CUDA_VISIBLE_DEVICES=1 python gen_pair2.py --model diffullama --data layout --every 2 [--conds a,b] [--shard i/n] [--bs 10]     출력: results/<model>_<data>_base[.sh<i>].jsonl
"""
import argparse, json, os, time
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaForCausalLM
from batchgen import gen_all
from gen import prompt, cond_docs, first_line, generate
from gen_fmt import select

HERE = os.path.dirname(os.path.abspath(__file__)); M = 16
NAMES = {"llama2": "NousResearch/Llama-2-7b-hf", "diffullama": "diffusionfamily/diffullama", "lladamoe": "inclusionAI/LLaDA-MoE-7B-A1B-Base", "dream": "Dream-org/Dream-v0-Base-7B"}


def load2(name):
    tok = AutoTokenizer.from_pretrained(NAMES[name], trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if name == "llama2": model = AutoModelForCausalLM.from_pretrained(NAMES[name], torch_dtype=torch.bfloat16)
    elif name == "diffullama": model = LlamaForCausalLM.from_pretrained(NAMES[name], torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    else: model = AutoModel.from_pretrained(NAMES[name], torch_dtype=torch.bfloat16, trust_remote_code=True)
    return tok, model.cuda().eval()


def full_hidden(name, model, x):
    """전체 attention forward.  [b, T, H]."""
    if name == "diffullama":                                                    # attention_patch.py 의 get_anneal_attn_mask(attn_mask_ratio=1.0) 와 같은 전부 0 마스크
        mask = torch.zeros(x.shape[0], 1, x.shape[1], x.shape[1], dtype=model.dtype, device=x.device)
        return model.model(input_ids=x, attention_mask=mask, use_cache=False)[0]
    if name == "dream": return model.model(input_ids=x, attention_mask="full", use_cache=False)[0]
    return model.model(input_ids=x, use_cache=False)[0]


@torch.inference_mode()
def slot_logits(name, model, x, n):
    """답 칸 M 개의 예측 logits [b, M, V].  Dream, DiffuLLaMA 는 한 칸 밀려 있고 LLaDA 는 같은 위치."""
    h = full_hidden(name, model, x); h = h[:, n:n + M] if name == "lladamoe" else h[:, n - 1:n + M - 1]
    return model.lm_head(h).float()


@torch.inference_mode()
def entropy_gen(name, model, tok, prompts):
    """temperature 0, entropy 순서, 한 걸음에 한 토큰.  Dream generation_utils 의 alg='entropy' 와 같은 식."""
    MID = tok.mask_token_id; ids = torch.tensor([tok(p)["input_ids"] for p in prompts], device="cuda"); b, n = ids.shape
    x = torch.cat([ids, torch.full((b, M), MID, device="cuda")], 1)
    for _ in range(M):
        lg = slot_logits(name, model, x, n); p = lg.softmax(-1); conf = (p * torch.log(p + 1e-10)).sum(-1); t = lg.argmax(-1)
        conf = conf.masked_fill(x[:, n:] != MID, -torch.inf); j = conf.argmax(1); r = torch.arange(b, device=x.device); x[r, n + j] = t[r, j]
    return tok.batch_decode(x[:, n:], skip_special_tokens=True)


@torch.inference_mode()
def l2r_gen(name, model, tok, prompts):
    """왼쪽부터 순차 확정, mask 16.  답 줄이 끝나면 멈춤 (앞쪽 줄바꿈은 건너뜀; LLaDA 는 "Answer:" 뒤에 줄바꿈부터 낸다)."""
    MID = tok.mask_token_id; ids = torch.tensor([tok(p)["input_ids"] for p in prompts], device="cuda"); b, n = ids.shape
    x = torch.cat([ids, torch.full((b, M), MID, device="cuda")], 1); out = [[] for _ in range(b)]; done = [False] * b; sp = set(tok.all_special_ids)
    for i in range(M):
        t = slot_logits(name, model, x, n)[:, i].argmax(-1); x[:, n + i] = t
        for k, v in enumerate(t.tolist()):
            if done[k]: continue
            if v in sp: done[k] = True; continue
            out[k].append(v)
            if "\n" in tok.decode(out[k]).lstrip(): done[k] = True
        if all(done): break
    return [tok.decode(o) for o in out]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--order", default="entropy", choices=["entropy", "l2r"]); ap.add_argument("--model", required=True, choices=list(NAMES)); ap.add_argument("--data", required=True); ap.add_argument("--conds", default="")
    ap.add_argument("--every", type=int, default=0); ap.add_argument("--shard", default=""); ap.add_argument("--bs", type=int, default=10); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per", type=int, default=0, help="틀마다 앞 per 문항만 (--every 적용 뒤)")
    a = ap.parse_args()
    rows = select([json.loads(l) for l in open(f"{HERE}/data/musique_{a.data}.jsonl")][: a.limit or None], a.every, a.shard, a.per)
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]; conds = a.conds.split(",") if a.conds else None
    sh = f".sh{a.shard.split('/')[0]}" if a.shard else ""; tag = "_check" if a.model == "dream" else ""; path = f"{HERE}/results/{a.model}_{a.data}_base{tag}{sh}.jsonl"; done = set()
    if os.path.exists(path): done = {(d["qid"], d["cond"]) for d in map(json.loads, open(path))}
    jobs = [(r, c, prompt(r["question"], cond_docs(r, c), shots)) for r in rows for c in (conds or r["ctx"]) if (r["qid"], c) not in done]
    print(f"[{a.model} {a.data}] {len(jobs)} 생성 (이미 {len(done)})", flush=True)
    if not jobs: return
    tok, model = load2(a.model); f = open(path, "a"); t0 = time.perf_counter()
    fn = (lambda ps: generate(model, tok, ps, M, False)[0]) if a.model == "llama2" else (lambda ps: (l2r_gen if a.order == "l2r" else entropy_gen)(a.model, model, tok, ps))
    def save(ix, texts):
        for i, t in zip(ix, texts):
            r, c, _ = jobs[i]; f.write(json.dumps(dict(qid=r["qid"], cond=c, raw=t, pred=first_line(t.lstrip() if a.model == "lladamoe" else t)), ensure_ascii=False) + "\n")
        f.flush()
    gen_all(model, tok, False, [p for _, _, p in jobs], bs=a.bs, log=a.model, on_batch=save, fn=fn)
    print(f"[{a.model} {a.data}] done {time.perf_counter() - t0:.0f}s, 프롬프트당 {(time.perf_counter() - t0) / len(jobs):.2f}s", flush=True)


if __name__ == "__main__":
    main()
