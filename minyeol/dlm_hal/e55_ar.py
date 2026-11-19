"""E55 LLaMA-2-7B (NousResearch/Llama-2-7b-hf, DiffuLLaMA 의 AR 부모) greedy, 새 토큰 32, 같은 프롬프트 (build_prompt).
같은 길이 프롬프트끼리 묶어 패딩 없이 생성 (HF generate, do_sample=False).  답 = 첫 특수 토큰 앞까지 decode -> extract_answer, 같은 채점기.
첫 생성 위치 (= 답 칸 0) 의 금 첫 토큰 순위와 log 확률 (E57 식 층화용, 첫 forward 의 로짓) 도 기록 (DiffuLLaMA step 0 칸 0 과 비교용).
usage: python followup_mk/e55_ar.py [--shard i --nshard n] [--limit N] [--bs 64]
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, "followup_mk")
from e55_common import rows, build_prompt, grade, gold_first, groups, decode_answer, oom_retry

MODEL = "NousResearch/Llama-2-7b-hf"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=64); args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    SP = set(tok.all_special_ids)
    items = list(range(len(rows)))
    if args.limit: items = items[:: max(1, len(items) // args.limit)][: args.limit]
    pids = {i: tok(build_prompt(rows[i]["question"]))["input_ids"] for i in items}; gold = {i: gold_first(tok, i, pids[i]) for i in items}
    batches = list(groups(items, lambda i: pids[i], args.bs)); mine = [(j, L, ix) for j, (L, ix) in enumerate(batches) if j % args.nshard == args.shard]
    fn = f"followup_mk/results_e55_ar{'_pilot' if args.limit else ''}_s{args.shard}.jsonl"; done = set()
    if os.path.exists(fn):
        for l in open(fn):
            try: done.add(json.loads(l)["row"])
            except Exception: pass
    fo = open(fn, "a", encoding="utf-8"); t0 = time.time(); nb = 0
    print(f"shard {args.shard}/{args.nshard}: 묶음 {len(mine)}, 이미 {len(done)} 행", flush=True)
    for j, L, ix in mine:
        if all(i in done for i in ix): continue
        inp = torch.tensor([pids[i] for i in ix], device="cuda")
        def gen_():
            with torch.inference_mode():
                return model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), max_new_tokens=32, do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=tok.eos_token_id, output_logits=True, return_dict_in_generate=True)
        o = oom_retry(gen_)
        z = o.logits[0].float(); g = torch.tensor([max(gold[i], 0) for i in ix], device="cuda"); gv = z[torch.arange(len(ix), device="cuda"), g]
        rk = (z > gv[:, None]).sum(-1).tolist(); top = z.argmax(-1).tolist(); glp = torch.log_softmax(z, -1)[torch.arange(len(ix), device="cuda"), g].tolist()
        for q, i in enumerate(ix):
            if i in done: continue
            gen = o.sequences[q, L:].tolist(); a = decode_answer(tok, gen, SP)
            fo.write(json.dumps(dict(row=i, batch=j, L=L, gold_id=gold[i], answer=a, type=grade(i, a), gen=gen, s0_rank=int(rk[q]), s0_top1=int(top[q]), s0_gold_logp=round(glp[q], 4)), ensure_ascii=False) + "\n")
        fo.flush(); nb += 1
        if nb % 20 == 0: print(f"  {nb}/{len(mine)} 묶음, {time.time() - t0:.0f}s", flush=True)
    print(f"끝: {time.time() - t0:.0f}s", flush=True); print("DONE", flush=True)

if __name__ == "__main__": main()
