"""E56 3 단계 (Qwen2.5-7B, Dream 의 AR 부모): greedy, 새 토큰 32, 같은 프롬프트 (common.build_prompt).
같은 길이 프롬프트끼리 묶어 패딩 없이 생성 (e55_ar.py 와 같은 방식).  답 = 첫 특수 토큰 앞까지 decode -> extract_answer.
첫 생성 위치 (= 한 번의 forward) 의 금 첫 토큰 순위 (s0_rank, 별칭 집합 최소 s0_rank_any), log 확률 (s0_lp, s0_lp_any), 금 답 토큰 수 (gold_len) 도 기록 (E57 층화용).
usage (repo 루트): python followup_mk/e56_qwen.py [--shard s --nshard n] [--limit N] [--bs 64]
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, "followup_mk")
from e56_common import load_items, first_tokens, groups, VERS
sys.path.insert(0, "data/1_raw")
from run_inference import build_prompt, extract_answer

MODEL = "Qwen/Qwen2.5-7B"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=64); args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    SP = set(tok.all_special_ids)
    items = load_items(); Q = {it["qid"]: it for it in items}
    R = [(it["qid"], v) for it in items for v in VERS]
    if args.limit: R = R[:: max(1, len(R) // args.limit)][: args.limit]
    pids = {r: tok(build_prompt(Q[r[0]][r[1]]))["input_ids"] for r in R}; ft = {q: first_tokens(tok, it["value"], it["aliases"]) for q, it in Q.items()}
    batches = list(groups(R, lambda r: pids[r], args.bs)); mine = [(j, L, ix) for j, (L, ix) in enumerate(batches) if j % args.nshard == args.shard]
    fn = f"followup_mk/results_e56_qwen{'_pilot' if args.limit else ''}_s{args.shard}.jsonl"; done = set()
    if os.path.exists(fn):
        for l in open(fn):
            try: d = json.loads(l); done.add((d["qid"], d["ver"]))
            except Exception: pass
    fo = open(fn, "a", encoding="utf-8"); t0 = time.time(); nb = 0
    print(f"shard {args.shard}/{args.nshard}: 묶음 {len(mine)}, 이미 {len(done)} 행", flush=True)
    for j, L, ix in mine:
        if all(r in done for r in ix): continue
        inp = torch.tensor([pids[r] for r in ix], device="cuda")
        with torch.inference_mode():
            o = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), max_new_tokens=32, do_sample=False, temperature=None, top_p=None, top_k=None,
                               pad_token_id=tok.eos_token_id, output_logits=True, return_dict_in_generate=True)
        z = o.logits[0].float(); lpz = torch.log_softmax(z, -1)
        for q_, r in enumerate(ix):
            if r in done: continue
            g, gs = ft[r[0]]; zr = z[q_]
            gen = o.sequences[q_, L:].tolist(); gen = gen[: next((p for p, t in enumerate(gen) if t in SP), len(gen))]
            a = extract_answer(tok.decode(gen, skip_special_tokens=True))
            fo.write(json.dumps(dict(qid=r[0], ver=r[1], batch=j, L=L, answer=a, gen=gen, gold_tok=g, s0_rank=int((zr > zr[g]).sum()), s0_rank_any=int(min((zr > zr[t]).sum().item() for t in gs)),
                                     s0_top1=int(zr.argmax()),
                                     s0_lp=float(lpz[q_, g]), s0_lp_any=float(max(lpz[q_, t].item() for t in gs)), gold_len=len(tok(" " + Q[r[0]]["value"].strip(), add_special_tokens=False)["input_ids"])), ensure_ascii=False) + "\n")
        fo.flush(); nb += 1
        if nb % 20 == 0: print(f"  {nb}/{len(mine)} 묶음, {time.time() - t0:.0f}s", flush=True)
    print(f"끝: {time.time() - t0:.0f}s", flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    main()
