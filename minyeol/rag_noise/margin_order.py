"""첫 답 토큰 logit 차 (PREREG_poison.md v16).  v15 데이터의 S, T, C 조건.  usage: CUDA_VISIBLE_DEVICES=0 python margin_order.py --model qwen|dream [--limit N] [--bs 8]
출력: results/margin_order_<model>.jsonl (qid, cond, a, o, la, lo, top)   Dream 은 gen.generate 와 같은 호출을 steps=1 로 하고 hook 으로 첫 forward 의 답 칸 0 을 읽는다."""
import argparse, json, os
import torch
from batchgen import load, gen_all
from gen import prompt, cond_docs

HERE = os.path.dirname(os.path.abspath(__file__)); MAX_NEW = 16


def first(tok, v): return tok(" " + v, add_special_tokens=False)["input_ids"][0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["qwen", "dream"]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args(); rows = [json.loads(l) for l in open(f"{HERE}/data/musique_order.jsonl")][: a.limit or None]; shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]
    tok, model, dream = load(a.model); jobs = [(r["qid"], c, r["ans"][c], r["other"][c], prompt(r["question"], cond_docs(r, c), shots)) for r in rows for c in r["ctx"]]
    assert len({j[4] for j in jobs}) == len(jobs); by = {j[4]: j for j in jobs}

    @torch.inference_mode()
    def fn(ps):
        x = tok(ps, return_tensors="pt", padding=True).to(model.device); keep = []
        if dream:
            def hook(i, xx, logits):
                keep.append(logits[:, -MAX_NEW, :].float().cpu()); return logits
            model.diffusion_generate(inputs=x["input_ids"], attention_mask=x["attention_mask"], max_new_tokens=MAX_NEW, steps=1, temperature=0.0, alg="entropy", mask_token_id=tok.mask_token_id,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id, generation_logits_hook_func=hook)
            lg = keep[0]
        else:
            lg = model(**x).logits[:, -1, :].float().cpu()
        out = []
        for p, v in zip(ps, lg):
            q, c, ans, oth, _ = by[p]; i, j = first(tok, ans), first(tok, oth); t = int(v.argmax())
            out.append(dict(qid=q, cond=c, a=i, o=j, la=round(float(v[i]), 4), lo=round(float(v[j]), 4), top=t, ltop=round(float(v[t]), 4)))
        return out
    with open(f"{HERE}/results/margin_order_{a.model}.jsonl", "w") as f:
        def on_batch(ix, items):
            for it in items: f.write(json.dumps(it) + "\n")
            f.flush()
        gen_all(model, tok, dream, [j[4] for j in jobs], bs=a.bs, log=a.model, on_batch=on_batch, fn=fn)
    print("done", a.model, len(jobs), flush=True)


if __name__ == "__main__":
    main()
