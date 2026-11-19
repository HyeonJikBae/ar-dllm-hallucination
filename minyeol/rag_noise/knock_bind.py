"""Dream attention knockout (PREREG_poison.md v7).  data/musique_bindswap.jsonl (S, U, U_sw, L, L_sw) 을 gen.py 와 같은 설정으로 생성하되,
모든 층의 attention 에 4차원 마스크를 넣는다 (self_attn forward pre-hook).
  none : 모든 칸 True (기준; knockout 과 같은 커널 경로)
  q    : 마지막 질문 앞의 모든 토큰 (few-shot 포함) -> 마지막 "Question:" 부터 답 칸 끝까지 를 보지 못함
  ctrl : 마지막 질문 앞의 토큰 (그 구간 자신 제외) -> 첫 few-shot "Question:" 부터 같은 토큰 수의 구간 을 보지 못함
usage: CUDA_VISIBLE_DEVICES=0 python knock_bind.py --ko none|q|ctrl [--limit N]     출력: results/knock_dream_<ko>.jsonl
"""
import argparse, json, os, time
import torch
from transformers import AutoTokenizer, AutoModel
from gen import MODELS, prompt, cond_docs, first_line

HERE = os.path.dirname(os.path.abspath(__file__)); MAX_NEW = 16


def tok_at(offs, c): return next(i for i, (s, e) in enumerate(offs) if e > c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ko", required=True, choices=["none", "q", "ctrl"]); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{HERE}/data/musique_bindswap.jsonl")][: a.limit or None]
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]
    qtok = AutoTokenizer.from_pretrained(MODELS["qwen"]); tok = AutoTokenizer.from_pretrained(MODELS["dream"], trust_remote_code=True)
    model = AutoModel.from_pretrained(MODELS["dream"], torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    st = {}
    def pre(mod, args, kwargs): kwargs["attention_mask"] = st["mask"]; return args, kwargs
    for layer in model.model.layers: layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True)
    out_f = open(f"{HERE}/results/knock_dream_{a.ko}.jsonl", "w"); t0 = time.perf_counter()
    for qi, r in enumerate(rows):
        for c in ("S", "U", "U_sw", "L", "L_sw"):
            P = prompt(r["question"], cond_docs(r, c), shots); enc = qtok(P, return_offsets_mapping=True); ids = enc["input_ids"]
            assert tok(P)["input_ids"] == ids
            n = len(ids); T = n + MAX_NEW; q0 = tok_at(enc["offset_mapping"], P.rfind("\n\nQuestion: ") + 2)
            m = torch.ones(1, 1, T, T, dtype=torch.bool, device="cuda")
            if a.ko == "q": m[0, 0, :q0, q0:] = False
            elif a.ko == "ctrl":
                c0 = tok_at(enc["offset_mapping"], P.find("\n\nQuestion: ") + 2); c1 = c0 + (T - q0); assert c1 < q0
                m[0, 0, :q0, c0:c1] = False; m[0, 0, c0:c1, c0:c1] = True
            st["mask"] = m
            x = torch.tensor([ids], device="cuda")
            with torch.inference_mode():
                o = model.diffusion_generate(inputs=x, attention_mask=None, max_new_tokens=MAX_NEW, steps=MAX_NEW, temperature=0.0, alg="entropy",
                                             mask_token_id=tok.mask_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            g = tok.decode((o.sequences if hasattr(o, "sequences") else o)[0, n:], skip_special_tokens=True)
            out_f.write(json.dumps(dict(qid=r["qid"], qi=r["qi"], cond=c, ko=a.ko, q0=q0, n_prompt_tok=n, raw=g, pred=first_line(g),
                                        answers=r["answers"]), ensure_ascii=False) + "\n")
        out_f.flush()
        if qi % 25 == 0: print(f"  {qi + 1}/{len(rows)}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[knock {a.ko}] done {len(rows)} 문항, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
