"""Dream 생성 궤적 (PREREG_poison.md v5).  data/musique_bindpatch.jsonl 의 src / dst 를 patch 없이 생성하며, 단계마다 첫 답 칸을 기록.
생성 설정은 gen.py 와 같음 (2-shot, 16 토큰, steps 16, alg=entropy, temperature 0, 배치 1).
단계 i 마다 (갱신 전 상태): 첫 답 칸의 값 6개 첫 토큰 logits, 어휘 전체 argmax, 답 칸 16개의 mask 여부.
usage: CUDA_VISIBLE_DEVICES=0 python traj_bind.py [--data bindpatch] [--limit N]     출력: results/traj_dream_<data>.jsonl
"""
import argparse, json, os, time
import torch
from transformers import AutoTokenizer, AutoModel
from gen import MODELS, prompt, first_line

HERE = os.path.dirname(os.path.abspath(__file__)); VALS = ["aX", "aY", "aZ", "bX", "bY", "bZ"]; MAX_NEW = 16


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="bindpatch"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{HERE}/data/musique_{a.data}.jsonl")][: a.limit or None]
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]
    tok = AutoTokenizer.from_pretrained(MODELS["dream"], trust_remote_code=True)
    model = AutoModel.from_pretrained(MODELS["dream"], torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    out_f = open(f"{HERE}/results/traj_dream_{a.data}.jsonl", "w"); t0 = time.perf_counter()
    for qi, r in enumerate(rows):
        cid = [tok(" " + r["v"][k], add_special_tokens=False)["input_ids"][0] for k in VALS]
        for s in ("src", "dst"):
            x = tok([prompt(r["question"], [r["docs"][d] for d in r["ctx"][s]], shots)], return_tensors="pt").to("cuda"); tr = []
            def hook(i, xx, logits):                                          # logits 는 위치 정렬된 상태 (generation_utils._sample)
                lg = logits[0, -MAX_NEW].float()
                tr.append(dict(cand=[round(float(lg[c]), 2) for c in cid], argmax=int(lg.argmax()),
                               masked="".join("1" if t == tok.mask_token_id else "0" for t in xx[0, -MAX_NEW:].tolist())))
                return logits
            with torch.inference_mode():
                o = model.diffusion_generate(inputs=x["input_ids"], attention_mask=x["attention_mask"], max_new_tokens=MAX_NEW, steps=MAX_NEW,
                                             temperature=0.0, alg="entropy", mask_token_id=tok.mask_token_id,
                                             pad_token_id=tok.pad_token_id or tok.eos_token_id, generation_logits_hook_func=hook)
            seq = (o.sequences if hasattr(o, "sequences") else o)[0, -MAX_NEW:].tolist(); text = tok.decode(seq, skip_special_tokens=True)
            out_f.write(json.dumps(dict(qid=r["qid"], side=s, template=r["template"], cand=cid, final=seq, pred=first_line(text), steps=tr)) + "\n")
        out_f.flush()
        if qi % 20 == 0: print(f"  {qi + 1}/{len(rows)}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[traj dream {a.data}] done {len(rows)} 쌍, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
