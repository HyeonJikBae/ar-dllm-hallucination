"""Dream 생성 조건 변화 (PREREG_poison.md v13).  양방향 attention 은 그대로 두고 확정 순서, 반복 횟수, mask 수만 바꾼다.
arm: l2r_m8, l2r_m16, l2r_m64 (왼쪽부터 순차 확정; 16 토큰까지, 줄바꿈이나 특수 토큰이 나오면 멈춤), ent_s1, ent_s4 (entropy, mask 16)
usage: CUDA_VISIBLE_DEVICES=0 python gen_dream_var.py --data layout --arm l2r_m16 [--conds a,b] [--every 2] [--shard i/n] [--bs 10]
출력: results/dream_<data>_<arm>[.sh<i>].jsonl
"""
import argparse, json, os, time
import torch
from batchgen import load, gen_all
from gen import prompt, cond_docs, first_line, generate
from gen_fmt import select

HERE = os.path.dirname(os.path.abspath(__file__)); MAX_TOK = 16


@torch.inference_mode()
def l2r(model, tok, prompts, m):
    """같은 길이 프롬프트 묶음.  위치 p 의 예측은 hidden p-1 에서 읽는다 (gen.py 의 logits 정렬과 같음)."""
    MID = tok.mask_token_id; ids = torch.tensor([tok(p)["input_ids"] for p in prompts], device="cuda"); b, n = ids.shape
    x = torch.cat([ids, torch.full((b, m), MID, device="cuda")], 1); out = [[] for _ in range(b)]; done = [False] * b; special = set(tok.all_special_ids)
    for i in range(min(m, MAX_TOK)):
        h = model.model(input_ids=x, attention_mask="full", use_cache=False)[0]
        t = model.lm_head(h[:, n + i - 1]).float().argmax(-1); x[:, n + i] = t
        for k, v in enumerate(t.tolist()):
            if done[k]: continue
            if v in special: done[k] = True; continue
            out[k].append(v)
            if "\n" in tok.decode(out[k]): done[k] = True
        if all(done): break
    return [tok.decode(o) for o in out]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--conds", default="")
    ap.add_argument("--every", type=int, default=0); ap.add_argument("--shard", default=""); ap.add_argument("--bs", type=int, default=10); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = select([json.loads(l) for l in open(f"{HERE}/data/musique_{a.data}.jsonl")][: a.limit or None], a.every, a.shard)
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]; conds = a.conds.split(",") if a.conds else None
    sh = f".sh{a.shard.split('/')[0]}" if a.shard else ""; path = f"{HERE}/results/dream_{a.data}_{a.arm}{sh}.jsonl"; done = set()
    if os.path.exists(path): done = {(d["qid"], d["cond"]) for d in map(json.loads, open(path))}
    jobs = [(r, c, prompt(r["question"], cond_docs(r, c), shots)) for r in rows for c in (conds or r["ctx"]) if (r["qid"], c) not in done]
    print(f"[dream {a.data} {a.arm}] {len(jobs)} 생성 (이미 {len(done)})", flush=True)
    if not jobs: return
    tok, model, _ = load("dream"); f = open(path, "a"); t0 = time.perf_counter()
    if a.arm.startswith("l2r_m"):
        m = int(a.arm[5:]); fn = lambda ps: l2r(model, tok, ps, m)
    else:
        s = int(a.arm[5:]); fn = lambda ps: generate(model, tok, ps, 16, True, s)[0]
    def save(ix, texts):
        for i, t in zip(ix, texts):
            r, c, _ = jobs[i]; f.write(json.dumps(dict(qid=r["qid"], cond=c, raw=t, pred=first_line(t)), ensure_ascii=False) + "\n")
        f.flush()
    gen_all(model, tok, True, [p for _, _, p in jobs], bs=a.bs, log=f"dream/{a.arm}", on_batch=save, fn=fn)
    print(f"[dream {a.data} {a.arm}] done {time.perf_counter() - t0:.0f}s, 프롬프트당 {(time.perf_counter() - t0) / len(jobs):.2f}s", flush=True)


if __name__ == "__main__":
    main()
