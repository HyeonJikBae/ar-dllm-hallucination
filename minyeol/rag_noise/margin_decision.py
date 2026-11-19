"""Dream 결정 시점 logit 차 (PREREG_poison.md v16.1).  실제 생성 경로를 따라가다 정답과 다른 값의 토큰이 다른 칸이 처음 확정되는 단계의 logit 을 읽는다.  batch 전부 결정되면 멈춘다.
usage: CUDA_VISIBLE_DEVICES=0 python margin_decision.py [--limit N] [--bs 6] [--shard i/n]   출력: results/margin_dec_dream[.sh<i>].jsonl"""
import argparse, json, os
import torch
from batchgen import load, gen_all
from gen import prompt, cond_docs

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"; MAX_NEW = 16


class Done(Exception): pass


def keep_qids():
    from grade_layout import load_data, load_pred, s_ok, M
    data = load_data("order"); P = {m: load_pred(m, "order", "base") for m in M}; G = {m: {(r["qid"], r["cond"]): r for r in map(json.loads, open(f"{RES}/margin_order_{m}.jsonl"))} for m in M}
    return {q for q in data if all(s_ok(P[m], data, q) for m in M) and all(G[m][(q, c)]["a"] != G[m][(q, c)]["o"] for m in M for c in data[q]["ctx"])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=6); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(); si, sn = map(int, a.shard.split("/")); kq = keep_qids(); rows = [r for r in map(json.loads, open(f"{HERE}/data/musique_order.jsonl")) if r["qid"] in kq][: a.limit or None][si::sn]
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]; tok, model, dream = load("dream")
    ids = lambda v: tok(" " + v, add_special_tokens=False)["input_ids"]
    jobs = [(r["qid"], c, ids(r["ans"][c]), ids(r["other"][c]), prompt(r["question"], cond_docs(r, c), shots)) for r in rows for c in r["ctx"]]; by = {j[4]: j for j in jobs}
    assert len(by) == len(jobs); print("문항", len(rows), "프롬프트", len(jobs), flush=True)
    fname = f"{RES}/margin_dec_dream" + (f".sh{si}" if sn > 1 else "") + ".jsonl"

    @torch.inference_mode()
    def fn(ps):
        x = tok(ps, return_tensors="pt", padding=True).to(model.device); n = len(ps); disc = []; res = [None] * n; st = {"mask": None, "v": None, "i": None}
        for p in ps:
            _, _, A, O, _ = by[p]; disc.append({j: (A[j], O[j]) for j in range(min(len(A), len(O), MAX_NEW)) if A[j] != O[j]})
        def settle(mask_now, seq_now, i_prev):
            for b in range(n):
                if res[b] is not None: continue
                filled = [j for j in disc[b] if st["mask"][b, j] and not mask_now[b, j]]
                if filled:
                    j = filled[0]; la, lo = st["v"][b][j]; res[b] = dict(step=i_prev, slot=j, la=la, lo=lo, tok=int(seq_now[b, j]), a_j=disc[b][j][0], o_j=disc[b][j][1])
        def hook(i, xx, logits):
            tail = xx[:, -MAX_NEW:]; mask = tail == tok.mask_token_id
            if st["mask"] is not None:
                settle(mask, tail, st["i"])
                if all(r is not None for r in res): raise Done
            lg = logits[:, -MAX_NEW:, :].float()
            st["v"] = [{j: (round(float(lg[b, j, A]), 4), round(float(lg[b, j, O]), 4)) for j, (A, O) in disc[b].items()} for b in range(n)]; st["mask"] = mask.clone(); st["i"] = i if isinstance(i, int) else int(i)
            return logits
        try:
            out = model.diffusion_generate(inputs=x["input_ids"], attention_mask=x["attention_mask"], max_new_tokens=MAX_NEW, steps=MAX_NEW, temperature=0.0, alg="entropy", mask_token_id=tok.mask_token_id,
                                           pad_token_id=tok.pad_token_id or tok.eos_token_id, generation_logits_hook_func=hook)
            seq = (out.sequences if hasattr(out, "sequences") else out)[:, -MAX_NEW:]; settle(seq == tok.mask_token_id, seq, st["i"])   # 마지막 단계에서 결정된 경우
        except Done:
            pass
        return [dict(qid=by[p][0], cond=by[p][1], n_disc=len(disc[b]), **(res[b] or dict(step=None))) for b, p in enumerate(ps)]
    with open(fname, "w") as f:
        def on_batch(ix, items):
            for it in items: f.write(json.dumps(it) + "\n")
            f.flush()
        gen_all(model, tok, dream, [j[4] for j in jobs], bs=a.bs, log=f"dec{si}", on_batch=on_batch, fn=fn)
    print("done", len(jobs), flush=True)


if __name__ == "__main__":
    main()
