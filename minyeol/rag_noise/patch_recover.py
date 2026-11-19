"""Qwen 오류 복구 patching (PREREG_poison.md v9).  data/musique_recover.jsonl (build_recover.py).
받는 입력 = v3 에서 Qwen 이 다른 값을 고른 경우, 주는 입력 = 값 두 개만 바꾼 같은 배치.  자리 last, qall x 층 29개 (임베딩 + 28).
usage: CUDA_VISIBLE_DEVICES=0 python patch_recover.py [--limit N]     출력: results/recover_qwen.jsonl
"""
import argparse, json, os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gen import MODELS, prompt, cond_docs
from build_recover import donor_docs

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_bindswap.jsonl")}
    cases = [json.loads(l) for l in open(f"{HERE}/data/musique_recover.jsonl")][: a.limit or None]
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]
    tok = AutoTokenizer.from_pretrained(MODELS["qwen"]); model = AutoModelForCausalLM.from_pretrained(MODELS["qwen"], torch_dtype=torch.bfloat16).cuda().eval()
    base = model.model; layers = [base.embed_tokens] + list(base.layers); NL = len(layers); st = {"mode": None}
    first = lambda v: tok(" " + v, add_special_tokens=False)["input_ids"][0]

    def mk(li):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if st["mode"] == "cap": st["cache"][li] = hs[0].clone()
            elif st["mode"] == "patch": hs[li, st["pos"]] = st["cache"][li][st["pos"]]
        return hook
    for li, m in enumerate(layers): m.register_forward_hook(mk(li))

    @torch.inference_mode()
    def run(x):
        return model.lm_head(base(input_ids=x, use_cache=False)[0][:, -1]).float()

    out_f = open(f"{HERE}/results/recover_qwen.jsonl", "w"); t0 = time.perf_counter()
    for i, cs in enumerate(cases):
        r = data[cs["qid"]]; c = cs["cond"]; sw = c.endswith("_sw")
        P = prompt(r["question"], cond_docs(r, c), shots); enc = tok(P, return_offsets_mapping=True); x_r = torch.tensor([enc["input_ids"]], device="cuda")
        qs = P.rfind("\n\nQuestion: ") + 2; qall = [k for k, (s, e) in enumerate(enc["offset_mapping"]) if e > qs]
        ans_r, oth_r = (r["V2"], r["V1"]) if sw else (r["V1"], r["V2"])
        rec = dict(qid=cs["qid"], cond=c, template=cs["template"], recv_argmax=int(run(x_r)[0].argmax()), ans=first(ans_r), oth=first(oth_r), donors=[])
        rec["same_first"] = rec["ans"] == rec["oth"]                       # 원래 v3 값은 첫 토큰 구분을 강제하지 않았음 -> 제외
        rec["recv_err"] = rec["recv_argmax"] == rec["oth"] and not rec["same_first"]
        for dn in (cs["donors"] if rec["recv_err"] else []):
            rr = dict(r, docs=donor_docs(r, dn["V1"], dn["V2"])); Pd = prompt(r["question"], cond_docs(rr, c), shots)
            x_d = torch.tensor([tok(Pd)["input_ids"]], device="cuda"); assert x_d.shape == x_r.shape
            ans_d, oth_d = (dn["V2"], dn["V1"]) if sw else (dn["V1"], dn["V2"])
            st.update(mode="cap", cache={}); lg = run(x_d)[0]; st["mode"] = None
            d = dict(ans=first(ans_d), oth=first(oth_d), argmax=int(lg.argmax())); d["ok"] = d["argmax"] == d["ans"]
            if d["ok"] and rec["recv_err"]:
                cache = st["cache"]
                for site, pos in (("last", [x_r.shape[1] - 1]), ("qall", qall)):
                    st.update(mode="patch", cache=cache, pos=torch.tensor(pos, device="cuda"))
                    d[site] = run(x_r.repeat(NL, 1)).argmax(-1).tolist(); st["mode"] = None
            rec["donors"].append(d)
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        if i % 20 == 0: print(f"  {i + 1}/{len(cases)}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[recover qwen] done {len(cases)} 경우, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
