"""같은 사실 donor 구간 patching (PREREG_poison.md v6).  data/musique_bindpatch.jsonl 의 src 사실로 연결만 다른 두 입력을 만든다.
  I1 = A<-X, B<-Y (정답 aX)   I2 = A<-Y, B<-X (정답 aY)   사실 문단 (X->aX, Y->aY, Z->aZ) 과 순서는 같음.
자리: fact, qall, last, fq (= fact + qall), ctrl_f / ctrl_fq (같은 토큰 수의 무관한 구간).  층: 임베딩 + decoder 0,2,..,26,27.
Qwen: 한 번의 forward.  Dream: t0 (프롬프트 + mask 16) 와 tc (받는 입력의 첫 답 칸이 확정되는 단계의 갱신 전 화면; 주는 입력은 같은 단계 번호의 화면).
usage: CUDA_VISIBLE_DEVICES=0 python patch_samefact.py --model qwen|dream [--limit N]     출력: results/samefact_<model>.jsonl
Dream 은 results/samefact_qwen.jsonl 에서 Qwen 이 I1, I2 를 모두 맞힌 쌍만 patching 한다.
"""
import argparse, json, os, time
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from gen import MODELS, prompt, HEAD_DOC, SEP

HERE = os.path.dirname(os.path.abspath(__file__)); VALS = ["aX", "aY", "aZ", "bX", "bY", "bZ"]; MAX_NEW = 16
LSEL = [0] + list(range(1, 28, 2)) + [28]                                  # 배열 번호 (0 = 임베딩, l+1 = decoder l)


def overlap(offs, a, b): return [i for i, (s, e) in enumerate(offs) if s < b and e > a]


def passage_span(P, docs, ids, d):
    i = ids.index(d); s = f"Passage {i + 1}: {docs[d]['title']}\n{docs[d]['text']}"; a = P.rfind(s); return a, a + len(s)


def sites(r, P, offs, ctx):
    n = len(offs); fact = []
    for d in ("F_aX", "F_aY", "F_aZ"): fact += overlap(offs, *passage_span(P, r["docs"], ctx, d))
    qs = P.rfind("\n\nQuestion: ") + 2; qall = overlap(offs, qs, len(P)); fq = sorted(set(fact) | set(qall))
    filler = overlap(offs, *passage_span(P, r["docs"], ctx, "r0"))
    fs = overlap(offs, len(HEAD_DOC), P.rfind(SEP))                          # few-shot 영역
    def ctrl(k):
        if len(filler) >= k: return filler[:k], "filler"
        m = len(fs) // 2 - k // 2; return fs[m:m + k], "fewshot"
    (cf, wf), (cfq, wfq) = ctrl(len(fact)), ctrl(len(fq))
    return dict(fact=sorted(fact), qall=qall, last=[n - 1], fq=fq, ctrl_f=cf, ctrl_fq=cfq), dict(ctrl_f=wf, ctrl_fq=wfq)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(MODELS)); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); dream = a.model == "dream"
    rows = [json.loads(l) for l in open(f"{HERE}/data/musique_bindpatch.jsonl")][: a.limit or None]
    if dream:
        qok = {r["qid"] for r in map(json.loads, open(f"{HERE}/results/samefact_qwen.jsonl")) if r["ok"]["ar"]}
        rows = [r for r in rows if r["qid"] in qok]
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]
    qtok = AutoTokenizer.from_pretrained(MODELS["qwen"]); mtok = AutoTokenizer.from_pretrained(MODELS[a.model], trust_remote_code=True)
    if dream: model = AutoModel.from_pretrained(MODELS[a.model], torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    else: model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], torch_dtype=torch.bfloat16).cuda().eval()
    base = model.model; layers = [base.embed_tokens] + list(base.layers); st = {"mode": None}; MID = mtok.mask_token_id if dream else None

    def mk(li):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if st["mode"] == "cap": st["cache"][li] = hs[0].clone()
            elif st["mode"] == "patch" and li in LSEL: hs[LSEL.index(li), st["pos"]] = st["cache"][li][st["pos"]]
        return hook
    for li, m in enumerate(layers): m.register_forward_hook(mk(li))

    @torch.inference_mode()
    def run(x, readpos):
        h = base(input_ids=x, attention_mask="full" if dream else None, use_cache=False)[0]
        return model.lm_head(h[:, readpos]).float()

    @torch.inference_mode()
    def traj(x):
        """Dream 생성 (gen.py 설정): 최종 답 칸 토큰과 단계별 (갱신 전) mask 여부 [steps, 16]."""
        tr = []
        def hook(i, xx, logits): tr.append(xx[0, -MAX_NEW:] == MID); return logits
        o = model.diffusion_generate(inputs=x, attention_mask=torch.ones_like(x), max_new_tokens=MAX_NEW, steps=MAX_NEW, temperature=0.0,
                                     alg="entropy", mask_token_id=MID, pad_token_id=mtok.pad_token_id or mtok.eos_token_id, generation_logits_hook_func=hook)
        return (o.sequences if hasattr(o, "sequences") else o)[0, -MAX_NEW:], torch.stack(tr)

    out_f = open(f"{HERE}/results/samefact_{a.model}.jsonl", "w"); t0 = time.perf_counter(); n_ok = 0
    for qi, r in enumerate(rows):
        cid = [qtok(" " + r["v"][k], add_special_tokens=False)["input_ids"][0] for k in VALS]
        ctx = {"I1": r["ctx"]["src"], "I2": ["LA_d" if d == "LA_s" else "LB_d" if d == "LB_s" else d for d in r["ctx"]["src"]]}
        ans = {"I1": cid[0], "I2": cid[1]}; ids, S = {}, {}
        for s in ("I1", "I2"):
            P = prompt(r["question"], [r["docs"][d] for d in ctx[s]], shots); enc = qtok(P, return_offsets_mapping=True); ids[s] = enc["input_ids"]
            assert mtok(P)["input_ids"] == ids[s], "토크나이저 불일치"
            S[s], where = sites(r, P, enc["offset_mapping"], ctx[s])
        assert len(ids["I1"]) == len(ids["I2"]) and S["I1"] == S["I2"], (r["qid"], "위치 불일치")
        n = len(ids["I1"]); S = S["I1"]
        diff = [i for i in range(n) if ids["I1"][i] != ids["I2"][i]]; assert not set(diff) & set(S["fact"]), "사실 구간이 다름"
        rec = dict(qid=r["qid"], template=r["template"], model=a.model, cand=cid, ctrl_from=where, n_site={k: len(v) for k, v in S.items()},
                   la_slot=r["slots"]["la"], p_slot=r["slots"]["p"], q_slot=r["slots"]["q"], ok={}, base={}, patch={})
        canv = {}                                                          # canv[시점][받는 입력][입력] = 화면
        if dream:
            seq, mk_ = {}, {}
            for s in ("I1", "I2"): seq[s], mk_[s] = traj(torch.tensor([ids[s]], device="cuda"))
            rec["commit"] = {s: int(mk_[s][:, 0].sum()) - 1 for s in ("I1", "I2")}
            x0 = {s: torch.tensor([ids[s] + [MID] * MAX_NEW], device="cuda") for s in ("I1", "I2")}
            canv["t0"] = {recv: x0 for recv in ("I1", "I2")}
            canv["tc"] = {recv: {s: torch.cat([torch.tensor(ids[s]), torch.where(mk_[s][rec["commit"][recv]].cpu(), MID, seq[s].cpu())])[None].cuda()
                                 for s in ("I1", "I2")} for recv in ("I1", "I2")}
        else:
            xa = {s: torch.tensor([ids[s]], device="cuda") for s in ("I1", "I2")}; canv["ar"] = {recv: xa for recv in ("I1", "I2")}
        times = ["t0", "tc"] if dream else ["ar"]
        for tm in times:
            rec["base"][tm] = {recv: int(run(canv[tm][recv][recv], n - 1)[0].argmax()) for recv in ("I1", "I2")}
            rec["ok"][tm] = all(rec["base"][tm][recv] == ans[recv] for recv in ("I1", "I2"))
        if all(rec["ok"].values()):
            n_ok += 1
            for tm in times:
                rec["patch"][tm] = {}
                for recv, donor in (("I2", "I1"), ("I1", "I2")):
                    st.update(mode="cap", cache={}); run(canv[tm][recv][donor], n - 1); st["mode"] = None; cache = st["cache"]
                    d = f"{donor}>{recv}"; rec["patch"][tm][d] = {}
                    for site, pos in S.items():
                        st.update(mode="patch", cache=cache, pos=torch.tensor(pos, device="cuda"))
                        rec["patch"][tm][d][site] = run(canv[tm][recv][recv].repeat(len(LSEL), 1), n - 1).argmax(-1).tolist(); st["mode"] = None
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        if qi % 20 == 0: print(f"  {qi + 1}/{len(rows)}  ok {n_ok}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[samefact {a.model}] done {len(rows)} 쌍, 모두 정답 {n_ok}, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
