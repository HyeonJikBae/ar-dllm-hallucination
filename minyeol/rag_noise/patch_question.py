"""질문만 다른 쌍 patching (PREREG_poison.md v12).  문맥은 토큰까지 같고 질문 대상만 다르다: QA (정답 aX), QB (정답 aY).
usage: CUDA_VISIBLE_DEVICES=0 python patch_question.py --model qwen|dream [--limit N]     출력: results/qpair_<model>.jsonl
Qwen: 모든 쌍의 base, sand 첫 토큰을 보고, 네 입력을 다 맞힌 쌍의 sand 를 patching.  처음 20 쌍은 구조 확인용으로 base 도 patching.
Dream: results/qpair_qwen.jsonl 에서 Qwen 이 모두 맞힌 쌍만.  t0 (프롬프트 + mask 16) 한 시점.
"""
import argparse, json, os, time
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from gen import MODELS, prompt, HEAD_DOC, SEP
from gen_fmt import make_prompt
from patch_samefact import overlap, passage_span, LSEL

HERE = os.path.dirname(os.path.abspath(__file__)); MAX_NEW = 16


def sites(r, P, offs, ctx, sand):
    n = len(offs); fact, link = [], []
    for d in ("F_aX", "F_aY", "F_aZ"): fact += overlap(offs, *passage_span(P, r["docs"], ctx, d))
    for d in ("LA_s", "LB_s", "LC"): link += overlap(offs, *passage_span(P, r["docs"], ctx, d))
    qs = P.rfind("\n\nQuestion: ") + 2; qall = overlap(offs, qs, len(P)); cx = sorted(set(fact) | set(link))
    filler = overlap(offs, *passage_span(P, r["docs"], ctx, "r0")); fs = overlap(offs, len(HEAD_DOC), P.rfind(SEP))
    def ctrl(k):
        if len(filler) >= k: return filler[:k], "filler"
        m = len(fs) // 2 - k // 2; return fs[m:m + k], "fewshot"
    (cf, wf), (cc, wc) = ctrl(len(fact)), ctrl(len(cx))
    S = dict(fact=sorted(fact), link=sorted(link), ctx=cx, qall=qall, last=[n - 1], ctrl_f=cf, ctrl_c=cc)
    if sand:
        i = P.rfind(SEP) + len(SEP); j = P.find("\n\nPassage 1:", i); S["qfront"] = overlap(offs, i, j)
    return S, dict(ctrl_f=wf, ctrl_c=wc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(MODELS)); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); dream = a.model == "dream"
    _q = AutoTokenizer.from_pretrained(MODELS["qwen"]); _n = lambda t: len(_q(" " + t, add_special_tokens=False)["input_ids"])
    rows = [r for r in map(json.loads, open(f"{HERE}/data/musique_bindpatch.jsonl")) if _n(r["A"]) == _n(r["B"])][: a.limit or None]   # v12.1: A, B 토큰 수가 같은 쌍만 (160 / 360)
    if dream:
        qok = {r["qid"] for r in map(json.loads, open(f"{HERE}/results/qpair_qwen.jsonl")) if r["ok"]["base"] and r["ok"]["sand"]}
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

    path = f"{HERE}/results/qpair_{a.model}.jsonl"; done = set()
    if os.path.exists(path): done = {json.loads(l)["qid"] for l in open(path)}
    out_f = open(path, "a"); t0 = time.perf_counter(); n_ok = 0; n_sanity = 0
    for qi, r in enumerate(rows):
        if r["qid"] in done: continue
        qa = r["question"]; qb = qa.replace(r["A"], r["B"]); assert qa != qb
        ctx = r["ctx"]["src"]; docs = [r["docs"][d] for d in ctx]
        cid = [qtok(" " + r["v"][k], add_special_tokens=False)["input_ids"][0] for k in ("aX", "aY", "aZ")]; ans = {"QA": cid[0], "QB": cid[1]}
        rec = dict(qid=r["qid"], template=r["template"], model=a.model, cand=cid, ok={}, base={}, patch={}, dist={}, ctrl_from={})
        fmts = ["base"] if dream else ["base", "sand"]; X, Sx = {}, {}
        for fm in fmts:
            mkp = (lambda q: prompt(q, docs, shots)) if fm == "base" else (lambda q: make_prompt("sand", q, docs, shots, None))
            P = {"QA": mkp(qa), "QB": mkp(qb)}; enc = {s: qtok(P[s], return_offsets_mapping=True) for s in P}
            ids = {s: enc[s]["input_ids"] for s in P}; assert mtok(P["QA"])["input_ids"] == ids["QA"], "토크나이저 불일치"
            assert len(ids["QA"]) == len(ids["QB"]), (r["qid"], "길이 불일치")
            S, wh = sites(r, P["QA"], enc["QA"]["offset_mapping"], ctx, fm == "sand"); rec["ctrl_from"][fm] = wh
            diff = [i for i in range(len(ids["QA"])) if ids["QA"][i] != ids["QB"][i]]
            assert not (set(diff) & set(S["ctx"])), (r["qid"], "문맥 토큰이 다름")
            n = len(ids["QA"]); rec.setdefault("n", {})[fm] = n
            tail = [MID] * MAX_NEW if dream else []
            X[fm] = {s: torch.tensor([ids[s] + tail], device="cuda") for s in P}; Sx[fm] = (S, n)
            rec["base"][fm] = {s: int(run(X[fm][s], n - 1)[0].argmax()) for s in P}
            rec["ok"][fm] = all(rec["base"][fm][s] == ans[s] for s in P)
        todo = []
        if dream:
            if rec["ok"]["base"]: todo = ["base"]
        elif rec["ok"]["base"] and rec["ok"]["sand"]:
            todo = ["sand"] + (["base"] if n_sanity < 20 else [])
            if "base" in todo: n_sanity += 1
        if todo: n_ok += 1
        for fm in todo:
            S, n = Sx[fm]; caches = {}
            for s in ("QA", "QB"):
                st.update(mode="cap", cache={}); run(X[fm][s], n - 1); st["mode"] = None; caches[s] = st["cache"]
            rec["dist"][fm] = {}
            for site, pos in S.items():
                p = torch.tensor(pos, device="cuda")
                rec["dist"][fm][site] = [round(float(((caches["QA"][li][p] - caches["QB"][li][p]).float().norm(dim=-1) / caches["QA"][li][p].float().norm(dim=-1).clamp_min(1e-6)).mean()), 5) for li in LSEL]
            rec["patch"][fm] = {}
            for donor, recv in (("QA", "QB"), ("QB", "QA")):
                d = f"{donor}>{recv}"; rec["patch"][fm][d] = {}
                for site, pos in S.items():
                    st.update(mode="patch", cache=caches[donor], pos=torch.tensor(pos, device="cuda"))
                    rec["patch"][fm][d][site] = run(X[fm][recv].repeat(len(LSEL), 1), n - 1).argmax(-1).tolist(); st["mode"] = None
            del caches
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        if qi % 20 == 0: print(f"  {qi + 1}/{len(rows)}  patch {n_ok}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[qpair {a.model}] done {len(rows)} 쌍, patch {n_ok}, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
