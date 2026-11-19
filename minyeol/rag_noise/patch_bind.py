"""연결 정보 / 최종 답 분리 patching (PREREG_poison.md v4).  data/musique_bindpatch{_pilot}.jsonl (build_bindpatch.py) 의 src / dst 쌍.

한 질의에서: src, dst 를 그대로 한 번씩 돌려 모든 층의 residual 을 저장하고, 첫 답 토큰으로 정답 여부를 본다.
두 모델 각자 src·dst 를 모두 맞힌 질의만 patching: 방향 2개 (src->dst, dst->src) x 자리 x 층 29개 (임베딩 + decoder 층 28개).
  층 l 개입 = 받는 실행의 층 l 출력 residual 을 그 자리에서 주는 실행의 값으로 바꿈 (층마다 배치 한 칸).
자리: link  = 연결 문장 A, B 의 사람 이름 토큰 (X <-> Y).  둘 다 바꿔야 임베딩 층 개입이 모순 없는 문맥 (A<-X, B<-Y + dst 사실) 이 됨
      qhead = 마지막 질문 속 머리 개체 A 의 토큰
      qall  = 마지막 "Question: ..." 부터 "Answer:" 끝까지
      last  = 프롬프트 마지막 토큰 (첫 답 토큰을 읽는 자리)
      masks = (Dream) 답 칸 mask 16개
읽기: Qwen 은 마지막 토큰의 다음 토큰 logits.  Dream 은 프롬프트 + mask 16개로 첫 디노이징 단계 한 번, 첫 답 칸의 logits
  (Dream 은 logits 를 한 칸 밀어 쓰므로 둘 다 프롬프트 마지막 위치의 hidden 에서 읽음).  어휘 전체 argmax 가 값 6개의 첫 토큰 중 무엇인지 기록.
usage: CUDA_VISIBLE_DEVICES=0 python patch_bind.py --model qwen|dream --data bindpatch_pilot|bindpatch [--limit N]
출력: results/patch_<model>_<data>.jsonl
"""
import argparse, json, os, time
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from gen import MODELS, prompt

HERE = os.path.dirname(os.path.abspath(__file__)); VALS = ["aX", "aY", "aZ", "bX", "bY", "bZ"]; MAX_NEW = 16


def overlap(offs, a, b): return [i for i, (s, e) in enumerate(offs) if s < b and e > a]


def sites(r, P, offs, name, dream):
    """자리 이름 -> 토큰 위치 목록 (P = 프롬프트 문자열, name = 대상 연결 문장의 사람 이름)."""
    qs = P.rfind("\n\nQuestion: ") + 2; a0 = qs + len("Question: ") + r["question"].index(r["A"]); n = len(offs)
    link = []
    for d, nm in ((("LA_s", r["X"]), ("LB_s", r["Y"])) if name == r["X"] else (("LA_d", r["Y"]), ("LB_d", r["X"]))):
        t = r["docs"][d]["text"]; k0 = P.rfind(t) + t.rindex(nm); link += overlap(offs, k0, k0 + len(nm))
    out = dict(link=sorted(link), qhead=overlap(offs, a0, a0 + len(r["A"])),
               qall=overlap(offs, qs, len(P)), last=[n - 1])
    if dream: out["masks"] = list(range(n, n + MAX_NEW))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--data", default="bindpatch_pilot"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); dream = a.model == "dream"
    rows = [json.loads(l) for l in open(f"{HERE}/data/musique_{a.data}.jsonl")][: a.limit or None]
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]
    qtok = AutoTokenizer.from_pretrained(MODELS["qwen"])                            # offset 계산용 (Dream 토크나이저와 id 가 같은지 확인)
    mtok = AutoTokenizer.from_pretrained(MODELS[a.model], trust_remote_code=True)
    if dream: model = AutoModel.from_pretrained(MODELS[a.model], torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    else: model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], torch_dtype=torch.bfloat16).cuda().eval()
    base = model.model; layers = [base.embed_tokens] + list(base.layers); NL = len(layers)
    st = {"mode": None}

    def mk(li):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if st["mode"] == "cap": st["cache"][li] = hs[0].clone()
            elif st["mode"] == "patch": hs[li, st["pos"]] = st["cache"][li][st["pos"]]
        return hook
    for li, m in enumerate(layers): m.register_forward_hook(mk(li))

    @torch.inference_mode()
    def run(x, readpos):
        h = base(input_ids=x, attention_mask="full" if dream else None, use_cache=False)[0]
        return model.lm_head(h[:, readpos]).float()

    out_f = open(f"{HERE}/results/patch_{a.model}_{a.data}.jsonl", "w"); t0 = time.perf_counter(); n_ok = 0
    for qi, r in enumerate(rows):
        cand = {k: qtok(" " + r["v"][k], add_special_tokens=False)["input_ids"][0] for k in VALS}; cid = [cand[k] for k in VALS]
        X, S, caches, rec = {}, {}, {}, dict(qid=r["qid"], template=r["template"], model=a.model, cand=cid)
        for s, name in (("src", r["X"]), ("dst", r["Y"])):
            P = prompt(r["question"], [r["docs"][d] for d in r["ctx"][s]], shots)
            enc = qtok(P, return_offsets_mapping=True); ids = enc["input_ids"]
            assert mtok(P)["input_ids"] == ids, "토크나이저 불일치"
            S[s] = sites(r, P, enc["offset_mapping"], name, dream)
            X[s] = torch.tensor([ids + ([mtok.mask_token_id] * MAX_NEW if dream else [])], device="cuda"); n = len(ids)
            st.update(mode="cap", cache={}); lg = run(X[s], n - 1)[0]; st["mode"] = None; caches[s] = st["cache"]
            rec[f"base_{s}"] = dict(argmax=int(lg.argmax()), cand=[round(float(lg[c]), 2) for c in cid])
        assert X["src"].shape == X["dst"].shape and S["src"] == S["dst"], (r["qid"], "위치 불일치")
        rec["n_diff_tok"] = int((X["src"] != X["dst"]).sum())
        ok = rec["base_src"]["argmax"] == cand["aX"] and rec["base_dst"]["argmax"] == cand["bY"]; rec["ok"] = ok
        if ok:
            n_ok += 1
            for d, (donor, recv) in (("s2d", ("src", "dst")), ("d2s", ("dst", "src"))):
                rec[d] = {}
                for site, pos in S["src"].items():
                    st.update(mode="patch", cache=caches[donor], pos=torch.tensor(pos, device="cuda"))
                    lg = run(X[recv].repeat(NL, 1), n - 1); st["mode"] = None
                    rec[d][site] = dict(argmax=lg.argmax(-1).tolist(), cand=[[round(float(v), 2) for v in row] for row in lg[:, cid]])
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        if qi % 10 == 0: print(f"  {qi + 1}/{len(rows)}  ok {n_ok}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[{a.model} {a.data}] done {len(rows)} 질의, src·dst 모두 정답 {n_ok}, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
