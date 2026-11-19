"""v14 코드 확인.  (1) 공통 샘플러로 돌린 Dream 답 = 기존 생성 답?  (2) DiffuLLaMA 전체 attention?  (3) 세 모델이 답을 내는가?"""
import json, torch
from gen import prompt, cond_docs, first_line
from gen_pair2 import load2, entropy_gen, full_hidden
rows = [json.loads(l) for l in open("data/musique_layout_pilot.jsonl")][:12]; shots = json.load(open("data/fewshot.json"))[:2]
ref = {(r["qid"], r["cond"]): r["pred"] for r in map(json.loads, open("results/dream_layout_pilot_base.jsonl"))}
tok, model = load2("dream"); same = tot = 0
for r in rows:
    cs = ["L1", "L1_sw", "L7", "L7_sw", "S_e", "S_l"]
    for gr in (cs[:4], cs[4:]):
        P = [prompt(r["question"], cond_docs(r, c), shots) for c in gr]
        if len({len(tok(p)["input_ids"]) for p in P}) > 1: continue
        for c, t in zip(gr, entropy_gen("dream", model, tok, P)): same += first_line(t) == ref[(r["qid"], c)]; tot += 1
print(f"공통 샘플러 Dream = 기존 생성: {same}/{tot}", flush=True); del model; torch.cuda.empty_cache()
tok, model = load2("diffullama"); x = torch.tensor([tok("The capital of France is Paris and the capital of Italy is Rome.")["input_ids"]], device="cuda")
with torch.inference_mode():
    h1 = full_hidden("diffullama", model, x).float(); y = x.clone(); y[0, -2] = tok.mask_token_id; h2 = full_hidden("diffullama", model, y).float()
print("DiffuLLaMA 뒤 토큰을 바꿨을 때 첫 토큰 hidden 변화 (0 이면 인과):", float((h1[0, 1] - h2[0, 1]).norm() / h1[0, 1].norm()), " mask id", tok.mask_token_id, flush=True)
r = rows[0]; P = [prompt(r["question"], cond_docs(r, c), shots) for c in ("L1", "L4", "L7")]
print("DiffuLLaMA", [first_line(t) for t in entropy_gen("diffullama", model, tok, P)], "정답", [r["ans"][c] for c in ("L1", "L4", "L7")], flush=True)
Se = [prompt(q["question"], cond_docs(q, "S_e"), shots) for q in rows[:6]]; acc = 0
for q, p in zip(rows[:6], Se): acc += q["ans"]["S_e"] in first_line(entropy_gen("diffullama", model, tok, [p])[0])
print("DiffuLLaMA S_e 포함", acc, "/ 6", flush=True); del model; torch.cuda.empty_cache()
tok, model = load2("lladamoe"); print("LLaDA mask id", tok.mask_token_id, "길이", len(tok(P[0])["input_ids"]), flush=True)
print("LLaDA", [first_line(t) for t in entropy_gen("lladamoe", model, tok, P)], flush=True); acc = 0
for q, p in zip(rows[:6], Se): acc += q["ans"]["S_e"] in first_line(entropy_gen("lladamoe", model, tok, [p])[0])
print("LLaDA S_e 포함", acc, "/ 6", flush=True); del model; torch.cuda.empty_cache()
from transformers import AutoTokenizer
tok, model = load2("llama2"); from gen import generate
print("Llama-2", [first_line(t) for t in generate(model, tok, P, 16, False)[0]], flush=True); acc = 0
for q, p in zip(rows[:6], Se): acc += q["ans"]["S_e"] in first_line(generate(model, tok, [p], 16, False)[0][0])
print("Llama-2 S_e 포함", acc, "/ 6", flush=True)
