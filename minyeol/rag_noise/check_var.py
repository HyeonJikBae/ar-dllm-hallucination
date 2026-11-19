"""v13 코드 확인: l2r 의 첫 토큰이 entropy 생성의 첫 단계 argmax 와 같은지, ent_s1 / ent_s4 / m8 / m64 가 도는지."""
import json, torch
from batchgen import load
from gen import prompt, cond_docs, first_line, generate
from gen_dream_var import l2r
rows = [json.loads(l) for l in open("data/musique_layout_pilot.jsonl")][:6]; shots = json.load(open("data/fewshot.json"))[:2]
tok, model, _ = load("dream")
same = tot = 0; agree = 0
for r in rows:
    P = [prompt(r["question"], cond_docs(r, c), shots) for c in ("L1", "L1_sw", "L4", "L4_sw", "L7", "L7_sw")]
    _, tr = generate(model, tok, P, 16, True, None, trace=True)
    canon, _ = generate(model, tok, P, 16, True)
    out = l2r(model, tok, P, 16)
    for k, p in enumerate(P):
        a0 = int(tr[k][0][0, 0]); t = tok(out[k], add_special_tokens=False)["input_ids"]; lead = tok.decode([a0])
        same += out[k].startswith(lead); tot += 1; agree += first_line(out[k]) == first_line(canon[k])
print(f"l2r 첫 토큰 = entropy 첫 단계 argmax: {same}/{tot},  l2r 답 = entropy 답: {agree}/{tot}")
P = [prompt(rows[0]["question"], cond_docs(rows[0], c), shots) for c in ("L1", "L4")]
for m in (8, 64): print("m", m, [first_line(x) for x in l2r(model, tok, P, m)])
for s in (1, 4): print("steps", s, [first_line(x) for x in generate(model, tok, P, 16, True, s)[0]])
print("정답", [rows[0]["ans"][c] for c in ("L1", "L4")])
