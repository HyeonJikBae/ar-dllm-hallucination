"""E52 sanity (GPU, 작은 규모):
 1) 패치한 모델 + e52_lib.generate (base, 주의 기록 켬/끔) 가 E44 (원본 diffusion_generate, 같은 묶음 구성) 와 답·칸 0 금 순위가 같은지.
 2) 빈 막기 (empty) 가 base 와 비트 단위로 같은지, empty_eager (eager 로 덮어쓰기만) 의 수치 잡음 크기.
 3) step0 DLA 선형화 점검: s * (u_g - u_c) . h_final 대 실제 로짓 차.
usage: python followup_mk/e52_sanity.py [--nb 6]"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
import e52_lib as E
NB = int(sys.argv[sys.argv.index("--nb") + 1]) if "--nb" in sys.argv else 6
m = Dream(); E.patch(m); tok = m.tok
_, Z7, F7 = e7_rows()
R44 = [int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]
ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in R44}
gold = {i: tok(" " + rows[i]["gold_value"], add_special_tokens=False)["input_ids"][0] for i in R44}
B44 = list(by_length(R44, lambda i: ids[i]))
sh = {}
for s in range(3):
    D = f"{OUT}/e44/shard{s}"; rr = np.load(f"{D}/rows.npy"); S0R = np.load(f"{D}/slot0_gold_rank.npy", mmap_mode="r"); S0M = np.load(f"{D}/slot0_mask.npy", mmap_mode="r")
    meta = {json.loads(l)["row"]: json.loads(l) for l in open(f"{D}/meta.jsonl", encoding="utf-8")}
    sh[s] = (rr, S0R, S0M, meta)
out = {"e44": [], "empty": [], "empty_eager": [], "cap_vs_nocap": [], "dla": []}
t0 = time.time()
for j in range(NB):
    L, ix = B44[j]; s = j % 3; rr, S0R, S0M, meta = sh[s]
    off = sum(len(B44[q][1]) for q in range(s, j, 3)); assert list(rr[off:off + len(ix)]) == ix, "E44 행 순서 불일치"
    g_ = [gold[i] for i in ix]
    t1 = time.time(); rb = E.generate(m, [ids[i] for i in ix], g_, cap=True); tb = time.time() - t1
    rn = E.generate(m, [ids[i] for i in ix], g_, cap=False)
    re_ = E.generate(m, [ids[i] for i in ix], g_, knock="empty")
    t1 = time.time(); ree = E.generate(m, [ids[i] for i in ix], g_, knock="empty_eager"); te = time.time() - t1
    for q, i in enumerate(ix):
        e_ans = meta[i]["answer"]; msk = S0M[off + q]; e_rank = S0R[off + q]
        my = rb["rank"][q]; okr = all((my[t] == e_rank[t]) for t in range(STEPS) if msk[t]) and all((my[t] == -2) for t in range(STEPS) if not msk[t])
        out["e44"].append(dict(row=i, ans_eq=rb["answers"][q] == e_ans, rank_eq=bool(okr), s0_last_eq=rb["s0_last"][q] == meta[i]["slot0_last_masked_step"]))
        out["cap_vs_nocap"].append(bool(rb["answers"][q] == rn["answers"][q] and (rb["rank"][q] == rn["rank"][q]).all() and (rb["canvas"][q] == rn["canvas"][q]).all()))
        out["empty"].append(bool((re_["canvas"][q] == rb["canvas"][q]).all() and (re_["rank"][q] == rb["rank"][q]).all()))
        m2 = rb["rank"][q][2] >= 0
        out["empty_eager"].append(dict(canvas_eq=bool((ree["canvas"][q] == rb["canvas"][q]).all()), ans_eq=ree["answers"][q] == rb["answers"][q],
                                       rank2_eq=bool(ree["rank"][q][2] == rb["rank"][q][2]) if m2 else None, rank2=(int(rb["rank"][q][2]), int(ree["rank"][q][2])) if m2 else None))
    # DLA 점검 (경쟁 = step 0 의 금 아닌 1 위)
    r0 = E.step0(m, [ids[i] for i in ix], g_, [int(t) for t in rb["s0_tok"]], [-1] * len(ix))
    for q, i in enumerate(ix):
        out["dla"].append(dict(row=i, logit_diff=float(r0["logit_diff"][q]), lin_diff=float(r0["lin_diff"][q]), heads_sum=float((r0["dla_g"][q] - r0["dla_c"][q]).sum()),
                               rank0_eq=int(r0["rank0"][q]) == int(rb["rank"][q][0])))
    print(f"묶음 {j} L={L} n={len(ix)}: base {tb:.1f}s, empty_eager {te:.1f}s", flush=True)
summ = {"n": len(out["e44"]), "e44_ans_eq": float(np.mean([d["ans_eq"] for d in out["e44"]])), "e44_rank_eq": float(np.mean([d["rank_eq"] for d in out["e44"]])),
        "e44_s0last_eq": float(np.mean([d["s0_last_eq"] for d in out["e44"]])), "cap_vs_nocap_eq": float(np.mean(out["cap_vs_nocap"])), "empty_eq_base": float(np.mean(out["empty"])),
        "empty_eager_canvas_eq": float(np.mean([d["canvas_eq"] for d in out["empty_eager"]])), "empty_eager_ans_eq": float(np.mean([d["ans_eq"] for d in out["empty_eager"]])),
        "empty_eager_rank2_eq": float(np.mean([d["rank2_eq"] for d in out["empty_eager"] if d["rank2_eq"] is not None])),
        "dla_lin_abs_err": float(np.mean([abs(d["lin_diff"] - d["logit_diff"]) for d in out["dla"]])), "dla_logit_diff_abs_mean": float(np.mean([abs(d["logit_diff"]) for d in out["dla"]])),
        "dla_heads_share": float(np.median([d["heads_sum"] / d["logit_diff"] for d in out["dla"] if abs(d["logit_diff"]) > 1])), "dla_rank0_eq": float(np.mean([d["rank0_eq"] for d in out["dla"]])),
        "time_s": time.time() - t0}
print(json.dumps(summ, indent=1, ensure_ascii=False))
json.dump({"summary": summ, "detail": out}, open(f"{OUT}/e52/sanity.json", "w"), ensure_ascii=False)
