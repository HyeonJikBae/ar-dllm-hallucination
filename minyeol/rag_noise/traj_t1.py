"""PREREG_traj.md T1: 기존 Dream bs1 궤적으로 '방해 개체를 예측하고 거르는가'.  usage: python traj_t1.py  -> results/traj_t1.json"""
import json, collections, numpy as np, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from grade_anchor import etype, f1
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", trust_remote_code=True)
def first_line(s):
    s = s.split("\n")[0]
    for cut in ("---", "Question:", "Passage", "user:", "assistant:"): s = s.split(cut)[0]
    return s.strip()
data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_anchor.jsonl")}
R = [json.loads(l) for l in open(f"{HERE}/results/dream_anchorbs1tr_s16.jsonl")]
Z = np.load(f"{HERE}/results/dream_anchorbs1tr_s16_trace.npz"); A, M, Fn = Z["argmax"], Z["masked"], Z["final"]
SP = set(tok.all_special_ids)
def dec(ids):
    ids = [int(t) for t in ids]; ids = ids[: next((q for q, t in enumerate(ids) if t in SP), len(ids))]; return first_line(tok.decode(ids, skip_special_tokens=True))
out = {}
for c in ("gold", "hard3"):
    tgt = "방해 문서 개체" if c == "hard3" else "근거 문서 속 다른 개체"
    rec = []
    for j, r in enumerate(R):
        if r["cond"] != c: continue
        d = data[r["qid"]]; T = []
        for s in range(A.shape[1]):
            T.append(etype(d, {"pred": dec(np.where(M[j, s], A[j, s], Fn[j])), "answers": d["answers"]}, c))
        fin = etype(d, r, c)
        tok0 = int(np.where(M[j, :, 0])[0].max()) + 1 if M[j, :, 0].any() else 0       # 첫 칸 확정 step (그 step forward 뒤 확정)
        rec.append(dict(fin=fin, T=T, commit0=tok0, t_at_commit=T[min(tok0, A.shape[1] - 1)] if tok0 < A.shape[1] else T[-1]))
    n = len(rec); fc = [x for x in rec if x["fin"] == "정답"]; fe = [x for x in rec if x["fin"] == tgt]
    ever = [x for x in rec if tgt in x["T"]]
    s0 = [x for x in rec if x["T"][0] == tgt]
    first_app = collections.Counter(x["T"].index(tgt) for x in fe)
    o = dict(n=n, 최종정답=len(fc), 최종표적오답=len(fe),
             P_rej=round(100 * np.mean([tgt in x["T"] for x in fc]), 1) if fc else None,
             초안에_한번이라도_표적=len(ever), 배수=round(len(ever) / max(len(fe), 1), 2),
             표적오답중_step0부터=round(100 * np.mean([x["T"][0] == tgt for x in fe]), 1) if fe else None,
             표적오답_처음나타난step=dict(sorted(first_app.items())),
             step0표적_결과=dict(collections.Counter(x["fin"] for x in s0)), step0표적_n=len(s0),
             step0_초안분류=dict(collections.Counter(x["T"][0] for x in rec)),
             첫칸확정step_중앙=dict(정답=float(np.median([x["commit0"] for x in fc])) if fc else None, 표적오답=float(np.median([x["commit0"] for x in fe])) if fe else None),
             확정순간초안_표적오답=dict(collections.Counter(x["t_at_commit"] for x in fe)))
    out[c] = o; print(c, json.dumps(o, ensure_ascii=False))
json.dump(out, open(f"{HERE}/results/traj_t1.json", "w"), ensure_ascii=False, indent=1)
