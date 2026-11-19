"""프롬프트 형식별 생성 (PREREG_poison.md v10, v11).  형식: base, sand, placebo, pre.
usage: CUDA_VISIBLE_DEVICES=0 python gen_fmt.py --model qwen|dream --data layout --fmt base [--conds a,b] [--bs 16] [--limit N]
출력: results/<model>_<data>_<fmt>.jsonl (이어 쓰기: 이미 있는 (qid, cond) 는 건너뜀)
"""
import argparse, json, os, time
from transformers import AutoTokenizer
from batchgen import load, gen_all
from gen import MODELS, HEAD_DOC, SEP, passages, cond_docs, first_line, prompt as base_prompt

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt_block(fmt, docs, q, ans, q_other=None):
    tail = f"Answer: {ans}" if ans else "Answer:"
    p = passages(docs)
    if fmt == "sand": return f"Question: {q}\n\n{p}\n\nQuestion: {q}\n{tail}"
    if fmt == "placebo": return f"Question: {q_other}\n\n{p}\n\nQuestion: {q}\n{tail}"
    if fmt == "pre": return f"Question: {q}\n\n{p}\n\n{tail}"
    raise ValueError(fmt)


def make_prompt(fmt, q, docs, shots, q_other=None):
    if fmt == "base": return base_prompt(q, docs, shots)
    n = len(shots)
    blocks = [fmt_block(fmt, s["docs"], s["question"], s["answer"], shots[(i + 1) % n]["question"]) for i, s in enumerate(shots)]
    return HEAD_DOC + SEP.join(blocks + [fmt_block(fmt, docs, q, None, q_other)])


def other_questions(rows, tok):
    """Q' = 토큰 수가 Q 와 정확히 같은 다른 질문.  같은 관계 틀의 뒤 문항부터, 없으면 전체에서.  대상 이름 A, B 가 들어 있으면 제외.  (자연 데이터는 틀이 없어 전체에서 찾음)"""
    n = [len(tok(r["question"], add_special_tokens=False)["input_ids"]) for r in rows]; by = {}
    for i, r in enumerate(rows): by.setdefault(r.get("template", 0), []).append(i)
    out = {}; fallback = 0
    def ok(i, j):
        r, s = rows[i], rows[j]
        return n[i] == n[j] and r["question"] != s["question"] and not any(r.get(k) and r[k] in s["question"] for k in ("A", "B"))
    for ix in by.values():
        pos = {i: p for p, i in enumerate(ix)}
        for i in ix:
            cands = [ix[(pos[i] + d) % len(ix)] for d in range(1, len(ix))] + [(i + d) % len(rows) for d in range(1, len(rows))]
            j = next((c for c in cands if ok(i, c)), None)
            if j is None: fallback += 1; j = min((c for c in range(len(rows)) if c != i), key=lambda c: abs(n[c] - n[i]))
            out[rows[i]["qid"]] = rows[j]["question"]
    if fallback: print(f"  Q' 길이를 못 맞춘 질문 {fallback}", flush=True)
    return out


def select(rows, every=0, shard="", per=0):
    if every: rows = [r for r in rows if (r["qi"] % 1000) % every == 0]
    if per:
        seen = {}; keep = []
        for r in rows:
            t = r.get("template", 0); seen[t] = seen.get(t, 0) + 1
            if seen[t] <= per: keep.append(r)
        rows = keep
    if shard:
        i, n = map(int, shard.split("/")); rows = rows[i::n]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--fmt", required=True, choices=["base", "sand", "placebo", "pre"])
    ap.add_argument("--conds", default=""); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="")
    ap.add_argument("--every", type=int, default=0, help="틀마다 every 번째 문항만"); ap.add_argument("--shard", default="", help="i/n: 문항 샤드, 출력은 .sh<i>.jsonl")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{HERE}/data/musique_{a.data}.jsonl")][: a.limit or None]
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]; Qo = other_questions(rows, AutoTokenizer.from_pretrained(MODELS["qwen"])) if a.fmt == "placebo" else {r["qid"]: None for r in rows}
    rows = select(rows, a.every, a.shard)
    conds = a.conds.split(",") if a.conds else None
    sh = f".sh{a.shard.split('/')[0]}" if a.shard else ""; path = a.out or f"{HERE}/results/{a.model}_{a.data}_{a.fmt}{sh}.jsonl"; done = set()
    if os.path.exists(path): done = {(d["qid"], d["cond"]) for d in map(json.loads, open(path))}
    jobs = [(r, c, make_prompt(a.fmt, r["question"], cond_docs(r, c), shots, Qo[r["qid"]])) for r in rows for c in (conds or r["ctx"]) if (r["qid"], c) not in done]
    print(f"[{a.model} {a.data} {a.fmt}] {len(jobs)} 생성 (이미 {len(done)})", flush=True)
    if not jobs: return
    tok, model, dream = load(a.model); f = open(path, "a"); t0 = time.perf_counter()
    def save(ix, texts):
        for i, t in zip(ix, texts):
            r, c, _ = jobs[i]; f.write(json.dumps(dict(qid=r["qid"], cond=c, raw=t, pred=first_line(t)), ensure_ascii=False) + "\n")
        f.flush()
    gen_all(model, tok, dream, [p for _, _, p in jobs], bs=a.bs, log=f"{a.model}/{a.fmt}", on_batch=save)
    print(f"[{a.model} {a.data} {a.fmt}] done {time.perf_counter() - t0:.0f}s, 프롬프트당 {(time.perf_counter() - t0) / len(jobs):.2f}s", flush=True)


if __name__ == "__main__":
    main()
