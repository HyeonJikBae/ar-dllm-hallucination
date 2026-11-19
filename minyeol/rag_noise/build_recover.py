"""Qwen 오류 복구 데이터 (PREREG_poison.md v9).  v3 에서 Qwen 이 다른 값을 고른 경우마다, 값 두 개만 토큰 수가 같은 새 값으로 바꾼 주는 입력 3개.
출력: data/musique_recover.jsonl  (행 = 경우; donors = [{V1, V2}, ...])
"""
import json, os, random
from transformers import AutoTokenizer
from build_bindswap import T, GENERIC
from gen import prompt, cond_docs
from grade_poison_main import lab

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"); ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]


def donor_docs(r, V1, V2):
    fact = T[r["template"]][3]; d = dict(r["docs"])
    d["FX"] = dict(title=r["X"], text=fact.format(P=r["X"], V=V1)); d["FY"] = dict(title=r["Y"], text=fact.format(P=r["Y"], V=V2)); return d


def main():
    data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_bindswap.jsonl")}
    R = {m: {(r["qid"], r["cond"]): r["pred"] for r in map(json.loads, open(f"{RES}/{m}_bindswap.jsonl"))} for m in ("qwen", "dream")}
    L = lambda m, q, c: lab(R[m][(q, c)], [data[q]["ans"][c]], data[q]["other"][c])
    Q = [q for q in data if all(L(m, q, "S") == "정답" for m in R)]
    shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]; r_ = random.Random(17); rows = []
    for q in Q:
        r = data[q]
        if r["template"] == 9: continue
        for c in ("U", "U_sw", "L", "L_sw"):
            if L("qwen", q, c) != "공격": continue
            used = set().union(*[set(r[k].split()) - GENERIC for k in ("A", "B", "X", "Y", "V1", "V2")])
            first = {ids(" " + r[k])[0] for k in ("V1", "V2")}; P0 = prompt(r["question"], cond_docs(r, c), shots); n0 = len(ids(P0))
            donors = []
            for _ in range(5000):
                if len(donors) == 3: break
                V = []
                for k in ("V1", "V2"):
                    for _ in range(500):
                        v = T[r["template"]][2](r_); w = set(v.split()) - GENERIC
                        if w and not w & used and len(ids(" " + v)) == len(ids(" " + r[k])) and ids(" " + v)[0] not in first | {ids(" " + x)[0] for x in V}:
                            V.append(v); break
                if len(V) < 2: continue
                rr = dict(r, docs=donor_docs(r, *V)); P = prompt(r["question"], cond_docs(rr, c), shots)
                if len(ids(P)) != n0: continue
                donors.append(dict(V1=V[0], V2=V[1])); used |= (set(V[0].split()) | set(V[1].split())) - GENERIC
                first |= {ids(" " + x)[0] for x in V}
            rows.append(dict(qid=q, cond=c, template=r["template"], donors=donors))
    with open(f"{HERE}/data/musique_recover.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"오류 경우 {len(rows)} (연도 틀 제외), 주는 입력 3개를 다 만든 경우 {sum(len(x['donors']) == 3 for x in rows)}")


if __name__ == "__main__":
    main()
