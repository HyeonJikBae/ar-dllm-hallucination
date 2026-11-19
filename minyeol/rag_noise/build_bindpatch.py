"""연결 정보 / 최종 답 분리 patching 데이터 (PREREG_poison.md v4).  출처(src)와 대상(dst) 문맥은 토큰 수가 같고, 이름·값 자리만 다르다.

질문: "Where did the founder of A study?"   (A, B, C = 머리 개체,  X, Y, Z = 다리 개체)
         연결 문장 (같은 칸)                         사실 문장 칸 p / q / r
  src :  A <- X,  B <- Y,  C <- Z                    X -> aX  /  Y -> aY  /  Z -> aZ       정답 aX
  dst :  A <- Y,  B <- X,  C <- Z                    Z -> bZ  /  X -> bX  /  Y -> bY       정답 bY
사실 문장은 칸 사이에서 한 칸씩 돌린다 (X: p->q, Y: q->r, Z: r->p).  그래서 한 방향으로 활성을 옮겼을 때 네 설명의 예측이 모두 갈린다.
  src -> dst :  바뀌지 않음 bY / 개체(설립자 X 를 받아 현재 문맥에서 X 의 값을 찾음) bX / 위치(출처에서 답이 있던 칸 p) bZ / 답 복사 aX
  dst -> src :  바뀌지 않음 aX / 개체(Y) aY / 위치(dst 에서 Y 가 있던 칸 r) aZ / 답 복사 bY
조건: 세 사람 이름은 토큰 수가 같고 (문장 안 " X" 와 줄 첫머리 "X" 모두), 값 6개도 토큰 수가 같으며 첫 토큰이 모두 다르다 (첫 토큰으로 답을 읽음).
문서 7개 = 연결 3 + 사실 3 + 무작위 문서 1, 칸 순서는 질의마다 무작위.  연도 틀(숫자는 한 자리씩 토큰)은 뺀다.
출력: data/musique_bindpatch{_pilot}.jsonl
"""
import json, os, random
from transformers import AutoTokenizer
from build_bindswap import T, person, GENERIC, CORPUS
from gen import prompt

HERE = os.path.dirname(os.path.abspath(__file__)); TEMPLATES = range(9)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]


def pick(r, gen, n, used, key, first=None):
    """gen 으로 n 개를 뽑는다: 이미 쓴 단어와 겹치지 않고, key 가 첫 번째와 같고, (first 가 있으면) 첫 토큰이 서로 다름."""
    out = []
    for _ in range(3000):
        c = gen(r); w = set(c.split()) - GENERIC
        if not w or w & used or (out and key(c) != key(out[0])) or (first is not None and ids(" " + c)[0] in first): continue
        out.append(c); used |= w
        if first is not None: first.add(ids(" " + c)[0])
        if len(out) == n: return out
    return None


def item(r, k, ti, filler, shots):
    head, link, val, fact, q = T[ti]
    while True:
        used = set()
        H = pick(r, head, 3, used, lambda c: 0)
        P = H and pick(r, person, 3, used, lambda c: (len(ids(" " + c)), len(ids(c))))
        V = P and pick(r, val, 6, used, lambda c: len(ids(" " + c)), first=set())
        if not V: continue
        (A, B, C), (X, Y, Z) = H, P; v = dict(zip(["aX", "aY", "aZ", "bX", "bY", "bZ"], V))
        L = lambda h, p: dict(title=h, text=link.format(H=h, P=p)); F = lambda p, x: dict(title=p, text=fact.format(P=p, V=x))
        docs = {"LA_s": L(A, X), "LB_s": L(B, Y), "LA_d": L(A, Y), "LB_d": L(B, X), "LC": L(C, Z),
                "F_aX": F(X, v["aX"]), "F_aY": F(Y, v["aY"]), "F_aZ": F(Z, v["aZ"]),
                "F_bX": F(X, v["bX"]), "F_bY": F(Y, v["bY"]), "F_bZ": F(Z, v["bZ"]), "r0": filler}
        slots = ["la", "lb", "lc", "p", "q", "r", "f"]; r.shuffle(slots)
        def ctx(fill):
            out = [None] * 7
            for role, d in fill.items(): out[slots.index(role)] = d
            return out
        c = {"src": ctx(dict(la="LA_s", lb="LB_s", lc="LC", p="F_aX", q="F_aY", r="F_aZ", f="r0")),
             "dst": ctx(dict(la="LA_d", lb="LB_d", lc="LC", p="F_bZ", q="F_bX", r="F_bY", f="r0"))}
        question = q.format(H=A)
        n = {s: len(ids(prompt(question, [docs[d] for d in c[s]], shots))) for s in c}
        if n["src"] == n["dst"]: break
    return dict(qid=f"bp{ti:02d}_{k:03d}", qi=k, template=ti, question=question, answers=[v["aX"]], ctx=c, docs=docs, v=v,
                A=A, B=B, C=C, X=X, Y=Y, Z=Z, slots={s: slots.index(s) for s in slots}, n_tok=n["src"])


def main():
    corpus = json.load(open(CORPUS)); shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]
    for name, n_per, seed in (("bindpatch_pilot", 3, 5), ("bindpatch", 40, 13)):
        r = random.Random(seed); rows = []
        for ti in TEMPLATES:
            for j in range(n_per):
                f = corpus[r.randrange(len(corpus))]
                rows.append(item(r, ti * 1000 + j, ti, dict(title=f["title"], text=f["text"]), shots))
        with open(f"{HERE}/data/musique_{name}.jsonl", "w") as fo:
            for x in rows: fo.write(json.dumps(x, ensure_ascii=False) + "\n")
        x = rows[0]
        print(name, len(rows), "토큰 평균", sum(r_["n_tok"] for r_ in rows) / len(rows), "|", x["question"])
        for s in ("src", "dst"): print(" ", s, [x["docs"][d]["text"] for d in x["ctx"][s] if d != "r0"])


if __name__ == "__main__":
    main()
