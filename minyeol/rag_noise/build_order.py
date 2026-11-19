"""순서 대 거리 데이터 (PREREG_poison.md v15).  같은 쪽 배치 T{g}_{a}{b}[_sw], 교차 배치 C{k}b{B}[_sw], S_e, S_l.  build_layout.item 으로 이름과 문서를 만들고 배치만 새로 짠다.  출력: data/musique_order{_pilot}.jsonl"""
import json, random
from build_layout import item, used_persons, HERE, CORPUS, n_tok

N = 9; GS = (1, 2, 4); ARR = ((7, 8), (8, 7), (0, 1), (1, 0))
CROSS = ((1, 8), (4, 0), (4, 8), (7, 0))    # (k=1,B=0), (k=7,B=8) 은 T4_10, T4_78 과 같은 배치


def fact_slots(g, a):
    return (6 - g, 6) if a >= 7 else (2, 2 + g)


def layout(fixed):
    out = [None] * N
    for s, d in fixed.items(): out[s] = d
    m = iter(f"r{i}" for i in range(5))
    return [d if d else next(m) for d in out]


def rebuild(r):
    ctx = {c: r["ctx"][c] for c in ("S_e", "S_l")}; ans = {c: r["ans"][c] for c in ctx}; other = {c: r["other"][c] for c in ctx}; meta = {c: r["meta"][c] for c in ctx}
    fE, fL = ("FX", "FY") if r["x_first"] else ("FY", "FX")
    def add(c, a, b, f1, f2, sw, kind, g=None):
        la, lb = ("LAs", "LBs") if sw else ("LA", "LB"); fa, fo = ("FY", "FX") if sw else ("FX", "FY")
        sa, so = (f1 if fa == fE else f2), (f1 if fo == fE else f2); assert {sa, so} == {f1, f2}
        ctx[c] = layout({a: la, b: lb, f1: fE, f2: fL}); ans[c] = r["V2"] if sw else r["V1"]; other[c] = r["V1"] if sw else r["V2"]
        cong = (a < b) == (sa < so); x = abs(a - sa) - abs(a - so); xB = abs(b - so) - abs(b - sa)
        tie = abs(a - sa) + abs(b - so) == abs(a - so) + abs(b - sa)
        meta[c] = dict(kind=kind, a=a, b=b, ans_slot=sa, oth_slot=so, x=x, xB=xB, cong=cong, tie=tie, g=g, side=("after" if a >= 7 else "before") if kind == "T" else None)
        assert ctx[c].index(la) == a and ctx[c].index(lb) == b and ctx[c].index(fa) == sa and ctx[c].index(fo) == so and len(set(ctx[c])) == N
        if kind == "T": assert tie and abs(x) == g and xB == -x, (c, x, xB, g)
    for g in GS:
        for a, b in ARR:
            f1, f2 = fact_slots(g, a)
            for sw in (False, True): add(f"T{g}_{a}{b}{'_sw' if sw else ''}", a, b, f1, f2, sw, "T", g)
    for k, b in CROSS:
        for sw in (False, True): add(f"C{k}b{b}{'_sw' if sw else ''}", k, b, 2, 6, sw, "C")
    r.update(ctx=ctx, ans=ans, other=other, meta=meta, qid=r["qid"].replace("lay", "ord"))
    return r


def main():
    corpus = json.load(open(CORPUS)); shots = json.load(open(f"{HERE}/data/fewshot.json")); shot_titles = {d["title"] for s in shots for d in s["docs"]}
    pool = [dict(title=c["title"], text=c["text"]) for c in corpus if c["title"] not in shot_titles and 70 <= n_tok(c["title"] + "\n" + c["text"]) <= 110]
    banned = used_persons()
    for f in ("layout", "layout_pilot", "layout_pilot2"):
        for l in open(f"{HERE}/data/musique_{f}.jsonl"):
            d = json.loads(l); banned |= {d["X"], d["Y"]}
    print("금지 이름", len(banned))
    for name, n_per, seed in (("order_pilot", 1, 37), ("order", 10, 31)):
        rnd = random.Random(seed); rows = []
        for ti in range(20):
            for j in range(n_per): rows.append(rebuild(item(rnd, ti * 1000 + j, ti, rnd.sample(pool, 7), banned)))
        with open(f"{HERE}/data/musique_{name}.jsonl", "w") as f:
            for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(name, len(rows), "문항", len(rows[0]["ctx"]), "조건")
    print(rows[0]["question"], rows[0]["x_first"]); [print(" ", c, rows[0]["ctx"][c], {k: rows[0]["meta"][c][k] for k in ("x", "xB", "cong")} if "T" in c or "C" in c else "") for c in ("T4_78", "T4_87", "T4_10", "T4_01", "T1_78", "T2_01_sw", "C4b0", "C4b8", "S_e")]


if __name__ == "__main__":
    main()
