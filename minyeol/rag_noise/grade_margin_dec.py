"""결정 시점 logit 차 채점 (PREREG_poison.md v16.1).  usage: python grade_margin_dec.py   출력: results/summary_margin_dec.json"""
import collections, glob, json
import numpy as np
from grade_layout import load_data, load_pred, L, RES
from grade_order import ALIAS


def main():
    data = load_data("order"); P = {m: load_pred(m, "order", "base") for m in ("qwen", "dream")}
    T0 = {m: {(r["qid"], r["cond"]): r for r in map(json.loads, open(f"{RES}/margin_order_{m}.jsonl"))} for m in ("qwen", "dream")}
    D = {}
    for f in glob.glob(f"{RES}/margin_dec_dream.sh*.jsonl"):
        for r in map(json.loads, open(f)): D[(r["qid"], r["cond"])] = r
    Q = sorted({q for q, _ in D}); f0 = Q[0]; conds = [c for c in data[f0]["meta"] if data[f0]["meta"][c].get("kind") in ("T", "C")]
    miss = [k for k, r in D.items() if r["step"] is None]; assert all((q, c) in D for q in Q for c in data[q]["ctx"]), "미완료"
    print(f"문항 {len(Q)}, 프롬프트 {len(D)}, 결정 없음 {len(miss)}")
    Q = [q for q in Q if all(D[(q, c)]["step"] is not None for c in conds)]; print("결정 있는 문항", len(Q))
    mq = lambda q, c: T0["qwen"][(q, c)]["la"] - T0["qwen"][(q, c)]["lo"]; md = lambda q, c: D[(q, c)]["la"] - D[(q, c)]["lo"]; m0 = lambda q, c: T0["dream"][(q, c)]["la"] - T0["dream"][(q, c)]["lo"]
    R = dict(n=len(Q)); pos = [md(q, c) for q in Q for c in conds if L(P["dream"], data, q, c) == "정답"]; neg = [md(q, c) for q in Q for c in conds if L(P["dream"], data, q, c) == "공격"]; pos, neg = np.array(pos), np.array(neg)
    rg = np.random.default_rng(0); auc = float(np.mean(rg.choice(pos, 200000) > rg.choice(neg, 200000))); R["G16.1"] = dict(공격중_음수=round(100 * float((neg < 0).mean()), 1), n공격=len(neg), 정답중_양수=round(100 * float((pos > 0).mean()), 1), n정답=len(pos), AUC=round(auc, 3)); ok = R["G16.1"]["공격중_음수"] >= 90 and R["G16.1"]["정답중_양수"] >= 90; R["G16.1_통과"] = ok; print("G16.1", R["G16.1"], "통과" if ok else "미달: 무효")
    tpl = np.array([data[q]["template"] for q in Q]); ts = sorted(set(tpl)); idx = rg.integers(0, len(Q), (10000, len(Q))); meta = lambda q, c: data[q]["meta"][c]
    def ci(a): return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    def item(f, pred): return np.array([np.mean([f(q, c) for c in conds if pred(meta(q, c), c)]) for q in Q])
    def cluster(v):
        mm = np.array([v[tpl == t].mean() for t in ts]); b = mm[np.random.default_rng(1).integers(0, len(ts), (10000, len(ts)))].mean(1); return [round(float(mm.mean()), 2), round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)]
    isT = lambda d, c: d["kind"] == "T"; isC = lambda d, c: d["kind"] == "C" or c in ALIAS.values()
    def section(name, base):
        out = {}; Db = {}; Sb = {}
        for lab, f in (("qwen", mq), ("dream_dec", md), ("dream_t0", m0)):
            o = item(f, lambda d, c: base(d, c) and d["cong"]); i = item(f, lambda d, c: base(d, c) and not d["cong"]); d = o - i; Db[lab] = d[idx].mean(1); Sb[lab] = Db[lab] / o[idx].mean(1)
            out[lab] = dict(m_일치=round(float(o.mean()), 2), m_불일치=round(float(i.mean()), 2), D=[round(float(d.mean()), 2), *[round(x, 2) for x in ci(Db[lab])]], D_틀=cluster(d), 몫=[round(float(d.mean() / o.mean()), 3), *ci(Sb[lab])], sd=round(float(np.std([f(q, c) for q in Q for c in conds if base(meta(q, c), c)])), 2))
            print(f"  {name} {lab}: {out[lab]}")
        out["r_D"] = [round(out["dream_dec"]["D"][0] / out["qwen"]["D"][0], 3), *ci(Db["dream_dec"] / Db["qwen"])]; out["r_s"] = [round(out["dream_dec"]["몫"][0] / out["qwen"]["몫"][0], 3), *ci(Sb["dream_dec"] / Sb["qwen"])]; out["D_dec-D_t0"] = [round(out["dream_dec"]["D"][0] - out["dream_t0"]["D"][0], 2), *ci(Db["dream_dec"] - Db["dream_t0"])]
        print(f"  {name} r_D {out['r_D']}  r_s {out['r_s']}  D_dec-D_t0 {out['D_dec-D_t0']}"); R[name] = out; return out
    print("[T x>0]"); h = section("T_x>0", lambda d, c: isT(d, c) and d["x"] > 0); print("[보조]"); section("T_x<0", lambda d, c: isT(d, c) and d["x"] < 0); section("T_전체", isT); section("C", isC)
    rd, rs = h["r_D"], h["r_s"]
    v = "무효" if not ok else "순서 효과 크기는 비슷하고 Dream 은 여유가 크다" if rd[1] >= .5 else "Dream 의 순서 의존은 실제로 작다" if rd[2] < .5 and rs[2] < .5 else "비례 축소" if rd[2] < .5 and rs[1] >= .5 else "판단 보류"
    R["판정"] = v; print("판정:", v)
    print("[결정 단계, 칸]"); R["단계"] = {}
    for name, pred in (("T x>0 일치", lambda d, c: isT(d, c) and d["x"] > 0 and d["cong"]), ("T x>0 불일치", lambda d, c: isT(d, c) and d["x"] > 0 and not d["cong"]), ("C 일치", lambda d, c: isC(d, c) and d["cong"]), ("C 불일치", lambda d, c: isC(d, c) and not d["cong"])):
        s = [D[(q, c)]["step"] for q in Q for c in conds if pred(meta(q, c), c)]; sl = collections.Counter(D[(q, c)]["slot"] for q in Q for c in conds if pred(meta(q, c), c)); R["단계"][name] = dict(평균=round(float(np.mean(s)), 2), 중앙=float(np.median(s)), 단계0=round(100 * float(np.mean(np.array(s) == 0)), 1), 칸0=round(100 * sl[0] / len(s), 1)); print(f"  {name}: {R['단계'][name]}")
    for lab in ("정답", "공격"):
        s = [D[(q, c)]["step"] for q in Q for c in conds if L(P["dream"], data, q, c) == lab]; R["단계"][lab] = dict(평균=round(float(np.mean(s)), 2), 중앙=float(np.median(s)), 단계0=round(100 * float(np.mean(np.array(s) == 0)), 1)); print(f"  {lab}: {R['단계'][lab]}")
    print("[C 6 칸의 m_dec, 탐색]"); R["C칸"] = {}
    for x in (-4, 0, 4):
        for xb in (-4, 4):
            p = lambda d, c: isC(d, c) and d["x"] == x and d["xB"] == xb; R["C칸"][f"{x},{xb}"] = dict(m_dec=round(float(item(md, p).mean()), 2), 공격=round(100 * float(np.mean([L(P["dream"], data, q, c) == "공격" for q in Q for c in conds if p(meta(q, c), c)])), 1)); print(f"  x {x:+d} xB {xb:+d}: {R['C칸'][f'{x},{xb}']}")
    json.dump(R, open(f"{RES}/summary_margin_dec.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
