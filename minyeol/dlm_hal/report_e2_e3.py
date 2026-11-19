"""E2, E3 집계 (PREREG.md 판정표)."""
import collections, json
import numpy as np
O = "followup_mk"; B = 10000
def load(f):
    by = collections.defaultdict(dict)
    for l in open(f):
        r = json.loads(l); by[r["cond"]][r["row"]] = r
    return by
def pboot(d, seed=0):
    d = np.asarray(d, float); ix = np.random.default_rng(seed).integers(0, len(d), (B, len(d))); m = d[ix].mean(1)
    return [round(100 * d.mean(), 1), round(100 * np.percentile(m, 2.5), 1), round(100 * np.percentile(m, 97.5), 1)]
E2 = load(f"{O}/results_e2.jsonl"); rows = sorted(E2["base"]); base = np.array([E2["base"][i]["new_type"] == "correct" for i in rows])
print(f"== E2 정답 행 {len(rows)}, 같은 실행 개입 없음 정답 유지 {100*base.mean():.1f}%")
def keep(c): return np.array([E2[c][i]["new_type"] == "correct" for i in rows])
def left(c):
    t = collections.Counter(E2[c][i]["new_type"] for i, b in zip(rows, base) if b and E2[c][i]["new_type"] != "correct"); n = sum(t.values())
    return n, {x: round(100 * t[x] / n, 1) for x in ("broken", "entity", "abstain")} if n else {}
CTRL = [f"{d} c1" for d in ("G0", "G1", "G2", "G3", "D0", "D1", "D2", "D3")]
for c in [c for c in E2 if c != "base"]:
    n, sh = left(c); print(f"  {c:9s} 정답 유지 {100*keep(c).mean():5.1f}  하락 {pboot(base.astype(float) - keep(c))}  벗어난 {n}행: {sh}")
ctrl_keep = np.mean([keep(c) for c in CTRL], 0); drops = {c: 100 * (base.mean() - keep(c).mean()) for c in CTRL}
d21 = pboot(keep("-B c1") * 0 + (ctrl_keep - keep("-B c1")))
H21 = d21[0] >= 10 and d21[1] > 0 and 100 * (base.mean() - keep("-B c1").mean()) > max(drops.values())
tl = collections.Counter(); 
for c in CTRL:
    for i, b in zip(rows, base):
        if b and E2[c][i]["new_type"] != "correct": tl[E2[c][i]["new_type"]] += 1
ctrl_br = 100 * tl["broken"] / max(sum(tl.values()), 1); nb, shb = left("-B c1"); n2, sh2 = left("+B c2")
H22 = shb.get("broken", 0) >= 50 and shb.get("broken", 0) >= 2 * ctrl_br and sh2.get("broken", 100) <= 20
pdrop = 100 * (base.mean() - keep("projB").mean()); ctrl_mean_drop = float(np.mean(list(drops.values())))
print(f"  대조 8개 하락: 평균 {ctrl_mean_drop:.1f}, 최대 {max(drops.values()):.1f} | 대조에서 벗어난 행 중 붕괴 {ctrl_br:.1f}% ({sum(tl.values())}행)")
print(f"  H2.1 (-B c1 하락 - 대조 평균 하락 {d21}, 대조 최대보다 큼 {100*(base.mean()-keep('-B c1').mean()):.1f} > {max(drops.values()):.1f}): {H21}")
print(f"  H2.2 (-B c1 벗어난 행 붕괴 {shb.get('broken')}% >=50 이고 >= 2x{ctrl_br:.1f}; +B c2 붕괴 {sh2.get('broken')}% <=20): {H22}")
print(f"  H2.3 B 성분 지우기 하락 {pdrop:.1f}점 -> {'필요하지 않음 (<5)' if pdrop < 5 else ('필요' if pdrop > ctrl_mean_drop else '대조 이하')}")
print("  판정:", "B 축 한쪽 끝은 붕괴, 반대쪽은 과잉 확약. 인과 그림 채택" if H21 and H22 else ("B 는 손상 크기에서 특별하지만 유형을 정하지 않음" if H21 else "빼는 쪽에서 B 가 특별하지 않음"))
E3 = collections.defaultdict(lambda: collections.defaultdict(dict))
for l in open(f"{O}/results_e3.jsonl"):
    r = json.loads(l); E3[r["cond"]][r["set"]][r["row"]] = r["new_type"]
print("== E3 (zero-shot 대 3-shot, 같은 행)")
res = {}
for s in ("broken", "entity", "correct"):
    for c in ("zero", "3shot"):
        v = E3[c][s]; t = collections.Counter(v.values()); n = len(v); res[(s, c)] = 100 * t["correct"] / n
        print(f"  원래 {s:8s} {c:6s} n={n} 정답 {100*t['correct']/n:5.1f}  entity {100*t['entity']/n:5.1f}  붕괴 {100*t['broken']/n:5.1f}")
    rs = sorted(E3["zero"][s]); print(f"    3shot - zero 정답 차이 {pboot([ (E3['3shot'][s][i]=='correct') - (E3['zero'][s][i]=='correct') for i in rs])}")
ok = res[("broken", "3shot")] >= 25 and res[("correct", "3shot")] >= 69.7
print("  판정:", "B 조향은 실용 처방으로는 few-shot 보다 못함, 메커니즘 주장으로 쓴다" if ok else "B 가 few-shot 이 못 하는 것을 한다")
