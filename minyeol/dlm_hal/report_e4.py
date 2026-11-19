"""E4 집계 (PREREG.md 판정표)."""
import collections, json
import numpy as np
O = "followup_mk"; Bn = 10000
by = collections.defaultdict(dict); W = {}
for l in open(f"{O}/results_e4.jsonl"):
    r = json.loads(l); by[r["cond"]][r["row"]] = r["new_type"]; W[r["row"]] = r["wdonor"]
rows = sorted(by["base"]); base = {i: by["base"][i] for i in rows}
def pb(d, seed=0):
    d = np.asarray(d, float); m = d[np.random.default_rng(seed).integers(0, len(d), (Bn, len(d)))].mean(1)
    return [round(100 * d.mean(), 1), round(100 * np.percentile(m, 2.5), 1), round(100 * np.percentile(m, 97.5), 1)]
fix = lambda c, rr=rows: {i for i in rr if by[c][i] == "correct" and base[i] != "correct"}
t = collections.Counter(base.values()); print(f"대상 {len(rows)}, 같은 실행 base: " + ", ".join(f"{k} {100*v/len(rows):.1f}" for k, v in t.most_common()))
print("조건별 (정답 / entity / 붕괴 %)")
for c in by:
    tt = collections.Counter(by[c].values()); n = len(by[c]); print(f"  {c:12s} 정답 {100*tt['correct']/n:5.1f}  entity {100*tt['entity']/n:5.1f}  붕괴 {100*tt['broken']/n:5.1f}  | 새로 고침 {len(fix(c))}")
hw = [i for i in rows if W[i] is not None]
d41 = pb([(by["SUBJ L0-8"][i] == "correct") - (by["SUBJw L0-8"][i] == "correct") for i in hw])
s_rate = 100 * np.mean([by["SUBJ L0-8"][i] == "correct" for i in rows])
H41 = s_rate >= 10 and d41[0] >= 5 and d41[1] > 0
print(f"H4.1: SUBJ 정답 {s_rate:.1f}% (>=10), 오답 공여 있는 {len(hw)}행에서 SUBJ - SUBJw {d41} (>=5, 하한>0) -> {H41}")
E_, S_, Bf, Df, SE = fix("END L20"), fix("SUBJ L0-8"), fix("B c1"), fix("D0 c1"), fix("SUBJ+END")
e_rate = len(E_) / len(rows)
ov_B = len(Bf & E_) / max(len(Bf), 1); ov_D = len(Df & E_) / max(len(Df), 1)
sonly = S_ - E_; b_all = len(Bf) / len(rows); b_sonly = len(Bf & sonly) / max(len(sonly), 1)
H42 = ov_B >= 2 * e_rate and ov_B > ov_D and b_sonly <= b_all / 2
print(f"H4.2: B 로 고친 {len(Bf)}행 중 END 로도 고친 비율 {100*ov_B:.1f}% (END 전체 {100*e_rate:.1f}% 의 2배 이상?), D0 로 고친 {len(Df)}행 중 {100*ov_D:.1f}%; "
      f"SUBJ 만 고친 {len(sonly)}행에서 B 고침 {100*b_sonly:.1f}% (B 전체 {100*b_all:.1f}% 의 절반 이하?) -> {H42}")
print("판정:", "아는 사실 실패 = 접근 실패 + 답 상태 실패. B 의 한계는 접근 실패 몫으로 설명됨" if H41 and H42 else ("두 단계는 있으나 B 의 한계를 이것으로 설명하지 못함" if H41 else "주어 위치 초기 층 교체로는 고쳐지지 않음. 접근 실패 몫을 주장하지 않음"))
n = len(rows); parts = dict(SUBJ만=len(S_ - E_), END만=len(E_ - S_), 둘다=len(S_ & E_), 합쳐야만=len(SE - S_ - E_), 안고쳐짐=n - len(S_ | E_ | SE))
print("분해 (행 수, %):", {k: (v, round(100 * v / n, 1)) for k, v in parts.items()})
print("겹침: B∩END", len(Bf & E_), "B∩SUBJ", len(Bf & S_), "D0∩END", len(Df & E_), "D0∩SUBJ", len(Df & S_), "B∩D0", len(Bf & Df))
