"""E48 (PREREG.md, CPU): 층 x 차원 x step x 토큰 위치 4 축 격자.  usage: python followup_mk/e48_grid.py <layer_index 0..6> | --combine
자료 E7 hid [n, step(0,4,8), 층(4..27), 위치(프롬프트 끝, 질문 끝, 주어 끝), 3584].  행 = 1~4/5 섞인 사실 (E7 실행 채점, abstain 제외).
지표: 같은 사실 안 정오 AUROC (사실별 평균).  사실 평균 제거, 주어 5 겹 겹밖 (PCA, 방향, 좌표 선택, probe 모두 학습 겹)."""
import collections, json, sys, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from common import k, fid, grp, NF, OUT
D7 = f"{OUT}/e7"; cfg = json.load(open(f"{D7}/config.json")); R7 = np.load(f"{D7}/rows.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
J = np.array([j for j, r in enumerate(R7) if int(r) in meta and 1 <= k[int(r)] <= 4 and meta[int(r)]["new_type"] != "abstain"])
RR = R7[J]; corr = np.array([meta[int(r)]["new_type"] == "correct" for r in RR]); ff = fid[RR]; gg = grp[RR]; kk = k[RR]
byf = collections.defaultdict(list)
for t, f in enumerate(ff): byf[f].append(t)
mixed = {f: v for f, v in byf.items() if 0 < corr[v].sum() < len(v)}
keep = np.array(sorted(t for v in mixed.values() for t in v)); J, RR, corr, ff, gg, kk = J[keep], RR[keep], corr[keep], ff[keep], gg[keep], kk[keep]
byf = collections.defaultdict(list)
for t, f in enumerate(ff): byf[f].append(t)
n = len(RR); METH = ["DM1", "PCA4", "PCA16", "PCA64", "PCA256", "PCA1024", "RAW16", "RAW64", "RAW256"]
STEPS, LAYERS, POS = cfg["steps"], cfg["layers"], cfg["positions"]

def center(X):
    X = X.copy()
    for v in byf.values(): X[v] -= X[v].mean(0)
    return X
def fact_diffs(Xc, idx):
    s = set(idx.tolist()); out = []
    for v in byf.values():
        v = [t for t in v if t in s]
        if v and 0 < corr[v].sum() < len(v): out.append(Xc[v][corr[v]].mean(0) - Xc[v][~corr[v]].mean(0))
    return np.array(out)
def lr(Ztr, ytr, Zte):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    return LogisticRegression(C=0.01, max_iter=2000).fit((Ztr - mu) / sd, ytr).decision_function((Zte - mu) / sd)

if "--combine" not in sys.argv:
    li = int(sys.argv[1]); H = np.load(f"{D7}/hid.npy", mmap_mode="r"); S = np.full((len(STEPS), len(POS), len(METH), n), np.nan, np.float32)
    for si in range(len(STEPS)):
        for pi in range(len(POS)):
            t0 = time.time(); Xc = center(np.asarray(H[J, si, li, pi], dtype=np.float32))
            for g in range(NF):
                tr, te = np.where(gg != g)[0], np.where(gg == g)[0]
                Dm = fact_diffs(Xc, tr); B = Dm.mean(0); B /= np.linalg.norm(B); S[si, pi, 0, te] = Xc[te] @ B
                p = PCA(1024, svd_solver="randomized", random_state=0).fit(Xc[tr]); Ztr, Zte = p.transform(Xc[tr]), p.transform(Xc[te])
                for q, r in enumerate([4, 16, 64, 256, 1024]): S[si, pi, 1 + q, te] = lr(Ztr[:, :r], corr[tr], Zte[:, :r])
                tst = np.abs(Dm.mean(0) / (Dm.std(0, ddof=1) / np.sqrt(len(Dm)) + 1e-9)); order = np.argsort(-tst)
                for q, r in enumerate([16, 64, 256]): c_ = order[:r]; S[si, pi, 6 + q, te] = lr(Xc[tr][:, c_], corr[tr], Xc[te][:, c_])
            print(f"층{LAYERS[li]} s{STEPS[si]} {POS[pi]}: {time.time() - t0:.0f}s", flush=True)
    np.save(f"{OUT}/e48_scores_L{LAYERS[li]}.npy", S); print("DONE", flush=True); sys.exit()

# ------------------------------------------------------------ 합치기
hi_f = {f: v for f, v in byf.items() if kk[v[0]] >= 3}
def per_fact(sc, facts): return {f: roc_auc_score(corr[v], sc[v]) for f, v in facts.items() if 0 < corr[v].sum() < len(v)}
rng = np.random.default_rng(0)
def boot_diff(a, b, B=2000):
    fs = sorted(set(a) & set(b)); d = np.array([a[f] - b[f] for f in fs]); bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
PF = {}; O = {"행": int(n), "섞인 사실": len(byf), "AT-high 사실": len(hi_f), "steps": STEPS, "layers": LAYERS, "positions": POS, "methods": METH, "격자": {}}
for li, L in enumerate(LAYERS):
    S = np.load(f"{OUT}/e48_scores_L{L}.npy")
    for si, s in enumerate(STEPS):
        for pi, pos in enumerate(POS):
            for mi, m in enumerate(METH):
                a = per_fact(S[si, pi, mi], byf); h = per_fact(S[si, pi, mi], hi_f); PF[(L, s, pos, m)] = a
                O["격자"][f"L{L}|s{s}|{pos}|{m}"] = [round(float(np.mean(list(a.values()))), 4), round(float(np.mean(list(h.values()))), 4)]
G = O["격자"]; best = lambda pred: max(((kk_, v[0]) for kk_, v in G.items() if pred(kk_)), key=lambda x: x[1])
O["최고"] = {"전체": best(lambda x: True), "step0": best(lambda x: "|s0|" in x), "step0 학습없음 DM1": best(lambda x: "|s0|" in x and x.endswith("DM1"))}
# H48.1: step 0 최고 층 (차원·위치 무관 최고)
b0 = O["최고"]["step0"][0]; O["H48.1"] = dict(step0_최고=O["최고"]["step0"], 판정=int(b0.split("|")[0][1:]) >= 20)
d = boot_diff(PF[(20, 0, "prompt_end", "PCA64")], PF[(20, 0, "prompt_end", "PCA1024")])
O["H48.2"] = dict(PCA64_minus_PCA1024=d, 판정=bool(d[0] >= -0.01))
wins = []
for L in LAYERS:
    v = {pos: max(G[f"L{L}|s0|{pos}|{m}"][0] for m in METH) for pos in POS}; wins.append(v["prompt_end"] > max(v["q_last"], v["subj_last"]))
O["H48.3"] = dict(층별_프롬프트끝_승=[bool(w) for w in wins], 판정=bool(sum(wins) > len(wins) / 2))
O["비교 CI"] = {
    "L20 s0 프롬프트끝: PCA1024 - DM1": boot_diff(PF[(20, 0, "prompt_end", "PCA1024")], PF[(20, 0, "prompt_end", "DM1")]),
    "L20 s0 프롬프트끝: RAW64 - PCA64": boot_diff(PF[(20, 0, "prompt_end", "RAW64")], PF[(20, 0, "prompt_end", "PCA64")]),
    "L20 PCA256 프롬프트끝: s8 - s0": boot_diff(PF[(20, 8, "prompt_end", "PCA256")], PF[(20, 0, "prompt_end", "PCA256")]),
    "L20 s0 PCA256: 프롬프트끝 - 주어끝": boot_diff(PF[(20, 0, "prompt_end", "PCA256")], PF[(20, 0, "subj_last", "PCA256")]),
    "L20 s0 PCA256: 프롬프트끝 - 질문끝": boot_diff(PF[(20, 0, "prompt_end", "PCA256")], PF[(20, 0, "q_last", "PCA256")]),
    "s0 프롬프트끝 PCA256: L24 - L12": boot_diff(PF[(24, 0, "prompt_end", "PCA256")], PF[(12, 0, "prompt_end", "PCA256")]),
}
json.dump(O, open(f"{OUT}/e48_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({x: O[x] for x in ("행", "섞인 사실", "AT-high 사실", "최고", "H48.1", "H48.2", "H48.3", "비교 CI")}, ensure_ascii=False, indent=1)); print("DONE")
