"""E52 분석 (CPU).  H52.1 (주의 막기), H52.2 (선택 헤드의 주어 주의), H52.3 (Brel 조향과 선택 헤드 기여).
경로: base 실행 채점으로 오답 (entity/broken) 인 행을 같은 문구 Qwen 정오로 'dlm' (Qwen 맞힘) / 'shared' (Qwen 틀림).  abstain 제외.
CI: 사실 (fid) 군집 부트스트랩 B=4000.
헤드 선택 (결과 보기 전 고정): 주어 5 겹 (grp) 중 시험 겹을 뺀 학습 겹에서, 사실 안 (맞힘 평균 - 틀림 평균) 의 step 0 헤드 DLA (금 - 경쟁) 를 사실 평균한 점수 상위 K=10 (양수 방향).  민감도 K=5, 20.
usage: python followup_mk/e52_analyze.py [--dir followup_mk/e52] [--out followup_mk/e52_summary.json]"""
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import rows, fid, grp, mid, OUT
arg = lambda k_, d=None: sys.argv[sys.argv.index(k_) + 1] if k_ in sys.argv else d
DIR = arg("--dir", f"{OUT}/e52"); FOUT = arg("--out", f"{OUT}/e52_summary.json"); BOOT = 4000; K = 10
POS = {int(k_): v for k_, v in json.load(open(f"{OUT}/e52/pos.json")).items()}

def load_gen(c, d=None):
    d = d or DIR; G = {}
    for f in sorted(glob.glob(f"{d}/gen_{c}_s*.jsonl")):
        for l in open(f, encoding="utf-8"): d = json.loads(l); G[d["row"]] = d
    return G
def load_arr(kind, c, d=None):
    d = d or DIR; A = {}
    for f in sorted(glob.glob(f"{d}/{kind}_{c}_s*.npy")):
        sh = f.rsplit("_s", 1)[1][:-4]; rr = json.load(open(f"{d}/rows_{'gen' if kind == 'att' else 'dla'}_s{sh}.json"))["rows"]; X = np.load(f, mmap_mode="r")
        for j, i in enumerate(rr): A[i] = np.asarray(X[j], dtype=np.float32)
    return A
def load_dlaj(c):
    G = {}
    for f in sorted(glob.glob(f"{DIR}/dla_{c}_s*.jsonl")):
        for l in open(f, encoding="utf-8"): d = json.loads(l); G[d["row"]] = d
    return G

# ---- 사실 군집 부트스트랩
def _W(nf, seed):
    rng = np.random.default_rng(seed); ix = rng.integers(0, nf, (BOOT, nf)); W = np.zeros((BOOT, nf), np.float32)
    np.add.at(W, (np.repeat(np.arange(BOOT), nf), ix.ravel()), 1); return W
_WC = {}
def W(nf, seed=0):
    if (nf, seed) not in _WC: _WC[(nf, seed)] = _W(nf, seed)
    return _WC[(nf, seed)]
r4 = lambda v: round(float(v), 4)
def cb_mean(f_, v, seed=0):
    """행 값의 평균, 사실 군집 부트스트랩."""
    f_ = np.asarray(f_); v = np.asarray(v, float)
    if len(v) == 0: return None
    u, inv = np.unique(f_, return_inverse=True); S = np.bincount(inv, v, len(u)); N = np.bincount(inv, None, len(u)); w = W(len(u), seed)
    b = (w @ S) / np.maximum(w @ N, 1e-9); return [r4(v.mean()), r4(np.percentile(b, 2.5)), r4(np.percentile(b, 97.5)), int(len(v)), int(len(u))]
def cb_diff(fa, va, fb, vb, seed=0):
    """두 행 집단 평균의 차 (A - B), 두 집단의 사실 합집합을 군집으로 재표집."""
    fa, fb = np.asarray(fa), np.asarray(fb); va, vb = np.asarray(va, float), np.asarray(vb, float)
    u = np.unique(np.concatenate([fa, fb])); ia = np.searchsorted(u, fa); ib = np.searchsorted(u, fb); w = W(len(u), seed)
    Sa, Na = np.bincount(ia, va, len(u)), np.bincount(ia, None, len(u)); Sb, Nb = np.bincount(ib, vb, len(u)), np.bincount(ib, None, len(u))
    b = (w @ Sa) / np.maximum(w @ Na, 1e-9) - (w @ Sb) / np.maximum(w @ Nb, 1e-9)
    return [r4(va.mean() - vb.mean()), r4(np.percentile(b, 2.5)), r4(np.percentile(b, 97.5)), int(len(va)), int(len(vb)), int(len(u))]
def within(rowsA, rowsB, val, seed=0):
    """같은 사실 안 (A 평균 - B 평균) 의 사실 평균.  두 집단이 모두 있는 사실만."""
    a = collections.defaultdict(list); b = collections.defaultdict(list)
    for i in rowsA:
        if val(i) is not None: a[fid[i]].append(val(i))
    for i in rowsB:
        if val(i) is not None: b[fid[i]].append(val(i))
    fs = sorted(set(a) & set(b)); d = [np.mean(a[f_]) - np.mean(b[f_]) for f_ in fs]
    if not fs: return None
    return cb_mean(fs, d, seed)[:3] + [len(fs)]

base = load_gen("base"); R = sorted(base)
lab = {i: base[i]["new_type"] for i in R}
route = {i: ("correct" if lab[i] == "correct" else (("dlm" if POS[i]["qwen_correct"] else "shared") if lab[i] in ("entity", "broken") else None)) for i in R}
S = {"n_rows": len(R), "labels": dict(collections.Counter(lab.values())), "routes": dict(collections.Counter(v for v in route.values() if v))}
# E7 meta 와 일치 (묶음 구성이 달라 완전 일치는 아님)
e7 = {}
for l in open(f"{OUT}/e7/meta.jsonl", encoding="utf-8"):
    d = json.loads(l); e7[d["row"]] = d
S["agree_e7_answer"] = r4(np.mean([e7[i]["answer"] == base[i]["answer"] for i in R if i in e7]))
S["agree_e7_correct"] = r4(np.mean([(e7[i]["new_type"] == "correct") == (lab[i] == "correct") for i in R if i in e7]))
lg = lambda r: np.log2(r + 1.0)

# ================= (a) 주의 막기
A = {}
SUBD = f"{DIR}/sub" if os.path.isdir(f"{DIR}/sub") else DIR   # 막기·조향 생성은 부분 집합 (오답 step2 마스크 전부 + 정답 800) 에서, 자체 묶음 구성
conds = [c for c in ["ko_all", "ko_0_9", "ko_10_19", "ko_20_27", "rnd_all", "rnd_20_27", "rnd_0_9", "rnd_10_19"] if glob.glob(f"{SUBD}/gen_{c}_s*.jsonl")]
G = {c: load_gen(c, SUBD) for c in conds}
BSUB = load_gen("basesub", SUBD) or base   # 부분 집합 묶음 구성의 plain base
REF = load_gen("empty_eager", SUBD) or base   # 막기 조건과 같은 계산 경로 (eager 다시 계산, 막는 키 없음).  없으면 base.
A["reference"] = "empty_eager" if REF is not base else "base"
if REF is not base:
    A["empty_eager_vs_basesub"] = {"answer_eq": r4(np.mean([REF[i]["answer"] == BSUB[i]["answer"] for i in REF if i in BSUB])),
                                "rank2_eq": r4(np.mean([REF[i]["rank"][2] == BSUB[i]["rank"][2] for i in REF if i in BSUB and BSUB[i]["rank"][2] >= 0])),
                                "mean_abs_dlog2rank2": r4(np.mean([abs(np.log2(REF[i]["rank"][2] + 1) - np.log2(BSUB[i]["rank"][2] + 1)) for i in REF if i in BSUB and BSUB[i]["rank"][2] >= 0])),
                                "correct_eq": r4(np.mean([(REF[i]["new_type"] == "correct") == (BSUB[i]["new_type"] == "correct") for i in REF if i in BSUB]))}
    A["basesub_vs_fullbase_answer_eq"] = r4(np.mean([BSUB[i]["answer"] == base[i]["answer"] for i in BSUB])) if BSUB is not base else None
elig = [i for i in R if i in REF and REF[i]["rank"][2] >= 0 and route[i] is not None]
A["n_eligible"] = {g: sum(route[i] == g for i in elig) for g in ("dlm", "shared", "correct")}
A["n_slot0_masked_step2_frac"] = r4(np.mean([base[i]["rank"][2] >= 0 for i in R]))
for c in conds:
    Gc = G[c]; ok = [i for i in elig if i in Gc]
    # 막기가 없던 행 (칸 0 이 step 1 에 확정) 은 base 와 같아야 함
    non = [i for i in Gc if i in REF and REF[i]["rank"][2] < 0]
    ok = [i for i in ok if i in REF]
    dd = {i: lg(REF[i]["rank"][2]) - lg(Gc[i]["rank"][2]) for i in ok}
    dbase = {i: lg(BSUB[i]["rank"][2]) - lg(Gc[i]["rank"][2]) for i in ok if i in BSUB}
    rec = {"noknock_rows_identical_to_ref": r4(np.mean([Gc[i]["canvas"] == REF[i]["canvas"] for i in non])) if non else None,
           "nk_step2_mean": r4(np.mean([Gc[i]["nk"][2] for i in ok]))}
    for g in ("dlm", "shared", "correct"):
        ig = [i for i in ok if route[i] == g]
        rec[g] = {"d_log2rank_step2": cb_mean([fid[i] for i in ig], [dd[i] for i in ig]), "d_log2rank_step2_vs_basesub": cb_mean([fid[i] for i in ig if i in dbase], [dbase[i] for i in ig if i in dbase]),
                  "top1_step2_ref": r4(np.mean([REF[i]["rank"][2] == 0 for i in ig])), "final_correct_ref": r4(np.mean([REF[i]["new_type"] == "correct" for i in ig])),
                  "final_correct_cond_minus_ref": cb_mean([fid[i] for i in ig], [float(Gc[i]["new_type"] == "correct") - float(REF[i]["new_type"] == "correct") for i in ig]),
                  "improved": r4(np.mean([dd[i] > 0 for i in ig])), "worse": r4(np.mean([dd[i] < 0 for i in ig])),
                  "top1_step2_base": r4(np.mean([base[i]["rank"][2] == 0 for i in ig])), "top1_step2_cond": r4(np.mean([Gc[i]["rank"][2] == 0 for i in ig])),
                  "final_correct_fullbase": r4(np.mean([lab[i] == "correct" for i in ig])),
                  "final_correct_cond": cb_mean([fid[i] for i in ig], [Gc[i]["new_type"] == "correct" for i in ig]),
                  "slot0_gold_final_ref": r4(np.mean([REF[i]["s0_tok"] == REF[i]["gold_id"] for i in ig])),
                  "slot0_gold_final_cond": r4(np.mean([Gc[i]["s0_tok"] == Gc[i]["gold_id"] for i in ig])),
                  "s0_last_mean_ref": r4(np.mean([REF[i]["s0_last"] for i in ig])), "s0_last_mean_cond": r4(np.mean([Gc[i]["s0_last"] for i in ig]))}
    D_ = lambda g: ([fid[i] for i in ok if route[i] == g], [dd[i] for i in ok if route[i] == g])
    rec["route_diff_dlm_minus_shared"] = cb_diff(*D_("dlm"), *D_("shared"))
    fc_ = lambda i: float(Gc[i]["new_type"] == "correct") - float(REF[i]["new_type"] == "correct")
    rec["route_diff_final_correct_change"] = cb_diff([fid[i] for i in ok if route[i] == "dlm"], [fc_(i) for i in ok if route[i] == "dlm"],
                                                     [fid[i] for i in ok if route[i] == "shared"], [fc_(i) for i in ok if route[i] == "shared"])
    A[c] = rec
for lay in ("all", "0_9", "10_19", "20_27"):
    ko, rn = f"ko_{lay}", f"rnd_{lay}"
    if ko in G and rn in G:
        for g in ("dlm", "shared", "correct", "wrong"):
            ig = [i for i in elig if (route[i] == g or (g == "wrong" and route[i] in ("dlm", "shared"))) and i in G[ko] and i in G[rn]]
            v = [(lg(G[rn][i]["rank"][2]) - lg(G[ko][i]["rank"][2])) for i in ig]   # ko 의 순위 개선 - rnd 의 순위 개선
            fc = [float(G[ko][i]["new_type"] == "correct") - float(G[rn][i]["new_type"] == "correct") for i in ig]
            A[f"{ko}_vs_{rn}_{g}"] = {"d_log2rank_step2": cb_mean([fid[i] for i in ig], v), "final_correct": cb_mean([fid[i] for i in ig], fc)}
if "ko_all" in G and "rnd_all" in G:
    rd = A["ko_all"]["route_diff_dlm_minus_shared"]; kv = A["ko_all_vs_rnd_all_dlm"]["d_log2rank_step2"]
    A["H52.1"] = {"route_diff_lo_gt0": bool(rd[1] > 0), "ko_gt_rnd_dlm_lo_gt0": bool(kv[1] > 0), "verdict": bool(rd[1] > 0 and kv[1] > 0)}
S["a_knockout"] = A

# ================= (b) 헤드 기여
Db = load_arr("dla", "base"); Jb = load_dlaj("base")
use = [i for i in R if i in Db and Jb[i]["comp"] >= 0 and route[i] is not None]
B_ = {"n_rows_with_comp": len(use), "n_facts_no_comp": len({fid[i] for i in R if i in Jb and Jb[i]["comp"] < 0})}
lin = np.array([[Jb[i]["logit_diff"], Jb[i]["lin_diff"]] for i in use])
hs = np.array([(Db[i][0] - Db[i][1]).sum() for i in use])
B_["lin_check"] = {"mean_abs_err": r4(np.abs(lin[:, 0] - lin[:, 1]).mean()), "mean_abs_logit_diff": r4(np.abs(lin[:, 0]).mean()),
                   "corr_heads_sum_vs_logit_diff": r4(np.corrcoef(hs, lin[:, 0])[0, 1]), "heads_sum_mean": r4(hs.mean()), "logit_diff_mean": r4(lin[:, 0].mean())}
dgc = {i: Db[i][0] - Db[i][1] for i in use}
iscor = {i: route[i] == "correct" for i in use}
def head_scores(train):
    a = collections.defaultdict(list); b = collections.defaultdict(list)
    for i in train: (a if iscor[i] else b)[fid[i]].append(dgc[i])
    fs = sorted(set(a) & set(b))
    return np.mean([np.mean(a[f_], 0) - np.mean(b[f_], 0) for f_ in fs], 0), len(fs)
SEL = {}; SC = {}
for kk in (5, 10, 20):
    SEL[kk] = {}
for g in range(5):
    sc, nf_ = head_scores([i for i in use if grp[i] != g]); SC[g] = sc
    order = np.argsort(-sc.ravel())
    for kk in (5, 10, 20): SEL[kk][g] = [(int(o // 28), int(o % 28)) for o in order[:kk]]
cnt = collections.Counter(h for g in range(5) for h in SEL[K][g])
B_["selected_by_fold"] = {g: [f"{l}.{h}" for l, h in SEL[K][g]] for g in range(5)}
B_["stable_heads"] = {f"{l}.{h}": c for (l, h), c in cnt.most_common() if c >= 3}
B_["score_allfold_top20"] = None
allsc, nfa = head_scores(use); top = np.argsort(-allsc.ravel())[:20]
B_["score_allfold_top20"] = [[f"{o // 28}.{o % 28}", r4(allsc.ravel()[o])] for o in top]
B_["score_allfold_bottom10"] = [[f"{o // 28}.{o % 28}", r4(allsc.ravel()[o])] for o in np.argsort(allsc.ravel())[:10]]
B_["layer_hist_selected"] = dict(collections.Counter(l for g in range(5) for l, h in SEL[K][g]))
def ssum(i, arr, kk=K, which=None):
    hh = SEL[kk][grp[i]]; return float(sum(arr[l, h] for l, h in hh))
cor = [i for i in use if route[i] == "correct"]; wr = [i for i in use if route[i] in ("dlm", "shared")]
dl = [i for i in use if route[i] == "dlm"]; shd = [i for i in use if route[i] == "shared"]
hasS = lambda i: POS[i]["se"] is not None
# 선택 헤드 DLA 의 겹밖 일반화 (확인용)
B_["heldout_sel_dla_within_fact_cor_minus_wrong"] = within(cor, wr, lambda i: ssum(i, dgc[i]))
B_["heldout_alldla_within_fact_cor_minus_wrong"] = within(cor, wr, lambda i: float(dgc[i].sum()))
# H52.2 주어 끝 주의 (step 0)
subj = lambda kk: (lambda i: ssum(i, Db[i][2], kk) if hasS(i) else None)
H2 = {}
for kk in (5, 10, 20):
    H2[f"K{kk}"] = {"cor_minus_wrong": within(cor, wr, subj(kk)), "cor_minus_dlm": within(cor, dl, subj(kk)), "cor_minus_shared": within(cor, shd, subj(kk)),
                    "dlm_minus_shared_rows": cb_diff([fid[i] for i in dl if hasS(i)], [subj(kk)(i) for i in dl if hasS(i)], [fid[i] for i in shd if hasS(i)], [subj(kk)(i) for i in shd if hasS(i)]),
                    "mean_by_route": {g: r4(np.mean([subj(kk)(i) for i in use if route[i] == g and hasS(i)])) for g in ("correct", "dlm", "shared")}}
# 대조: 같은 층의 무작위 헤드 (200 회)
rng = np.random.default_rng(52); null = []
for t in range(200):
    rs = {g: [(l, int(rng.integers(28))) for l, h in SEL[K][g]] for g in range(5)}
    v = lambda i: float(sum(Db[i][2][l, h] for l, h in rs[grp[i]])) if hasS(i) else None
    null.append(within(cor, wr, v)[0])
H2["random_same_layer_null_K10"] = {"mean": r4(np.mean(null)), "p2.5": r4(np.percentile(null, 2.5)), "p97.5": r4(np.percentile(null, 97.5)),
                                    "frac_null_ge_selected": r4(np.mean(np.array(null) >= H2["K10"]["cor_minus_wrong"][0]))}
H2["all_heads_mean_cor_minus_wrong"] = within(cor, wr, lambda i: float(Db[i][2].mean()) if hasS(i) else None)
H2["verdict"] = bool(H2["K10"]["cor_minus_wrong"][1] > 0)
B_["H52.2_subject_attention"] = H2
# 확정 칸 주의 (첫 확정 뒤 step 2, 칸 0 마스크인 행), 칸 0 확정 step 까지 평균
Ab = load_arr("att", "base")
m2 = lambda i: base[i]["rank"][2] >= 0
comm2 = lambda i: ssum(i, Ab[i][0]) if (i in Ab and m2(i)) else None
commM = lambda i: ssum(i, Ab[i][2]) if (i in Ab and m2(i)) else None
B_["committed_attention"] = {
    "step2": {"cor_minus_wrong": within(cor, wr, comm2), "cor_minus_dlm": within(cor, dl, comm2), "cor_minus_shared": within(cor, shd, comm2),
              "dlm_minus_shared_rows": cb_diff([fid[i] for i in dl if comm2(i) is not None], [comm2(i) for i in dl if comm2(i) is not None],
                                               [fid[i] for i in shd if comm2(i) is not None], [comm2(i) for i in shd if comm2(i) is not None]),
              "mean_by_route": {g: r4(np.mean([comm2(i) for i in use if route[i] == g and comm2(i) is not None])) for g in ("correct", "dlm", "shared")}},
    "mean_step2_to_commit": {"cor_minus_wrong": within(cor, wr, commM), "mean_by_route": {g: r4(np.mean([commM(i) for i in use if route[i] == g and commM(i) is not None])) for g in ("correct", "dlm", "shared")}},
    "all_heads_step2_cor_minus_wrong": within(cor, wr, lambda i: float(Ab[i][0].mean()) if (i in Ab and m2(i)) else None)}
# 안정 헤드 설명
desc = {}
for hn in B_["stable_heads"]:
    l, h = map(int, hn.split("."))
    def mm_(f, g): v = [f(i) for i in use if route[i] == g]; v = [x for x in v if x is not None]; return r4(np.mean(v)) if v else None
    desc[hn] = {k_: {g: mm_(f, g) for g in ("correct", "dlm", "shared")} for k_, f in [
        ("dla_gold_minus_comp", lambda i: float(dgc[i][l, h])), ("dla_gold", lambda i: float(Db[i][0][l, h])), ("dla_comp", lambda i: float(Db[i][1][l, h])),
        ("att_subj_last", lambda i: float(Db[i][2][l, h]) if hasS(i) else None), ("att_question_last", lambda i: float(Db[i][4][l, h])),
        ("att_pos0", lambda i: float(Db[i][5][l, h])), ("att_canvas_masks", lambda i: float(Db[i][3][l, h])),
        ("att_committed_step2", lambda i: float(Ab[i][0][l, h]) if (i in Ab and m2(i)) else None)]}
B_["stable_head_profiles"] = desc
S["b_heads"] = B_

# ================= (c) 조향
C = {}
if glob.glob(f"{DIR}/dla_brel_s*.npy") and glob.glob(f"{DIR}/dla_rand_s*.npy"):
    Dr = load_arr("dla", "brel"); Dn = load_arr("dla", "rand")
    uc = [i for i in use if i in Dr and i in Dn]
    sel_layers = collections.Counter(l for g in range(5) for l, h in SEL[K][g])
    C["frac_selected_heads_after_layer20"] = r4(sum(v for l, v in sel_layers.items() if l > 20) / max(1, sum(sel_layers.values())))
    dB = lambda i: ssum(i, Dr[i][0] - Dr[i][1]) - ssum(i, dgc[i]); dN = lambda i: ssum(i, Dn[i][0] - Dn[i][1]) - ssum(i, dgc[i])
    gB = lambda i: ssum(i, Dr[i][0]) - ssum(i, Db[i][0]); gN = lambda i: ssum(i, Dn[i][0]) - ssum(i, Db[i][0])
    sB = lambda i: ssum(i, Dr[i][2]) - ssum(i, Db[i][2]); sN = lambda i: ssum(i, Dn[i][2]) - ssum(i, Db[i][2])
    for g in ("all", "wrong", "dlm", "shared", "correct"):
        ig = [i for i in uc if g == "all" or route[i] == g or (g == "wrong" and route[i] in ("dlm", "shared"))]; fi = [fid[i] for i in ig]
        C[g] = {"dDLA_gc_brel": cb_mean(fi, [dB(i) for i in ig]), "dDLA_gc_rand": cb_mean(fi, [dN(i) for i in ig]), "dDLA_gc_brel_minus_rand": cb_mean(fi, [dB(i) - dN(i) for i in ig]),
                "dDLA_gold_brel_minus_rand": cb_mean(fi, [gB(i) - gN(i) for i in ig]),
                "dSubjAtt_brel": cb_mean([fid[i] for i in ig if hasS(i)], [sB(i) for i in ig if hasS(i)]), "dSubjAtt_rand": cb_mean([fid[i] for i in ig if hasS(i)], [sN(i) for i in ig if hasS(i)]),
                "dSubjAtt_brel_minus_rand": cb_mean([fid[i] for i in ig if hasS(i)], [sB(i) - sN(i) for i in ig if hasS(i)]),
                "dDLA_gc_allheads_brel_minus_rand": cb_mean(fi, [float((Dr[i][0] - Dr[i][1]).sum() - (Dn[i][0] - Dn[i][1]).sum()) for i in ig])}
    Jr, Jn = load_dlaj("brel"), load_dlaj("rand")
    for g in ("wrong", "dlm", "shared", "correct"):
        ig = [i for i in uc if (route[i] == g or (g == "wrong" and route[i] in ("dlm", "shared")))]
        C[g]["rank0_top1"] = {"base": r4(np.mean([Jb[i]["rank0"] == 0 for i in ig])), "brel": r4(np.mean([Jr[i]["rank0"] == 0 for i in ig])), "rand": r4(np.mean([Jn[i]["rank0"] == 0 for i in ig]))}
    C["H52.3"] = {"verdict": bool(C["all"]["dDLA_gc_brel_minus_rand"][1] > 0)}
    Gb, Gn = load_gen("brel", SUBD), load_gen("rand", SUBD); GB0 = BSUB
    if Gb and Gn:
        for g in ("wrong", "dlm", "shared", "correct"):
            ig = [i for i in R if route[i] == g or (g == "wrong" and route[i] in ("dlm", "shared"))]; ig = [i for i in ig if i in Gb and i in Gn and i in GB0]; fi = [fid[i] for i in ig]
            C[g]["s0_last"] = {"base": r4(np.mean([GB0[i]["s0_last"] for i in ig])), "brel_minus_base": cb_mean(fi, [Gb[i]["s0_last"] - GB0[i]["s0_last"] for i in ig]),
                               "rand_minus_base": cb_mean(fi, [Gn[i]["s0_last"] - GB0[i]["s0_last"] for i in ig]),
                               "brel_minus_rand": cb_mean(fi, [Gb[i]["s0_last"] - Gn[i]["s0_last"] for i in ig])}
            C[g]["final_correct"] = {"base": r4(np.mean([GB0[i]["new_type"] == "correct" for i in ig])), "brel": cb_mean(fi, [Gb[i]["new_type"] == "correct" for i in ig]),
                                     "rand": cb_mean(fi, [Gn[i]["new_type"] == "correct" for i in ig])}
        # 조향 뒤 확정 칸 주의 (선택 헤드, step 2)
        Ar, An, A0 = load_arr("att", "brel", SUBD), load_arr("att", "rand", SUBD), (load_arr("att", "basesub", SUBD) or Ab)
        for g in ("dlm", "shared", "correct"):
            ig = [i for i in R if route[i] == g and i in use and i in A0 and i in Ar and i in An and GB0[i]["rank"][2] >= 0 and Gb[i]["rank"][2] >= 0 and Gn[i]["rank"][2] >= 0]
            C[g]["committed_att_step2"] = {"base": r4(np.mean([ssum(i, A0[i][0]) for i in ig])), "brel": r4(np.mean([ssum(i, Ar[i][0]) for i in ig])), "rand": r4(np.mean([ssum(i, An[i][0]) for i in ig])), "n": len(ig)}
S["c_steering"] = C
json.dump(S, open(FOUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False, indent=1)[:12000])
