"""Three questions on the same data layout (Dream or Qwen, 4,000 extension facts x 5 wordings), one pipeline.

  1. Pick-the-wording: for EVERY fact (not only mixed ones) answer with the wording that scores highest. Overall accuracy vs
     the original wording / a random wording / majority vote / the ceiling "any wording right".
  2. Dream vs Qwen: run the identical pipeline on both models' features (--model dream|qwen).
  3. Decomposition: out-of-fold scores of confidence, wording-text-only (TF-IDF), SAE-20 and raw hidden state are stacked in every
     subset; unique contribution = full model minus the model without that signal; plus within-fact correlations of the scores.

Protocol: subject-level 5-fold CV. Signals are fitted on the mixed (>=1 right, >=1 wrong) facts of the training subjects and scored on
ALL rows of the test subjects. Run twice: "all" (5 wordings) and "para" (paraphrases v1-v4 only).
Label = "contain" (gold alias as a whole-word span of a short answer).

Usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=sae_analysis:grounded_dream python sae_analysis/signals_unified.py --model dream --dir sae_analysis/features_dream_step0_ext
"""
import argparse
import collections
import itertools
import json
import os
import sys
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "grounded_dream"); sys.path.insert(0, "sae_analysis")
from string_grader import norm  # noqa: E402

warnings.filterwarnings("ignore")
VERS = ("question", "v1", "v2", "v3", "v4")
DREAM_SAE = "sae/runs/dream_pilot_200M/dream-7b_L20_shared-0.2-0.5-0.8_s5_200M"
QWEN_SAE = "sae/qwen2.5-7b_L20_batchtopk_16k_k100_20M"
N_BOOT = 2000


def within_auc(s, y):
    a, b = s[y == 1], s[y == 0]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    d = a[:, None] - b[None, :]
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size


def center(X, f):
    Xc = X.astype(np.float64).copy()
    for u in np.unique(f):
        m = f == u; Xc[m] -= Xc[m].mean(0)
    return Xc


def lr_scores(Xtr, ytr, ftr, Xte, C):
    sc = StandardScaler().fit(center(Xtr, ftr))
    clf = LogisticRegression(C=C, max_iter=400).fit(sc.transform(center(Xtr, ftr)), ytr)
    return clf.decision_function(sc.transform(Xte))


def load_sae(kind):
    import torch
    if kind == "dream":
        from sae_lens import SAE
        sae = SAE.load_from_disk(DREAM_SAE, device="cpu").eval()
        enc = sae.encode
    else:
        from extract_qwen_features import JumpReLUSAE
        enc = JumpReLUSAE(QWEN_SAE, "cpu").encode

    def f(H):
        out = []
        with torch.no_grad():
            for b in range(0, len(H), 512):
                out.append(enc(torch.from_numpy(H[b:b + 512])).numpy())
        return np.concatenate(out)
    return f


def boot_ci(d, B):
    lo, hi = np.percentile(d[B].mean(1), [2.5, 97.5])
    return f"{d.mean():+.1%} [{lo:+.1%},{hi:+.1%}]"


def run(a, meta, conf, hl, A_last, txt, subj_uri, setting, outdir, extra=None):
    sid = np.array([m["sample_id"] for m in meta]); ver = np.array([m["version"] for m in meta])
    y = np.array([m["contain"] for m in meta]).astype(int)
    ans = np.array([norm(m["answer"]) for m in meta])
    keep_row = np.ones(len(meta), bool) if setting == "all" else ver != "question"
    need = 5 if setting == "all" else 4
    cnt = collections.Counter(sid[keep_row])
    complete = {f for f, c in cnt.items() if c == need}
    keep_row &= np.isin(sid, list(complete))
    idx = np.where(keep_row)[0]
    r_sid, r_ver, r_y = sid[idx], ver[idx], y[idx]
    facts = np.array(sorted(complete))
    fidx = {f: i for i, f in enumerate(facts)}
    rf = np.array([fidx[f] for f in r_sid])                      # fact index per kept row
    ysum = np.bincount(rf, weights=r_y, minlength=len(facts))
    paired = (ysum > 0) & (ysum < need)
    subjects = np.unique([subj_uri[f] for f in facts])
    fold_of_sub = {s: i % 5 for i, s in enumerate(np.random.default_rng(0).permutation(subjects))}
    ffold = np.array([fold_of_sub[subj_uri[f]] for f in facts])
    rfold = ffold[rf]
    rpair = paired[rf]
    n = len(idx)
    print(f"\n{'=' * 20} [{a.model}] 설정={setting}: 사실 {len(facts)}개 (맞힘·틀림 섞인 사실 {paired.sum()}개), 행 {n}개 {'=' * 20}")
    C_ = conf[idx].copy(); C_[:, 3] = np.where(np.isnan(C_[:, 3]), np.nanmin(C_[:, 3]), C_[:, 3])
    M = A_last[idx]; keepf = (M[rpair] > 0).mean(0) >= 0.005; M = M[:, keepf]
    H = hl[idx]
    order = ["확신도 log p(top1)", "확신도 top1-top2 격차", "확신도 -엔트로피", "문구 텍스트만 (TF-IDF)", "SAE 20개", "내부 상태 전체", "확신도 + SAE 20개"]
    ex_names = [] if extra is None else list(extra[0])
    order = order + ex_names
    S = {k: np.zeros(n) for k in order}
    for j, k in enumerate(ex_names):
        S[k] = extra[1][idx, j]
    S["확신도 log p(top1)"], S["확신도 top1-top2 격차"], S["확신도 -엔트로피"] = C_[:, 0], C_[:, 1], -C_[:, 2]
    Mc, Mw = np.zeros((len(facts), M.shape[1])), np.zeros((len(facts), M.shape[1]))
    nc, nw = np.bincount(rf[r_y == 1], minlength=len(facts)), np.bincount(rf[r_y == 0], minlength=len(facts))
    np.add.at(Mc, rf[r_y == 1], M[r_y == 1]); np.add.at(Mw, rf[r_y == 0], M[r_y == 0])
    Dm = np.divide(Mc, np.maximum(nc, 1)[:, None]) - np.divide(Mw, np.maximum(nw, 1)[:, None])
    texts = np.array([txt[(s, v)] for s, v in zip(r_sid, r_ver)])
    for fold in range(5):
        tr = (rfold != fold) & rpair; te = rfold == fold
        trf = (ffold != fold) & paired
        Dt = Dm[trf]; t = Dt.mean(0) / (Dt.std(0, ddof=1) / np.sqrt(len(Dt)) + 1e-9)
        sd = M[tr].std(0) + 1e-6; o_ = np.argsort(-np.abs(t))[:20]
        s20 = M[:, o_] @ (np.sign(t[o_]) / sd[o_])
        S["SAE 20개"][te] = s20[te]
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=3000).fit(texts[tr])
        T = tf.transform(texts).toarray()
        S["문구 텍스트만 (TF-IDF)"][te] = lr_scores(T[tr], r_y[tr], r_sid[tr], T[te], 0.1)
        S["내부 상태 전체"][te] = lr_scores(H[tr], r_y[tr], r_sid[tr], H[te], 0.01)
        X1 = np.c_[C_[:, 0], s20]
        S["확신도 + SAE 20개"][te] = lr_scores(X1[tr], r_y[tr], r_sid[tr], X1[te], 1.0)
        print(f"  fold {fold} 끝", flush=True)
    bs = np.random.default_rng(1).integers(0, len(facts), (N_BOOT, len(facts)))
    rows_of = [np.where(rf == i)[0] for i in range(len(facts))]
    # ---------- 실험 1: 모든 사실에서 1등 문구로 답했을 때 전체 정답률 ----------
    rand = np.array([r_y[m].mean() for m in rows_of]); oracle = np.array([r_y[m].max() for m in rows_of])
    base = {"무작위 문구 (기대값)": rand, "천장 (하나라도 맞음)": oracle}
    if setting == "all":
        base["원래 문구"] = np.array([r_y[m][r_ver[m] == "question"][0] for m in rows_of])
    vote = []
    for m in rows_of:
        g = collections.defaultdict(list)
        for j in m:
            g[ans[idx[j]]].append(j)
        best = max(g.values(), key=lambda L: (len(L), any(r_ver[j] == "question" for j in L)))
        vote.append(float(np.mean([r_y[j] for j in best]) > 0.5))
    base["다수결 (같은 답 많은 쪽)"] = np.array(vote)
    pick_acc, picks = {}, {}
    for k in order:
        sel = [m[np.argmax(S[k][m])] for m in rows_of]
        pick_acc[k] = np.array([r_y[j] for j in sel]); picks[k] = {facts[i]: str(r_ver[j]) for i, j in enumerate(sel)}
    print(f"\n[실험 1] 사실 {len(facts)}개 전체에서, 점수 1등 문구로 답했을 때 정답률 (기준: 무작위 문구와의 차이 [95% 구간])")
    print(f"  {'방법':<28}{'정답률':>8}{'무작위 대비':>26}" + (f"{'원래 문구 대비':>26}" if setting == "all" else ""))
    for k, v in {**base, **pick_acc}.items():
        line = f"  {k:<28}{v.mean():>8.1%}{boot_ci(v - rand, bs):>26}"
        if setting == "all":
            line += f"{boot_ci(v - base['원래 문구'], bs):>26}"
        print(line)
    mp = paired
    print(f"  (참고: 맞힘·틀림 섞인 사실 {mp.sum()}개만: 무작위 {rand[mp].mean():.1%}, 천장 {oracle[mp].mean():.1%}, "
          + ", ".join(f"{k} {v[mp].mean():.1%}" for k, v in pick_acc.items() if k in ("확신도 log p(top1)", "SAE 20개", "확신도 + SAE 20개")) + ")")
    # ---------- 같은 사실 안 AUROC (섞인 사실만) ----------
    pf = np.where(paired)[0]
    def per_fact_auc(sc):
        return np.array([within_auc(sc[rows_of[i]], r_y[rows_of[i]]) for i in pf])
    print(f"\n[같은 사실 안 AUROC] 섞인 사실 {len(pf)}개 (0.5 = 구분 못 함)")
    aucs = {k: per_fact_auc(S[k]) for k in order}
    for k, v in aucs.items():
        print(f"  {k:<28}{np.nanmean(v):>8.3f}")
    # ---------- 실험 3: 분해 ----------
    names = ["확신도 log p(top1)", "문구 텍스트만 (TF-IDF)", "SAE 20개", "내부 상태 전체"] + ex_names[:1]
    short = ["확신도", "문구텍스트", "SAE20", "내부상태"] + (["Dream확신도"] if ex_names else [])
    K = len(names)
    Z = np.c_[[ (S[k] - S[k].mean()) / (S[k].std() + 1e-9) for k in names]].T
    res = {}
    for r in range(1, K + 1):
        for sub in itertools.combinations(range(K), r):
            sc = np.zeros(n)
            for fold in range(5):
                tr = (rfold != fold) & rpair; te = rfold == fold
                sc[te] = lr_scores(Z[tr][:, list(sub)], r_y[tr], r_sid[tr], Z[te][:, list(sub)], 1.0)
            res[sub] = per_fact_auc(sc)
    pm = np.arange(len(pf))
    bsp = np.random.default_rng(2).integers(0, len(pf), (N_BOOT, len(pf)))
    def mean_ci(d):
        d = np.nan_to_num(d, nan=0.5)
        lo, hi = np.percentile(d[bsp].mean(1), [2.5, 97.5]); return f"{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]"
    print("\n[실험 3] 신호 조합별 같은 사실 안 AUROC (신호 점수는 교차검증으로 얻은 값, 그 위에서 다시 교차검증으로 합침)")
    for r in range(1, K + 1):
        for sub in itertools.combinations(range(K), r):
            print(f"  {' + '.join(short[i] for i in sub):<34}{np.nanmean(res[sub]):.3f}")
    full = res[tuple(range(K))]
    print("  각 신호의 고유 기여 = 네 개 전부 − 그 신호만 뺀 세 개 (AUROC 차이 [95% 구간])")
    for i in range(K):
        rest = tuple(j for j in range(K) if j != i)
        print(f"    {short[i]:<12} 고유 {mean_ci(full - res[rest])}   (혼자 {np.nanmean(res[(i,)]):.3f})")
    Zc = center(Z[rpair], r_sid[rpair])
    cc = np.corrcoef(Zc.T)
    print("  같은 사실 안 점수끼리 상관 (사실 평균을 뺀 뒤)")
    print("  " + " " * 12 + "".join(f"{s:>12}" for s in short))
    for i in range(K):
        print(f"  {short[i]:<12}" + "".join(f"{cc[i, j]:>12.2f}" for j in range(K)))
    os.makedirs(outdir, exist_ok=True)
    json.dump({k: {f: v for f, v in d.items()} for k, d in picks.items()}, open(f"{outdir}/picks_{a.model}{a.tag}_{setting}.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dream", "qwen"], required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="sae_analysis/results_twelve/unified")
    ap.add_argument("--native", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--settings", nargs="+", default=["all", "para"])
    a = ap.parse_args()
    F = a.dir + "/"
    meta = [json.loads(l) for l in open(F + "meta.jsonl", encoding="utf-8")]
    conf = np.load(F + "conf.npy"); hl = np.load(F + "hid_last.npy").astype(np.float32)
    subj_uri = {}
    for l in open("data/popqa/1. popqa_raw.jsonl", encoding="utf-8"):
        x = json.loads(l); subj_uri[str(x["id"])] = x["s_uri"].rsplit("/", 1)[-1]
    txt = {}
    for l in open("data/popqa/4. popqa_paraphrased.jsonl", encoding="utf-8"):
        r = json.loads(l)
        for v in VERS:
            txt[(str(r["id"]), v)] = r[v]
    A_last = load_sae(a.model)(hl)
    extra = None
    if a.native:
        z = np.load(a.native, allow_pickle=True); extra = (list(z["names"]), z["X"].astype(np.float64))
        assert len(extra[1]) == len(meta), (len(extra[1]), len(meta))
    for setting in a.settings:
        run(a, meta, conf, hl, A_last, txt, subj_uri, setting, a.out, extra)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
