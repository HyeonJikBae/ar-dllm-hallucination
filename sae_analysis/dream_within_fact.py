"""Dream: can a pre-decoding signal tell the correct paraphrase of a fact from a wrong one?  (same design as the Qwen analyses)

Labels: Dream answers decoded with topk_margin, string-graded ("contain" = gold alias as a whole-word span in a short answer).
Facts: the 1,000-fact PopQA sample, facts with >=1 correct and >=1 wrong paraphrase. Evaluated twice:
  all    all five versions
  para   paraphrases v1-v4 only (removes the original-wording axis)
Signals at denoising step 0 (prompt + all-masked canvas; dream_step0_extract.py):
  confidence of the first answer token (log p top1, top1-top2 margin, -entropy, log p of the row's own first token),
  Dream's shared layer-20 SAE (top-n latents chosen on train subjects; prompt_last and subject_last positions),
  raw layer-20 hidden state (fact-centred logistic regression), surface wording features, and combinations.
Subject-level 5-fold CV; per fact the top-scored paraphrase is picked and compared with picking at random.

Usage:  PYTHONPATH=sae_analysis:grounded_dream /home/minyeol/.conda/envs/sae/bin/python sae_analysis/dream_within_fact.py
"""
import collections
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "grounded_dream")
from string_grader import norm  # noqa: E402

warnings.filterwarnings("ignore")
F = os.environ.get("WF_DIR", "sae_analysis/features_dream_step0") + "/"
SAE_DIR = "sae/runs/dream_pilot_200M/dream-7b_L20_shared-0.2-0.5-0.8_s5_200M"
VERS = ("question", "v1", "v2", "v3", "v4")
LABEL = "contain"
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


def main():
    meta = [json.loads(l) for l in open(F + "meta.jsonl", encoding="utf-8")]
    conf = np.load(F + "conf.npy"); hl = np.load(F + "hid_last.npy").astype(np.float32); hs = np.load(F + "hid_subj.npy").astype(np.float32)
    sid = np.array([m["sample_id"] for m in meta]); ver = np.array([m["version"] for m in meta])
    y = np.array([m[LABEL] for m in meta]).astype(int)
    found = np.array([m["subj_found"] for m in meta])
    bad = {m["sample_id"] for m in meta if m["gold_issue"]}
    ok = ~np.isin(sid, list(bad))
    conf[:, 3] = np.where(np.isnan(conf[:, 3]), np.nanmin(conf[:, 3]), conf[:, 3])
    subj = {}
    for l in open("data/popqa/1. popqa_raw.jsonl", encoding="utf-8"):
        x = json.loads(l); subj[str(x["id"])] = x["s_uri"].rsplit("/", 1)[-1]
    txt = {}
    for l in open("data/popqa/4. popqa_paraphrased.jsonl", encoding="utf-8"):
        r = json.loads(l)
        for v in VERS:
            txt[(str(r["id"]), v)] = r[v]
    from sae_lens import SAE
    sae = SAE.load_from_disk(SAE_DIR, device="cpu").eval()
    import torch

    def enc(H):
        out = []
        with torch.no_grad():
            for b in range(0, len(H), 512):
                out.append(sae.encode(torch.from_numpy(H[b : b + 512])).numpy())
        return np.concatenate(out)

    A_last, A_subj = enc(hl), enc(hs)
    print(f"행 {len(meta)}, gold_issue 사실 제외 후 {ok.sum()}행, 주어 위치 찾음 {found.mean():.1%}")
    print("버전별 정답률(Dream topk_margin, contain):", {v: f"{y[ok & (ver == v)].mean():.1%}" for v in VERS})
    print("버전별 평균 첫 토큰 log p(top1):", {v: f"{conf[ok & (ver == v), 0].mean():.2f}" for v in VERS})
    o, p_ = ok & (ver == "question"), ok & (ver != "question")
    d = conf[o, 0][:, None] - conf[p_, 0][None, :500]
    print(f"원래 vs 변형 확신도 AUROC {np.mean(conf[o, 0][:, None] > conf[p_, 0][None, :]):.3f} (0.5 = 구분 안 됨)")
    wc = collections.Counter(norm(txt[(s, v)]).split()[0] for s, v in zip(sid, ver))
    top_first = [w for w, _ in wc.most_common(10)]

    def surface(i, with_orig):
        q, orig = txt[(sid[i], ver[i])], txt[(sid[i], "question")]
        w, wo = norm(q).split(), norm(orig).split()
        f = [len(w), len(q)] + [float(w[0] == t) for t in top_first]
        if with_orig:
            f += [len(set(w) & set(wo)) / max(len(set(w) | set(wo)), 1), float(ver[i] == "question")]
        return f

    for title, mask, with_orig in (("모든 5개 문구", ok, True), ("변형 v1~v4만 (원래 문구 제외)", ok & (ver != "question"), False)):
        idx = np.where(mask)[0]
        facts = np.unique(sid[idx])
        paired = [f for f in facts if y[idx][sid[idx] == f].min() == 0 and y[idx][sid[idx] == f].max() == 1]
        idx = idx[np.isin(sid[idx], paired)]
        r_sid, r_y, r_ver = sid[idx], y[idx], ver[idx]
        subjects = np.unique([subj[f] for f in paired])
        rng = np.random.default_rng(0)
        fold_of = {s: i % 5 for i, s in enumerate(rng.permutation(subjects))}
        r_fold = np.array([fold_of[subj[f]] for f in r_sid])
        C_, SF = conf[idx], np.array([surface(i, with_orig) for i in idx])
        feats = {}
        for nm, A in (("last", A_last), ("subj", A_subj)):
            M = A[idx]; keep = (M > 0).mean(0) >= 0.005; feats[nm] = M[:, keep]
        pf = {f: i for i, f in enumerate(paired)}
        rf = np.array([pf[f] for f in r_sid])
        scores = {}
        for k in ("확신도: log p(top1)", "확신도: top1−top2 격차", "확신도: −엔트로피", "확신도: 자기 답 첫 토큰 log p",
                  "표면(문구 특징)", "SAE 5개 (마지막 프롬프트 위치)", "SAE 20개 (마지막 프롬프트 위치)", "SAE 5개 (주어 마지막 위치)",
                  "SAE 20개 (주어 마지막 위치)", "원래 내부 상태 (마지막 프롬프트 위치)", "원래 내부 상태 (주어 마지막 위치)",
                  "확신도 + SAE 20개(마지막 위치)", "확신도 + 표면", "문구 텍스트만 (TF-IDF)"):
            scores[k] = np.zeros(len(idx))
        scores["확신도: log p(top1)"], scores["확신도: top1−top2 격차"] = C_[:, 0], C_[:, 1]
        scores["확신도: −엔트로피"], scores["확신도: 자기 답 첫 토큰 log p"] = -C_[:, 2], C_[:, 3]
        for fold in range(5):
            tr, te = r_fold != fold, r_fold == fold
            ftr = np.array([fold_of[subj[f]] != fold for f in paired])
            sel = {}
            for nm in ("last", "subj"):
                M = feats[nm]
                Mc, Mw = np.zeros((len(paired), M.shape[1])), np.zeros((len(paired), M.shape[1]))
                nc, nw = np.bincount(rf[r_y == 1], minlength=len(paired)), np.bincount(rf[r_y == 0], minlength=len(paired))
                np.add.at(Mc, rf[r_y == 1], M[r_y == 1]); np.add.at(Mw, rf[r_y == 0], M[r_y == 0])
                Dm = Mc / nc[:, None] - Mw / nw[:, None]
                Dt = Dm[ftr]
                t = Dt.mean(0) / (Dt.std(0, ddof=1) / np.sqrt(len(Dt)) + 1e-9)
                sd = M[tr].std(0) + 1e-6
                o_ = np.argsort(-np.abs(t))
                for n in (5, 20):
                    scores[f"SAE {n}개 ({'마지막 프롬프트 위치' if nm == 'last' else '주어 마지막 위치'})"][te] = (M[:, o_[:n]] @ (np.sign(t[o_[:n]]) / sd[o_[:n]]))[te]
                sel[nm] = M[:, o_[:20]] @ (np.sign(t[o_[:20]]) / sd[o_[:20]])
            scores["표면(문구 특징)"][te] = lr_scores(SF[tr], r_y[tr], r_sid[tr], SF[te], 1.0)
            scores["원래 내부 상태 (마지막 프롬프트 위치)"][te] = lr_scores(hl[idx][tr], r_y[tr], r_sid[tr], hl[idx][te], 0.01)
            scores["원래 내부 상태 (주어 마지막 위치)"][te] = lr_scores(hs[idx][tr], r_y[tr], r_sid[tr], hs[idx][te], 0.01)
            tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=3000).fit([txt[(r_sid[i], r_ver[i])] for i in np.where(tr)[0]])
            T = tf.transform([txt[(a, b)] for a, b in zip(r_sid, r_ver)]).toarray()
            scores["문구 텍스트만 (TF-IDF)"][te] = lr_scores(T[tr], r_y[tr], r_sid[tr], T[te], 0.1)
            X1 = np.c_[C_[:, 0], sel["last"]]; X2 = np.c_[C_[:, 0], SF]
            scores["확신도 + SAE 20개(마지막 위치)"][te] = lr_scores(X1[tr], r_y[tr], r_sid[tr], X1[te], 1.0)
            scores["확신도 + 표면"][te] = lr_scores(X2[tr], r_y[tr], r_sid[tr], X2[te], 1.0)
        per = {}
        for nm, sc in scores.items():
            top, base, au = [], [], []
            for f in paired:
                m = np.where(r_sid == f)[0]
                top.append(r_y[m][np.argmax(sc[m])]); base.append(r_y[m].mean()); au.append(within_auc(sc[m], r_y[m]))
            per[nm] = (np.array(top), np.array(base), np.nanmean(au))
        bs = np.random.default_rng(1).integers(0, len(paired), (N_BOOT, len(paired)))
        print(f"\n##### {title} | 맞힘·틀림이 모두 있는 사실 {len(paired)}개, 행 {len(idx)}개")
        if with_orig:
            has = [f for f in paired if ((r_sid == f) & (r_ver == "question")).any()]
            print(f"  원래 문구만 쓰면: {np.mean([r_y[(r_sid == f) & (r_ver == 'question')][0] for f in has]):.1%} "
                  f"(그 사실들의 무작위 기준 {np.mean([r_y[r_sid == f].mean() for f in has]):.1%}, 사실 {len(has)}개)")
        print(f"  {'신호':<34}{'1등 맞힘':>9}{'무작위 대비 [95% 구간]':>26}{'같은 사실 안 AUROC':>20}")
        for nm, (top, base, au) in per.items():
            dd = top - base; lo, hi = np.percentile(dd[bs].mean(1), [2.5, 97.5])
            print(f"  {nm:<34}{top.mean():>9.1%}{dd.mean():>+11.1%} [{lo:+.1%},{hi:+.1%}]{au:>14.3f}")
        print(f"  (무작위 기준 {np.mean(per['확신도: log p(top1)'][1]):.1%})")


if __name__ == "__main__":
    main()
