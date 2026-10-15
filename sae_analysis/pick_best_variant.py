"""Simulate the intended use: the same fact is asked in several paraphrases; use the signal to rank them.

Signal: top-n SAE latents chosen on train subjects for "correct vs wrong inside a fact" (as in minimal_signal.py),
position = answer prediction (prompt_last) or question end (q_last). Evaluation: held-out subjects, 1-4/5 facts that have
both correct and wrong rows among their (analysis_use) paraphrases.

  pick-best   accuracy of the paraphrase with the HIGHEST score vs picking one at random (= share of correct paraphrases)
  flag-worst  share of wrong rows among the paraphrase with the LOWEST score vs the base share
  overlap     row-level correct-vs-wrong AUROC over all partial-fact rows, and inside the fact
  fact level  does the mean score over the paraphrases of a fact separate 5/5 facts from 0/5 facts
              (relation-stratified AUROC), for these n latents and for the 430-latent knowledge score

Usage:  PYTHONPATH=sae_analysis python3 sae_analysis/pick_best_variant.py
"""

import collections
import json

import numpy as np

from know_axis_by_k import fact_matrix
from signal_wrong_vs_correct import load, within_auc
from twelve_cells import auroc, strat_auroc, strat_diff

SPLITS = 10
NS = (5, 20)


def fact_means(score, mask, sid, rel):
    f, inv = np.unique(sid[mask], return_inverse=True)
    m = np.bincount(inv, weights=score[mask]) / np.bincount(inv)
    fr = np.empty(len(f), dtype=object)
    fr[inv] = rel[mask]
    return m, fr


def main():
    out = {}
    for position in ("prompt_last", "q_last"):
        M, freq, corr, sid, sub, rel, k, cols = load(position)
        el = np.where(freq >= 0.005)[0]
        B = (M[:, el] > 0).astype(np.float32)
        part = (k >= 1) & (k <= 4)
        isK5, isU = corr & (k == 5), (k == 0) & ~corr
        PW, PC = part & ~corr, part & corr
        paired = np.array([f for f in np.unique(sid[part]) if PW[sid == f].any() and PC[sid == f].any()])
        pp = part & np.isin(sid, paired)
        subjects = np.unique(sub)
        y = corr.astype(int)
        fi = {f: i for i, f in enumerate(paired)}
        idx = np.where(pp)[0]
        row_f = np.array([fi[f] for f in sid[idx]])
        f_sub = np.array([sub[sid == f][0] for f in paired])
        Mc, Mw = np.zeros((len(paired), M.shape[1]), np.float32), np.zeros((len(paired), M.shape[1]), np.float32)
        nc, nw = np.zeros(len(paired)), np.zeros(len(paired))
        for sel, Ms, n in ((corr[idx], Mc, nc), (~corr[idx], Mw, nw)):
            np.add.at(Ms, row_f[sel], M[idx[sel]])
            np.add.at(n, row_f[sel], 1)
        D = Mc / nc[:, None] - Mw / nw[:, None]
        R = collections.defaultdict(list)
        for seed in range(SPLITS):
            rng = np.random.default_rng(seed)
            trs = set(rng.permutation(subjects)[: int(0.6 * len(subjects))])
            tr = np.array([s in trs for s in sub]); te = ~tr
            ftr = np.array([s in trs for s in f_sub])
            Dt = D[ftr]
            t = Dt.mean(0) / (Dt.std(0, ddof=1) / np.sqrt(len(Dt)) + 1e-9)
            t[np.setdiff1d(np.arange(M.shape[1]), el)] = 0
            sd = M[tr & part].std(0) + 1e-6
            order = np.argsort(-np.abs(t))
            # knowledge score (430 binary latents, K5 - U weights), for the fact-level comparison
            _, XK, rK = fact_matrix(B[tr & isK5], sid[tr & isK5], rel[tr & isK5])
            _, XU, rU = fact_matrix(B[tr & isU], sid[tr & isU], rel[tr & isU])
            d, _ = strat_diff(XK, rK, XU, rU, 3)
            s_know = B @ d.astype(np.float32)
            te_pp = pp & te
            tf = np.unique(sid[te_pp])
            for n in NS:
                sel = order[:n]
                sc = M[:, sel] @ (np.sign(t[sel]) / sd[sel])
                hit_top, base, hit_bot, base_w, kk_ = [], [], [], [], []
                for f in tf:
                    m = np.where(te_pp & (sid == f))[0]
                    c = y[m]
                    hit_top.append(c[np.argmax(sc[m])]); hit_bot.append(1 - c[np.argmin(sc[m])])
                    base.append(c.mean()); base_w.append(1 - c.mean()); kk_.append(int(k[m[0]]))
                hit_top, base, hit_bot, base_w, kk_ = map(np.array, (hit_top, base, hit_bot, base_w, kk_))
                R[(n, "top")].append(hit_top.mean()); R[(n, "base")].append(base.mean())
                R[(n, "bot")].append(hit_bot.mean()); R[(n, "base_w")].append(base_w.mean())
                for kv in (1, 2, 3, 4):
                    mk = kk_ == kv
                    if mk.sum() >= 5:
                        R[(n, "top", kv)].append(hit_top[mk].mean()); R[(n, "base", kv)].append(base[mk].mean())
                R[(n, "pooled")].append(auroc(sc[te & PC], sc[te & PW]))
                R[(n, "within")].append(within_auc(sc[te_pp], y[te_pp], sid[te_pp]))
                a, ra = fact_means(sc, te & isK5, sid, rel); b, rb = fact_means(sc, te & isU, sid, rel)
                R[(n, "fact_ku")].append(strat_auroc(a, ra, b, rb, 5))
            a, ra = fact_means(s_know, te & isK5, sid, rel); b, rb = fact_means(s_know, te & isU, sid, rel)
            R[("know", "fact_ku")].append(strat_auroc(a, ra, b, rb, 5))
        print(f"\n##### {position} | 시험용 사실 평균 {np.mean([0]) if False else len(paired) * 0.4:.0f}개 (1~4/5 사실, 맞힌 행·틀린 행 모두 있는 것), 10회 평균")
        print("\n[1] 한 사실의 변형들 중 점수가 가장 높은 변형을 골랐을 때 맞힌 비율 (무작위로 고르면 그 사실의 맞힌 비율)")
        print(f"  {'특징 수':<8}{'점수 1등 변형':>14}{'무작위':>10}{'차이':>8}   k별(점수 1등 / 무작위): " )
        for n in NS:
            ks = "  ".join(f"{kv}/5 {np.mean(R[(n, 'top', kv)]):.0%}/{np.mean(R[(n, 'base', kv)]):.0%}" for kv in (1, 2, 3, 4) if R[(n, 'top', kv)])
            print(f"  {n}개{'':<5}{np.mean(R[(n, 'top')]):>13.1%}{np.mean(R[(n, 'base')]):>10.1%}{np.mean(R[(n, 'top')]) - np.mean(R[(n, 'base')]):>+8.1%}   {ks}")
        print("\n[2] 점수가 가장 낮은 변형이 틀린 비율 (무작위로 고르면 그 사실의 틀린 비율)")
        for n in NS:
            print(f"  {n}개   점수 꼴찌 변형: {np.mean(R[(n, 'bot')]):.1%}   무작위: {np.mean(R[(n, 'base_w')]):.1%}")
        print("\n[3] 개별 행 수준 겹침 (맞힌 행이 틀린 행보다 점수가 높을 확률)")
        for n in NS:
            print(f"  {n}개   전체 행 섞어서 {np.mean(R[(n, 'pooled')]):.3f}   같은 사실 안 {np.mean(R[(n, 'within')]):.3f}")
        print("\n[4] 사실 단위 (변형 점수 평균): 모두 정답 사실 vs 모두 오답 사실 AUROC, 관계 보정")
        for n in NS:
            print(f"  이 신호 {n}개: {np.mean(R[(n, 'fact_ku')]):.3f}")
        print(f"  앎 점수(430개 합): {np.mean(R[('know', 'fact_ku')]):.3f}")
        out[position] = {"|".join(map(str, key)): float(np.mean(v)) for key, v in R.items()}
    json.dump(out, open("sae_analysis/results_twelve/pick_best_variant.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
