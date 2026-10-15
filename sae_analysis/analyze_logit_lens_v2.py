"""Step 3: logit lens at prompt_last (all layers) and at the subject's last token, within fact correct vs wrong.

Input: logit_lens_extract_v2.py (guess-risk facts removed). Groups: K5, U, P+ / P- (correct / wrong rows of 1-4/5 facts).
margin = log p(gold first token) - log p(best non-gold token); persistence = #layers with gold in the top 10.
Rows whose wrong answer shares its first token with the gold ("overlap") are shown separately.

Usage:
    python3 sae_analysis/analyze_logit_lens_v2.py
"""
import json

import numpy as np

D = "sae_analysis/features_lens_v2"


def within_auc(score, pos, neg, sid):
    out = []
    for f in np.unique(sid[pos | neg]):
        a, b = score[pos & (sid == f)], score[neg & (sid == f)]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) and len(b):
            d = a[:, None] - b[None, :]
            out.append(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)
    return float(np.mean(out)), len(out)


def main():
    meta = [json.loads(l) for l in open(f"{D}/meta.jsonl", encoding="utf-8")]
    z = np.load(f"{D}/lens.npz")
    sid = np.array([m["sample_id"] for m in meta])
    lab = np.array([m["final_label"] for m in meta]).astype(str)
    grp = np.array([m["fact_group"] for m in meta])
    corr = np.char.startswith(lab, "correct")
    k5, U = grp == "5/5", grp == "0/5"
    part = ~k5 & ~U
    Pp, Pm = part & corr, part & ~corr
    paired = np.isin(sid, [f for f in np.unique(sid[part]) if (Pp & (sid == f)).any() and (Pm & (sid == f)).any()])
    gl, gr, ol, cl = z["gold_lp"], z["gold_rk"], z["out_lp"], z["comp_lp"]
    sgl, sgr = z["sgold_lp"], z["sgold_rk"]
    margin = gl - cl
    rep = z["out_rk"][:, -1] == 1
    ov = z["overlap"]
    nl = gl.shape[1]
    rep_out = {}
    print(f"rows {len(meta):,} | K5 {k5.sum():,} U {U.sum():,} | 부분 fact 정답 {Pp.sum():,} 오답 {Pm.sum():,} | 짝지은 fact {len(np.unique(sid[paired & part])):,}")
    print(f"재현 점검(마지막 층에서 실제 출력 첫 토큰이 1위): K5 {rep[k5].mean():.1%}, U {rep[U].mean():.1%}, 부분 정답 {rep[Pp].mean():.1%}, 부분 오답 {rep[Pm].mean():.1%}")
    print(f"첫 토큰 겹침(오답 첫 토큰 = gold 첫 토큰 후보): 부분 오답 {ov[Pm].mean():.1%}, U {ov[U & ~corr].mean():.1%}")

    print("\n[1] 오답 행, 마지막 층: gold 첫 토큰 순위와 margin (gold − 최강 경쟁 토큰, 로그확률; 재현된 행, 겹침 제외)")
    print(f"  {'':<16}{'행':>6}{'순위2':>8}{'2~3':>8}{'≤10':>8}{'>100':>8}{'margin중앙값':>13}{'margin>-1':>11}{'margin>-0.5':>12}")
    for name, m in [("부분 fact 오답", Pm)] + [(f"  유형 {t}", Pm & (lab == str(t))) for t in (1, 2, 3, 6)] + [("U 오답", U & ~corr)]:
        m = m & rep & ~ov
        r, mg = gr[m, -1], margin[m, -1]
        rep_out[f"last|{name.strip()}"] = {"n": int(m.sum()), "rank2": float((r == 2).mean()), "rank2_3": float(((r >= 2) & (r <= 3)).mean()),
                                           "top10": float((r <= 10).mean()), "median_margin": float(np.median(mg)), "margin_gt_-1": float((mg > -1).mean())}
        print(f"  {name:<16}{m.sum():>6}{(r == 2).mean():>8.1%}{((r >= 2) & (r <= 3)).mean():>8.1%}{(r <= 10).mean():>8.1%}{(r > 100).mean():>8.1%}"
              f"{np.median(mg):>13.2f}{(mg > -1).mean():>11.1%}{(mg > -0.5).mean():>12.1%}")

    show = [4, 8, 12, 16, 18, 20, 22, 24, 26, 27, 28]
    print("\n[2] 층별 곡선 (답 예측 위치): gold top-10 비율 / 평균 margin")
    print(f"  {'':<14}" + "".join(f"{f'L{L}':>8}" for L in show))
    for name, m in (("K5", k5), ("부분 정답", Pp), ("부분 오답", Pm & ~ov), ("U 오답", U & ~corr)):
        print(f"  {name + ' top10':<14}" + "".join(f"{(gr[m, L - 1] <= 10).mean():>8.1%}" for L in show))
        print(f"  {name + ' margin':<14}" + "".join(f"{margin[m, L - 1].mean():>8.2f}" for L in show))
        pers = (gr[m] <= 10).sum(1)
        print(f"  {name}: gold가 top-10에 있는 층 수 평균 {pers.mean():.1f} / 28, 한 번도 없음 {np.mean(pers == 0):.1%}")

    print("\n[3] 오답 행에서 gold vs 실제 오답 첫 토큰 (재현·겹침 제외): 층별로 gold가 앞서는 비율")
    m = Pm & rep & ~ov
    lead = gl[m] > ol[m]
    print("  부분 오답 " + "".join(f"{lead[:, L - 1].mean():>8.1%}" for L in show))
    mu = U & ~corr & rep & ~ov
    print("  U 오답    " + "".join(f"{(gl[mu] > ol[mu])[:, L - 1].mean():>8.1%}" for L in show))
    led_late = lead[:, 19:27].any(1)  # gold ahead somewhere in L20-L27 but loses at L28
    print(f"  부분 오답 중 L20~27에서 한 번이라도 gold가 앞섰다가 마지막 층에서 진 비율: {led_late.mean():.1%} "
          f"(U 오답 {(gl[mu] > ol[mu])[:, 19:27].any(1).mean():.1%})")
    rep_out["gold_led_L20_27_then_lost"] = {"partial": float(led_late.mean()), "U": float((gl[mu] > ol[mu])[:, 19:27].any(1).mean())}

    print("\n[4] 주어 마지막 위치 vs 답 예측 위치: gold top-100 비율 (주어 위치는 gold 속성이 드러나는지 보는 지표)")
    print(f"  {'':<18}" + "".join(f"{f'L{L}':>8}" for L in show))
    for name, m in (("K5", k5), ("부분 정답", Pp), ("부분 오답", Pm & ~ov), ("U 오답", U & ~corr)):
        print(f"  {name + ' 주어위치':<18}" + "".join(f"{(sgr[m, L - 1] <= 100).mean():>8.1%}" for L in show))
        print(f"  {name + ' 답위치':<18}" + "".join(f"{(gr[m, L - 1] <= 100).mean():>8.1%}" for L in show))

    print("\n[5] 같은 fact 안 AUROC (정답 > 오답), 층별; 겹침 오답 제외")
    print(f"  {'':<22}" + "".join(f"{f'L{L}':>7}" for L in show))
    feats = {"답위치 gold logp": gl, "답위치 margin": margin, "주어위치 gold logp": sgl}
    for name, X in feats.items():
        a = [within_auc(X[:, L - 1], Pp & paired, Pm & paired & ~ov, sid)[0] for L in show]
        rep_out[f"within_auc|{name}"] = dict(zip(show, a))
        print(f"  {name:<22}" + "".join(f"{v:>7.3f}" for v in a))
    tf = within_auc(z["gold_mean"], Pp & paired, Pm & paired & ~ov, sid)
    print(f"  teacher forcing gold 전체 답 평균 logp: {tf[0]:.3f} (fact {tf[1]})")
    json.dump(rep_out, open("sae_analysis/results_twelve/logit_lens_v2.json", "w"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
