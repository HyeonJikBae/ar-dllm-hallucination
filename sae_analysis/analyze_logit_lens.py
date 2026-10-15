"""Within-fact correct vs wrong rows: logit lens, teacher forcing and raw-hidden probes (outputs of logit_lens_extract.py).

Groups (guess-risk facts already removed): K5 (5/5 rows), U (0/5 rows, sampled facts), P+ / P- = correct / wrong rows
of partially known facts (1-4/5). Within-fact comparisons use facts that have both P+ and P- rows.

  [0] sanity: the model's own first answer token is rank 1 at the last layer ("reproduced" rows)
  [1] last-layer rank of the gold first token in wrong rows (near miss = rank 2-3), by error type, vs U
  [2] layer curves (Jiang et al. style): mean gold log-prob, share of rows with gold at rank 1 / in top 10, per layer
  [3] within-fact AUROC (correct > wrong) per layer from the gold log-prob, plus teacher-forced full gold answer
      log-prob. The last layer is near-trivially high (greedy picks rank 1), so the informative part is WHERE in
      depth the separation appears.
  [4] raw hidden probes, within fact (fact-centred logistic regression; null = labels shuffled inside facts),
      at prompt_last for layers 4..28 and at q_last layer 20 -> compare with the SAE result (AUROC ~0.55)
Rows whose wrong answer shares its first token with the gold ("overlap") are reported separately.

Usage:
    python3 sae_analysis/analyze_logit_lens.py
"""

import argparse
import collections
import json
import warnings

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
D = "sae_analysis/features_lens"
HID_LAYERS = [4, 8, 12, 16, 20, 24, 28]


def within_auc(score, pos, neg, sid):
    """Mean over facts with both classes of P(score_pos > score_neg)."""
    out = []
    for f in np.unique(sid[pos | neg]):
        a, b = score[pos & (sid == f)], score[neg & (sid == f)]
        if len(a) and len(b):
            d = a[:, None] - b[None, :]
            out.append(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)
    return float(np.mean(out)), len(out)


def center(X, sid):
    Xc = X.copy()
    for f in np.unique(sid):
        m = sid == f
        Xc[m] -= X[m].mean(0)
    return Xc


def probe(X, y, sid, splits=10, C=0.01):
    res, null = [], []
    facts = np.unique(sid)
    for seed in range(splits):
        rng = np.random.default_rng(seed)
        tr = np.isin(sid, rng.permutation(facts)[: int(0.6 * len(facts))])
        te = ~tr
        Xc = center(X[tr], sid[tr])
        sc = StandardScaler().fit(Xc)
        clf = LogisticRegression(C=C, max_iter=300).fit(sc.transform(Xc), y[tr])
        s = clf.decision_function(sc.transform(X[te]))
        res.append(within_auc(s, y[te] == 1, y[te] == 0, sid[te])[0])
        yn = y.copy()
        for f in facts:
            m = np.where(sid == f)[0]
            yn[m] = rng.permutation(y[m])
        clf = LogisticRegression(C=C, max_iter=300).fit(sc.transform(Xc), yn[tr])
        null.append(within_auc(clf.decision_function(sc.transform(X[te])), yn[te] == 1, yn[te] == 0, sid[te])[0])
    return np.mean(res), np.std(res), np.mean(null)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=10)
    args = ap.parse_args()
    meta = [json.loads(l) for l in open(f"{D}/meta.jsonl", encoding="utf-8")]
    z = np.load(f"{D}/lens.npz")
    sid = np.array([m["sample_id"] for m in meta])
    lab = np.array([m["final_label"] for m in meta])
    grp = np.array([m["fact_group"] for m in meta])
    corr = np.char.startswith(lab.astype(str), "correct")
    k5, U = grp == "5/5", grp == "0/5"
    part = ~k5 & ~U
    Pp, Pm = part & corr, part & ~corr
    paired = np.isin(sid, [f for f in np.unique(sid[part]) if (Pp & (sid == f)).any() and (Pm & (sid == f)).any()])
    gl, gr, orank = z["gold_lp"], z["gold_rk"], z["out_rk"]
    rep = orank[:, -1] == 1
    ov = z["overlap"]
    nl = gl.shape[1]
    report = {}

    print(f"rows {len(meta):,} | K5 {k5.sum():,} U {U.sum():,} | 부분 fact 정답 {Pp.sum():,} 오답 {Pm.sum():,} | 짝지은 fact {len(np.unique(sid[paired])):,}")
    print("\n[0] 재현 점검: 마지막 층에서 실제 출력 첫 토큰이 1위인 비율")
    for name, m in (("K5", k5), ("U", U), ("부분 정답", Pp), ("부분 오답", Pm)):
        print(f"  {name:<8} {rep[m].mean():.1%}  (정답 행이면 gold 1위 비율 {(gr[m, -1] == 1).mean():.1%})")

    print("\n[1] 오답 행에서 gold 첫 토큰의 마지막 층 순위 (재현된 행만; 겹침 = 오답과 gold의 첫 토큰이 같음)")
    bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 100), (101, 10 ** 9)]
    head = "".join(f"{(f'{a}' if a == b else (f'{a}-{b}' if b < 10 ** 9 else f'>{a - 1}')):>8}" for a, b in bins)
    print(f"  {'':<22}{'행':>6}{head}")
    rows_def = [("부분 fact 오답 전체", Pm)] + [(f"  유형 {t}", Pm & (lab == str(t))) for t in (1, 2, 3, 6)] + \
               [("  └ 겹침 제외", Pm & ~ov), ("모르는 fact(U) 오답", U & ~corr)]
    for name, m in rows_def:
        m = m & rep
        r = gr[m, -1]
        print(f"  {name:<22}{m.sum():>6}" + "".join(f"{((r >= a) & (r <= b)).mean():>8.1%}" for a, b in bins))
        report[f"rank_last|{name.strip()}"] = {f"{a}-{b}": float(((r >= a) & (r <= b)).mean()) for a, b in bins}
    print(f"  겹침 비율: 부분 오답 {ov[Pm].mean():.1%} (유형1 {ov[Pm & (lab == '1')].mean():.1%}, 유형2 {ov[Pm & (lab == '2')].mean():.1%}), U 오답 {ov[U & ~corr].mean():.1%}")

    print("\n[2] 층별 곡선 (gold 첫 토큰): 평균 log-prob / gold 1위 비율 / gold top-10 비율")
    show = [4, 8, 12, 16, 18, 20, 22, 24, 26, 27, 28]
    print(f"  {'':<10}" + "".join(f"{f'L{L}':>8}" for L in show))
    for name, m in (("K5", k5), ("부분 정답", Pp), ("부분 오답", Pm), ("U", U)):
        for stat, f in (("logp", lambda L: gl[m, L - 1].mean()), ("1위", lambda L: (gr[m, L - 1] == 1).mean()),
                        ("top10", lambda L: (gr[m, L - 1] <= 10).mean())):
            fmt = "{:>8.2f}" if stat == "logp" else "{:>8.1%}"
            print(f"  {name + ' ' + stat:<10}" + "".join(fmt.format(f(L)) for L in show))
        first10 = np.array([np.argmax(gr[i] <= 10) + 1 if (gr[i] <= 10).any() else np.nan for i in np.where(m)[0]])
        print(f"  {name}: gold가 처음 top-10에 드는 층 중앙값 {np.nanmedian(first10):.0f} (한 번이라도 드는 비율 {np.mean(~np.isnan(first10)):.1%})")

    print("\n[3] 같은 fact 안 AUROC (정답 > 오답), 층별 gold log-prob")
    aucs = {L: within_auc(gl[:, L - 1], Pp & paired, Pm & paired, sid) for L in range(1, nl + 1)}
    aucs_no = {L: within_auc(gl[:, L - 1], Pp & paired, Pm & paired & ~ov, sid) for L in range(1, nl + 1)}
    print("  " + "".join(f"{f'L{L}':>7}" for L in show))
    print("  " + "".join(f"{aucs[L][0]:>7.3f}" for L in show) + "   (전체)")
    print("  " + "".join(f"{aucs_no[L][0]:>7.3f}" for L in show) + "   (겹침 오답 제외)")
    first = next((L for L in range(1, nl + 1) if aucs[L][0] >= 0.6), None)
    print(f"  짝지은 fact {aucs[nl][1]} | AUROC가 처음 0.60을 넘는 층: L{first}")
    tf_auc = within_auc(z["gold_mean"], Pp & paired, Pm & paired, sid)
    print(f"  teacher forcing: gold 전체 답 평균 log-prob의 같은 fact 안 AUROC {tf_auc[0]:.3f}")
    report["within_auc_by_layer"] = {L: v[0] for L, v in aucs.items()}
    report["within_auc_by_layer_no_overlap"] = {L: v[0] for L, v in aucs_no.items()}
    report["tf_gold_auc"] = tf_auc[0]

    print("\n[4] raw hidden probe, 같은 fact 안 (fact 중심화 로지스틱 회귀; null = fact 안에서 라벨 섞음)")
    H = np.load(f"{D}/hidden_prompt_last.npy", mmap_mode="r")
    Hq = np.load(f"{D}/hidden_q_last_L20.npy", mmap_mode="r")
    m = paired & part
    y = corr[m].astype(int)
    s = sid[m]
    idx = np.where(m)[0]
    for name, X in [(f"prompt_last L{L}", np.asarray(H[idx, i]).astype(np.float32)) for i, L in enumerate(HID_LAYERS)] + \
                   [("q_last L20", np.asarray(Hq[idx]).astype(np.float32))]:
        a, sd, nu = probe(X, y, s, args.splits)
        report[f"probe|{name}"] = [float(a), float(sd), float(nu)]
        print(f"  {name:<16} AUROC {a:.3f}±{sd:.3f}  (null {nu:.3f})")
    json.dump(report, open("sae_analysis/results_twelve/logit_lens.json", "w"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
