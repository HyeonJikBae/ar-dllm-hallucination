"""Baselines every "internal signal" has to beat when choosing / combining paraphrases of the same fact.

A. Surface baseline (same 843 partially-known PopQA facts, same subject-level 5-fold CV and scores as combine_signals.py):
   logistic regression on wording features only (is-original, length, word overlap with the original question, first word),
   alone and together with the model's own confidence. Per fact: pick the top-scored paraphrase, compare with random.
B. Voting: take the most frequent normalised answer among a fact's paraphrases (tie -> highest confidence / original wording),
   on the 843 facts and on whole datasets. Reports, over all facts: random paraphrase, original wording, majority vote,
   and the ceiling "any paraphrase correct" (oracle best-of-5).

Usage:  PYTHONPATH=sae_analysis:grounded_dream python3 sae_analysis/baseline_battery.py
"""
import collections
import json
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "grounded_dream")
from string_grader import norm  # noqa: E402

warnings.filterwarnings("ignore")
D = "sae_analysis/features_lens_v2/"
VERS = ("question", "v1", "v2", "v3", "v4")
N_BOOT = 2000


def within_auc(s, y):
    a, b = s[y == 1], s[y == 0]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    d = a[:, None] - b[None, :]
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size


def part_a():
    meta = [json.loads(l) for l in open(D + "meta.jsonl", encoding="utf-8")]
    Z = np.load("sae_analysis/results_twelve/combine_scores_prompt_last.npz")
    rows = Z["rows"]
    sid = np.array([meta[i]["sample_id"] for i in rows]); ver = np.array([meta[i]["version"] for i in rows])
    y = np.char.startswith(np.array([meta[i]["final_label"] for i in rows]).astype(str), "correct").astype(int)
    subj = {}
    for l in open("data/popqa/1. popqa_raw.jsonl", encoding="utf-8"):
        x = json.loads(l); subj[str(x["id"])] = x["s_uri"].rsplit("/", 1)[-1]
    txt = {}
    for l in open("data/popqa/4. popqa_paraphrased.jsonl", encoding="utf-8"):
        r = json.loads(l)
        for v in VERS:
            txt[(str(r["id"]), v)] = r[v]
    facts = np.unique(sid)
    f_sub = np.array([subj[f] for f in facts])
    subjects = np.unique(f_sub)
    rng = np.random.default_rng(0)
    fold_of_sub = {s: i % 5 for i, s in enumerate(rng.permutation(subjects))}      # same folds as combine_signals.py
    row_fold = np.array([fold_of_sub[subj[f]] for f in sid])
    first = collections.Counter(norm(txt[(s, v)]).split()[0] for s, v in zip(sid, ver))
    top_first = [w for w, _ in first.most_common(10)]
    F = []
    for s, v in zip(sid, ver):
        q, o = txt[(s, v)], txt[(s, "question")]
        w, wo = norm(q).split(), norm(o).split()
        jac = len(set(w) & set(wo)) / max(len(set(w) | set(wo)), 1)
        F.append([len(w), len(q), jac, float(v == "question")] + [float(w[0] == t) for t in top_first])
    F = np.array(F)
    sets = {"원래 질문 여부만": [3], "문구 특징(길이, 시작 단어)": [0, 1] + list(range(4, 4 + len(top_first))), "표면 특징 전부": list(range(F.shape[1]))}
    conf = Z["conf"]
    scores = {k: np.zeros(len(rows)) for k in ["conf", "sae20", "conf_sae20"] + list(sets) + ["conf + 표면 전부"]}
    scores["conf"], scores["sae20"], scores["conf_sae20"] = Z["conf"], Z["sae20"], Z["conf_sae20"]

    def fit_pred(X, tr, te):
        Xt = X[tr].astype(float).copy()
        for f in np.unique(sid[tr]):
            m = sid[tr] == f; Xt[m] -= Xt[m].mean(0)
        sc = StandardScaler().fit(Xt)
        clf = LogisticRegression(C=1.0, max_iter=400).fit(sc.transform(Xt), y[tr])
        return clf.decision_function(sc.transform(X[te].astype(float)))

    for fold in range(5):
        tr, te = row_fold != fold, row_fold == fold
        for name, cols in sets.items():
            scores[name][te] = fit_pred(F[:, cols], tr, te)
        scores["conf + 표면 전부"][te] = fit_pred(np.c_[conf, F], tr, te)
    per = {}
    for name, sc in scores.items():
        top, base, aucs = [], [], []
        for f in facts:
            m = np.where(sid == f)[0]
            top.append(y[m][np.argmax(sc[m])]); base.append(y[m].mean()); aucs.append(within_auc(sc[m], y[m]))
        per[name] = (np.array(top), np.array(base), np.nanmean(aucs))
    bs = np.random.default_rng(1).integers(0, len(facts), (N_BOOT, len(facts)))
    print(f"[A] 부분적으로 아는 PopQA 사실 {len(facts)}개 (맞힘·틀림 모두 있음), 주어 단위 5-fold. 점수 1등 변형이 맞은 비율")
    print(f"  {'신호':<26}{'1등 맞힘':>9}{'무작위 대비 [95% 구간]':>26}{'같은 사실 안 AUROC':>20}")
    for name, (top, base, auc) in per.items():
        d = top - base; lo, hi = np.percentile(d[bs].mean(1), [2.5, 97.5])
        print(f"  {name:<26}{top.mean():>9.1%}{d.mean():>+11.1%} [{lo:+.1%},{hi:+.1%}]{auc:>14.3f}")
    for a, b in (("표면 전부" if False else "표면 특징 전부", "conf"), ("conf_sae20", "표면 특징 전부"), ("conf + 표면 전부", "conf_sae20")):
        d = per[a][0] - per[b][0]; lo, hi = np.percentile(d[bs].mean(1), [2.5, 97.5])
        print(f"  {a} − {b}: {d.mean():+.1%} [{lo:+.1%},{hi:+.1%}]")
    # voting on the same facts
    sidr = {f: np.where(sid == f)[0] for f in facts}
    ans = np.array([norm(meta[i]["model_answer"]) for i in rows])
    vote_hits = []
    for f in facts:
        m = sidr[f]
        g = collections.defaultdict(list)
        for j in m:
            g[ans[j]].append(j)
        best = max(g.values(), key=lambda js: (len(js), max(conf[j] for j in js)))
        vote_hits.append(float(np.mean([y[j] for j in best]) > 0.5))
    base = per["conf"][1]
    d = np.array(vote_hits) - base; lo, hi = np.percentile(d[bs].mean(1), [2.5, 97.5])
    print(f"  {'다수결 (동률은 confidence)':<26}{np.mean(vote_hits):>9.1%}{d.mean():>+11.1%} [{lo:+.1%},{hi:+.1%}]")
    d2 = np.array(vote_hits) - per["conf"][0]; lo, hi = np.percentile(d2[bs].mean(1), [2.5, 97.5])
    print(f"  다수결 − conf 고르기: {d2.mean():+.1%} [{lo:+.1%},{hi:+.1%}]")


def overall(name, facts):
    """facts: {fact: [(version, norm_answer, correct)]}; only facts with all 5 versions."""
    singles, origs, votes, oracle = [], [], [], []
    for f, rs in facts.items():
        if len(rs) != 5:
            continue
        c = np.array([r[2] for r in rs], float)
        singles.append(c.mean()); oracle.append(float(c.max()))
        orig = [r for r in rs if r[0] == "question"]
        origs.append(float(orig[0][2]) if orig else np.nan)
        g = collections.defaultdict(list)
        for r in rs:
            g[r[1]].append(r)
        best = max(g.values(), key=lambda L: (len(L), any(r[0] == "question" for r in L)))
        votes.append(float(np.mean([r[2] for r in best]) > 0.5))
    n = len(singles)
    mid = np.array(oracle) - 0
    print(f"  {name:<28}{n:>7,}{np.mean(singles):>10.1%}{np.nanmean(origs):>10.1%}{np.mean(votes):>10.1%}{np.mean(oracle):>10.1%}")


def part_b():
    print("\n[B] 전체 사실에서 (5개 문구가 모두 있는 사실): 문구 하나 / 원래 문구 / 다수결 / 천장(하나라도 맞음)")
    print(f"  {'데이터':<28}{'사실':>7}{'무작위':>10}{'원래문구':>10}{'다수결':>10}{'천장':>10}")
    # PopQA Qwen (final labels)
    by = collections.defaultdict(list); bad = set()
    for l in open("analysis/graded/7. final/qwen2.5-7b_popqa_final.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("gold_issue") or r["final_label"] == "uncertain" or r.get("audit_flag"):
            bad.add(r["sample_id"])
        by[r["sample_id"]].append((r["version"], norm(r["model_answer"]), r["final_label"] in ("correct", "correct_lenient")))
    overall("PopQA · Qwen (최종 라벨)", {f: v for f, v in by.items() if f not in bad})
    for model, tag in (("qwen2.5-7b", "Qwen"), ("dream-7b", "Dream")):
        for ds in ("popqa", "triviaqa"):
            if model == "qwen2.5-7b" and ds == "popqa":
                continue
            by = collections.defaultdict(list); bad = set()
            for l in open(f"analysis/graded/1. reviewed 1/{model}_{ds}_graded.jsonl", encoding="utf-8"):
                r = json.loads(l)
                if r.get("gold_issue") or r["verdict"] == "uncertain":
                    bad.add(r["sample_id"])
                by[r["sample_id"]].append((r["version"], norm(r["model_answer"]), r["verdict"] == "correct"))
            overall(f"{ds.replace('popqa', 'PopQA').replace('triviaqa', 'TriviaQA')} · {tag} (1차 채점)", {f: v for f, v in by.items() if f not in bad})


if __name__ == "__main__":
    part_a()
    part_b()
