"""Is "knows-but-wrong" just in between?  Fit a KNOW(5/5 correct) vs UNK(0/5 wrong) probe on dev subjects, project ALL test rows on that one axis.
Reports the mean position (0 = typical UNK, 1 = typical KNOW) by fact group k/5 x correct/wrong.   Dream, raw hidden state, prompt end.
Usage: python3 analysis/5_onoff_vs_strength/between_axis.py"""
import json, warnings, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
cor = np.array([r["correct"] for r in rows]); k = np.array([r["k_correct"] for r in rows])
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
H = np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r")
for S, name in ((0, "step 0"), (1, "step 8")):
    di = np.where(dev & (((k == 5) & cor) | ((k == 0) & ~cor)))[0]; di = np.concatenate([rng.permutation(di[k[di] == c])[:3000] for c in (0, 5)])
    X = np.stack([np.asarray(H[i, S], dtype=np.float32) for i in di]); sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.01, max_iter=300).fit(sc.transform(X), (k[di] == 5).astype(int))
    ti = np.where(te)[0]; s = m.decision_function(sc.transform(np.stack([np.asarray(H[i, S], dtype=np.float32) for i in ti])))
    ref0, ref1 = np.median(s[(k[ti] == 0)]), np.median(s[(k[ti] == 5) & cor[ti]])
    pos = (s - ref0) / (ref1 - ref0)
    print(f"\n{name}: position on the KNOW-vs-UNK axis (0 = median of 0/5 rows, 1 = median of correct 5/5 rows); test rows")
    print("  fact group | correct rows: n, median, | wrong rows: n, median")
    for g in range(6):
        c = (k[ti] == g) & cor[ti]; w = (k[ti] == g) & ~cor[ti]
        f = lambda m_: f"n={m_.sum():>5d} median {np.median(pos[m_]):.2f}" if m_.sum() else "      -        "
        print(f"  {g}/5        | {f(c)} | {f(w)}")
