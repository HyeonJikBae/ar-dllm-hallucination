"""BY FACT GROUP: the same analysis as within_fact_types.py, but the within-fact AUROC is reported separately for facts with k = 4, 3, 2, 1 (of 5).  Models are fit once on all dev facts 1/5..4/5 and scored per k on test facts.

Inside the same fact (1/5..4/5), what differs between a correct row and an entity-wrong row, between a correct row and a broken row, and between an entity-wrong and a broken row?
Dream, prompt-end token, steps 0 and 8.  Inputs: SAE on/off (top-20), SAE strength (top-20, t-stat), raw hidden-state probe, answer-slot confidence (log-prob / margin, all denoising steps up to the saved step), question wording (tf-idf of the question text).
Everything is fit on dev subjects and scored on test subjects.  Score = within-fact AUROC (AUROC computed inside each fact that has both kinds of rows, averaged over test facts); 95% CI by resampling test facts.
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/5_onoff_vs_strength/within_fact_types_by_k.py"""
import json, sys, warnings
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis")
import sae_io
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"; K = 20
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
k_all = np.array([r["k_correct"] for r in rows]); ty_all = np.array([r["type3"] if not r["correct"] else "correct" for r in rows])
mid = np.where((k_all >= 1) & (k_all <= 4))[0]
ty = ty_all[mid]; fid = np.array([rows[i]["sample_id"] for i in mid]); sub = np.array([popqa[rows[i]["sample_id"]] for i in mid]); quest = [rows[i]["question"] for i in mid]
rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
w = {a: b.cuda() for a, b in sae_io._load(sae_io.DREAM).items()}
enc = sae_io._encoder(w, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
Hm = np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r")
RAW = [np.asarray(Hm[mid, s], dtype=np.float32) for s in (0, 1)]
ACT = [np.concatenate([enc(torch.from_numpy(R[b:b + 1024]).cuda()).half().cpu().numpy() for b in range(0, len(R), 1024)]) for R in RAW]
LP = np.load(f"{F}/conf_lp.npy", mmap_mode="r")[mid].astype(np.float32); MG = np.load(f"{F}/conf_mg.npy", mmap_mode="r")[mid].astype(np.float32)
CONF = np.concatenate([LP.mean(2), LP.min(2), MG.mean(2), MG.min(2)], 1)
def conf_cols(smax): return np.concatenate([CONF[:, j * 32: j * 32 + smax + 1] for j in range(4)], 1)
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3).fit([q for q, d in zip(quest, dev) if d]); TXT = tf.transform(quest)

def within(score, y, m, rng_b=np.random.default_rng(0)):
    if m.sum() == 0: return float('nan'), float('nan'), float('nan'), 0
    per = []
    for u in np.unique(fid[m]):
        k = m & (fid == u)
        if y[k].min() != y[k].max(): per.append(roc_auc_score(y[k], score[k]))
    if len(per) == 0: return float('nan'), float('nan'), float('nan'), 0
    per = np.array(per); bs = [rng_b.choice(per, len(per)).mean() for _ in range(500)]
    return per.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(per)
def topk(X, y, mk, on):
    P, Q = X[mk & dev & (y == 1)], X[mk & dev & (y == 0)]
    if on:
        sep = (P > 0).mean(0) - (Q > 0).mean(0); B = (X > 0).astype(np.float32)
        return B[:, np.argsort(-sep)[:K]].sum(1) - B[:, np.argsort(sep)[:K]].sum(1)
    t = (P.mean(0) - Q.mean(0)) / np.sqrt(P.var(0) / len(P) + Q.var(0) / len(Q) + 1e-8); mu, sd = X[mk & dev].mean(0), X[mk & dev].std(0) + 1e-6; Z = (X - mu) / sd
    return Z[:, np.argsort(-t)[:K]].sum(1) - Z[:, np.argsort(t)[:K]].sum(1)
def lr(X, y, mk, C):
    sc = StandardScaler(with_mean=not hasattr(X, "tocsr")).fit(X[mk & dev]); m = LogisticRegression(C=C, max_iter=400).fit(sc.transform(X[mk & dev]), y[mk & dev]); return m.decision_function(sc.transform(X))

kk = k_all[mid]
def run(title, pos, neg):
    mk = np.isin(ty, [pos, neg]); y = (ty == pos).astype(int)
    out = {"question wording (tf-idf)": lr(TXT, y, mk, 0.3)}
    A0 = ACT[0].astype(np.float32); A1 = ACT[1].astype(np.float32)
    out["SAE on/off, step 0"] = topk(A0, y, mk, True); out["SAE strength, step 0"] = topk(A0, y, mk, False); out["raw probe, step 0"] = lr(RAW[0], y, mk, 0.001)
    out["SAE on/off, step 8"] = topk(A1, y, mk, True); out["SAE strength, step 8"] = topk(A1, y, mk, False); out["slot confidence, step 0"] = lr(conf_cols(0), y, mk, 0.1); out["raw probe, step 8"] = lr(RAW[1], y, mk, 0.001)
    print(f"\n=== {title}  (label 1 = {pos}, 0 = {neg})")
    for g in (4, 3, 2, 1):
        m = mk & te & (kk == g); nf = within(out["raw probe, step 0"], y, m)[3]
        print(f"  -- facts {g}/5: test facts with both kinds {nf}, test rows {int(m.sum())}")
        if nf < 8: print("     (too few facts)"); continue
        for nm, sc in out.items():
            a, lo, hi, n = within(sc, y, m); print(f"     {nm:<28s} {a:.3f} [{lo:.3f}, {hi:.3f}]")
run("correct vs entity-wrong", "correct", "entity"); run("correct vs broken", "correct", "broken"); run("entity-wrong vs broken (both wrong)", "broken", "entity")
