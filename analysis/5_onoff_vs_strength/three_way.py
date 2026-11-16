"""3-way classification of the knowledge state of a ROW: KNOW (correct, 5/5 fact) / UNK (wrong, 0/5 fact) / KBW (wrong, 1/5..4/5 fact).  Dream, layer 20, prompt-end token.
Multinomial logistic regression, class-balanced, fit on dev subjects, scored on test subjects: balanced accuracy, per-class recall, confusion matrix (chance = 0.333).
Inputs: SAE on/off (activation>0, all 16k latents) | SAE strength | raw hidden state | relation only (baseline).   Steps 0 and 8.
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/5_onoff_vs_strength/three_way.py"""
import json, sys, warnings
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis")
import sae_io
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
cor = np.array([r["correct"] for r in rows]); k = np.array([r["k_correct"] for r in rows]); rel = np.array([[r["relation"]] for r in rows])
lab = np.full(len(rows), -1); lab[cor & (k == 5)] = 0; lab[~cor & (k == 0)] = 1; lab[~cor & (k >= 1) & (k <= 4)] = 2
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
use = lab >= 0
# balance dev: cap each class at the size of the smallest*? -> keep class_weight balanced but cap the largest class for speed
di = np.where(dev & use)[0]; cap = 6000
di = np.concatenate([rng.permutation(di[lab[di] == c])[:cap] for c in range(3)]); ti = np.where(te & use)[0]
print("dev (used)", {c: int((lab[di] == c).sum()) for c in range(3)}, "| test", {c: int((lab[ti] == c).sum()) for c in range(3)}, flush=True)
def report(name, Xd, Xt):
    m = LogisticRegression(C=0.01, max_iter=300, class_weight="balanced").fit(Xd, lab[di]); p = m.predict(Xt); y = lab[ti]
    r = recall_score(y, p, average=None); cm = confusion_matrix(y, p, normalize="true")
    print(f"{name:<26s} balanced acc {balanced_accuracy_score(y, p):.3f} | recall KNOW {r[0]:.2f} UNK {r[1]:.2f} KBW {r[2]:.2f}")
    return cm
oh = OneHotEncoder(handle_unknown="ignore").fit(rel[di]); report("relation only (baseline)", oh.transform(rel[di]), oh.transform(rel[ti]))
w = {a: b.cuda() for a, b in sae_io._load(sae_io.DREAM).items()}
enc = sae_io._encoder(w, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
H = np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r")
for S in (0, 1):
    print(f"\n--- step {[0, 8][S]}")
    R = {i: np.asarray(H[i, S], dtype=np.float32) for i in np.concatenate([di, ti])}
    Rd = np.stack([R[i] for i in di]); Rt = np.stack([R[i] for i in ti])
    def sae(X): return np.concatenate([enc(torch.from_numpy(X[b:b + 1024]).cuda()).cpu().numpy() for b in range(0, len(X), 1024)])
    Ad, At = sae(Rd), sae(Rt)
    report("SAE on/off", (Ad > 0).astype(np.float32), (At > 0).astype(np.float32))
    sc = StandardScaler().fit(Ad); report("SAE strength", sc.transform(Ad), sc.transform(At))
    sc = StandardScaler().fit(Rd); cm = report("raw hidden state", sc.transform(Rd), sc.transform(Rt))
    print("  confusion (raw, rows = true KNOW/UNK/KBW, columns = predicted):"); print(np.round(cm, 2))
