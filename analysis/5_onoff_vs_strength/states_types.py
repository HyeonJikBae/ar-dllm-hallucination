"""Paper-style (MaxMin over groups) separation, with groups = wrong-answer TYPE (entity / broken) x knowledge STATE (knows / does not know / knows-but-wrong).  Dream, layer 20, prompt-end token.
States:  KNOW = correct rows of 5/5 facts   UNK = wrong rows of 0/5 facts   KBW = wrong rows of 1/5..4/5 facts
Comparisons, each done separately for wrong type t in {entity, broken} (the paper's four entity types play this role) and ranked by the MIN over types, like the paper's MaxMin:
  KNOW vs UNK_t    KNOW vs KBW_t    UNK_t vs KBW_t   (does-not-know-and-wrong vs knows-but-wrong)
Latents picked on dev subjects, reported on test subjects.  on/off = separation score of activation>0 ; strength = Welch t of the activation.
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/5_onoff_vs_strength/states_types.py [step_index 0..4]   (0 = step 0, 1 = step 8)"""
import json, sys, warnings
import numpy as np, torch
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis")
import sae_io
S = int(sys.argv[1]) if len(sys.argv) > 1 else 0
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
cor = np.array([r["correct"] for r in rows]); k = np.array([r["k_correct"] for r in rows]); ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows])
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
know = cor & (k == 5); unk = ~cor & (k == 0); kbw = ~cor & (k >= 1) & (k <= 4)
w = {a: b.cuda() for a, b in sae_io._load(sae_io.DREAM).items()}
enc = sae_io._encoder(w, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
H = np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r"); N = len(rows)
A = np.zeros((N, 16384), np.float16)
for b in range(0, N, 1024): A[b:b + 1024] = enc(torch.from_numpy(np.asarray(H[b:b + 1024, S], dtype=np.float32)).cuda()).half().cpu().numpy()
TYPES = ("entity", "broken")
def sep_t(pos, neg):
    P, Q = A[pos].astype(np.float32), A[neg].astype(np.float32)
    return (P > 0).mean(0) - (Q > 0).mean(0), (P.mean(0) - Q.mean(0)) / np.sqrt(P.var(0) / len(P) + Q.var(0) / len(Q) + 1e-8)
def run(name, posf, negf):
    ps = {t: posf(t) for t in TYPES}; ns = {t: negf(t) for t in TYPES}
    sizes = {t: (int((ps[t] & dev).sum()), int((ns[t] & dev).sum()), int((ps[t] & te).sum()), int((ns[t] & te).sum())) for t in TYPES}
    print(f"\n=== {name}   (n dev pos/neg, test pos/neg per type: {sizes})")
    dv = {t: sep_t(ps[t] & dev, ns[t] & dev) for t in TYPES}
    for kind, ix in (("on/off", 0), ("strength(t)", 1)):
        M = np.stack([dv[t][ix] for t in TYPES])                       # [type, latent]
        mm_pos, mm_neg = M.min(0), (-M).min(0)                         # MaxMin: best worst-case type
        ip, im = np.argsort(-mm_pos), np.argsort(-mm_neg)
        print(f"  [{kind}] best MaxMin on dev: positive side {mm_pos[ip[0]]:.2f}, negative side {mm_neg[im[0]]:.2f}")
        for K in (1, 20):
            line = []
            for t in TYPES:
                m = (ps[t] | ns[t]) & te; y = ps[t][m].astype(int)
                X = A[m].astype(np.float32)
                if kind == "on/off": X = (X > 0).astype(np.float32); sc = X[:, ip[:K]].sum(1) - X[:, im[:K]].sum(1)
                else:
                    mu, sd = A[(ps[t] | ns[t]) & dev].astype(np.float32).mean(0), A[(ps[t] | ns[t]) & dev].astype(np.float32).std(0) + 1e-6
                    Z = (X - mu) / sd; sc = Z[:, ip[:K]].sum(1) - Z[:, im[:K]].sum(1)
                line.append(f"{t} test AUROC {roc_auc_score(y, sc):.3f}")
            print(f"     top-{K:<2d} MaxMin latents: " + " | ".join(line))
print("step", [0, 8, 16, 24, 31][S])
run("KNOW vs UNK  (knows vs does-not-know, wrong type t)", lambda t: know, lambda t: unk & (ty == t))
run("KNOW vs KBW  (knows vs knows-but-wrong type t)", lambda t: know, lambda t: kbw & (ty == t))
run("UNK vs KBW  (does-not-know-wrong vs knows-but-wrong, type t)", lambda t: unk & (ty == t), lambda t: kbw & (ty == t))
print("\nabstain sizes (not analysed per type): UNK", int((unk & (ty == 'abstain')).sum()), " KBW", int((kbw & (ty == 'abstain')).sum()))
