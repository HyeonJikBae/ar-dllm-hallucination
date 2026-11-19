"""E53 (PREREG.md): tuned lens 번역기 학습 (Belrose et al. 2023).  층 l (0~26) 출력의 프롬프트 끝 상태 h 를
T_l(h) = h + A_l h + b_l (A, b 는 0 에서 시작 = 항등) 로 옮긴 뒤 고정된 final norm + lm_head 로 읽은 분포가
같은 자리의 실제 마지막 층 분포 (층 27 출력 -> norm -> lm_head = 실제 칸 0 로짓) 와 가깝도록 KL(final || lens) 최소화.
층 27 은 항등 (정확).  자료: e53_capture.py --mode train (분석에 쓰지 않는 5/5·0/5 문구, step 0/2/4/8).
사실 (fid) 단위로 20% 를 보고용 hold-out, 나머지의 10% 사실을 조기 종료용 val.
usage: python followup_mk/e53_lens_train.py --data DIR --out DIR [--sweep] [--lr 1e-4] [--layers 0,1,..]"""
import json, os, sys, time
import numpy as np, torch
from safetensors import safe_open
arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
_SC = "${SCRATCH:-/tmp/scratch}/e53"
DATA = arg("--data", f"{_SC}/train"); OUTD = arg("--out", f"{_SC}/lens"); os.makedirs(OUTD, exist_ok=True)
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--Dream-org--Dream-v0-Base-7B/snapshots/6572adb5535263e4d1a337b56942ba48b6dee2a9")
NL = 28; dev = os.environ.get("E53_DEV", "cuda"); T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

def load_head():
    with safe_open(f"{SNAP}/model-00004-of-00004.safetensors", "pt", device=dev) as f: W = f.get_tensor("lm_head.weight")
    with safe_open(f"{SNAP}/model-00003-of-00004.safetensors", "pt", device=dev) as f: g = f.get_tensor("model.norm.weight")
    return W, g
W, G = load_head()
def head(h):
    """DreamRMSNorm (fp32 계산 -> bf16) + lm_head (bf16).  h: fp32 [n, d] -> fp32 로짓."""
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    return ((G * y.to(torch.bfloat16)) @ W.T).float()

def split_facts(fids, seed=530):
    u = np.unique(fids); p = np.random.default_rng(seed).permutation(u)
    rep = set(p[: int(0.2 * len(u))]); val = set(p[int(0.2 * len(u)): int(0.28 * len(u))])
    return np.array([f in rep for f in fids]), np.array([f in val for f in fids])

class Tr(torch.nn.Module):
    def __init__(self, d=3584):
        super().__init__(); self.lin = torch.nn.Linear(d, d); torch.nn.init.zeros_(self.lin.weight); torch.nn.init.zeros_(self.lin.bias)
    def forward(self, h): return h + self.lin(h)

def kl_top1(lp_t, logit_q):
    lq = torch.log_softmax(logit_q, -1)
    kl = (lp_t.exp() * (lp_t - lq)).sum(-1)
    return kl, (logit_q.argmax(-1) == lp_t.argmax(-1))

if __name__ == "__main__":
    meta = [json.loads(l) for l in open(f"{DATA}/meta.jsonl")]; R = np.load(f"{DATA}/rows.npy"); done = np.load(f"{DATA}/done.npy")
    mrow = {m_["row"]: m_ for m_ in meta}; J = np.where(done)[0]; fids = np.array([mrow[int(R[j])]["fid"] for j in J])
    rep, val = split_facts(fids); trn = ~rep & ~val
    S = json.load(open(f"{DATA}/config.json"))["steps"]; nS = len(S)
    log(f"행 {len(J)}: 학습 {trn.sum()}, val {val.sum()}, 보고 {rep.sum()} (사실 단위), steps {S}")
    def load(l): return torch.tensor(np.asarray(np.load(f"{DATA}/hid_L{l}.npy", mmap_mode="r")[J], dtype=np.float32), device=dev)   # [n, S, d]
    H27 = load(NL - 1)
    def targets(H, sel):
        out = []
        for b0 in range(0, len(sel), 512):
            with torch.no_grad(): out.append(torch.log_softmax(head(H[sel[b0:b0 + 512]]), -1).to(torch.bfloat16))
        return out
    flat = lambda mask: torch.tensor(np.where(np.repeat(mask, nS))[0], device=dev)
    I_tr, I_val, I_rep = flat(trn), flat(val), flat(rep)
    H27f = H27.reshape(-1, H27.shape[-1])

    def evaluate(T, Hf, idx):
        kls, t1s = [], []
        with torch.no_grad():
            for b0 in range(0, len(idx), 512):
                ii = idx[b0:b0 + 512]; lp_t = torch.log_softmax(head(H27f[ii]), -1)
                kl, t1 = kl_top1(lp_t, head(T(Hf[ii]) if T is not None else Hf[ii])); kls.append(kl); t1s.append(t1)
        return torch.cat(kls), torch.cat(t1s)

    def train_layer(l, lr, max_ep=40, patience=3, bs=256, wd=0.0):
        Hf = load(l).reshape(-1, H27.shape[-1]); T = Tr().to(dev); opt = torch.optim.AdamW(T.parameters(), lr=lr, weight_decay=wd)
        best = (float(evaluate(T, Hf, I_val)[0].mean()), 0, {k_: v.clone() for k_, v in T.state_dict().items()}); hist = [best[0]]; bad = 0
        g = torch.Generator(device=dev); g.manual_seed(l)
        for ep in range(1, max_ep + 1):
            perm = I_tr[torch.randperm(len(I_tr), device=dev, generator=g)]
            for b0 in range(0, len(perm), bs):
                ii = perm[b0:b0 + bs]
                with torch.no_grad(): lp_t = torch.log_softmax(head(H27f[ii]), -1)
                kl, _ = kl_top1(lp_t, head(T(Hf[ii]))); loss = kl.mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            v = float(evaluate(T, Hf, I_val)[0].mean()); hist.append(v)
            if v < best[0] - 1e-4: best = (v, ep, {k_: x.clone() for k_, x in T.state_dict().items()}); bad = 0
            else:
                bad += 1
                if bad >= patience: break
        T.load_state_dict(best[2]); return T, Hf, best, hist

    if "--sweep" in sys.argv:
        res = {}
        for l in [3, 12, 20, 25]:
            for lr in [1e-5, 3e-5, 1e-4, 3e-4]:
                T, Hf, best, hist = train_layer(l, lr, max_ep=12, patience=2)
                res[f"L{l}_lr{lr}"] = dict(best_val=round(best[0], 4), best_ep=best[1], hist=[round(x, 4) for x in hist], logit_lens_val=round(float(evaluate(None, Hf, I_val)[0].mean()), 4))
                log(l, lr, res[f"L{l}_lr{lr}"]); del T, Hf; torch.cuda.empty_cache()
        json.dump(res, open(f"{OUTD}/sweep.json", "w"), indent=1); sys.exit()

    if "--lr" in sys.argv: LR = float(arg("--lr", "1e-4"))
    else:   # 스윕에서 4 개 층 평균 val KL 이 가장 낮은 lr
        sw = json.load(open(f"{OUTD}/sweep.json")); lrs = sorted({k_.split("_lr")[1] for k_ in sw})
        LR = float(min(lrs, key=lambda x: np.mean([v["best_val"] for k_, v in sw.items() if k_.endswith("_lr" + x)]))); log("lr (스윕 선택)", LR)
    LAYS = [int(x) for x in arg("--layers", ",".join(map(str, range(NL - 1)))).split(",")]
    Q = json.load(open(f"{OUTD}/quality.json")) if os.path.exists(f"{OUTD}/quality.json") else {}
    stepv = torch.tensor(np.tile(np.arange(nS), len(J)), device=dev)
    for l in LAYS:
        if os.path.exists(f"{OUTD}/T_L{l}.pt") and str(l) in Q: log("skip", l); continue
        T, Hf, best, hist = train_layer(l, LR)
        torch.save({k_: v.cpu() for k_, v in T.state_dict().items()}, f"{OUTD}/T_L{l}.pt")
        q = {"best_ep": best[1], "val_hist": [round(x, 4) for x in hist]}
        for name, TT in (("tuned", T), ("logit", None)):
            kl, t1 = evaluate(TT, Hf, I_rep); st = stepv[I_rep]
            q[name] = dict(kl=round(float(kl.mean()), 4), top1=round(float(t1.float().mean()), 4),
                           by_step={str(S[s]): dict(kl=round(float(kl[st == s].mean()), 4), top1=round(float(t1[st == s].float().mean()), 4)) for s in range(nS)})
        Q[str(l)] = q; json.dump(Q, open(f"{OUTD}/quality.json", "w"), indent=1)
        log(f"L{l}: ep {best[1]}, 보고 KL tuned {q['tuned']['kl']} / logit {q['logit']['kl']}, top1 {q['tuned']['top1']} / {q['logit']['top1']}")
        del T, Hf; torch.cuda.empty_cache()
    log("DONE")
