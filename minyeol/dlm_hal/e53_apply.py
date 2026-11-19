"""E53 (PREREG.md): 분석 행 (e53_capture.py --mode ana) 의 28 층 x step (0~8, 16, 31) 프롬프트 끝 상태에 tuned lens 와 logit lens 를 적용.
저장 (followup_mk/e53/, 작은 배열만): rows.npy, {tl,ll}_gold_rank [n, step, layer] int32, {tl,ll}_gold_logit fp16, {tl,ll}_final_rank int32 (최종 확정 칸 0 토큰의 순위),
{tl,ll}_top1 int32, {tl,ll}_kl fp16 (KL(실제 마지막 층 || lens)), meta 복사.  금 = tok(' ' + gold_value)[0].
usage: python followup_mk/e53_apply.py --ana DIR --lens DIR"""
import json, os, shutil, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
SC = "${SCRATCH:-/tmp/scratch}/e53"
ANA = arg("--ana", f"{SC}/ana"); LENS = arg("--lens", f"{SC}/lens"); OD = arg("--out", "followup_mk/e53"); os.makedirs(OD, exist_ok=True)
from e53_lens_train import head, Tr, dev, NL
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
R = np.load(f"{ANA}/rows.npy"); done = np.load(f"{ANA}/done.npy"); meta = {}
for l in open(f"{ANA}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
J = np.where(done & np.array([int(r) in meta for r in R]))[0]; RR = R[J]; n = len(J)
cfg = json.load(open(f"{ANA}/config.json")); S = cfg["steps"]; nS = len(S)
gold = torch.tensor([meta[int(r)]["gold_id"] for r in RR], device=dev)
fin = torch.tensor([meta[int(r)]["slot0_final_token"] if meta[int(r)]["slot0_final_token"] is not None else 0 for r in RR], device=dev)
log(f"행 {n}, steps {S}")
np.save(f"{OD}/rows.npy", RR); shutil.copy(f"{ANA}/meta.jsonl", f"{OD}/ana_meta.jsonl"); json.dump(cfg, open(f"{OD}/ana_config.json", "w"))
for nm in ("slot0_gold_rank", "slot0_argmax", "slot0_mask", "committed"): np.save(f"{OD}/{nm}.npy", np.asarray(np.load(f"{ANA}/{nm}.npy", mmap_mode="r")[J]))
O = {f"{p}_{x}": np.zeros((n, nS, NL), dt) for p in ("tl", "ll") for x, dt in (("gold_rank", np.int32), ("gold_logit", np.float16), ("final_rank", np.int32), ("top1", np.int32), ("kl", np.float16))}
H27 = np.load(f"{ANA}/hid_L{NL - 1}.npy", mmap_mode="r")
CH = 64   # 행 묶음 (x 11 step = 704 벡터)
with torch.no_grad():
    for l in range(NL):
        Hl = np.load(f"{ANA}/hid_L{l}.npy", mmap_mode="r")
        T = None
        if l < NL - 1:
            T = Tr().to(dev); T.load_state_dict({k_: v.to(dev) for k_, v in torch.load(f"{LENS}/T_L{l}.pt").items()})
        for b0 in range(0, n, CH):
            jj = J[b0:b0 + CH]; nb = len(jj)
            h = torch.tensor(np.asarray(Hl[jj], dtype=np.float32), device=dev).reshape(-1, 3584)
            hf = torch.tensor(np.asarray(H27[jj], dtype=np.float32), device=dev).reshape(-1, 3584)
            lp_t = torch.log_softmax(head(hf), -1)
            g = gold[b0:b0 + nb].repeat_interleave(nS); fz = fin[b0:b0 + nb].repeat_interleave(nS); ar = torch.arange(len(g), device=dev)
            for p, z in (("tl", head(T(h)) if T is not None else None), ("ll", head(h))):
                if z is None: z = head(h)
                gv = z[ar, g]; fv = z[ar, fz]
                lq = torch.log_softmax(z, -1); kl = (lp_t.exp() * (lp_t - lq)).sum(-1)
                sl = slice(b0, b0 + nb)
                O[f"{p}_gold_rank"][sl, :, l] = (z > gv[:, None]).sum(-1).reshape(nb, nS).cpu().numpy()
                O[f"{p}_gold_logit"][sl, :, l] = gv.reshape(nb, nS).cpu().numpy()
                O[f"{p}_final_rank"][sl, :, l] = (z > fv[:, None]).sum(-1).reshape(nb, nS).cpu().numpy()
                O[f"{p}_top1"][sl, :, l] = z.argmax(-1).reshape(nb, nS).cpu().numpy()
                O[f"{p}_kl"][sl, :, l] = kl.reshape(nb, nS).cpu().numpy()
        log(f"L{l}: KL tuned {float(O['tl_kl'][:, :, l].astype(np.float32).mean()):.3f} / logit {float(O['ll_kl'][:, :, l].astype(np.float32).mean()):.3f}")
for k_, v in O.items(): np.save(f"{OD}/{k_}.npy", v)
log("DONE")
