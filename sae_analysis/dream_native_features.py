"""Turn dream_native chunks into per-row confidence features aligned with dream_ext/rows.jsonl (= features_dream_step0_ext order).
  nat_mean/min/sum  log-prob (at the step it was committed) of the answer tokens (positions before the first EOS)
  nat_first         same for the first canvas position
  nat_s0_max8       max top1-top2 margin over the first 8 canvas positions at step 0 (pre-decoding)
  nat_s0_top3       mean of the 3 largest margins among the first 8 canvas positions at step 0 (pre-decoding)
Output: sae_analysis/results_twelve/dream_native/features.npz (names, X [N, k], answers)."""
import glob, json, sys
import numpy as np
D = "sae_analysis/results_twelve/dream_native"
files = sorted(glob.glob(f"{D}/chunk_*.npz")); X, ans = [], []
for f in files:
    z = np.load(f, allow_pickle=True); lp, mg, mk, fx, eos = z["lp"].astype(np.float32), z["mg"].astype(np.float32), z["mk"], z["fx"], int(z["eos"])
    n = len(fx)
    commit = mk.sum(1) - 1                                  # [n, 32]: step at which each canvas position was committed
    commit = np.clip(commit, 0, lp.shape[1] - 1)
    clp = np.take_along_axis(lp, commit[:, None, :], 1)[:, 0, :]   # [n, 32]
    has = (fx == eos); first_eos = np.where(has.any(1), has.argmax(1), fx.shape[1])
    ans_mask = np.arange(fx.shape[1])[None, :] < first_eos[:, None]
    cnt = ans_mask.sum(1)
    s = (clp * ans_mask).sum(1)
    mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    mn = np.where(cnt > 0, np.where(ans_mask, clp, np.inf).min(1), np.nan)
    m0 = mg[:, 0, :8]
    X.append(np.c_[mean, mn, np.where(cnt > 0, s, np.nan), clp[:, 0], m0.max(1), np.sort(m0, 1)[:, -3:].mean(1)])
    ans += list(z["ans"])
X = np.concatenate(X)
for j in range(X.shape[1]):
    X[:, j] = np.where(np.isnan(X[:, j]), np.nanmin(X[:, j]), X[:, j])
names = ["Dream 확신도: 답 토큰 평균 log p", "Dream 확신도: 답 토큰 최소 log p", "Dream 확신도: 답 토큰 log p 합", "Dream 확신도: 첫 칸 확정 log p",
         "Dream 확신도(디코딩 전): 앞 8칸 최대 격차", "Dream 확신도(디코딩 전): 앞 8칸 상위3 격차"]
np.savez(f"{D}/features.npz", names=np.array(names), X=X, ans=np.array(ans, dtype=object))
print("rows", len(X), "features", names)
