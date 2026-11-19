"""E58b 방향 (CPU): 겹 g 마다 학습 겹 (grp != g) 1~4/5 사실 중 맞힘·틀림이 다 있는 사실의 (맞힘 평균 − 틀림 평균) 층 27 step 0 프롬프트 끝 상태를 사실끼리 평균, 단위화.
크기 = 0.5 x 층 27 평균 ‖h‖ (분석 행 전체).  -> e58b/dirs.npz"""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from e58b_load import load_base
R, B, H, k, grp, fid = load_base()
hb = H[:, 0].astype(np.float32); ok = np.array([B[i]["type"] == "correct" for i in R]); mid = (k >= 1) & (k <= 4)
dirs = np.zeros((5, hb.shape[1]), np.float32); nf = []
for g in range(5):
    by = collections.defaultdict(lambda: ([], []))
    for j in np.where(mid & (grp != g))[0]: by[fid[j]][0 if ok[j] else 1].append(j)
    ds = [hb[c].mean(0) - hb[w].mean(0) for c, w in by.values() if c and w]; v = np.mean(ds, 0); dirs[g] = v / np.linalg.norm(v); nf.append(len(ds))
scale = 0.5 * float(np.linalg.norm(hb, axis=1).mean())
np.savez("followup_mk/e58b/dirs.npz", dirs=dirs, scale=scale); print("사실 수 (겹별)", nf, "scale", round(scale, 3), "겹 방향 cos", np.round(dirs @ dirs.T, 3)[0])
