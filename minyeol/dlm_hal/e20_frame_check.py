"""E20 사전 점검 (CPU, 관찰): 답 길이 틀 (첫 특수 토큰 칸) 과 내용 칸의 확정 순서.  E7 궤적."""
import collections, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import k, fid, OUT, MODEL, REV
D7 = f"{OUT}/e7"; R = np.load(f"{D7}/rows.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")); n = len(R)
from transformers import AutoTokenizer
SP = np.array(sorted(AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True).all_special_ids))
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); com = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
isSP = np.isin(com, SP); L = np.where(isSP.any(1), isSP.argmax(1), 32)
t0 = lastm[:, 0]                                                   # 칸 0 확정 step
tF = np.array([lastm[j, L[j]] if L[j] < 32 else 99 for j in range(n)])   # 길이를 정한 칸 (첫 특수 칸) 확정 step
ncont_before = np.array([(lastm[j, :L[j]] < tF[j]).sum() if L[j] < 32 else -1 for j in range(n)])   # 틀 확정 전에 이미 확정된 내용 칸 수
sp_before0 = np.array([np.where(isSP[j] & (lastm[j] < t0[j]))[0].min() if (isSP[j] & (lastm[j] < t0[j])).any() else 32 for j in range(n)])   # 칸 0 확정 전 상한
kk = k[R]; ty = np.array([meta[int(i)]["new_type"] for i in R]); f7 = fid[R]
G = {"5/5 정답": (kk == 5) & (ty == "correct"), "1~4/5 정답": (kk >= 1) & (kk <= 4) & (ty == "correct"), "1~4/5 entity": (kk >= 1) & (kk <= 4) & (ty == "entity"),
     "1~4/5 붕괴": (kk >= 1) & (kk <= 4) & (ty == "broken"), "0/5": kk == 0}
gl = collections.defaultdict(list)
for j in np.where(G["1~4/5 정답"])[0]: gl[f7[j]].append(L[j])
for nm, g in G.items():
    ix = np.where(g)[0]; o = {"n": len(ix), "답 길이 중앙값": float(np.median(L[ix])), "칸0 확정 step 중앙값": float(np.median(t0[ix])), "틀 확정 step 중앙값": float(np.median(tF[ix])),
        "틀이 내용보다 먼저(내용 0칸 확정)%": round(100 * float((ncont_before[ix] == 0).mean()), 1), "틀이 칸0보다 먼저%": round(100 * float((tF[ix] < t0[ix]).mean()), 1)}
    if nm.startswith("1~4/5") and "정답" not in nm:
        jj = [j for j in ix if f7[j] in gl]; lo = np.array([min(gl[f7[j]]) for j in jj]); hi = np.array([max(gl[f7[j]]) for j in jj]); Lw = L[jj]
        o.update({"같은 사실 정답 길이보다 짧음%": round(100 * float((Lw < lo).mean()), 1), "같음(범위 안)%": round(100 * float(((Lw >= lo) & (Lw <= hi)).mean()), 1), "김%": round(100 * float((Lw > hi).mean()), 1),
                  "칸0 확정 전 상한이 정답 길이보다 짧음%": round(100 * float((sp_before0[jj] < lo).mean()), 1)})
    print(nm, json.dumps(o, ensure_ascii=False), flush=True)
