"""E53 (PREREG.md): tuned lens 용 프롬프트 끝 (L-1 = 답 칸 0 을 예측하는 자리) 상태 수집, 28 층 전부.
--mode train : 분석에 쓰지 않는 행 (E7 밖의 5/5 문구 전부 3,165 + E7 밖의 0/5 문구 무작위 4,835 = 8,000) 의 step 0, 2, 4, 8.
               개입 없는 생성 (topk_margin, 32/32) 을 step 8 forward 까지만 돌리고 멈춤 (그 뒤 step 은 필요 없음).
--mode ana   : 분석 행.  E44 와 같은 묶음 구성 (R = 1~4/5 5,950 + E7 5/5 2,000, by_length) 으로 끝까지 생성, step 0~8, 16, 31.
               칸 0 의 step 별 argmax / 마스크 / 실제 로짓 금 순위도 저장 (E44 와 같은 계산).  최종 답과 채점.
상태는 층별 파일 hid_L{l}.npy [n, step, 3584] fp16 로 --dir (기본: 로컬 디스크) 에 저장.  재개 가능.
usage: python followup_mk/e53_capture.py --mode train|ana [--limit 32] [--dir D]"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
MODE = sys.argv[sys.argv.index("--mode") + 1]
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
DEF = "${SCRATCH:-/tmp/scratch}/e53"   # 실행 서버의 로컬 디스크
D = (sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else DEF) + f"/{MODE}{'_pilot' if LIM else ''}"
os.makedirs(D, exist_ok=True)
NL = 28
_, Z7, F7 = e7_rows()
if MODE == "train":
    SV = [0, 2, 4, 8]
    five = np.setdiff1d(np.where(k == 5)[0], F7); zero = np.setdiff1d(np.where(k == 0)[0], Z7)
    zero = np.random.default_rng(53).permutation(zero)[:8000 - len(five)]
    R = [int(i) for i in np.concatenate([five, zero])]
else:
    SV = list(range(9)) + [16, 31]
    R = [int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]          # E44 의 R 과 같은 순서
setof = {i: ("mid" if mid[i] else ("five" if k[i] == 5 else "zero")) for i in R}
m = Dream(); tok = m.tok
ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in R}
batches = list(by_length(R, lambda i: ids[i]))
if LIM: batches = batches[:max(1, LIM // BS)]
rows_mine = [i for _, ix in batches for i in ix]; n = len(rows_mine)
np.save(f"{D}/rows.npy", np.array(rows_mine, dtype=np.int64))
json.dump(dict(mode=MODE, steps=SV, layers=list(range(NL)), position="prompt_end (L-1)", n=n, gold_rule="tok(' ' + gold_value)[0]",
               note="hid = decoder layer l 출력 (layer 27 출력에 final norm + lm_head = 실제 칸 0 로짓)"), open(f"{D}/config.json", "w"), ensure_ascii=False)
mm = lambda name, shape, dt: np.lib.format.open_memmap(f"{D}/{name}.npy", mode="r+" if os.path.exists(f"{D}/{name}.npy") else "w+", dtype=dt, shape=shape)
HID = [mm(f"hid_L{l}", (n, len(SV), 3584), np.float16) for l in range(NL)]
if MODE == "ana":
    S0M = mm("slot0_mask", (n, STEPS), bool); S0A = mm("slot0_argmax", (n, STEPS), np.int32); S0R = mm("slot0_gold_rank", (n, STEPS), np.int32)
    S0G = mm("slot0_gold_logit", (n, STEPS), np.float32); COM = mm("committed", (n, STEPS), np.int16)
done = np.load(f"{D}/done.npy") if os.path.exists(f"{D}/done.npy") else np.zeros(n, bool)
where = {i: j for j, i in enumerate(rows_mine)}
gold = {i: tok(" " + rows[i]["gold_value"], add_special_tokens=False)["input_ids"][0] for i in rows_mine}
meta = open(f"{D}/meta.jsonl", "a", encoding="utf-8"); t0 = time.time(); cnt = 0
class Stop(Exception): pass
print(f"{MODE}: 묶음 {len(batches)}, 행 {n}, 이미 {int(done.sum())}", flush=True)
for L, ix in batches:
    J = np.array([where[i] for i in ix])
    if done[J].all(): continue
    nb = len(ix); g = torch.tensor([gold[i] for i in ix], device="cuda"); ar = torch.arange(nb, device="cuda")
    hb = torch.zeros(nb, len(SV), NL, 3584, dtype=torch.float16, device="cuda")
    def mk(l_):
        def f(h, step):
            if step in SV:
                hb[:, SV.index(step), l_] = h[:, L - 1].to(torch.float16)
                if MODE == "train" and l_ == NL - 1 and step == SV[-1]: raise Stop()
        return f
    fns = {l_: mk(l_) for l_ in range(NL)}
    if MODE == "train":
        for att in range(120):          # 다른 사용자와 GPU 를 나눠 쓰므로 OOM 이면 기다렸다 다시
            try: m.generate([ids[i] for i in ix], fns)
            except Stop: break
            except (torch.OutOfMemoryError, RuntimeError) as _e:
                if "out of memory" not in str(_e): raise
                m.fns = {}; torch.cuda.empty_cache(); print("OOM, 60s 대기", flush=True); time.sleep(60)
        m.fns = {}
    else:
        com = torch.zeros(nb, STEPS, dtype=torch.int16, device="cuda"); s0m = torch.zeros(nb, STEPS, dtype=torch.bool, device="cuda"); s0a = torch.zeros(nb, STEPS, dtype=torch.int32, device="cuda")
        s0r = torch.zeros(nb, STEPS, dtype=torch.int32, device="cuda"); s0g = torch.zeros(nb, STEPS, device="cuda")
        def lh(step, x, logits):
            cv = x[:, -CANVAS:] == m.mask_id; com[:, step] = (CANVAS - cv.sum(-1)).to(torch.int16); s0m[:, step] = cv[:, 0]
            z = logits[:, -CANVAS].float(); s0a[:, step] = torch.softmax(logits[:, -CANVAS], -1).max(-1).indices.int()   # E44 와 같은 동점 처리
            gv = z[ar, g]; s0g[:, step] = gv; s0r[:, step] = (z > gv[:, None]).sum(-1).int()
            return logits
        for att in range(120):
            try: ans = m.generate([ids[i] for i in ix], fns, logits_hook=lh); break
            except (torch.OutOfMemoryError, RuntimeError) as _e:
                if "out of memory" not in str(_e): raise
                m.fns = {}; torch.cuda.empty_cache(); print("OOM, 60s 대기", flush=True); time.sleep(60)
        COM[J] = com.cpu().numpy(); S0M[J] = s0m.cpu().numpy(); S0A[J] = s0a.cpu().numpy(); S0R[J] = s0r.cpu().numpy(); S0G[J] = s0g.cpu().numpy()
        s0m_ = s0m.cpu().numpy(); s0a_ = s0a.cpu().numpy()
    hb_ = hb.cpu().numpy()
    for l_ in range(NL): HID[l_][J] = hb_[:, :, l_]
    for j, i in enumerate(ix):
        rec = {"row": i, "set": setof[i], "k": int(k[i]), "fid": fid[i], "gold_id": int(gold[i])}
        if MODE == "ana":
            lm_ = np.where(s0m_[j])[0]; last = int(lm_.max()) if len(lm_) else -1
            rec.update(answer=ans[j], new_type=m.grade(i, ans[j]), slot0_last_masked_step=last, slot0_final_token=int(s0a_[j, last]) if last >= 0 else None)
        meta.write(json.dumps(rec, ensure_ascii=False) + "\n")
    done[J] = True; cnt += nb; meta.flush()
    if cnt % 320 < nb:
        for a_ in HID: a_.flush()
        if MODE == "ana":
            for a_ in (S0M, S0A, S0R, S0G, COM): a_.flush()
        np.save(f"{D}/done.npy", done); print(f"  {int(done.sum())}/{n} {time.time() - t0:.0f}s", flush=True)
for a_ in HID: a_.flush()
if MODE == "ana":
    for a_ in (S0M, S0A, S0R, S0G, COM): a_.flush()
np.save(f"{D}/done.npy", done); print(f"DONE {time.time() - t0:.0f}s", flush=True)
