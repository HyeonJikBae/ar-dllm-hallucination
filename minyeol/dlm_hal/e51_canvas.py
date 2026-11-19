"""E51 (PREREG.md 'E51 캔버스 교환 2x2' + 00:45 보강 '경계 보존 중립').  문구 효과 대 확정 내용 효과.
단위: 같은 사실 (1~4/5) 의 (S = Dream 맞힌 문구, F = Dream 틀린 문구) 짝.  F 의 경로: 같은 문구를 Qwen 이 맞히면 D (DLM 고유), 틀리면 H (공유).
phase base : 1~4/5 행 전부를 새로 생성 (끝에 새 라벨로 짝 선택) (같은 디코딩, 행별 옮기는 개수 = 원본과 같음).  step 2, 4 입력 캔버스, 최종 캔버스, 칸 0 금 순위 저장.
phase cont : step tau 의 forward 직전 캔버스를 바꿔 넣고 tau 부터 남은 일정대로 끝까지 생성.  조건 (확정 칸 수 같음):
   own (자기 캔버스), swap (짝 상대의 tau 확정 칸 위치·토큰), neut (자기 확정 위치에 다른 관계 무작위 행 최종 캔버스의 같은 위치 토큰),
   neutB (경계 보존 중립: 자기 확정 토큰 중 줄바꿈 포함/특수 토큰은 그대로, 나머지만 다른 관계 행 토큰), reset (F 만, 확정 칸 전부 마스크; 마스크 수 다름).
usage (repo 루트):  python followup_mk/e51_canvas.py --phase base|cont [--tau 2|4] [--pilot NFACT]"""
import collections, json, os, sys, time
import numpy as np, torch, torch.nn.functional as Fnn
sys.path.insert(0, "followup_mk")
from common import *
import dloop as dl0

PHASE = sys.argv[sys.argv.index("--phase") + 1]
TAU = int(sys.argv[sys.argv.index("--tau") + 1]) if "--tau" in sys.argv else None
PIL = int(sys.argv[sys.argv.index("--pilot") + 1]) if "--pilot" in sys.argv else 0
SFX = "_pilot" if PIL else ""; TAUS = (2, 4)
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); w7 = {int(r): j for j, r in enumerate(R7)}
meta7 = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta7[d["row"]] = d
QW = {}
for l in open(f"{D_}/3_classification/full/qwen_final_full.jsonl", encoding="utf-8"):
    d = json.loads(l); QW[(d["sample_id"], d["version"])] = d
WRONG = ("entity", "broken")


def qroute(r):
    """F 행의 경로.  Qwen 정답 -> D, Qwen entity/broken -> H, 그 밖 (abstain) -> None (E49 와 같이 제외)."""
    q = QW[(rows[r]["sample_id"], rows[r]["version"])]
    return "D" if q["correct"] else ("H" if q["type3"] in WRONG else None)


def select_pairs(B_):
    """새 base (B_: row -> 기록) 의 라벨과 궤적으로 짝을 고른다.  S = 새 base 정답, F = 새 base entity/broken (경로 D/H, Qwen abstain 제외).
    두 행 모두 step 2 입력에서 칸 0 이 마스크.  사실당 최대 2 짝, D 먼저 (드문 경로), 남는 자리에 H.  F 는 step 4 에도 마스크인 행 먼저, S 는 짝끼리 다른 행 먼저."""
    MIDT = B_[next(iter(B_))]["mask_id"]
    el = lambda r, t: B_[r][f"canvas_t{t}"][0] == MIDT
    byf = collections.defaultdict(list)
    for r in sorted(B_): byf[fid[r]].append(r)
    pairs = []
    for f in sorted(byf):
        rng = np.random.default_rng(int(f) % (2 ** 31))
        Sc = [r for r in byf[f] if B_[r]["new_type"] == "correct" and el(r, 2)]
        if not Sc: continue
        Fc = {ro: [r for r in byf[f] if B_[r]["new_type"] in WRONG and qroute(r) == ro and el(r, 2)] for ro in "DH"}
        picks = []
        for ro in "DH":
            c = [int(r) for r in rng.permutation(Fc[ro])] if Fc[ro] else []; c.sort(key=lambda r: -el(r, 4))
            picks += [(ro, r) for r in c]
        used = collections.Counter()
        for ro, r in picks[:2]:
            sc = [int(s) for s in rng.permutation(Sc)]; sc.sort(key=lambda s: (used[s], -(el(s, 4) and el(r, 4))))
            s = sc[0]; used[s] += 1
            pairs.append(dict(pid=len(pairs), fact=str(f), route=ro, F=r, S=s, rel=rows[r]["relation"]))
    return pairs


@torch.inference_mode()
def run(m, id_lists, canvases=None, start=0, golds=None, snap=(), steps=STEPS, eps=1e-3):
    """dloop.generate (mode orig, topk_margin, temperature 0) 을 복사해 고친 고리.  옮기는 개수는 행별 마스크 수 기준
    (모든 행의 마스크 수가 같으면 원본의 배치 평균 규칙과 같음).  canvases [B, 32] (마스크 = mask_id) 를 주면 step start 부터 이어서 생성.
    반환: 답, 최종 캔버스, {step: 입력 캔버스} (snap), {step: 칸 0 정보} (start 와 snap step 의 forward), 칸 0 이 확정된 step."""
    inp = torch.tensor(id_lists, device="cuda"); B, L = inp.shape; MID = m.mask_id; m.fns = {}; m.step = -1
    x = Fnn.pad(inp, (0, CANVAS), value=MID) if canvases is None else torch.cat([inp, torch.as_tensor(np.asarray(canvases), device="cuda").long()], 1)
    ts = torch.linspace(1, eps, steps + 1, device="cuda"); rows_ = torch.arange(B, device="cuda")
    g = torch.tensor(golds, device="cuda") if golds is not None else None
    snaps, s0 = {}, {}; c0 = [None] * B
    for i in range(start, steps):
        if i in snap: snaps[i] = x[:, L:].clone().cpu().numpy()
        mask_index = (x == MID)
        logits = m.model(x, "full", None).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        if g is not None and (i == start or i in snap or i == 0):
            l0 = logits[:, L].float(); gl = l0.gather(1, g[:, None])[:, 0]
            s0[i] = dict(rank=(l0 > gl[:, None]).sum(1).tolist(), logit=gl.tolist(), logp=torch.log_softmax(l0, -1).gather(1, g[:, None])[:, 0].tolist(),
                         top1=l0.argmax(-1).tolist(), top1_logit=l0.max(-1).values.tolist(), masked=mask_index[:, L].tolist())
        mask_logits = logits[mask_index]
        t = ts[i]; s = ts[i + 1]
        confidence, x0 = dl0._scores(mask_logits, "topk_margin")
        full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype); full_confidence[mask_index] = confidence
        nm = mask_index.sum(1)
        kr = [int(nm[r] * (1 - s / t)) if i < steps - 1 else int(nm[r]) for r in range(B)]; kmax = max(kr)
        was = mask_index[:, L].tolist()
        if kmax > 0:
            _, transfer_index = torch.topk(full_confidence, kmax)
            x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + MID; x_[mask_index] = x0.clone()
            valid = torch.arange(kmax, device=x.device)[None, :] < torch.tensor(kr, device=x.device)[:, None]
            ri = rows_.unsqueeze(1).expand_as(transfer_index)
            x[ri[valid], transfer_index[valid]] = x_[ri[valid], transfer_index[valid]]
        now = (x[:, L] == MID).tolist()
        for b in range(B):
            if was[b] and not now[b]: c0[b] = i
    out = []
    for j in range(B):
        gen = x[j, L:].tolist(); gen = gen[: next((q for q, t_ in enumerate(gen) if t_ in m.SP), len(gen))]; out.append(extract_answer(m.tok.decode(gen, skip_special_tokens=True)))
    return out, x[:, L:].cpu().numpy(), snaps, s0, c0


def readj(fn):
    R = []
    if os.path.exists(fn):
        for l in open(fn, encoding="utf-8"):
            try: R.append(json.loads(l))
            except Exception: pass
    return R


PFN = f"{OUT}/e51_pairs{SFX}.json"; BFN = f"{OUT}/results_e51{SFX}_base.jsonl"
if PHASE == "base":
    # 1~4/5 전체 행을 새로 생성 (묶음 = by_length, 묶음 번호 bid 와 자리 pos 를 저장해 cont 에서 같은 모양의 묶음을 다시 만든다)
    need = [int(i) for i in np.where(mid)[0]]
    if PIL:
        fs = sorted(set(fid[need].tolist())); keep = set(np.random.default_rng(51).permutation(fs)[:PIL].tolist()); need = [i for i in need if fid[i] in keep]
    m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in need}
    gold = {i: m.tok(" " + rows[i]["gold_value"], add_special_tokens=False)["input_ids"][0] for i in need}
    batches = list(by_length(need, lambda i: ids[i]))
    have = {d["bid"] for d in readj(BFN)}; fo = open(BFN, "a", encoding="utf-8"); t0 = time.time(); nb = 0; chk = [0, 0]
    print(f"base: 행 {len(need)}, 묶음 {len(batches)} (남음 {len(batches) - len(have)})", flush=True)
    for bid, (L, ix) in enumerate(batches):
        if bid in have: continue
        b = [ids[i] for i in ix]
        ans, fin, snaps, s0, c0 = run(m, b, golds=[gold[i] for i in ix], snap=TAUS)
        if PIL and nb < 4:   # sanity: 원본 dloop.generate 와 같은 묶음에서 최종 캔버스 일치
            a2, cv2, _ = dl0.generate(m, b, return_canvas=True); chk[0] += int((cv2.numpy() == fin).all(1).sum()); chk[1] += len(ix)
        for j, i in enumerate(ix):
            fo.write(json.dumps(dict(row=i, bid=bid, pos=j, fact=fid[i], k=int(k[i]), L=L, gold_id=int(gold[i]), mask_id=int(m.mask_id), answer=ans[j], new_type=m.grade(i, ans[j]),
                                     e7_type=meta7[i]["new_type"], e7_answer=meta7[i]["answer"], slot0_commit_step=c0[j], final=fin[j].tolist(),
                                     **{f"canvas_t{t}": snaps[t][j].tolist() for t in TAUS}, **{f"s0_t{t}": {kk: s0[t][kk][j] for kk in s0[t]} for t in s0}), ensure_ascii=False) + "\n")
        fo.flush(); nb += 1
        if nb % 50 == 0: print(f"  묶음 {nb}, {time.time() - t0:.0f}s", flush=True)
    if PIL: print(f"sanity 원본 dloop.generate 와 최종 캔버스 일치 {chk[0]}/{chk[1]}", flush=True)
    B_ = {d["row"]: d for d in readj(BFN)}; pairs = select_pairs(B_); json.dump(pairs, open(PFN, "w"), ensure_ascii=False)
    print(f"짝 {len(pairs)}: {collections.Counter(p['route'] for p in pairs)}, 사실 {len({p['fact'] for p in pairs})}", flush=True)
    print(f"base 끝 {time.time() - t0:.0f}s", flush=True); print("DONE", flush=True)

elif PHASE == "cont":
    # 같은 묶음 모양: base 의 묶음 bid (같은 행, 같은 순서, 같은 크기) 를 조건마다 다시 만든다.  해당 조건이 없는 행은 자기 캔버스로 채움 (기록 안 함).
    # 같은 모양이면 행별 계산이 다른 행 내용과 무관하므로 'own' 은 base 를 그대로 재현해야 한다 (sanity).
    assert TAU in TAUS
    pairs = json.load(open(PFN)); B_ = {d["row"]: d for d in readj(BFN)}
    m = Dream(); MID = m.mask_id; SP = m.SP; tok = m.tok
    _bd = {}
    def isbound(t):
        if t not in _bd: _bd[t] = (t in SP) or ("\n" in tok.decode([t]))
        return _bd[t]
    def committed(cv): return [(p, int(t)) for p, t in enumerate(cv) if t != MID]
    def tok_ids(s): return tok(" " + s, add_special_tokens=False)["input_ids"]
    hintset = {}
    def hset(i):
        if i not in hintset:
            r = rows[i]; S_ = set()
            for v in [r["gold_value"]] + list(r["gold_aliases"] or []):
                if v: S_ |= set(tok_ids(v))
            S_.discard(tok_ids(r["gold_value"])[0]); hintset[i] = S_
        return hintset[i]
    ok = [p for p in pairs if B_[p["F"]][f"canvas_t{TAU}"][0] == MID and B_[p["S"]][f"canvas_t{TAU}"][0] == MID]
    print(f"tau={TAU}: 짝 {len(ok)} / {len(pairs)} ({collections.Counter(p['route'] for p in ok)})", flush=True)
    pool = sorted(B_); poolrel = np.array([rows[i]["relation"] for i in pool])
    def donors(i):
        """다른 관계 무작위 행 순서 (행·tau 로 시드)."""
        rng = np.random.default_rng(1000003 * i + TAU); cand = np.where(poolrel != rows[i]["relation"])[0]
        return [pool[j] for j in rng.permutation(cand)]
    def neutral(i):
        own = committed(B_[i][f"canvas_t{TAU}"]); dl = donors(i)
        d0 = dl[0]; cvN = [MID] * CANVAS
        for p, t in own: cvN[p] = B_[d0]["final"][p]
        rep = [p for p, t in own if not isbound(t)]; cvB = [MID] * CANVAS
        for p, t in own: cvB[p] = t
        dB = None
        if rep:
            dB = next((d for d in dl if all(not isbound(B_[d]["final"][p]) for p in rep)), dl[0])
            for p in rep: cvB[p] = B_[dB]["final"][p]
        return cvN, d0, cvB, dB
    role = {}; partners = collections.defaultdict(list)
    for p in ok:
        role[p["F"]] = "F"; role[p["S"]] = "S"; partners[p["F"]].append((p["S"], p["pid"])); partners[p["S"]].append((p["F"], p["pid"]))
    byb = collections.defaultdict(list)
    for d in B_.values(): byb[d["bid"]].append(d)
    jobs = []   # (bid, variant, [(row, canvas, extra or None)])
    for bid in sorted(byb):
        mem = sorted(byb[bid], key=lambda d: d["pos"]); rs = [d["row"] for d in mem]
        if not any(r in role for r in rs): continue
        own = {r: B_[r][f"canvas_t{TAU}"] for r in rs}; neu = {r: neutral(r) for r in rs if r in role}
        V = {"own": [(r, own[r], dict(cond="own", role=role[r]) if r in role else None) for r in rs],
             "neut": [(r, neu[r][0], dict(cond="neut", role=role[r], donor=neu[r][1])) if r in role else (r, own[r], None) for r in rs],
             "neutB": [(r, neu[r][2], dict(cond="neutB", role=role[r], donor=neu[r][3])) if r in role else (r, own[r], None) for r in rs]}
        for q in range(max(len(partners[r]) for r in rs)):
            V[f"swap{q}"] = [(r, B_[partners[r][q][0]][f"canvas_t{TAU}"], dict(cond="swap", role=role[r], partner=partners[r][q][0], pid=partners[r][q][1]))
                             if len(partners[r]) > q else (r, own[r], None) for r in rs]
        if any(role.get(r) == "F" for r in rs):
            V["reset"] = [(r, [MID] * CANVAS, dict(cond="reset", role="F")) if role.get(r) == "F" else (r, own[r], None) for r in rs]
        for vn, lst in V.items(): jobs.append((bid, vn, lst))
    fn = f"{OUT}/results_e51{SFX}_t{TAU}.jsonl"; have = {(d["bid"], d["variant"]) for d in readj(fn)}
    todo = [j for j in jobs if (j[0], j[1]) not in have]
    print(f"묶음-조건 {len(jobs)} (남음 {len(todo)}): {collections.Counter(j[1] for j in jobs)}", flush=True)
    fo = open(fn, "a", encoding="utf-8"); t0 = time.time(); nb = 0
    for bid, vn, lst in todo:
        rs = [r for r, _, _ in lst]; L = B_[rs[0]]["L"]; idl = [m.ids(build_prompt(rows[r]["question"])) for r in rs]; assert all(len(x) == L for x in idl)
        ans, fin, _, s0, _ = run(m, idl, canvases=[cv for _, cv, _ in lst], start=TAU, golds=[B_[r]["gold_id"] for r in rs])
        for j, (i, cv, ex) in enumerate(lst):
            if ex is None: continue
            cm = committed(cv); hs = hset(i); bnd = [p for p, t in cm if isbound(t)]
            rec = dict(tau=TAU, bid=bid, variant=vn, row=i, **ex, fact=fid[i], answer=ans[j], new_type=m.grade(i, ans[j]),
                       base_answer=B_[i]["answer"], base_type=B_[i]["new_type"], same_as_base=(ans[j] == B_[i]["answer"]), final_same_as_base=bool((fin[j] == np.array(B_[i]["final"])).all()),
                       committed=cm, n_mask=CANVAS - len(cm), hint=any(t in hs for p, t in cm), hint_toks=[t for p, t in cm if t in hs],
                       bound_pos=bnd, first_bound=(min(bnd) if bnd else None), gold_len=len(tok_ids(rows[i]["gold_value"])), special=[p for p, t in cm if t in SP], final=fin[j].tolist(), final0_is_gold=bool(int(fin[j][0]) == B_[i]["gold_id"]),
                       **{f"s0_{kk}": s0[TAU][kk][j] for kk in ("rank", "logit", "logp", "top1", "top1_logit")})
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush(); nb += 1
        if nb % 100 == 0: print(f"  묶음-조건 {nb}/{len(todo)}, {time.time() - t0:.0f}s", flush=True)
    print(f"cont tau={TAU} 끝 {time.time() - t0:.0f}s", flush=True); print("DONE", flush=True)
