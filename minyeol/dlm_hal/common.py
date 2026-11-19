"""후속 실험 공용 코드.  repo 루트에서 실행한다.  방향, 겹, 생성, 채점은 analysis/6_steering 스크립트와 같은 식."""
import collections, json, sys
import numpy as np, torch
sys.path.insert(0, "data/1_raw"); sys.path.insert(0, "data/2_grading"); sys.path.insert(0, "data/3_classification")
import wd
from run_inference import build_prompt, extract_answer, INSTRUCTION_PROMPT
from run_grading import is_correct
from classify_rules import classify

D_ = "data"; MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; LAYER, CANVAS, STEPS, BS, NF, ND = 20, 32, 32, 16, 5, 4
OUT = "followup_mk"
rows = [json.loads(l) for l in open(f"{D_}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
POP = {str(json.loads(l)["id"]): json.loads(l) for l in open(f"{D_}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([POP[r["sample_id"]]["s_uri"] for r in rows]); _rng = np.random.default_rng(0); _subs = _rng.permutation(np.unique(sub)); _order = {x: j for j, x in enumerate(_subs)}; grp = np.array([_order[x] % NF for x in sub])
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k >= 1) & (k <= 4)
unit = lambda v: v / np.linalg.norm(v)

def load_X():
    H = np.load(f"{D_}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r"); X = np.asarray(H[:, 0], dtype=np.float32)
    norm = float(np.linalg.norm(X[np.random.default_rng(1).choice(len(X), 500, replace=False)], axis=1).mean())
    return X, norm

def directions(X):
    """caa_directions.py 와 같은 B (정답-붕괴), D0..D3 (붕괴 반반).  겹마다."""
    dirs = {}
    for g in range(NF):
        tr = grp != g; byf = collections.defaultdict(lambda: collections.defaultdict(list))
        for i in np.where(tr & mid & np.isin(ty, ["correct", "broken"]))[0]: byf[fid[i]][ty[i]].append(i)
        dirs[g] = {"B": unit(np.mean([X[v["correct"]].mean(0) - X[v["broken"]].mean(0) for v in byf.values() if v["correct"] and v["broken"]], 0))}
        bi = np.where(tr & (ty == "broken"))[0]
        for q in range(ND):
            p = np.random.default_rng(100 * q + g).permutation(bi); dirs[g][f"D{q}"] = unit(X[p[: len(p) // 2]].mean(0) - X[p[len(p) // 2:]].mean(0))
    return dirs

class Dream:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
        self.SP = set(self.tok.all_special_ids); self.mask_id = self.tok.mask_token_id
        names = wd._cache.get("names", {}); self.has_cand = lambda n: (bool(names.get(n)) if n in names else None)
        self.fns = {}; self.step = -1; self.nl = len(self.model.model.layers)
        for l_, layer in enumerate(self.model.model.layers): layer.register_forward_hook(self._mk(l_))
    def _mk(self, l_):
        def hook(module, a, output):
            if l_ == 0: self.step += 1
            f = self.fns.get(l_)
            if f is not None:
                h = output[0] if isinstance(output, tuple) else output; f(h, self.step)
        return hook
    def ids(self, prompt): return self.tok(prompt)["input_ids"]
    @torch.inference_mode()
    def generate(self, id_lists, fns=None, steps=STEPS, alg="topk_margin", canvas=CANVAS, logits_hook=None, seed=None):
        """같은 길이 프롬프트 묶음.  fns: {층: f(h, step)} 를 h 에 제자리로 적용.  logits_hook(step, x, logits) -> logits."""
        self.fns = fns or {}; self.step = -1
        if seed is not None: torch.manual_seed(seed)
        inp = torch.tensor(id_lists, device="cuda"); kw = {} if logits_hook is None else {"generation_logits_hook_func": logits_hook}
        o = self.model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=canvas, steps=steps, temperature=0.0, alg=alg, alg_temp=0.0,
                                          mask_token_id=self.mask_id, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id, **kw)
        self.fns = {}; L = inp.shape[1]; out = []
        for j in range(len(id_lists)):
            gen = o[j][L:].tolist(); gen = gen[: next((q for q, t in enumerate(gen) if t in self.SP), len(gen))]; out.append(extract_answer(self.tok.decode(gen, skip_special_tokens=True)))
        return out
    def grade(self, i, ans):
        r = rows[i]; ok = is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]})
        return "correct" if ok else classify(ans, r["question"], r["relation"], self.has_cand)[0]

def by_length(items, idf):
    """items 를 프롬프트 길이로 묶어 BS 씩 자른다.  idf(item) -> 토큰 목록."""
    g = collections.defaultdict(list)
    for it in items: g[len(idf(it))].append(it)
    for L, xs in sorted(g.items()):
        for b0 in range(0, len(xs), BS): yield L, xs[b0:b0 + BS]

def add_end(vecs, L):
    """프롬프트 끝 토큰에 행별 벡터를 더함.  vecs [n, D] float32 cuda."""
    def f(h, step): h[:, L - 1] = (h[:, L - 1].float() + vecs).to(h.dtype)
    return f

def boot_mean(x, B=10000, seed=0):
    x = np.asarray(x, dtype=float); ix = np.random.default_rng(seed).integers(0, len(x), (B, len(x))); m = x[ix].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]

# ---- 야간 묶음 공용 (E7~E10)
def e7_rows():
    z = np.random.default_rng(21).permutation(np.where(k == 0)[0])[:2000]; f = np.random.default_rng(22).permutation(np.where(k == 5)[0])[:2000]
    return np.concatenate([np.where(mid)[0], z, f]).astype(int), z, f

def set_S():
    r5 = np.random.default_rng(5); _, z, f = e7_rows()
    S = {"broken": np.where(mid & (ty == "broken"))[0], "entity": r5.permutation(np.where(mid & (ty == "entity"))[0])[:300], "correct": r5.permutation(np.where(mid & (ty == "correct"))[0])[:300],
         "zero": z[:300], "five": f[:300]}
    return [(s, int(i)) for s, v in S.items() for i in v]

_tq = None
def positions(i):
    """(토큰 목록, 질문 끝, 주어 끝 또는 None).  dream_steps_extract.py 와 같은 규칙 (Qwen 빠른 토크나이저 offset)."""
    global _tq
    if _tq is None:
        from transformers import AutoTokenizer
        _tq = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    r = rows[i]; p = build_prompt(r["question"]); enc = _tq(p, return_offsets_mapping=True); off = enc["offset_mapping"]
    q0 = p.find(r["question"]); qe = q0 + len(r["question"]); ql = max(t for t, (a, b) in enumerate(off) if a < qe and b > q0)
    x = POP[r["sample_id"]]; low = p.lower(); se = None
    for nm in sorted([m_ for m_ in [x["subj"]] + json.loads(x["s_aliases"] or "[]") if m_], key=len, reverse=True):
        j = low.find(nm.lower())
        if j >= 0: se = max(t for t, (a, b) in enumerate(off) if a < j + len(nm) and b > j); break
    return enc["input_ids"], ql, se
