"""v2: adds per-layer gold-vs-best-competitor margin and the logit lens at the subject's last token.

Logit lens + teacher forcing + raw hidden states for the within-fact analysis (Qwen2.5-7B, PopQA, 7. final).

Rows (analysis_use only, gold_issue / audit-flagged facts already excluded by analysis_use):
  partial  every row of 1-4/5 facts
  K5       every row of 5/5 facts
  U        rows of a fixed random sample of 0/5 facts (--n-u-facts)
minus GUESS-RISK facts: in sport / country / religion / occupation / color, facts whose gold set contains one of the
relation's default answers (model's top-3 outputs in the relation, or the 3 most common golds of the relation).

Per row, at prompt_last (the position that predicts the first answer token), for every layer L = 1..28:
  logit lens  log_softmax(lm_head(norm(h_L))) (the last layer is already normed by the model)
  gold        max log-prob / min rank over the first tokens of every gold alias, with and without a leading space
  out         log-prob / rank of the first token of the model's own answer (" " + model_answer)
Also: teacher-forced log-prob of the full gold answer (gold_value) and of the full model answer, and
raw hidden states at prompt_last for layers 4, 8, ..., 28 and at q_last for layer 20.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 sae_analysis/logit_lens_extract.py
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "TrivaQA_ex")
from extract_qwen_features import MODEL, REVISION, positions  # noqa: E402
from run_inference import build_prompt  # noqa: E402
from scoring import normalize_answer, strip_answer_wrapper  # noqa: E402

FINAL = "analysis/graded/7. final/qwen2.5-7b_popqa_final.jsonl"
RISK_RELATIONS = {"sport", "country", "religion", "occupation", "color"}
HID_LAYERS = [4, 8, 12, 16, 20, 24, 28]


def norm(x):
    return normalize_answer(strip_answer_wrapper(x or ""))


def select_rows(n_u_facts, seed=0):
    rows = [json.loads(l) for l in open(FINAL, encoding="utf-8")]
    golds = {r["sample_id"]: {norm(g) for g in [r["gold_value"], *(r.get("gold_aliases") or [])] if norm(g)} for r in rows}
    defaults = {}
    for rel in RISK_RELATIONS:
        out = collections.Counter(norm(r["model_answer"]) for r in rows if r["relation"] == rel and norm(r["model_answer"]))
        gc = collections.Counter(g for f in {r["sample_id"] for r in rows if r["relation"] == rel} for g in golds[f])
        defaults[rel] = {a for a, _ in out.most_common(3)} | {g for g, _ in gc.most_common(3)}
    risky = {r["sample_id"] for r in rows if r["relation"] in RISK_RELATIONS and golds[r["sample_id"]] & defaults[r["relation"]]}
    use = [r for r in rows if r["analysis_use"] and r["sample_id"] not in risky]
    grp = {r["sample_id"]: r["fact_group"] for r in use}
    u_facts = sorted({s for s, g in grp.items() if g == "0/5"})
    rng = np.random.default_rng(seed)
    u_keep = set(rng.permutation(u_facts)[:n_u_facts])
    sel = [r for r in use if grp[r["sample_id"]] != "0/5" or r["sample_id"] in u_keep]
    print(f"guess-risk facts removed: {len(risky)} | defaults: { {k: sorted(v) for k, v in defaults.items()} }")
    print("selected rows by fact_group:", dict(collections.Counter(r["fact_group"] for r in sel)), flush=True)
    return sel, golds


def first_token_ids(tok, texts):
    ids = set()
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        for v in {t, t[:1].upper() + t[1:]}:
            for pre in (" ", ""):
                enc = tok(pre + v, add_special_tokens=False)["input_ids"]
                if enc:
                    ids.add(enc[0])
    return sorted(ids)


@torch.no_grad()
def answer_logprob(model, tok, prompts, answers, device):
    """Sum and mean log-prob of " " + answer given prompt (teacher forcing). Empty answers -> nan."""
    out_sum, out_mean = np.full(len(prompts), np.nan), np.full(len(prompts), np.nan)
    items = [(i, p, a.strip()) for i, (p, a) in enumerate(zip(prompts, answers)) if a and a.strip()]
    if not items:
        return out_sum, out_mean
    p_ids = [tok(p)["input_ids"] for _, p, _ in items]
    a_ids = [tok(" " + a, add_special_tokens=False)["input_ids"][:24] for _, _, a in items]
    seqs = [p + a for p, a in zip(p_ids, a_ids)]
    L = max(map(len, seqs))
    pad = tok.pad_token_id
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, : len(s)] = torch.tensor(s)
        att[j, : len(s)] = 1
    logp = torch.log_softmax(model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.float(), -1)
    for j, ((i, _, _), p, a) in enumerate(zip(items, p_ids, a_ids)):
        pos = torch.arange(len(p) - 1, len(p) - 1 + len(a), device=device)
        lp = logp[j, pos, torch.tensor(a, device=device)]
        out_sum[i], out_mean[i] = lp.sum().item(), lp.mean().item()
    return out_sum, out_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-u-facts", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", default="sae_analysis/features_lens_v2")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows, golds = select_rows(args.n_u_facts)
    subj_names = {}
    for l in open("data/popqa/1. popqa_raw.jsonl"):
        x = json.loads(l)
        subj_names[str(x["id"])] = [x["subj"]] + json.loads(x["s_aliases"] or "[]")
    if args.limit:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0}, low_cpu_mem_usage=True).eval()
    norm_layer, head = model.model.norm, model.lm_head
    n, nl, d = len(rows), model.config.num_hidden_layers, model.config.hidden_size
    gold_lp, gold_rk = np.zeros((n, nl), np.float32), np.zeros((n, nl), np.int32)
    out_lp, out_rk = np.zeros((n, nl), np.float32), np.zeros((n, nl), np.int32)
    argmax_last = np.zeros(n, np.int64)
    overlap = np.zeros(n, bool)
    H = np.zeros((n, len(HID_LAYERS), d), np.float16)
    Hq = np.zeros((n, d), np.float16)
    tf = {k: np.zeros(n, np.float32) for k in ("gold_sum", "gold_mean", "out_sum", "out_mean")}
    comp_lp, comp_id = np.zeros((n, nl), np.float32), np.zeros((n, nl), np.int64)
    sgold_lp, sgold_rk = np.full((n, nl), np.nan, np.float32), np.zeros((n, nl), np.int32)
    subj_last = np.full(n, -1, np.int64)

    for start in range(0, n, args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [build_prompt(r["question"]) for r in batch]
        pos = [positions(tok, r["question"], p) for r, p in zip(batch, prompts)]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
            ar = torch.arange(len(batch), device="cuda")
            pl = torch.tensor([p[2] for p in pos], device="cuda")
            sl_pos = []
            for r, pr in zip(batch, prompts):
                low, ends = pr.lower(), None
                for nm in sorted(subj_names.get(r["sample_id"], []), key=len, reverse=True):
                    i0 = low.find(nm.lower())
                    if i0 >= 0:
                        offs = tok(pr, return_offsets_mapping=True)["offset_mapping"]
                        ends = max(j for j, (a, b) in enumerate(offs) if a < i0 + len(nm) and b > i0)
                        break
                sl_pos.append(-1 if ends is None else ends)
            subj_last[start : start + len(batch)] = sl_pos
            has_s = torch.tensor([x >= 0 for x in sl_pos], device="cuda")
            spos = torch.tensor([max(x, 0) for x in sl_pos], device="cuda")
            ql = torch.tensor([p[1] for p in pos], device="cuda")
            gids = [first_token_ids(tok, [r["gold_value"], *(r.get("gold_aliases") or [])]) for r in batch]
            oids = [tok(" " + (r["model_answer"] or "").strip(), add_special_tokens=False)["input_ids"][:1] or [tok.eos_token_id] for r in batch]
            B = len(batch)
            G = max(map(len, gids))
            gpad = torch.tensor([g + [g[0]] * (G - len(g)) for g in gids], device="cuda")  # [B, G]
            ot = torch.tensor([o[0] for o in oids], device="cuda")
            hl = torch.stack([hs[L][ar, pl] for L in range(1, nl + 1)], 1)  # [B, nl, d]
            hn = torch.cat([norm_layer(hl[:, :-1]), hl[:, -1:]], 1)  # last layer already normed
            hs_ = torch.stack([hs[L][ar, spos] for L in range(1, nl + 1)], 1)
            hsn = torch.cat([norm_layer(hs_[:, :-1]), hs_[:, -1:]], 1)  # last layer already normed
            for L0 in range(0, nl, 7):  # subject-last position: gold log-prob / rank
                lps = torch.log_softmax(head(hsn[:, L0 : L0 + 7]).float(), -1)
                gs = lps.gather(2, gpad[:, None, :].expand(-1, lps.shape[1], -1)).max(-1).values
                rs = (lps > gs[..., None]).sum(-1) + 1
                ss = slice(L0, L0 + lps.shape[1])
                sgold_lp[start : start + B, ss] = torch.where(has_s[:, None], gs, torch.full_like(gs, float("nan"))).cpu().numpy()
                sgold_rk[start : start + B, ss] = rs.cpu().numpy()
                del lps
            for L0 in range(0, nl, 7):  # chunks of layers to bound memory
                logp = torch.log_softmax(head(hn[:, L0 : L0 + 7]).float(), -1)  # [B, l, V]
                gsel = logp.gather(2, gpad[:, None, :].expand(-1, logp.shape[1], -1)).max(-1).values  # [B, l]
                masked = logp.scatter(2, gpad[:, None, :].expand(-1, logp.shape[1], -1), float("-inf"))
                cv, ci = masked.max(-1)
                comp_lp[start : start + B, L0 : L0 + logp.shape[1]] = cv.cpu().numpy()
                comp_id[start : start + B, L0 : L0 + logp.shape[1]] = ci.cpu().numpy()
                del masked
                osel = logp.gather(2, ot[:, None, None].expand(-1, logp.shape[1], 1)).squeeze(-1)
                sl = slice(L0, L0 + logp.shape[1])
                gold_lp[start : start + B, sl] = gsel.cpu().numpy()
                gold_rk[start : start + B, sl] = ((logp > gsel[..., None]).sum(-1) + 1).cpu().numpy()
                out_lp[start : start + B, sl] = osel.cpu().numpy()
                out_rk[start : start + B, sl] = ((logp > osel[..., None]).sum(-1) + 1).cpu().numpy()
                if L0 + logp.shape[1] == nl:
                    argmax_last[start : start + B] = logp[:, -1].argmax(-1).cpu().numpy()
                del logp
            for i, L in enumerate(HID_LAYERS):
                H[start : start + B, i] = hs[L][ar, pl].float().cpu().numpy().astype(np.float16)
            Hq[start : start + len(batch)] = hs[20][ar, ql].float().cpu().numpy().astype(np.float16)
            for j in range(len(batch)):
                overlap[start + j] = oids[j][0] in gids[j]
        gs, gm = answer_logprob(model, tok, prompts, [r["gold_value"] for r in batch], "cuda")
        os_, om = answer_logprob(model, tok, prompts, [r["model_answer"] for r in batch], "cuda")
        tf["gold_sum"][start : start + len(batch)], tf["gold_mean"][start : start + len(batch)] = gs, gm
        tf["out_sum"][start : start + len(batch)], tf["out_mean"][start : start + len(batch)] = os_, om
        if (start // args.batch_size) % 25 == 0:
            print(f"{start + len(batch)}/{n}", flush=True)

    np.savez(os.path.join(args.output_dir, "lens.npz"), gold_lp=gold_lp, gold_rk=gold_rk, out_lp=out_lp, out_rk=out_rk,
             argmax_last=argmax_last, overlap=overlap, comp_lp=comp_lp, comp_id=comp_id,
             sgold_lp=sgold_lp, sgold_rk=sgold_rk, subj_last=subj_last, **tf)
    np.save(os.path.join(args.output_dir, "hidden_prompt_last.npy"), H)
    np.save(os.path.join(args.output_dir, "hidden_q_last_L20.npy"), Hq)
    with open(os.path.join(args.output_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in ("sample_id", "version", "relation", "question", "model_answer", "gold_value",
                                                      "final_label", "fact_group", "k_correct")}, ensure_ascii=False) + "\n")
    print(f"done {n} rows; hidden layers {HID_LAYERS}", flush=True)


if __name__ == "__main__":
    main()
