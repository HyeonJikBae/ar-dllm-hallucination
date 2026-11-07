"""Load the two layer-20 SAEs used in the analyses and encode hidden states.  Weights live in the repo's sae/ folder (not in git).

  dream_sae()  Dream-v0-Base-7B, shared SAE trained on masking rates 0.2 / 0.5 / 0.8  (sae/dream_pilot_200M/dream-7b_L20_shared-0.2-0.5-0.8_s5_200M)
  qwen_sae()   Qwen2.5-7B, BatchTopK 16k k=100                                        (sae/qwen2.5-7b_L20_batchtopk_16k_k100_20M)
Both return a function  encode(x: torch.Tensor [n, 3584]) -> torch.Tensor [n, 16384]  (JumpReLU activations).
"""
import json
import os

import torch
from safetensors.torch import load_file

SAE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sae")
DREAM = os.path.join(SAE_ROOT, "dream_pilot_200M", "dream-7b_L20_shared-0.2-0.5-0.8_s5_200M")
QWEN = os.path.join(SAE_ROOT, "qwen2.5-7b_L20_batchtopk_16k_k100_20M")


def _load(path):
    w = load_file(os.path.join(path, "sae_weights.safetensors"))
    return {k: w[k].float() for k in ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")}


def _encoder(w, subtract_b_dec):
    @torch.no_grad()
    def encode(x):
        x = x.float()
        pre = ((x - w["b_dec"]) if subtract_b_dec else x) @ w["W_enc"] + w["b_enc"]
        return pre * (pre > w["threshold"])
    encode.W_dec = w["W_dec"]          # decoder directions (rows), for steering
    return encode


def dream_sae():
    cfg = json.load(open(os.path.join(DREAM, "cfg.json")))
    return _encoder(_load(DREAM), cfg.get("apply_b_dec_to_input", True))


def qwen_sae():
    return _encoder(_load(QWEN), True)
