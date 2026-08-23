#!/usr/bin/env python3
"""Smoke-test loading and factual completion generation for three base language models.

Select a physical GPU outside this script, for example:

    CUDA_VISIBLE_DEVICES=2 python test_model_loading.py

The models are deliberately loaded one at a time. This is only a loading and
generation-API smoke test; it does not perform evaluation or save activations.
"""

from __future__ import annotations

import gc
import traceback
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


MODELS = (
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B", "llama"),
    ("LLaDA-8B-Base", "GSAI-ML/LLaDA-8B-Base", "llada"),
    ("Dream-v0-Base-7B", "Dream-org/Dream-v0-Base-7B", "dream"),
)

# Base models are tested with factual completion prompts rather than
# Question/Answer instruction-style prompts.
PROMPTS = (
    "LeBron James was born in",
    "The movie 12 Angry Men was directed by",
    "The song Yellow Submarine was performed by",
    "Michael Jordan played the sport of",
)

# Shared generation setting.
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0

# LLaDA generation settings.
LLADA_STEPS = MAX_NEW_TOKENS
LLADA_BLOCK_LENGTH = MAX_NEW_TOKENS
LLADA_REMASKING = "low_confidence"

# LLaDA's official generate.py identifies 126336 as its [MASK] token ID.
LLADA_MASK_TOKEN_ID = 126336

# Dream generation settings.
DREAM_STEPS = MAX_NEW_TOKENS
DREAM_ALG = "entropy"
DREAM_ALG_TEMP = 0.0


def preferred_dtype() -> torch.dtype:
    """Use BF16 when the visible CUDA device supports it, otherwise FP16."""
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def input_device(model: torch.nn.Module) -> torch.device:
    """Find the device holding the model's input parameters."""
    return next(model.parameters()).device


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def print_model_info(
    model_id: str, model: torch.nn.Module, tokenizer: Any
) -> None:
    print(f"model name:       {model_id}")
    print(f"model class:      {type(model).__module__}.{type(model).__name__}")
    print(
        f"tokenizer class:  {type(tokenizer).__module__}."
        f"{type(tokenizer).__name__}"
    )
    print(f"parameter count:  {model_parameter_count(model):,}")


def print_generation_config() -> None:
    print("generation config:")
    print(f"  max_new_tokens:      {MAX_NEW_TOKENS}")
    print(f"  temperature:         {TEMPERATURE}")
    print(f"  LLaDA steps:         {LLADA_STEPS}")
    print(f"  LLaDA block_length:  {LLADA_BLOCK_LENGTH}")
    print(f"  LLaDA remasking:     {LLADA_REMASKING}")
    print(f"  Dream steps:         {DREAM_STEPS}")
    print(f"  Dream alg:           {DREAM_ALG}")
    print(f"  Dream alg_temp:      {DREAM_ALG_TEMP}")


def load_model(model_id: str, implementation: str) -> tuple[Any, Any]:
    """Load one official HF checkpoint onto the single visible CUDA device."""
    common = dict(
        pretrained_model_name_or_path=model_id,
        torch_dtype=preferred_dtype(),
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if implementation == "llama":
        model = AutoModelForCausalLM.from_pretrained(**common)
    else:
        # The official LLaDA and Dream repositories load their custom base model
        # classes with AutoModel, not with an AR causal-LM generation wrapper.
        model = AutoModel.from_pretrained(**common)

    model = model.to("cuda").eval()
    return model, tokenizer


@torch.inference_mode()
def generate_llama(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy autoregressive continuation."""
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {
        key: value.to(input_device(model))
        for key, value in encoded.items()
    }

    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    continuation = output_ids[
        0,
        encoded["input_ids"].shape[1] :,
    ]

    return tokenizer.decode(
        continuation,
        skip_special_tokens=True,
    ).strip()


def _llada_transfer_schedule(
    mask: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Official LLaDA linear-noise schedule: unmask nearly equally per step."""
    mask_count = mask.sum(dim=1, keepdim=True)

    schedule = torch.zeros(
        mask_count.shape[0],
        steps,
        dtype=torch.long,
        device=mask.device,
    ) + mask_count // steps

    remainder = mask_count % steps

    for row in range(mask_count.shape[0]):
        schedule[row, : int(remainder[row].item())] += 1

    return schedule


@torch.inference_mode()
def llada_diffusion_generate(
    model: Any,
    prompt_ids: torch.Tensor,
    *,
    generation_length: int = MAX_NEW_TOKENS,
    steps: int = LLADA_STEPS,
    block_length: int = LLADA_BLOCK_LENGTH,
    temperature: float = TEMPERATURE,
    remasking: str = LLADA_REMASKING,
) -> torch.Tensor:
    """LLaDA's masked-diffusion decoding algorithm.

    Current setting:
        generation_length = 32
        steps = 32
        block_length = 32
        temperature = 0
        remasking = low_confidence

    Therefore, the full continuation region is decoded as one diffusion block
    with one denoising step per target token position.
    """
    if generation_length % block_length:
        raise ValueError(
            "LLaDA requires generation_length to be divisible by block_length"
        )

    block_count = generation_length // block_length

    if steps % block_count:
        raise ValueError(
            "LLaDA requires steps to be divisible by the number of blocks"
        )

    if remasking not in {"low_confidence", "random"}:
        raise ValueError(
            f"Unsupported official LLaDA remasking strategy: {remasking}"
        )

    device = input_device(model)
    prompt_ids = prompt_ids.to(device)

    tokens = torch.full(
        (
            prompt_ids.shape[0],
            prompt_ids.shape[1] + generation_length,
        ),
        LLADA_MASK_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )

    tokens[:, : prompt_ids.shape[1]] = prompt_ids

    block_steps = steps // block_count

    for block in range(block_count):
        start = prompt_ids.shape[1] + block * block_length
        end = start + block_length

        schedule = _llada_transfer_schedule(
            tokens[:, start:end] == LLADA_MASK_TOKEN_ID,
            block_steps,
        )

        for step in range(block_steps):
            is_mask = tokens == LLADA_MASK_TOKEN_ID
            logits = model(tokens).logits

            if temperature > 0:
                # Official LLaDA Gumbel-max parameterization.
                noise = torch.rand_like(
                    logits,
                    dtype=torch.float64,
                )

                scores = (
                    logits.double().exp()
                    / (-torch.log(noise)).pow(temperature)
                )
            else:
                scores = logits

            predictions = scores.argmax(dim=-1)

            if remasking == "low_confidence":
                probabilities = F.softmax(
                    logits,
                    dim=-1,
                )

                confidence = probabilities.gather(
                    -1,
                    predictions.unsqueeze(-1),
                ).squeeze(-1)
            else:
                confidence = torch.rand(
                    predictions.shape,
                    device=device,
                )

            predictions = torch.where(
                is_mask,
                predictions,
                tokens,
            )

            confidence = torch.where(
                is_mask,
                confidence,
                -torch.inf,
            )

            confidence[:, :start] = -torch.inf
            confidence[:, end:] = -torch.inf

            transfer = torch.zeros_like(
                tokens,
                dtype=torch.bool,
            )

            for row in range(tokens.shape[0]):
                count = int(
                    schedule[row, step].item()
                )

                if count <= 0:
                    continue

                indices = torch.topk(
                    confidence[row],
                    k=count,
                ).indices

                transfer[row, indices] = True

            tokens[transfer] = predictions[transfer]

    return tokens


@torch.inference_mode()
def generate_llada(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate a factual continuation with LLaDA."""
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
    )["input_ids"]

    output_ids = llada_diffusion_generate(
        model,
        prompt_ids,
        generation_length=max_new_tokens,
        steps=max_new_tokens,
        block_length=max_new_tokens,
    )

    continuation = output_ids[
        0,
        prompt_ids.shape[1] :,
    ]

    return tokenizer.decode(
        continuation,
        skip_special_tokens=True,
    ).strip()


@torch.inference_mode()
def generate_dream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate a factual continuation with Dream."""
    if not hasattr(model, "diffusion_generate"):
        raise AttributeError(
            "Dream custom model has no diffusion_generate(); the downloaded "
            "remote code is incompatible with the official Dream interface."
        )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(input_device(model))
        for key, value in encoded.items()
    }

    # diffusion_generate creates Dream's masked continuation region internally.
    # One diffusion step per target token and deterministic minimum-entropy
    # decoding are used here.
    output = model.diffusion_generate(
        encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        steps=max_new_tokens,
        temperature=TEMPERATURE,
        alg=DREAM_ALG,
        alg_temp=DREAM_ALG_TEMP,
        output_history=False,
        return_dict_in_generate=True,
    )

    continuation = output.sequences[
        0,
        encoded["input_ids"].shape[1] :,
    ]

    text = tokenizer.decode(
        continuation,
        skip_special_tokens=True,
    )

    # Dream generates a fixed-size masked region; trim its decoded EOS suffix.
    if tokenizer.eos_token and tokenizer.eos_token in text:
        text = text.split(
            tokenizer.eos_token,
            1,
        )[0]

    return text.strip()


def generate_answer(
    model_name: str,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Unified entry point that dispatches to the model's native paradigm."""
    if model_name == "llama":
        return generate_llama(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
        )

    if model_name == "llada":
        return generate_llada(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
        )

    if model_name == "dream":
        return generate_dream(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
        )

    raise ValueError(
        f"Unknown model implementation: {model_name}"
    )


def compatibility_hint(
    implementation: str,
) -> str:
    if implementation == "llama":
        return (
            "Meta requires Transformers >= 4.43.0. The checkpoint is gated, so "
            "also accept its license and authenticate with `hf auth login` or HF_TOKEN."
        )

    if implementation == "llada":
        return (
            "The official LLaDA repository specifies transformers==4.38.2 and "
            "loads the checkpoint with trust_remote_code=True."
        )

    return (
        "The official Dream repository tests transformers==4.46.2 with "
        "torch==2.5.1 and requires trust_remote_code=True (plus an SDPA-capable GPU)."
    )


def clear_cuda_memory() -> None:
    """Collect Python objects and release unused CUDA allocator blocks."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def main() -> None:
    print(f"transformers version: {transformers.__version__}")
    print(f"torch version:        {torch.__version__}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Expose a GPU (optionally select it with "
            "CUDA_VISIBLE_DEVICES) and rerun this script."
        )

    print(f"CUDA device:          cuda:0 ({torch.cuda.get_device_name(0)})")
    print(f"CUDA BF16 supported:  {torch.cuda.is_bf16_supported()}")
    print(f"selected dtype:       {preferred_dtype()}")

    print_generation_config()

    for display_name, model_id, implementation in MODELS:
        model = None
        tokenizer = None

        try:
            print("\n" + "=" * 50)
            print(f"MODEL: {display_name}")
            print("=" * 50)

            model, tokenizer = load_model(
                model_id,
                implementation,
            )

            print_model_info(
                model_id,
                model,
                tokenizer,
            )

            if implementation == "llada":
                reported_mask_id = getattr(
                    tokenizer,
                    "mask_token_id",
                    None,
                )

                print(
                    f"LLaDA diffusion mask ID: "
                    f"{LLADA_MASK_TOKEN_ID}"
                )

                if (
                    reported_mask_id is not None
                    and reported_mask_id != LLADA_MASK_TOKEN_ID
                ):
                    raise RuntimeError(
                        f"Tokenizer reports mask_token_id={reported_mask_id}, but the "
                        f"official LLaDA sampler requires {LLADA_MASK_TOKEN_ID}."
                    )

            for prompt in PROMPTS:
                continuation = generate_answer(
                    implementation,
                    model,
                    tokenizer,
                    prompt,
                )

                print(f"Prompt: {prompt}")
                print(f"Continuation: {continuation}\n")

        except Exception as exc:
            print(
                f"ERROR while testing {model_id}: "
                f"{type(exc).__name__}: {exc}"
            )

            print("Exact traceback follows:")
            traceback.print_exc()

            print(
                f"Compatibility guidance: "
                f"{compatibility_hint(implementation)}"
            )

        finally:
            # Delete the references in this scope before clearing CUDA cache.
            del model
            del tokenizer
            clear_cuda_memory()


if __name__ == "__main__":
    main()
