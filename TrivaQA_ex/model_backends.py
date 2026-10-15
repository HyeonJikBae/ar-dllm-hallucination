"""Native single-GPU generation backends for Dream, LLaDA, and Qwen."""

from __future__ import annotations

import gc
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment_utils import QA_INSTRUCTION, render_plain_prompt


PARENT_REPO = Path(__file__).resolve().parent.parent
DEFAULT_LLADA_MODEL = "GSAI-ML/LLaDA-8B-Base"
DEFAULT_DREAM_MODEL = "Dream-org/Dream-v0-Base-7B"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-7B"


@dataclass
class LoadedBackend:
    name: str
    model_id: str
    model: Any
    tokenizer: Any
    prompt_mode: str
    requested_revision: str | None = None
    resolved_revision: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    raw_output: str
    scored_output: str
    continuation_token_ids: tuple[int, ...]
    first_eos_index: int | None


def _require_cuda() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for model generation") from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Select a GPU with CUDA_VISIBLE_DEVICES and retry."
        )
    return torch


def _import_parent_generation() -> Any:
    parent = str(PARENT_REPO)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import model_generation  # pylint: disable=import-outside-toplevel

    return model_generation


def _token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return {int(value)}


def _decode_continuation(
    backend: LoadedBackend,
    continuation: Any,
) -> GenerationResult:
    token_ids = tuple(int(token_id) for token_id in continuation.tolist())
    eos_ids = _token_id_set(getattr(backend.tokenizer, "eos_token_id", None))
    generation_config = getattr(backend.model, "generation_config", None)
    eos_ids.update(_token_id_set(getattr(generation_config, "eos_token_id", None)))
    first_eos_index = next(
        (index for index, token_id in enumerate(token_ids) if token_id in eos_ids),
        None,
    )
    scored_ids = (
        token_ids[:first_eos_index] if first_eos_index is not None else token_ids
    )
    raw_output = backend.tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    ).strip()
    scored_output = backend.tokenizer.decode(
        scored_ids,
        skip_special_tokens=True,
    ).strip()
    return GenerationResult(
        prompt="",
        raw_output=raw_output,
        scored_output=scored_output,
        continuation_token_ids=token_ids,
        first_eos_index=first_eos_index,
    )


def load_backend(
    name: str,
    model_id: str,
    *,
    qwen_use_chat_template: bool = False,
    revision: str | None = None,
) -> LoadedBackend:
    torch = _require_cuda()
    if name in {"llada", "dream"}:
        parent_generation = _import_parent_generation()
        model, tokenizer = parent_generation.load_model(
            model_id,
            name,
            revision=revision,
        )
        reported_mask_id = getattr(tokenizer, "mask_token_id", None)
        if (
            name == "llada"
            and reported_mask_id is not None
            and reported_mask_id != parent_generation.LLADA_MASK_TOKEN_ID
        ):
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"LLaDA mask token mismatch: expected "
                f"{parent_generation.LLADA_MASK_TOKEN_ID}, "
                f"tokenizer reports {reported_mask_id}"
            )
        if name == "dream" and not hasattr(model, "diffusion_generate"):
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(
                "Dream checkpoint does not expose the required diffusion_generate API"
            )
        return LoadedBackend(
            name=name,
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            prompt_mode="plain",
            requested_revision=revision,
            resolved_revision=(
                getattr(getattr(model, "config", None), "_commit_hash", None)
                or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            ),
        )

    if name == "qwen":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Transformers is required for Qwen generation") from error
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=False,
            revision=revision,
        )
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        parent_generation = _import_parent_generation()
        with parent_generation.direct_safetensors_to_cuda():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=False,
                revision=revision,
                low_cpu_mem_usage=True,
                device_map={"": 0},
            ).eval()
        if qwen_use_chat_template and not getattr(tokenizer, "chat_template", None):
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"--qwen-use-chat-template was set, but {model_id} has no chat template"
            )
        return LoadedBackend(
            name=name,
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            prompt_mode="chat" if qwen_use_chat_template else "plain",
            requested_revision=revision,
            resolved_revision=(
                getattr(getattr(model, "config", None), "_commit_hash", None)
                or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            ),
        )

    raise ValueError(f"Unsupported backend: {name!r}")


def format_prompt(backend: LoadedBackend, question: str) -> tuple[str, bool]:
    if backend.name == "qwen" and backend.prompt_mode == "chat":
        prompt = backend.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"{QA_INSTRUCTION}\n\nQuestion: {question.strip()}",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, False
    return render_plain_prompt(question), True


def generate(
    backend: LoadedBackend,
    question: str,
    *,
    max_new_tokens: int,
    diffusion_steps: int | None = None,
) -> GenerationResult:
    """Generate one answer.

    `diffusion_steps` decouples the denoising-iteration count from the mask
    canvas length for the diffusion backends. It defaults to max_new_tokens,
    which is one token committed per step.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    steps = max_new_tokens if diffusion_steps is None else diffusion_steps
    if steps < 1:
        raise ValueError("diffusion_steps must be positive")
    if steps > max_new_tokens:
        raise ValueError("diffusion_steps cannot exceed max_new_tokens")
    prompt, add_special_tokens = format_prompt(backend, question)

    import torch

    if backend.name == "llada":
        parent_generation = _import_parent_generation()
        prompt_ids = backend.tokenizer(
            prompt,
            return_tensors="pt",
        )["input_ids"]
        with torch.inference_mode():
            output_ids = parent_generation.llada_diffusion_generate(
                backend.model,
                prompt_ids,
                generation_length=max_new_tokens,
                steps=steps,
                block_length=max_new_tokens,
            )
        continuation = output_ids[0, prompt_ids.shape[1] :]
        result = _decode_continuation(backend, continuation)
        return GenerationResult(
            prompt=prompt,
            raw_output=result.raw_output,
            scored_output=result.scored_output,
            continuation_token_ids=result.continuation_token_ids,
            first_eos_index=result.first_eos_index,
        )

    if backend.name == "dream":
        parent_generation = _import_parent_generation()
        encoded = backend.tokenizer(prompt, return_tensors="pt")
        encoded = {
            key: value.to(next(backend.model.parameters()).device)
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            output = backend.model.diffusion_generate(
                encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                steps=steps,
                temperature=parent_generation.TEMPERATURE,
                alg=parent_generation.DREAM_ALG,
                alg_temp=parent_generation.DREAM_ALG_TEMP,
                output_history=False,
                return_dict_in_generate=True,
            )
        continuation = output.sequences[0, encoded["input_ids"].shape[1] :]
        result = _decode_continuation(backend, continuation)
        return GenerationResult(
            prompt=prompt,
            raw_output=result.raw_output,
            scored_output=result.scored_output,
            continuation_token_ids=result.continuation_token_ids,
            first_eos_index=result.first_eos_index,
        )

    if backend.name == "qwen":
        encoded = backend.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        encoded = {
            key: value.to(next(backend.model.parameters()).device)
            for key, value in encoded.items()
        }
        pad_token_id = backend.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = backend.tokenizer.eos_token_id
        generation_config = copy.deepcopy(backend.model.generation_config)
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        generation_config.pad_token_id = pad_token_id
        with torch.inference_mode():
            output_ids = backend.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                generation_config=generation_config,
            )
        continuation = output_ids[0, encoded["input_ids"].shape[1] :]
        result = _decode_continuation(backend, continuation)
        return GenerationResult(
            prompt=prompt,
            raw_output=result.raw_output,
            scored_output=result.scored_output,
            continuation_token_ids=result.continuation_token_ids,
            first_eos_index=result.first_eos_index,
        )

    raise ValueError(f"Unsupported loaded backend: {backend.name!r}")


def unload_backend(backend: LoadedBackend) -> None:
    backend.model = None
    backend.tokenizer = None
    gc.collect()
    try:
        _import_parent_generation().clear_cuda_memory()
    except (ImportError, RuntimeError):
        # Model objects are already released; this fallback is enough when the
        # parent helper cannot call CUDA IPC collection in the current runtime.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def runtime_metadata() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_bf16_supported": (
            torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None
        ),
    }
