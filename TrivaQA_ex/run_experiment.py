#!/usr/bin/env python3
"""Run frozen five-question facts through Dream, LLaDA, and/or Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from experiment_utils import (
    QA_INSTRUCTION,
    SCHEMA_VERSION,
    append_jsonl,
    iter_jsonl,
    load_jsonl,
    sha256_file,
    utc_now,
    validate_question_set,
    write_json,
    write_jsonl,
)
from model_backends import (
    DEFAULT_DREAM_MODEL,
    DEFAULT_LLADA_MODEL,
    DEFAULT_QWEN_MODEL,
    generate,
    load_backend,
    runtime_metadata,
    unload_backend,
)


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "data" / "triviaqa_questions_5.jsonl"
DEFAULT_OUTPUT_DIR = HERE / "results" / "triviaqa_smoke"
MODEL_ORDER = ("dream", "llada", "qwen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=["dream", "qwen"],
    )
    parser.add_argument("--dream-model", default=DEFAULT_DREAM_MODEL)
    parser.add_argument("--dream-revision", default=None)
    parser.add_argument("--llada-model", default=DEFAULT_LLADA_MODEL)
    parser.add_argument("--llada-revision", default=None)
    parser.add_argument(
        "--qwen-model",
        default=DEFAULT_QWEN_MODEL,
        help="Hugging Face checkpoint; defaults to the paired Qwen2.5-7B Base model.",
    )
    parser.add_argument("--qwen-use-chat-template", action="store_true")
    parser.add_argument("--qwen-revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=0,
        help=(
            "Denoising iterations for Dream/LLaDA. 0 means one step per canvas "
            "token (steps = --max-new-tokens), which is the original behaviour."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N frozen facts; 0 means all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Run model shards in parallel on the listed physical GPUs. Extra GPUs "
            "are assigned round-robin in Dream, Qwen, LLaDA priority order."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Allow unreviewed paraphrases for smoke tests only.",
    )
    return parser.parse_args()


def load_question_sets(
    path: Path,
    limit: int = 0,
    *,
    allow_unreviewed: bool = False,
) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("--limit must be 0 or positive")
    rows = load_jsonl(path)
    if limit:
        rows = rows[:limit]
    seen: set[str] = set()
    for row in rows:
        validate_question_set(row)
        if not allow_unreviewed:
            if row.get("review_status") != "approved":
                raise ValueError(
                    f"{row['fact_id']} is not manually approved; set review_status="
                    "'approved' after review or use --allow-unreviewed for a smoke test"
                )
            if row.get("validation_warnings"):
                raise ValueError(
                    f"{row['fact_id']} has unresolved validation warnings"
                )
        if row["fact_id"] in seen:
            raise ValueError(f"Duplicate fact_id in question sets: {row['fact_id']}")
        seen.add(row["fact_id"])
    if not rows:
        raise ValueError("No question sets were selected")
    return rows


def parallel_shard_counts(
    models: list[str],
    gpu_ids: list[int] | None,
    fact_count: int,
) -> dict[str, int]:
    """Balance extra GPUs across models while keeping one GPU per model."""
    counts = {model_name: 1 for model_name in models}
    if not gpu_ids:
        return counts
    extra_gpu_count = max(0, len(gpu_ids) - len(models))
    priority = [name for name in ("dream", "qwen", "llada") if name in counts]
    while extra_gpu_count > 0:
        assigned_this_round = False
        for model_name in priority:
            if extra_gpu_count <= 0:
                break
            if counts[model_name] >= fact_count:
                continue
            counts[model_name] += 1
            extra_gpu_count -= 1
            assigned_this_round = True
        if not assigned_this_round:
            break
    return counts


def build_settings(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_ids = {
        "dream": args.dream_model,
        "llada": args.llada_model,
        "qwen": args.qwen_model,
    }
    revisions = {
        "dream": args.dream_revision,
        "llada": args.llada_revision,
        "qwen": args.qwen_revision,
    }
    selected_fact_digest = hashlib.sha256(
        "\n".join(row["fact_id"] for row in rows).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "selected_fact_digest": selected_fact_digest,
        "selected_fact_count": len(rows),
        "limit": args.limit,
        "models": args.models,
        "model_ids": {name: model_ids[name] for name in args.models},
        "requested_revisions": {name: revisions[name] for name in args.models},
        "qwen_prompt_mode": "chat" if args.qwen_use_chat_template else "plain",
        "plain_prompt_template": (
            f"{QA_INSTRUCTION}\n\nQuestion: {{question}}\nAnswer:"
        ),
        "plain_prompt_template_sha256": hashlib.sha256(
            f"{QA_INSTRUCTION}\n\nQuestion: {{question}}\nAnswer:".encode("utf-8")
        ).hexdigest(),
        "max_new_tokens": args.max_new_tokens,
        "diffusion_steps": args.diffusion_steps or args.max_new_tokens,
        "seed": args.seed,
        "parallel_shard_counts": parallel_shard_counts(
            args.models, args.gpu_ids, len(rows)
        ),
        "allow_unreviewed": args.allow_unreviewed,
        "decoding": "deterministic_greedy_or_entropy_diffusion",
    }


def initialize_run(
    output_dir: Path,
    settings: dict[str, Any],
    *,
    resume: bool,
) -> None:
    config_path = output_dir / "run_config.json"
    cache_dir = output_dir / "cache"
    if config_path.exists():
        if not resume:
            raise FileExistsError(
                f"{config_path} already exists; pass --resume for the same run"
            )
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("settings") != settings:
            raise ValueError(
                "Resume settings differ from run_config.json. Use a new output directory."
            )
        return
    if resume and cache_dir.exists() and any(cache_dir.glob("*.jsonl")):
        raise FileNotFoundError(
            "Generation caches exist without run_config.json; refusing an unverified resume"
        )
    if not resume and cache_dir.exists() and any(cache_dir.glob("*.jsonl")):
        raise FileExistsError(
            f"Generation caches already exist in {cache_dir}; use a new directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        config_path,
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "settings": settings,
            "runtime": runtime_metadata(),
        },
    )


def load_completed_cache(
    cache_path: Path,
    *,
    model_name: str,
    model_id: str,
    prompt_mode: str,
    max_new_tokens: int,
    requested_revision: str | None,
    input_sha256: str,
    expected_tasks: dict[tuple[str, str], dict[str, Any]],
    seed: int,
    shard_index: int,
    shard_count: int,
) -> tuple[set[tuple[str, str]], str | None]:
    completed: set[tuple[str, str]] = set()
    resolved_revisions: set[str] = set()
    if not cache_path.exists():
        return completed, None
    for row in iter_jsonl(cache_path):
        if row.get("model_name") != model_name:
            raise ValueError(f"Wrong model record in {cache_path}: {row.get('model_name')}")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Schema version mismatch in cached record {cache_path}")
        if row.get("model_id") != model_id:
            raise ValueError(f"Model ID mismatch in cached record {cache_path}")
        if row.get("requested_revision") != requested_revision:
            raise ValueError(f"Requested revision mismatch in cached record {cache_path}")
        if row.get("resolved_revision"):
            resolved_revisions.add(str(row["resolved_revision"]))
        if row.get("prompt_mode") != prompt_mode:
            raise ValueError(f"Prompt mode mismatch in cached record {cache_path}")
        if row.get("max_new_tokens") != max_new_tokens:
            raise ValueError(f"Generation length mismatch in cached record {cache_path}")
        if row.get("question_set_sha256") != input_sha256:
            raise ValueError(f"Input hash mismatch in cached record {cache_path}")
        key = (str(row.get("fact_id")), str(row.get("variant_id")))
        if key not in expected_tasks:
            raise ValueError(f"Unknown task key in cached record {cache_path}: {key}")
        task = expected_tasks[key]
        expected_hash = hashlib.sha256(task["question"].encode("utf-8")).hexdigest()
        if row.get("question") != task["question"] or row.get("question_sha256") != expected_hash:
            raise ValueError(f"Question mismatch in cached record {cache_path}: {key}")
        expected_seed = task_seed(seed, model_name, key[0], key[1])
        if row.get("generation_seed") != expected_seed:
            raise ValueError(f"Generation seed mismatch in cached record {cache_path}: {key}")
        if row.get("shard_index") != shard_index or row.get("shard_count") != shard_count:
            raise ValueError(f"Shard assignment mismatch in cached record {cache_path}: {key}")
        if "scored_output" not in row or "continuation_token_ids" not in row:
            raise ValueError(
                f"Cached record predates EOS-safe scoring fields in {cache_path}: {key}"
            )
        if key in completed:
            raise ValueError(f"Duplicate generation cache key in {cache_path}: {key}")
        completed.add(key)
    if len(resolved_revisions) > 1:
        raise ValueError(
            f"Multiple resolved checkpoint revisions in {cache_path}: "
            f"{sorted(resolved_revisions)}"
        )
    return completed, next(iter(resolved_revisions), None)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_seed(
    base_seed: int,
    model_name: str,
    fact_id: str,
    variant_id: str,
) -> int:
    """Return a stable per-generation seed, independent of sharding and resume order."""
    payload = f"{base_seed}\0{model_name}\0{fact_id}\0{variant_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def task_rows(question_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for fact in question_sets:
        for variant in fact["variants"]:
            tasks.append(
                {
                    "fact_id": fact["fact_id"],
                    "question_id": fact["question_id"],
                    "variant_id": variant["variant_id"],
                    "variant_index": variant["variant_index"],
                    "question": variant["question"],
                }
            )
    return tasks


def shard_tasks(
    tasks: list[dict[str, Any]],
    shard_count: int,
) -> list[list[dict[str, Any]]]:
    """Round-robin whole facts so all five variants stay on the same worker."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        grouped.setdefault(task["fact_id"], []).append(task)
    if shard_count > max(1, len(grouped)):
        raise ValueError("Cannot create more non-empty shards than facts")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for fact_index, fact_tasks in enumerate(grouped.values()):
        shards[fact_index % shard_count].extend(fact_tasks)
    return shards


def shard_cache_path(
    cache_dir: Path,
    model_name: str,
    shard_index: int,
    shard_count: int,
) -> Path:
    return cache_dir / (
        f"{model_name}.shard_{shard_index:03d}_of_{shard_count:03d}.jsonl"
    )


def merge_model_shards(
    *,
    model_name: str,
    tasks: list[dict[str, Any]],
    shard_paths: list[Path],
    output_path: Path,
) -> None:
    """Validate complete, disjoint shard output and atomically write canonical cache."""
    task_order = {
        (task["fact_id"], task["variant_id"]): index
        for index, task in enumerate(tasks)
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    resolved_revisions: set[str] = set()
    for shard_path in shard_paths:
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing completed shard cache: {shard_path}")
        for row in iter_jsonl(shard_path):
            if row.get("model_name") != model_name:
                raise ValueError(
                    f"Wrong model in shard {shard_path}: {row.get('model_name')}"
                )
            key = (str(row.get("fact_id")), str(row.get("variant_id")))
            if key not in task_order:
                raise ValueError(f"Unknown task in shard {shard_path}: {key}")
            if key in seen:
                raise ValueError(f"Duplicate task across {model_name} shards: {key}")
            seen.add(key)
            if row.get("resolved_revision"):
                resolved_revisions.add(str(row["resolved_revision"]))
            rows.append(row)
    missing = set(task_order).difference(seen)
    if missing:
        preview = ", ".join(f"{fact_id}:{variant_id}" for fact_id, variant_id in sorted(missing)[:10])
        raise ValueError(
            f"Missing {len(missing)} tasks while merging {model_name}; first: {preview}"
        )
    if len(resolved_revisions) > 1:
        raise ValueError(
            f"{model_name} shards resolved to different revisions: "
            f"{sorted(resolved_revisions)}"
        )
    rows.sort(key=lambda row: task_order[(row["fact_id"], row["variant_id"])])
    write_jsonl(output_path, rows)


def run_model(
    *,
    model_name: str,
    model_id: str,
    tasks: list[dict[str, Any]],
    cache_path: Path,
    input_sha256: str,
    max_new_tokens: int,
    diffusion_steps: int,
    qwen_use_chat_template: bool,
    revision: str | None,
    seed: int,
    shard_index: int = 0,
    shard_count: int = 1,
    worker_gpu_id: int | None = None,
    load_lock: Any = None,
) -> None:
    completed, cached_resolved_revision = load_completed_cache(
        cache_path,
        model_name=model_name,
        model_id=model_id,
        prompt_mode=("chat" if model_name == "qwen" and qwen_use_chat_template else "plain"),
        max_new_tokens=max_new_tokens,
        requested_revision=revision,
        input_sha256=input_sha256,
        expected_tasks={
            (task["fact_id"], task["variant_id"]): task for task in tasks
        },
        seed=seed,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    pending = [
        task
        for task in tasks
        if (task["fact_id"], task["variant_id"]) not in completed
    ]
    print(
        f"{model_name}: {len(tasks):,} total, {len(completed):,} cached, "
        f"{len(pending):,} pending"
    )
    if not pending:
        return

    backend = None
    try:
        load_context = load_lock if load_lock is not None else nullcontext()
        with load_context:
            backend = load_backend(
                model_name,
                model_id,
                qwen_use_chat_template=qwen_use_chat_template,
                revision=revision,
            )
        if (
            cached_resolved_revision is not None
            and backend.resolved_revision != cached_resolved_revision
        ):
            raise RuntimeError(
                f"Resolved {model_name} revision changed from "
                f"{cached_resolved_revision} to {backend.resolved_revision}; "
                "use a new output directory or pin the old commit"
            )
        for index, task in enumerate(pending, start=1):
            generation_seed = task_seed(
                seed,
                model_name,
                task["fact_id"],
                task["variant_id"],
            )
            set_deterministic_seed(generation_seed)
            result = generate(
                backend,
                task["question"],
                max_new_tokens=max_new_tokens,
                diffusion_steps=diffusion_steps or None,
            )
            question_sha256 = hashlib.sha256(
                task["question"].encode("utf-8")
            ).hexdigest()
            append_jsonl(
                cache_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "model_name": model_name,
                    "model_id": model_id,
                    "requested_revision": revision,
                    "resolved_revision": backend.resolved_revision,
                    "prompt_mode": backend.prompt_mode,
                    "fact_id": task["fact_id"],
                    "question_id": task["question_id"],
                    "variant_id": task["variant_id"],
                    "variant_index": task["variant_index"],
                    "question": task["question"],
                    "question_sha256": question_sha256,
                    "prompt": result.prompt,
                    "raw_output": result.raw_output,
                    "scored_output": result.scored_output,
                    "continuation_token_ids": list(result.continuation_token_ids),
                    "first_eos_index": result.first_eos_index,
                    "eos_trim_applied": result.first_eos_index is not None,
                    "max_new_tokens": max_new_tokens,
                    "diffusion_steps": diffusion_steps or max_new_tokens,
                    "question_set_sha256": input_sha256,
                    "created_at_utc": utc_now(),
                    "generation_seed": generation_seed,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "worker_gpu_id": worker_gpu_id,
                },
            )
            print(
                f"{model_name} [{index}/{len(pending)}] "
                f"{task['fact_id']}:{task['variant_id']}"
            )
    finally:
        if backend is not None:
            unload_backend(backend)


def run_model_on_gpu(gpu_id: int, kwargs: dict[str, Any]) -> None:
    """Spawn target: isolate one model on one physical GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    run_model(**kwargs, worker_gpu_id=gpu_id)


def main() -> None:
    args = parse_args()
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models cannot contain duplicates")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.diffusion_steps < 0:
        raise ValueError("--diffusion-steps must be 0 or positive")
    if args.diffusion_steps > args.max_new_tokens:
        raise ValueError("--diffusion-steps cannot exceed --max-new-tokens")
    if "qwen" in args.models and not args.qwen_model:
        raise ValueError("--qwen-model is required when qwen is selected")
    if args.qwen_use_chat_template and any(
        model_name != "qwen" for model_name in args.models
    ):
        raise ValueError(
            "Do not mix Qwen chat formatting with plain Dream/LLaDA prompts in one "
            "comparison run; run the chat condition in a separate output directory"
        )
    if args.gpu_ids:
        if len(set(args.gpu_ids)) != len(args.gpu_ids):
            raise ValueError("--gpu-ids cannot contain duplicates")
        if len(args.gpu_ids) < len(args.models):
            raise ValueError("Provide at least one distinct --gpu-ids value per model")

    question_sets = load_question_sets(
        args.input,
        args.limit,
        allow_unreviewed=args.allow_unreviewed,
    )
    settings = build_settings(args, question_sets)
    initialize_run(args.output_dir, settings, resume=args.resume)
    input_sha256 = settings["input_sha256"]
    tasks = task_rows(question_sets)
    model_ids = {
        "dream": args.dream_model,
        "llada": args.llada_model,
        "qwen": args.qwen_model,
    }
    revisions = {
        "dream": args.dream_revision,
        "llada": args.llada_revision,
        "qwen": args.qwen_revision,
    }

    cache_dir = args.output_dir / "cache"
    if args.gpu_ids:
        jobs: list[dict[str, Any]] = []
        model_shard_paths: dict[str, list[Path]] = {}
        for model_name in args.models:
            model_id = model_ids[model_name]
            assert model_id is not None
            shard_count = settings["parallel_shard_counts"][model_name]
            model_shard_paths[model_name] = []
            for shard_index, task_shard in enumerate(shard_tasks(tasks, shard_count)):
                cache_path = shard_cache_path(
                    cache_dir, model_name, shard_index, shard_count
                )
                model_shard_paths[model_name].append(cache_path)
                jobs.append(
                    {
                        "model_name": model_name,
                        "model_id": model_id,
                        "tasks": task_shard,
                        "cache_path": cache_path,
                        "input_sha256": input_sha256,
                        "max_new_tokens": args.max_new_tokens,
                        "diffusion_steps": args.diffusion_steps or args.max_new_tokens,
                        "qwen_use_chat_template": (
                            args.qwen_use_chat_template
                            if model_name == "qwen"
                            else False
                        ),
                        "revision": revisions[model_name],
                        "seed": args.seed,
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                    }
                )
        if len(jobs) > len(args.gpu_ids):
            raise RuntimeError("Internal GPU shard allocation exceeds available GPUs")
        context = multiprocessing.get_context("spawn")
        checkpoint_load_lock = context.Lock()
        for job in jobs:
            job["load_lock"] = checkpoint_load_lock
        workers = [
            context.Process(
                target=run_model_on_gpu,
                args=(gpu_id, job),
                name=(
                    f"triviaqa-{job['model_name']}-shard{job['shard_index']}"
                    f"-gpu{gpu_id}"
                ),
            )
            for gpu_id, job in zip(args.gpu_ids, jobs)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        failed = [
            f"{worker.name} (exit {worker.exitcode})"
            for worker in workers
            if worker.exitcode != 0
        ]
        if failed:
            raise RuntimeError(f"Parallel model worker failure: {', '.join(failed)}")
        for model_name in args.models:
            merge_model_shards(
                model_name=model_name,
                tasks=tasks,
                shard_paths=model_shard_paths[model_name],
                output_path=cache_dir / f"{model_name}_generations.jsonl",
            )
    else:
        for model_name in args.models:
            model_id = model_ids[model_name]
            assert model_id is not None
            run_model(
                model_name=model_name,
                model_id=model_id,
                tasks=tasks,
                cache_path=cache_dir / f"{model_name}_generations.jsonl",
                input_sha256=input_sha256,
                max_new_tokens=args.max_new_tokens,
                diffusion_steps=args.diffusion_steps,
                qwen_use_chat_template=(
                    args.qwen_use_chat_template if model_name == "qwen" else False
                ),
                revision=revisions[model_name],
                seed=args.seed,
            )

    if not args.skip_scoring:
        from score_results import analyze

        analyze(
            questions_path=args.input,
            cache_dir=args.output_dir / "cache",
            output_dir=args.output_dir / "analysis",
            expected_models=args.models,
            limit=args.limit,
            require_complete=True,
        )

    print(f"Experiment complete -> {args.output_dir}")


if __name__ == "__main__":
    main()
