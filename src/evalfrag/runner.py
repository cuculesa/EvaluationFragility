from __future__ import annotations

import importlib.metadata
import json
import uuid
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

from .aggregate import aggregate_records
from .config import Config
from .datasets import validate_data
from .dashboard import build_dashboard
from .schema import validate_results
from .tasks import TASK_VERSION, build_task
from .util import (
    atomic_write_json,
    atomic_write_text,
    build_file_manifest,
    canonical_json,
    git_revision,
    redact_secrets,
    runtime_fingerprint,
    sha256_bytes,
    sha256_file,
    utc_now,
)


def _run_id(name: str) -> str:
    timestamp = utc_now().replace("-", "").replace(":", "").replace(".", "")
    timestamp = timestamp.replace("Z", "")
    return f"{name}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _score_value(score: Any) -> str:
    value = getattr(score, "value", score)
    return str(value)


def _records_from_logs(logs: list[Any], *, store_completion_text: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for log in logs:
        condition = (log.eval.metadata or {}).get("evalfrag_condition")
        if not condition:
            raise ValueError(
                f"Inspect log {log.eval.eval_id} is missing evalfrag condition metadata"
            )
        if log.status != "success":
            continue
        for sample in log.samples or []:
            completion = sample.output.completion if sample.output else ""
            score_payload: dict[str, Any] = {}
            for parser in ("parse_strict", "parse_flexible", "parse_last_number"):
                if parser not in sample.scores:
                    raise ValueError(f"sample {sample.id} is missing scorer {parser}")
                score = sample.scores[parser]
                metadata = score.metadata or {}
                score_payload[parser] = {
                    "correct": _score_value(score) == "C",
                    "answer": score.answer,
                    "unparsed": bool(metadata.get("unparsed")),
                }
            record: dict[str, Any] = {
                "schema_version": 1,
                "suite": condition["suite"],
                "fmt": condition["fmt"],
                "temp": float(condition["temperature"]),
                "seed": int(condition["generation_seed"]),
                "sample_id": str(sample.id),
                "epoch": int(getattr(sample, "epoch", 1) or 1),
                "question_sha256": (sample.metadata or {}).get("question_sha256"),
                "completion_sha256": sha256_bytes(completion.encode("utf-8")),
                "scores": score_payload,
            }
            if store_completion_text:
                record["completion"] = completion
            records.append(record)
    return records


def _validate_record_grid(records: list[dict[str, Any]], config: Config) -> None:
    expected_conditions = set(
        product(
            config.experiment.suites,
            config.experiment.prompt_formats,
            config.experiment.temperatures,
            config.experiment.seeds,
        )
    )
    grouped: dict[tuple[str, str, float, int], list[dict[str, Any]]] = defaultdict(list)
    seen_records: set[tuple[str, str, float, int, str, int]] = set()
    for record in records:
        condition = (
            str(record["suite"]),
            str(record["fmt"]),
            float(record["temp"]),
            int(record["seed"]),
        )
        grouped[condition].append(record)
        record_key = condition + (str(record["sample_id"]), int(record["epoch"]))
        if record_key in seen_records:
            raise ValueError(f"duplicate sample record: {record_key}")
        seen_records.add(record_key)
        if not record.get("question_sha256"):
            raise ValueError(f"sample {record['sample_id']} is missing question provenance")

    actual_conditions = set(grouped)
    missing = sorted(expected_conditions - actual_conditions)
    unexpected = sorted(actual_conditions - expected_conditions)
    if missing or unexpected:
        raise ValueError(
            f"incomplete result grid: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    ids_by_suite: dict[str, set[str]] = {}
    for condition, group in grouped.items():
        sample_ids = {str(record["sample_id"]) for record in group}
        if len(group) != config.experiment.n or len(sample_ids) != config.experiment.n:
            raise ValueError(
                f"condition {condition} has {len(group)} records / {len(sample_ids)} unique "
                f"items; expected {config.experiment.n}"
            )
        suite = condition[0]
        reference = ids_by_suite.setdefault(suite, sample_ids)
        if sample_ids != reference:
            raise ValueError(
                f"condition {condition} does not use the same item IDs as other {suite} runs"
            )


def run_experiment(config: Config, *, project_root: Path) -> Path:
    try:
        from inspect_ai import eval_set
    except ImportError as exc:
        raise RuntimeError(
            "Inspect AI is not installed. Install the project with 'pip install -e .'"
        ) from exc

    dataset_manifest = validate_data(config.experiment.data_dir)
    run_id = _run_id(config.experiment.name)
    run_dir = config.experiment.output_dir / run_id
    log_dir = run_dir / "inspect-logs"
    bundle_dir = run_dir / "inspect-viewer"
    run_dir.mkdir(parents=True, mode=0o750)
    log_dir.mkdir(mode=0o750)

    resolved_config = config.model_dump(mode="json")
    artifact_config = redact_secrets(resolved_config)
    config_hash = sha256_bytes(canonical_json(artifact_config).encode("utf-8"))
    provenance = {
        "run_id": run_id,
        "created_at": utc_now(),
        "config_sha256": config_hash,
        "git": git_revision(project_root),
        "runtime": runtime_fingerprint(),
        "dataset_manifest_sha256": sha256_file(
            config.experiment.data_dir / "manifest.json"
        ),
        "dataset_manifest": dataset_manifest,
    }
    atomic_write_json(run_dir / "config.resolved.json", artifact_config)
    atomic_write_json(run_dir / "provenance.json", provenance)
    atomic_write_json(
        run_dir / "run_state.json",
        {"status": "running", "run_id": run_id, "started_at": provenance["created_at"]},
    )

    synthetic = config.experiment.mode == "synthetic"

    try:
        tasks = [
            build_task(
                suite=suite,
                fmt=fmt,
                temperature=temperature,
                generation_seed=seed,
                n=config.experiment.n,
                dataset_seed=config.experiment.dataset_seed,
                data_dir=config.experiment.data_dir,
                max_tokens=config.experiment.max_tokens,
                top_p=config.experiment.top_p,
                synthetic=synthetic,
                model=config.experiment.model,
            )
            for suite in config.experiment.suites
            for fmt in config.experiment.prompt_formats
            for temperature in config.experiment.temperatures
            for seed in config.experiment.seeds
        ]
        success, logs = eval_set(
            tasks=tasks,
            model=None if synthetic else config.experiment.model,
            model_base_url=None if synthetic else config.experiment.model_base_url,
            model_args={} if synthetic else config.experiment.model_args,
            log_dir=str(log_dir),
            bundle_dir=str(bundle_dir),
            embed_viewer=True,
            retry_attempts=config.runtime.eval_set_retry_attempts,
            checkpoint=config.runtime.checkpoint,
            display="full",
            log_format="eval",
            log_samples=True,
            log_realtime=True,
            log_model_api=False,
            log_dir_allow_dirty=False,
            max_tasks=config.runtime.max_tasks,
            max_samples=config.runtime.max_samples,
            fail_on_error=config.runtime.fail_on_error,
            continue_on_fail=True,
            retry_on_error=config.runtime.sample_retry_on_error,
            max_retries=config.runtime.model_max_retries,
            timeout=config.runtime.timeout_seconds,
            attempt_timeout=config.runtime.attempt_timeout_seconds,
            max_connections=config.runtime.max_connections,
        )
        failed = [
            {
                "status": log.status,
                "task": log.eval.task,
                "eval_id": log.eval.eval_id,
                "error": getattr(log.error, "message", None) if log.error else None,
            }
            for log in logs
            if log.status != "success"
        ]
        atomic_write_json(run_dir / "failures.json", failed)
        if not success or failed:
            detail = (
                f"{len(failed)} evaluation tasks failed"
                if failed
                else "evaluation set did not complete successfully"
            )
            raise RuntimeError(f"{detail}; inspect {run_dir / 'failures.json'}")

        # eval_set()'s returned log objects are not guaranteed to have full
        # sample data loaded in memory even when log.status == "success" --
        # observed directly: a log with status "success" and real, correctly
        # scored samples on disk came back with log.samples empty from
        # eval_set() itself, which silently produced zero records for that
        # condition downstream despite the run genuinely succeeding. Re-read
        # every log fresh from disk to force full sample loading rather than
        # trusting whatever eval_set() handed back in memory.
        from inspect_ai.log import read_eval_log

        logs = [read_eval_log(str(log.location)) for log in logs]

        records = _records_from_logs(
            logs, store_completion_text=config.experiment.store_completion_text
        )
        _validate_record_grid(records, config)
        atomic_write_text(
            run_dir / "records.jsonl",
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        )
        inspect_version = importlib.metadata.version("inspect-ai")
        meta = {
            "run_id": run_id,
            "created_at": provenance["created_at"],
            "model": config.experiment.model,
            "model_base_url_configured": bool(config.experiment.model_base_url),
            "model_revision": config.experiment.model_revision,
            "tokenizer_revision": config.experiment.tokenizer_revision,
            "serving_engine": config.experiment.serving_engine,
            "serving_engine_version": config.experiment.serving_engine_version,
            "quantization": config.experiment.quantization,
            "chat_template_sha256": config.experiment.chat_template_sha256,
            "synthetic": synthetic,
            "n_per_condition": config.experiment.n,
            "seeds": config.experiment.seeds,
            "temperatures": config.experiment.temperatures,
            "prompt_formats": config.experiment.prompt_formats,
            "suites": config.experiment.suites,
            "dataset_seed": config.experiment.dataset_seed,
            "config_sha256": config_hash,
            "inspect_ai_version": inspect_version,
            "evalfrag_version": importlib.metadata.version("evalfrag"),
            "task_version": TASK_VERSION,
            "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
            "records_file": "records.jsonl",
            "completion_text_stored": config.experiment.store_completion_text,
        }
        results = aggregate_records(
            records=records,
            meta=meta,
            confidence=config.statistics.confidence_level,
            bootstrap_resamples=config.statistics.bootstrap_resamples,
            bootstrap_seed=config.statistics.bootstrap_seed,
        )
        payload = results.model_dump(mode="json")
        validate_results(payload)
        atomic_write_json(run_dir / "results.json", payload)
        build_dashboard(
            results_path=run_dir / "results.json",
            output_path=run_dir / "dashboard.html",
            title=config.dashboard.title,
            high_unparsed_threshold=config.dashboard.high_unparsed_threshold,
        )
        atomic_write_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "run_id": run_id,
                "started_at": provenance["created_at"],
                "completed_at": utc_now(),
            },
        )
        artifact_manifest = build_file_manifest(
            run_dir, exclude={"artifacts.manifest.json"}
        )
        atomic_write_json(run_dir / "artifacts.manifest.json", artifact_manifest)
        return run_dir
    except Exception as exc:
        if not (run_dir / "failures.json").exists():
            atomic_write_json(
                run_dir / "failures.json",
                [{"status": "error", "error_type": type(exc).__name__}],
            )
        atomic_write_json(
            run_dir / "run_state.json",
            {
                "status": "failed",
                "run_id": run_id,
                "started_at": provenance["created_at"],
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "message": "Evaluation run failed; inspect failures.json and Inspect logs.",
            },
        )
        raise
