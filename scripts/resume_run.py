#!/usr/bin/env python3
"""
Resume an existing evalfrag run instead of starting a new one, and finish
the pipeline (aggregation + results.json + dashboard.html) the same way
`evalfrag run` does.

Why this script exists: `evalfrag run` mints a brand-new run_id (and
therefore a brand-new log_dir) on every invocation, so re-running it after
a partial failure re-does the ENTIRE grid at full cost, even with
runtime.checkpoint = true in the config -- that flag only lets Inspect
resume samples *within* one log_dir, and evalfrag's CLI never points two
invocations at the same one.

This script re-builds the exact same tasks the failed run used, calls
eval_set() directly against the EXISTING run's inspect-logs directory (so
Inspect only re-attempts what didn't already succeed there), and then runs
the same aggregation/validation/dashboard steps evalfrag's own `run()`
does -- which a bare eval_set() call does NOT do on its own.

Usage:
    python3 scripts/resume_run.py --config configs/live-claude-smoke.toml \
        --run-dir runs/evalfrag-live-claude-20260818T224310738315-158a91b2
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from evalfrag.aggregate import aggregate_records
from evalfrag.config import load_config
from evalfrag.dashboard import build_dashboard
from evalfrag.runner import _records_from_logs, _validate_record_grid
from evalfrag.schema import validate_results
from evalfrag.tasks import build_task
from evalfrag.util import atomic_write_json, atomic_write_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="same config the failed run used")
    ap.add_argument("--run-dir", required=True, help="the existing run directory to resume")
    args = ap.parse_args()

    config = load_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    log_dir = run_dir / "inspect-logs"
    if not log_dir.exists():
        raise SystemExit(f"no inspect-logs directory found at {log_dir}")

    from inspect_ai import eval_set

    synthetic = config.experiment.mode == "synthetic"
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

    print(f"resuming into existing log_dir: {log_dir}")
    print(f"({len(tasks)} tasks total -- only incomplete/missing work should actually call the API)\n")

    success, logs = eval_set(
        tasks=tasks,
        model=None if synthetic else config.experiment.model,
        model_base_url=None if synthetic else config.experiment.model_base_url,
        model_args={} if synthetic else config.experiment.model_args,
        log_dir=str(log_dir),
        embed_viewer=True,
        retry_attempts=config.runtime.eval_set_retry_attempts,
        checkpoint=config.runtime.checkpoint,
        display="full",
        log_format="eval",
        log_samples=True,
        log_realtime=True,
        log_model_api=False,
        log_dir_allow_dirty=True,  # resuming an existing (non-empty) dir on purpose
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

    print(f"\nsuccess={success}  total_logs={len(logs)}  still_failed={len(failed)}")
    for f in failed:
        print(f"  FAILED: {f['task']}  ({f['error']})")

    if not success or failed:
        print(f"\nStill incomplete -- see {run_dir / 'failures.json'}. Not building results.json yet.")
        return

    # --- everything below mirrors what evalfrag run's runner.py does after eval_set ---
    print("\nAll tasks succeeded -- aggregating into results.json ...")

    # IMPORTANT: eval_set()'s returned log objects are not guaranteed to have
    # full sample data loaded in memory even when log.status == "success" --
    # this was confirmed directly: a log with status "success" and 10 real,
    # correctly-scored samples on disk came back with log.samples empty from
    # eval_set() itself, causing _records_from_logs to silently produce zero
    # records for that (and 4 other) conditions despite them having genuinely
    # succeeded. Re-read every log fresh from disk to force full sample
    # loading rather than trusting whatever eval_set() handed back in memory.
    from inspect_ai.log import read_eval_log

    reloaded_logs = [read_eval_log(str(log.location)) for log in logs]
    missing_samples = [
        log.eval.task for log in reloaded_logs if not (log.samples or [])
    ]
    if missing_samples:
        raise SystemExit(
            "Even after re-reading from disk, these tasks have zero samples "
            f"-- this is a real problem, not a stale-object artifact: {missing_samples}"
        )

    records = _records_from_logs(
        reloaded_logs, store_completion_text=config.experiment.store_completion_text
    )
    _validate_record_grid(records, config)
    atomic_write_text(
        run_dir / "records.jsonl",
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )

    # reuse provenance already written by the original run rather than re-deriving it
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    config_hash = provenance.get("config_sha256", "")

    inspect_version = importlib.metadata.version("inspect-ai")
    meta = {
        "run_id": run_dir.name,
        "created_at": provenance.get("created_at", ""),
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
        "task_version": tasks[0].version if hasattr(tasks[0], "version") else "",
        "dataset_manifest_sha256": provenance.get("dataset_manifest_sha256", ""),
        "records_file": "records.jsonl",
        "completion_text_stored": config.experiment.store_completion_text,
        "resumed": True,
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

    print(f"\nDone. Wrote:")
    print(f"  {run_dir / 'results.json'}")
    print(f"  {run_dir / 'dashboard.html'}")


if __name__ == "__main__":
    main()
