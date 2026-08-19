from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from .schema import Results, validate_results
from .util import read_json_object, sha256_file

PARSERS = ("parse_strict", "parse_flexible", "parse_last_number")


def release_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a machine-readable publication-readiness report for results v2."""
    results: Results = validate_results(payload)
    meta = results.meta
    errors: list[str] = []
    notices: list[str] = []

    if meta.get("synthetic"):
        errors.append("synthetic results cannot pass the live publication gate")

    required_identity = (
        "model_revision",
        "tokenizer_revision",
        "serving_engine",
        "serving_engine_version",
        "task_version",
    )
    missing_identity = [field for field in required_identity if not meta.get(field)]
    if missing_identity:
        errors.append("missing deployment identity: " + ", ".join(missing_identity))

    suites = [str(value) for value in meta.get("suites", [])]
    formats = [str(value) for value in meta.get("prompt_formats", [])]
    temperatures = [float(value) for value in meta.get("temperatures", [])]
    seeds = [int(value) for value in meta.get("seeds", [])]
    n = int(meta.get("n_per_condition") or 0)
    if not suites or not formats or not temperatures or not seeds or n < 1:
        errors.append("experiment grid metadata is incomplete")
    else:
        cell_map = {
            (cell.suite, cell.fmt, float(cell.temp), cell.seed, cell.parser): cell
            for cell in results.cells
        }
        expected_seeded = set(product(suites, formats, temperatures, seeds, PARSERS))
        expected_pooled = set(product(suites, formats, temperatures, [None], PARSERS))
        expected = expected_seeded | expected_pooled
        actual = set(cell_map)
        missing = sorted(expected - actual, key=str)
        unexpected = sorted(actual - expected, key=str)
        if missing:
            errors.append(f"missing {len(missing)} expected result cells")
        if unexpected:
            errors.append(f"found {len(unexpected)} unexpected result cells")
        for key in expected_seeded & actual:
            cell = cell_map[key]
            if cell.n != n or cell.unique_items != n:
                errors.append(
                    f"seeded cell {key} has n={cell.n}, unique_items={cell.unique_items}; "
                    f"expected {n}"
                )
                break
        pooled_n = n * len(seeds)
        for key in expected_pooled & actual:
            cell = cell_map[key]
            if cell.n != pooled_n or cell.unique_items != n:
                errors.append(
                    f"pooled cell {key} has n={cell.n}, unique_items={cell.unique_items}; "
                    f"expected n={pooled_n}, unique_items={n}"
                )
                break

    if len(temperatures) > 1 and not results.condition_contrasts:
        errors.append("paired condition contrasts are missing")
    if not results.parser_contrasts:
        errors.append("paired parser contrasts are missing")

    if not meta.get("chat_template_sha256"):
        notices.append("chat_template_sha256 is not recorded")
    if not meta.get("quantization"):
        notices.append("quantization is not recorded; use an explicit value such as 'none'")
    high_unparsed = [
        cell
        for cell in results.cells
        if cell.seed is None and cell.unparsed_rate > 0.25
    ]
    if high_unparsed:
        notices.append(
            f"{len(high_unparsed)} pooled cells exceed 25% unparsed outputs; review raw logs"
        )

    return {
        "schema_version": 1,
        "run_id": meta.get("run_id"),
        "ready": not errors,
        "errors": errors,
        "notices": notices,
    }


def require_release_ready(payload: dict[str, Any]) -> dict[str, Any]:
    report = release_report(payload)
    if not report["ready"]:
        raise ValueError("release check failed: " + "; ".join(report["errors"]))
    return report


def verify_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "artifacts.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing artifact manifest: {manifest_path}")
    payload = read_json_object(manifest_path)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), dict):
        raise ValueError("unsupported artifact manifest")

    errors: list[str] = []
    expected_files: dict[str, Any] = payload["files"]
    for relative, expected in expected_files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"unsafe manifest path: {relative}")
            continue
        path = run_dir / relative_path
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing or unsafe artifact: {relative}")
            continue
        if path.stat().st_size != expected.get("bytes"):
            errors.append(f"byte-size mismatch: {relative}")
            continue
        if sha256_file(path) != expected.get("sha256"):
            errors.append(f"checksum mismatch: {relative}")

    symlinks = [
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        errors.append(f"symbolic links are not permitted in run artifacts: {symlinks[:5]}")
    actual_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name != "artifacts.manifest.json"
    }
    unexpected = sorted(actual_files - set(expected_files))
    if unexpected:
        errors.append(f"unexpected artifacts not covered by manifest: {unexpected[:5]}")

    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "ok": not errors,
        "checked_files": len(expected_files),
        "errors": errors,
    }
