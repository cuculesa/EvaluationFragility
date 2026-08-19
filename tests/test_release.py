import json
from pathlib import Path

from evalfrag.aggregate import aggregate_records
from evalfrag.release import release_report, verify_artifact_manifest
from evalfrag.util import atomic_write_json, build_file_manifest


def _records() -> list[dict[str, object]]:
    records = []
    for fmt in ("cot_tagged", "fewshot_tagged"):
        for temp in (0.0, 0.7):
            for seed in (42, 43, 44):
                for item in ("a", "b"):
                    records.append(
                        {
                            "suite": "gsm8k",
                            "fmt": fmt,
                            "temp": temp,
                            "seed": seed,
                            "sample_id": item,
                            "scores": {
                                parser: {"correct": item == "a", "unparsed": False}
                                for parser in (
                                    "parse_strict",
                                    "parse_flexible",
                                    "parse_last_number",
                                )
                            },
                        }
                    )
    return records


def _payload(*, synthetic: bool = False) -> dict[str, object]:
    meta = {
        "run_id": "run",
        "created_at": "2026-01-01T00:00:00Z",
        "model": "vllm/model",
        "model_revision": "abc123",
        "tokenizer_revision": "abc123",
        "serving_engine": "vllm",
        "serving_engine_version": "1.0",
        "quantization": "none",
        "chat_template_sha256": "f" * 64,
        "task_version": "1.0.0",
        "synthetic": synthetic,
        "n_per_condition": 2,
        "seeds": [42, 43, 44],
        "temperatures": [0.0, 0.7],
        "prompt_formats": ["cot_tagged", "fewshot_tagged"],
        "suites": ["gsm8k"],
        "config_sha256": "c" * 64,
        "inspect_ai_version": "0.3.251",
        "evalfrag_version": "1.0.0",
        "dataset_manifest_sha256": "d" * 64,
        "completion_text_stored": False,
    }
    return aggregate_records(
        records=_records(),
        meta=meta,
        confidence=0.95,
        bootstrap_resamples=1000,
        bootstrap_seed=1,
    ).model_dump(mode="json")


def test_release_report_accepts_complete_live_artifact() -> None:
    report = release_report(_payload())
    assert report["ready"] is True
    assert report["errors"] == []


def test_release_report_rejects_synthetic_artifact() -> None:
    report = release_report(_payload(synthetic=True))
    assert report["ready"] is False
    assert any("synthetic" in error for error in report["errors"])


def test_artifact_manifest_detects_tampering(tmp_path: Path) -> None:
    (tmp_path / "results.json").write_text(json.dumps({"a": 1}))
    atomic_write_json(
        tmp_path / "artifacts.manifest.json",
        build_file_manifest(tmp_path, exclude={"artifacts.manifest.json"}),
    )
    assert verify_artifact_manifest(tmp_path)["ok"] is True
    (tmp_path / "results.json").write_text(json.dumps({"a": 2}))
    report = verify_artifact_manifest(tmp_path)
    assert report["ok"] is False
    assert any("mismatch" in error for error in report["errors"])
