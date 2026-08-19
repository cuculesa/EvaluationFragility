from pathlib import Path

import pytest

from evalfrag.config import Config, ExperimentConfig
from evalfrag.runner import _validate_record_grid


def _config() -> Config:
    return Config(
        experiment=ExperimentConfig(
            name="grid-test",
            mode="synthetic",
            model="mockllm/model",
            suites=["gsm8k"],
            prompt_formats=["cot_tagged", "fewshot_tagged"],
            temperatures=[0.0, 0.7],
            seeds=[42],
            n=2,
            data_dir=Path("data"),
            output_dir=Path("runs"),
        )
    )


def _record(fmt: str, temp: float, sample_id: str, epoch: int = 1) -> dict[str, object]:
    return {
        "suite": "gsm8k",
        "fmt": fmt,
        "temp": temp,
        "seed": 42,
        "sample_id": sample_id,
        "epoch": epoch,
        "question_sha256": f"hash-{sample_id}",
    }


def _complete_records() -> list[dict[str, object]]:
    return [
        _record(fmt, temp, sample_id)
        for fmt in ("cot_tagged", "fewshot_tagged")
        for temp in (0.0, 0.7)
        for sample_id in ("item-a", "item-b")
    ]


def test_grid_accepts_complete_paired_design() -> None:
    _validate_record_grid(_complete_records(), _config())


def test_grid_rejects_missing_condition() -> None:
    records = [
        record
        for record in _complete_records()
        if not (record["fmt"] == "fewshot_tagged" and record["temp"] == 0.7)
    ]
    with pytest.raises(ValueError, match="incomplete result grid"):
        _validate_record_grid(records, _config())


def test_grid_rejects_duplicate_sample_record() -> None:
    records = _complete_records()
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="duplicate sample record"):
        _validate_record_grid(records, _config())


def test_grid_rejects_item_mismatch_between_conditions() -> None:
    records = _complete_records()
    for record in records:
        if (
            record["fmt"] == "fewshot_tagged"
            and record["temp"] == 0.7
            and record["sample_id"] == "item-b"
        ):
            record["sample_id"] = "item-c"
            record["question_sha256"] = "hash-item-c"
    with pytest.raises(ValueError, match="same item IDs"):
        _validate_record_grid(records, _config())


def test_grid_rejects_missing_question_provenance() -> None:
    records = _complete_records()
    records[0]["question_sha256"] = None
    with pytest.raises(ValueError, match="missing question provenance"):
        _validate_record_grid(records, _config())
