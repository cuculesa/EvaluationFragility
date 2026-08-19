from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import Cell, ParserContrast, Results
from .util import atomic_write_json, read_json_object, utc_now


def migrate_v1(input_path: Path, output_path: Path) -> None:
    old = read_json_object(input_path)
    old_meta = old.get("meta", {})
    formats = old_meta.get("formats", [])
    temperatures = old_meta.get("temps", [])
    suites = sorted({cell["suite"] for cell in old.get("cells", [])})
    cells = [
        Cell(
            suite=cell["suite"],
            fmt=cell["fmt"],
            temp=float(cell["temp"]),
            seed=None,
            parser=cell["parser"],
            n=int(cell["n"]),
            unique_items=int(cell["n"]),
            acc=float(cell["acc"]),
            ci_lo=float(cell["ci_lo"]),
            ci_hi=float(cell["ci_hi"]),
            unparsed_rate=float(cell["unparsed_rate"]),
        )
        for cell in old.get("cells", [])
    ]
    contrasts = [
        ParserContrast(
            suite=row["suite"],
            fmt=row["fmt"],
            temp=float(row["temp"]),
            seed=None,
            parser_a=row["a"],
            parser_b=row["b"],
            acc_a=float(row["acc_a"]),
            acc_b=float(row["acc_b"]),
            delta=float(row["acc_b"]) - float(row["acc_a"]),
            only_b=int(row["only_b"]),
            only_a=int(row["only_a"]),
            p_value=float(row["p"]),
            q_value=None,
        )
        for row in old.get("contrasts", [])
    ]
    meta: dict[str, Any] = {
        "run_id": "legacy-synthetic-migration",
        "created_at": utc_now(),
        "model": old_meta.get("model", "unknown"),
        "synthetic": bool(old_meta.get("offline_stub", True)),
        "n_per_condition": old_meta.get("n_per_condition"),
        "seeds": [],
        "temperatures": temperatures,
        "prompt_formats": formats,
        "suites": suites,
        "dataset_seed": None,
        "config_sha256": "legacy-unavailable",
        "inspect_ai_version": old_meta.get("harness_version", "legacy-unavailable"),
        "evalfrag_version": "1.0.0",
        "dataset_manifest_sha256": "legacy-unavailable",
        "records_file": None,
        "completion_text_stored": "unknown",
    }
    result = Results(
        meta=meta,
        cells=cells,
        parser_contrasts=contrasts,
        condition_contrasts=[],
        warnings=[
            "Migrated legacy summary: no sample-level records are available "
            "for paired condition intervals.",
            "Synthetic backend: absolute scores and temperature effects are stipulated, "
            "not model measurements.",
        ],
    )
    atomic_write_json(output_path, result.model_dump(mode="json"))
