from __future__ import annotations

import json
import random
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from .prompts import build_prompt, load_bbh_fewshot_prefix
from .util import (
    atomic_write_bytes,
    atomic_write_json,
    read_json_object,
    sha256_file,
    stable_id,
    utc_now,
)

SOURCES: dict[str, dict[str, str]] = {
    "gsm8k_test.jsonl": {
        "url": (
            "https://raw.githubusercontent.com/openai/grade-school-math/b0bb162/"
            "grade_school_math/data/test.jsonl"
        ),
        "source": "openai/grade-school-math",
        "license": "MIT",
    },
    "bbh_date_understanding.json": {
        "url": (
            "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/9ee07bd/"
            "bbh/date_understanding.json"
        ),
        "source": "suzgunmirac/BIG-Bench-Hard",
        "license": "MIT",
    },
    "bbh_date_understanding_cot.txt": {
        "url": (
            "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/9ee07bd/"
            "cot-prompts/date_understanding.txt"
        ),
        "source": "suzgunmirac/BIG-Bench-Hard",
        "license": "MIT",
    },
}


def prepare_data(data_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return validate_data(data_dir)

    existing_sources = [
        data_dir / filename for filename in SOURCES if (data_dir / filename).exists()
    ]
    if existing_sources and not overwrite:
        raise ValueError(
            "refusing to trust pre-existing benchmark files without a matching manifest; "
            "rerun with --overwrite"
        )

    with tempfile.TemporaryDirectory(prefix=".evalfrag-data-", dir=data_dir.parent) as temp:
        staging_dir = Path(temp)
        entries: dict[str, Any] = {}
        for filename, source in SOURCES.items():
            request = urllib.request.Request(
                source["url"], headers={"User-Agent": "evalfrag/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            path = staging_dir / filename
            atomic_write_bytes(path, content)
            entries[filename] = {
                **source,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_at": utc_now(),
            "files": entries,
        }
        atomic_write_json(staging_dir / "manifest.json", manifest)
        validate_data(staging_dir)

        for filename in SOURCES:
            atomic_write_bytes(data_dir / filename, (staging_dir / filename).read_bytes())
        atomic_write_json(manifest_path, manifest)

    validate_data(data_dir)
    return manifest


def validate_data(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing dataset manifest at {manifest_path}; run 'evalfrag prepare-data'"
        )
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported dataset manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(SOURCES):
        raise ValueError(
            "dataset manifest must contain exactly the benchmark files pinned by this release"
        )
    for filename, source in SOURCES.items():
        expected = files[filename]
        if not isinstance(expected, dict):
            raise ValueError(f"invalid dataset manifest entry for {filename}")
        for field in ("url", "source", "license"):
            if expected.get(field) != source[field]:
                raise ValueError(f"dataset manifest source mismatch for {filename}:{field}")
        path = data_dir / filename
        if not path.exists() or not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing or unsafe dataset file: {path}")
        actual = sha256_file(path)
        if actual != expected.get("sha256"):
            raise ValueError(f"dataset checksum mismatch for {filename}: {actual}")
        if path.stat().st_size != expected.get("bytes"):
            raise ValueError(f"dataset byte-size mismatch for {filename}")
    gsm_rows = _read_gsm8k(data_dir / "gsm8k_test.jsonl")
    bbh_rows = _read_bbh(data_dir / "bbh_date_understanding.json")
    if len(gsm_rows) < 1000:
        raise ValueError(f"expected at least 1000 GSM8K test rows, found {len(gsm_rows)}")
    if len(bbh_rows) < 100:
        raise ValueError(f"expected at least 100 BBH rows, found {len(bbh_rows)}")
    load_bbh_fewshot_prefix(data_dir, "date_understanding")
    return manifest


def _read_gsm8k(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                parsed: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"invalid GSM8K row at {path}:{line_no}")
            question = parsed.get("question")
            answer = parsed.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str) or "####" not in answer:
                raise ValueError(f"invalid GSM8K row at {path}:{line_no}")
            rows.append({"question": question, "answer": answer})
    return rows


def _read_bbh(path: Path) -> list[dict[str, str]]:
    payload = read_json_object(path)
    rows = payload.get("examples")
    if not isinstance(rows, list):
        raise ValueError(f"invalid BBH payload: {path}")
    typed_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"invalid BBH row {index}: {path}")
        question = row.get("input")
        target = row.get("target")
        if not isinstance(question, str) or not isinstance(target, str):
            raise ValueError(f"invalid BBH row {index}: {path}")
        typed_rows.append({"input": question, "target": target})
    return typed_rows


def select_rows(rows: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    if n > len(rows):
        raise ValueError(f"requested n={n}, but dataset contains only {len(rows)} rows")
    indexed = list(enumerate(rows))
    random.Random(seed).shuffle(indexed)
    return [row | {"_source_index": str(index)} for index, row in indexed[:n]]


def make_samples(
    *,
    suite: str,
    fmt: str,
    n: int,
    dataset_seed: int,
    data_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    if suite == "gsm8k":
        source = data_dir / "gsm8k_test.jsonl"
        rows = select_rows(_read_gsm8k(source), n, dataset_seed)
        samples: list[dict[str, Any]] = []
        for row in rows:
            question = row["question"]
            gold = row["answer"].rsplit("####", 1)[-1].strip()
            samples.append(
                {
                    "id": stable_id("gsm8k", question),
                    "input": build_prompt(question, fmt, "math"),
                    "target": gold,
                    "metadata": {
                        "gold": gold,
                        "kind": "math",
                        "fmt": fmt,
                        "source_index": int(row["_source_index"]),
                        "question_sha256": stable_id("q", question).split("-", 1)[1],
                    },
                }
            )
        return samples, str(source)
    if suite == "bbh_date_understanding":
        source = data_dir / "bbh_date_understanding.json"
        rows = select_rows(_read_bbh(source), n, dataset_seed)
        prefix = (
            load_bbh_fewshot_prefix(data_dir, "date_understanding")
            if fmt == "fewshot_tagged"
            else None
        )
        samples: list[dict[str, Any]] = []
        for row in rows:
            question = row["input"]
            gold = row["target"].strip()
            samples.append(
                {
                    "id": stable_id("bbh-date", question),
                    "input": build_prompt(question, fmt, "mc", bbh_fewshot_prefix=prefix),
                    "target": gold,
                    "metadata": {
                        "gold": gold,
                        "kind": "mc",
                        "fmt": fmt,
                        "source_index": int(row["_source_index"]),
                        "question_sha256": stable_id("q", question).split("-", 1)[1],
                    },
                }
            )
        return samples, str(source)
    raise ValueError(f"unsupported suite: {suite}")
