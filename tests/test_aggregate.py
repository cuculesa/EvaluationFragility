from evalfrag.aggregate import aggregate_records


def score(correct: bool, unparsed: bool = False) -> dict[str, object]:
    return {"correct": correct, "answer": "1" if not unparsed else None, "unparsed": unparsed}


def record(
    sample: str,
    fmt: str,
    temp: float,
    seed: int,
    strict: bool,
    flex: bool,
    permissive: bool,
):
    return {
        "suite": "gsm8k",
        "fmt": fmt,
        "temp": temp,
        "seed": seed,
        "sample_id": sample,
        "scores": {
            "parse_strict": score(strict, not strict and flex),
            "parse_flexible": score(flex),
            "parse_last_number": score(permissive),
        },
    }


def test_aggregation_produces_seed_and_pooled_cells_and_paired_contrasts() -> None:
    records = []
    for seed in (1, 2):
        records += [
            record("a", "bare", 0.0, seed, False, True, True),
            record("b", "bare", 0.0, seed, True, True, True),
            record("a", "bare", 1.0, seed, False, False, True),
            record("b", "bare", 1.0, seed, False, True, True),
        ]
    results = aggregate_records(
        records=records,
        meta={
            "run_id": "test",
            "created_at": "2026-07-30T00:00:00Z",
            "model": "test/model",
            "synthetic": False,
            "n_per_condition": 2,
            "seeds": [1, 2],
            "prompt_formats": ["bare"],
            "temperatures": [0.0, 1.0],
            "suites": ["gsm8k"],
            "dataset_seed": 1,
            "config_sha256": "x",
            "inspect_ai_version": "test",
            "evalfrag_version": "test",
            "dataset_manifest_sha256": "y",
            "records_file": "records.jsonl",
            "completion_text_stored": False,
        },
        confidence=0.95,
        bootstrap_resamples=1000,
        bootstrap_seed=3,
    )
    pooled = [c for c in results.cells if c.seed is None]
    per_seed = [c for c in results.cells if c.seed is not None]
    assert len(pooled) == 6
    assert len(per_seed) == 12
    assert results.parser_contrasts
    assert {contrast.seed for contrast in results.parser_contrasts} == {1, 2}
    assert results.condition_contrasts
    strict_t0 = next(c for c in pooled if c.temp == 0 and c.parser == "parse_strict")
    assert strict_t0.acc == 0.5
    assert strict_t0.unique_items == 2
