from types import SimpleNamespace

from evalfrag.runner import _records_from_logs


def test_log_conversion_preserves_condition_and_hashes_completion() -> None:
    scores = {
        name: SimpleNamespace(value="C", answer="72", metadata={"unparsed": False})
        for name in ("parse_strict", "parse_flexible", "parse_last_number")
    }
    sample = SimpleNamespace(
        id="gsm8k-abc",
        epoch=1,
        metadata={"question_sha256": "abc"},
        output=SimpleNamespace(completion="reasoning\n#### 72"),
        scores=scores,
    )
    log = SimpleNamespace(
        status="success",
        eval=SimpleNamespace(
            eval_id="e1",
            metadata={
                "evalfrag_condition": {
                    "suite": "gsm8k",
                    "fmt": "cot_tagged",
                    "temperature": 0.0,
                    "generation_seed": 42,
                }
            },
        ),
        samples=[sample],
    )
    records = _records_from_logs([log], store_completion_text=False)
    assert len(records) == 1
    assert records[0]["seed"] == 42
    assert "completion" not in records[0]
    assert len(records[0]["completion_sha256"]) == 64
