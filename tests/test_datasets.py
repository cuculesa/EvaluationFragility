from evalfrag.datasets import select_rows


def test_selection_is_reproducible_and_keeps_source_index() -> None:
    rows = [{"question": str(i), "answer": "#### 1"} for i in range(20)]
    first = select_rows(rows, 5, 42)
    second = select_rows(rows, 5, 42)
    assert first == second
    assert len({row["_source_index"] for row in first}) == 5


def test_selection_rejects_oversampling() -> None:
    try:
        select_rows([{"x": "1"}], 2, 0)
    except ValueError as exc:
        assert "contains only" in str(exc)
    else:
        raise AssertionError("expected oversampling failure")


def test_manifest_rejects_unpinned_files(tmp_path) -> None:
    import json

    from evalfrag.datasets import validate_data

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    "../../outside": {
                        "url": "https://example.invalid",
                        "source": "unknown",
                        "license": "unknown",
                        "sha256": "0" * 64,
                        "bytes": 0,
                    }
                },
            }
        )
    )
    try:
        validate_data(tmp_path)
    except ValueError as exc:
        assert "exactly the benchmark files" in str(exc)
    else:
        raise AssertionError("expected manifest allowlist failure")


def test_prepare_data_rejects_unmanifested_existing_file(tmp_path) -> None:
    from evalfrag.datasets import prepare_data

    (tmp_path / "gsm8k_test.jsonl").write_text("untrusted")
    try:
        prepare_data(tmp_path)
    except ValueError as exc:
        assert "without a matching manifest" in str(exc)
    else:
        raise AssertionError("expected existing-file trust failure")


def test_prepare_data_stages_and_reuses_approved_manifest(tmp_path, monkeypatch) -> None:
    import io
    import json

    from evalfrag.datasets import prepare_data, validate_data

    gsm = "".join(
        json.dumps({"question": f"q{i}", "answer": "work #### 1"}) + "\n"
        for i in range(1000)
    ).encode()
    bbh = json.dumps(
        {
            "examples": [
                {"input": f"question {i}\n(A) one\n(B) two", "target": "(A)"}
                for i in range(100)
            ]
        }
    ).encode()
    cot = ("Official example and explanation.\n" * 10).encode()
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        if request.full_url.endswith("test.jsonl"):
            return io.BytesIO(gsm)
        if request.full_url.endswith("date_understanding.json"):
            return io.BytesIO(bbh)
        return io.BytesIO(cot)

    monkeypatch.setattr("evalfrag.datasets.urllib.request.urlopen", fake_urlopen)
    manifest = prepare_data(tmp_path)
    assert len(calls) == 3
    assert set(manifest["files"]) == {
        "gsm8k_test.jsonl",
        "bbh_date_understanding.json",
        "bbh_date_understanding_cot.txt",
    }
    assert validate_data(tmp_path) == manifest

    def unexpected_call(*args, **kwargs):
        raise AssertionError("approved manifest should be reused without network")

    monkeypatch.setattr("evalfrag.datasets.urllib.request.urlopen", unexpected_call)
    assert prepare_data(tmp_path) == manifest


def test_make_samples_keeps_ids_paired_across_formats(tmp_path) -> None:
    import json

    from evalfrag.datasets import make_samples

    (tmp_path / "gsm8k_test.jsonl").write_text(
        "".join(
            json.dumps({"question": f"q{i}", "answer": "work #### 1"}) + "\n"
            for i in range(10)
        )
    )
    bare, _ = make_samples(
        suite="gsm8k", fmt="bare", n=5, dataset_seed=7, data_dir=tmp_path
    )
    tagged, _ = make_samples(
        suite="gsm8k", fmt="cot_tagged", n=5, dataset_seed=7, data_dir=tmp_path
    )
    assert [sample["id"] for sample in bare] == [sample["id"] for sample in tagged]
    assert [sample["input"] for sample in bare] != [sample["input"] for sample in tagged]
