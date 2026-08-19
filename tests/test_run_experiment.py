import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from evalfrag.config import Config, ExperimentConfig, RuntimeConfig, StatisticsConfig
from evalfrag.release import verify_artifact_manifest
from evalfrag.runner import run_experiment


def test_run_experiment_consumes_eval_set_tuple_and_writes_auditable_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text('{"files": {}}')
    output_dir = tmp_path / "runs"

    config = Config(
        experiment=ExperimentConfig(
            name="contract-test",
            mode="live",
            model="vllm/test-model",
            suites=["gsm8k"],
            prompt_formats=["cot_tagged"],
            temperatures=[0.0],
            seeds=[42],
            n=1,
            data_dir=data_dir,
            output_dir=output_dir,
        ),
        runtime=RuntimeConfig(
            max_tasks=1,
            max_samples=1,
            max_connections=1,
            eval_set_retry_attempts=0,
            model_max_retries=0,
            sample_retry_on_error=0,
        ),
        statistics=StatisticsConfig(bootstrap_resamples=1000),
    )

    score = SimpleNamespace(value="C", answer="72", metadata={"unparsed": False})
    sample = SimpleNamespace(
        id="gsm8k-item",
        epoch=1,
        metadata={"question_sha256": "qhash"},
        output=SimpleNamespace(completion="work\n#### 72"),
        scores={
            "parse_strict": score,
            "parse_flexible": score,
            "parse_last_number": score,
        },
    )
    log = SimpleNamespace(
        status="success",
        error=None,
        eval=SimpleNamespace(
            task="task",
            eval_id="eval-id",
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
    called: dict[str, object] = {}

    def fake_eval_set(**kwargs):
        called.update(kwargs)
        return True, [log]

    inspect_module = ModuleType("inspect_ai")
    inspect_module.eval_set = fake_eval_set  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    monkeypatch.setattr("evalfrag.runner.validate_data", lambda _: {"files": {}})
    monkeypatch.setattr("evalfrag.runner.build_task", lambda **_: "task")
    monkeypatch.setattr(
        "evalfrag.runner.importlib.metadata.version",
        lambda name: "0.3.251" if name == "inspect-ai" else "1.0.0",
    )

    run_dir = run_experiment(config, project_root=tmp_path)
    assert called["tasks"] == ["task"]
    assert called["model"] == "vllm/test-model"
    assert called["log_format"] == "eval"
    assert (run_dir / "records.jsonl").is_file()
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "dashboard.html").is_file()
    assert (run_dir / "artifacts.manifest.json").is_file()
    assert json.loads((run_dir / "failures.json").read_text()) == []
    assert json.loads((run_dir / "run_state.json").read_text())["status"] == "complete"
    assert verify_artifact_manifest(run_dir)["ok"] is True


def test_synthetic_run_does_not_construct_a_model_provider(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text('{"files": {}}')
    config = Config(
        experiment=ExperimentConfig(
            name="synthetic-contract",
            mode="synthetic",
            model="mockllm/model",
            suites=["gsm8k"],
            prompt_formats=["cot_tagged"],
            temperatures=[0.0],
            seeds=[42],
            n=1,
            data_dir=data_dir,
            output_dir=tmp_path / "runs",
        ),
        runtime=RuntimeConfig(
            max_tasks=1,
            max_samples=1,
            max_connections=1,
            eval_set_retry_attempts=0,
            model_max_retries=0,
            sample_retry_on_error=0,
        ),
        statistics=StatisticsConfig(bootstrap_resamples=1000),
    )
    score = SimpleNamespace(value="C", answer="72", metadata={"unparsed": False})
    sample = SimpleNamespace(
        id="gsm8k-item",
        epoch=1,
        metadata={"question_sha256": "qhash"},
        output=SimpleNamespace(completion="work\n#### 72"),
        scores={name: score for name in ("parse_strict", "parse_flexible", "parse_last_number")},
    )
    log = SimpleNamespace(
        status="success",
        error=None,
        eval=SimpleNamespace(
            task="task",
            eval_id="eval-id",
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
    called: dict[str, object] = {}

    def fake_eval_set(**kwargs):
        called.update(kwargs)
        return True, [log]

    inspect_module = ModuleType("inspect_ai")
    inspect_module.eval_set = fake_eval_set  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    monkeypatch.setattr("evalfrag.runner.validate_data", lambda _: {"files": {}})
    monkeypatch.setattr("evalfrag.runner.build_task", lambda **_: "task")
    monkeypatch.setattr(
        "evalfrag.runner.importlib.metadata.version",
        lambda name: "0.3.251" if name == "inspect-ai" else "1.0.0",
    )

    run_experiment(config, project_root=tmp_path)
    assert called["model"] is None
    assert called["model_base_url"] is None
    assert called["model_args"] == {}
