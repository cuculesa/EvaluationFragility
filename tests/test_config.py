from pathlib import Path

import pytest
from pydantic import ValidationError

from evalfrag.config import Config, ExperimentConfig, load_config


def test_synthetic_config_is_explicit_and_paths_are_resolved() -> None:
    config = load_config(Path("configs/synthetic.toml"))
    assert config.experiment.mode == "synthetic"
    assert config.experiment.model == "mockllm/model"
    assert config.experiment.data_dir.is_absolute()
    assert config.experiment.output_dir.is_absolute()


def test_unknown_configuration_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(
            {
                "experiment": {
                    "mode": "synthetic",
                    "model": "mockllm/model",
                    "typo": 1,
                }
            }
        )


def test_unknown_provider_with_multiple_seeds_requires_opt_in() -> None:
    with pytest.raises(ValidationError, match="not known to honor generation seeds"):
        ExperimentConfig(model="custom/model", mode="live", seeds=[1, 2])


def test_live_mode_rejects_mock_model() -> None:
    with pytest.raises(ValidationError, match="non-mock"):
        ExperimentConfig(model="mockllm/model", mode="live")


def test_experiment_name_cannot_escape_output_directory() -> None:
    import pytest
    from pydantic import ValidationError

    from evalfrag.config import ExperimentConfig

    with pytest.raises(ValidationError, match="name must be"):
        ExperimentConfig(name="../../escape")


def test_nested_model_args_are_supported_and_secret_values_are_redacted() -> None:
    from evalfrag.config import ExperimentConfig
    from evalfrag.util import redact_secrets

    config = ExperimentConfig(
        model_args={
            "extra_headers": {"Authorization": "Bearer secret"},
            "endpoint_url": "https://user:pass@example.com/v1",
        }
    )
    redacted = redact_secrets(config.model_dump())
    assert redacted["model_args"]["extra_headers"]["Authorization"] == "<redacted>"
    assert redacted["model_args"]["endpoint_url"] == "https://<redacted>@example.com/v1"


def test_runtime_and_statistics_bounds_fail_early() -> None:
    import pytest
    from pydantic import ValidationError

    from evalfrag.config import RuntimeConfig, StatisticsConfig

    with pytest.raises(ValidationError):
        RuntimeConfig(max_tasks=0)
    with pytest.raises(ValidationError, match="confidence_level"):
        StatisticsConfig(confidence_level=0.93)
