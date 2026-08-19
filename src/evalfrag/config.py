from __future__ import annotations

import math
import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROMPT_FORMATS = ("bare", "cot_zero_shot", "cot_tagged", "fewshot_tagged")
SUITES = ("gsm8k", "bbh_date_understanding")
SEED_SUPPORTED_PROVIDERS = {
    "openai",
    "google",
    "mistral",
    "groq",
    "hf",
    "huggingface",
    "vllm",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(StrictModel):
    name: str = "evalfrag"
    mode: Literal["live", "synthetic"] = "live"
    model: str = "vllm/Qwen/Qwen2.5-7B-Instruct"
    model_base_url: str | None = None
    model_args: dict[str, Any] = Field(default_factory=dict)
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    serving_engine: str | None = None
    serving_engine_version: str | None = None
    quantization: str | None = None
    chat_template_sha256: str | None = None
    suites: list[str] = Field(default_factory=lambda: list(SUITES))
    prompt_formats: list[str] = Field(default_factory=lambda: list(PROMPT_FORMATS))
    temperatures: list[float] = Field(default_factory=lambda: [0.0, 0.7, 1.0])
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44])
    n: int = 200
    dataset_seed: int = 2026
    max_tokens: int = 512
    top_p: float = 1.0
    output_dir: Path = Path("runs")
    data_dir: Path = Path("data")
    allow_unseeded_provider: bool = False
    store_completion_text: bool = False

    @field_validator("name")
    @classmethod
    def name_is_safe_slug(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
            raise ValueError(
                "name must be a 1-64 character slug using letters, digits, '.', '_', or '-'"
            )
        return value

    @field_validator("model")
    @classmethod
    def model_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be empty")
        return value

    @field_validator("suites")
    @classmethod
    def suites_known(cls, values: list[str]) -> list[str]:
        unknown = sorted(set(values) - set(SUITES))
        if unknown:
            raise ValueError(f"unknown suites: {unknown}")
        if not values:
            raise ValueError("at least one suite is required")
        return list(dict.fromkeys(values))

    @field_validator("prompt_formats")
    @classmethod
    def formats_known(cls, values: list[str]) -> list[str]:
        unknown = sorted(set(values) - set(PROMPT_FORMATS))
        if unknown:
            raise ValueError(f"unknown prompt formats: {unknown}")
        if not values:
            raise ValueError("at least one prompt format is required")
        return list(dict.fromkeys(values))

    @field_validator("temperatures")
    @classmethod
    def temperatures_valid(cls, values: list[float]) -> list[float]:
        if not values or any(not math.isfinite(t) or t < 0 or t > 2 for t in values):
            raise ValueError("temperatures must be finite and between 0 and 2")
        return list(dict.fromkeys(values))

    @field_validator("seeds")
    @classmethod
    def seeds_valid(cls, values: list[int]) -> list[int]:
        if not values:
            raise ValueError("at least one generation seed is required")
        return list(dict.fromkeys(values))

    @field_validator("n")
    @classmethod
    def n_valid(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n must be positive")
        return value

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_valid(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_tokens must be positive")
        return value

    @field_validator("top_p")
    @classmethod
    def top_p_valid(cls, value: float) -> float:
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        return value

    @model_validator(mode="after")
    def live_model_required(self) -> ExperimentConfig:
        if self.mode == "live" and self.model.startswith("mock"):
            raise ValueError("live mode requires a non-mock model")
        if self.mode == "synthetic" and self.model != "mockllm/model":
            raise ValueError("synthetic mode must use model='mockllm/model'")
        if len(self.seeds) > 1 and self.mode == "live":
            provider = self.model.split("/", 1)[0].lower()
            if provider not in SEED_SUPPORTED_PROVIDERS and not self.allow_unseeded_provider:
                raise ValueError(
                    f"provider {provider!r} is not known to honor generation seeds; "
                    "set allow_unseeded_provider=true only after verifying provider behavior"
                )
        return self


class RuntimeConfig(StrictModel):
    max_tasks: int = Field(default=4, ge=1)
    max_samples: int = Field(default=8, ge=1)
    max_connections: int | None = Field(default=16, ge=1)
    timeout_seconds: int = Field(default=180, ge=1)
    attempt_timeout_seconds: int = Field(default=120, ge=1)
    model_max_retries: int = Field(default=3, ge=0)
    sample_retry_on_error: int = Field(default=2, ge=0)
    eval_set_retry_attempts: int = Field(default=3, ge=0)
    fail_on_error: bool | float = True
    checkpoint: bool = True

    @field_validator("fail_on_error")
    @classmethod
    def fail_on_error_valid(cls, value: bool | float) -> bool | float:
        if isinstance(value, bool):
            return value
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("fail_on_error must be finite and in [0, 1]")
        return value


class StatisticsConfig(StrictModel):
    confidence_level: float = 0.95
    bootstrap_resamples: int = 5000
    bootstrap_seed: int = 1729

    @field_validator("confidence_level")
    @classmethod
    def confidence_valid(cls, value: float) -> float:
        if round(value, 2) not in {0.90, 0.95, 0.99}:
            raise ValueError("confidence_level must be 0.90, 0.95, or 0.99")
        return value

    @field_validator("bootstrap_resamples")
    @classmethod
    def bootstrap_valid(cls, value: int) -> int:
        if value < 1000:
            raise ValueError("bootstrap_resamples must be at least 1000")
        return value


class DashboardConfig(StrictModel):
    title: str = "Evaluation methodology sensitivity"
    high_unparsed_threshold: float = 0.10

    @field_validator("high_unparsed_threshold")
    @classmethod
    def threshold_is_probability(cls, value: float) -> float:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("high_unparsed_threshold must be in [0, 1]")
        return value


class Config(StrictModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def load_config(path: str | Path) -> Config:
    cfg_path = Path(path).resolve()
    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)
    config = Config.model_validate(raw)
    base = cfg_path.parent
    exp = config.experiment.model_copy(
        update={
            "output_dir": (base / config.experiment.output_dir).resolve()
            if not config.experiment.output_dir.is_absolute()
            else config.experiment.output_dir,
            "data_dir": (base / config.experiment.data_dir).resolve()
            if not config.experiment.data_dir.is_absolute()
            else config.experiment.data_dir,
            "model_base_url": config.experiment.model_base_url
            or os.getenv("EVALFRAG_MODEL_BASE_URL"),
        }
    )
    return config.model_copy(update={"experiment": exp})
