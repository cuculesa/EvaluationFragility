from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cell(StrictModel):
    suite: str
    fmt: str
    temp: float
    seed: int | None = None
    parser: str
    n: int
    unique_items: int
    acc: float
    ci_lo: float
    ci_hi: float
    unparsed_rate: float

    @field_validator("acc", "ci_lo", "ci_hi", "unparsed_rate")
    @classmethod
    def probability(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("probability must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def coherent_counts_and_interval(self) -> Cell:
        if self.n < 1 or self.unique_items < 1 or self.unique_items > self.n:
            raise ValueError("cell counts must satisfy 1 <= unique_items <= n")
        if not self.ci_lo <= self.acc <= self.ci_hi:
            raise ValueError("cell interval must contain accuracy")
        return self


class ParserContrast(StrictModel):
    suite: str
    fmt: str
    temp: float
    seed: int | None = None
    parser_a: str
    parser_b: str
    acc_a: float
    acc_b: float
    delta: float
    only_b: int
    only_a: int
    p_value: float
    q_value: float | None = None

    @field_validator("acc_a", "acc_b", "p_value", "q_value")
    @classmethod
    def bounded_probability(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 1:
            raise ValueError("probability must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def coherent_contrast(self) -> ParserContrast:
        if self.only_a < 0 or self.only_b < 0:
            raise ValueError("discordant counts must be non-negative")
        if abs(self.delta - (self.acc_b - self.acc_a)) > 1e-9:
            raise ValueError("parser contrast delta must equal acc_b - acc_a")
        return self


class ConditionContrast(StrictModel):
    suite: str
    parser: str
    fmt_a: str
    temp_a: float
    fmt_b: str
    temp_b: float
    n_pairs: int
    delta: float
    ci_lo: float
    ci_hi: float
    inference_scope: str

    @model_validator(mode="after")
    def coherent_interval(self) -> ConditionContrast:
        if self.n_pairs < 1:
            raise ValueError("condition contrast requires paired items")
        if not -1 <= self.ci_lo <= self.delta <= self.ci_hi <= 1:
            raise ValueError("condition contrast interval must contain delta within [-1, 1]")
        return self


class Results(StrictModel):
    schema_version: Literal[2] = 2
    meta: dict[str, Any]
    cells: list[Cell]
    parser_contrasts: list[ParserContrast]
    condition_contrasts: list[ConditionContrast] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def artifact_invariants(self) -> Results:
        keys = [(c.suite, c.fmt, c.temp, c.seed, c.parser) for c in self.cells]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate result cells")
        if not self.cells:
            raise ValueError("results contain no cells")
        required_meta = {
            "run_id",
            "created_at",
            "model",
            "synthetic",
            "n_per_condition",
            "seeds",
            "temperatures",
            "prompt_formats",
            "suites",
            "config_sha256",
            "inspect_ai_version",
            "evalfrag_version",
            "dataset_manifest_sha256",
            "completion_text_stored",
        }
        missing = sorted(required_meta - set(self.meta))
        if missing:
            raise ValueError(f"missing result metadata: {missing}")
        return self


def validate_results(payload: dict[str, Any]) -> Results:
    return Results.model_validate(payload)
