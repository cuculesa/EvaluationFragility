from pathlib import Path

import pytest

from evalfrag.prompts import build_prompt, load_bbh_fewshot_prefix
from evalfrag.stub import synthetic_completion


@pytest.mark.parametrize(
    ("fmt", "needle"),
    [
        ("bare", "Answer:"),
        ("cot_zero_shot", "step by step"),
        ("cot_tagged", "#### <numeric answer>"),
        ("fewshot_tagged", "Natalia sold clips"),
    ],
)
def test_math_prompt_formats_are_distinct(fmt: str, needle: str) -> None:
    prompt = build_prompt("What is 1 + 1?", fmt, "math")
    assert needle in prompt


def test_bbh_fewshot_requires_official_prefix() -> None:
    with pytest.raises(ValueError, match="official task few-shot prefix"):
        build_prompt("Question\n(A) one\n(B) two", "fewshot_tagged", "mc")
    prompt = build_prompt(
        "Question\n(A) one\n(B) two",
        "fewshot_tagged",
        "mc",
        bbh_fewshot_prefix="Official demonstrations " * 10,
    )
    assert prompt.startswith("Official demonstrations")
    assert prompt.rstrip().endswith("#### (X)")


def test_bbh_prefix_loader_rejects_short_files(tmp_path: Path) -> None:
    path = tmp_path / "bbh_date_understanding_cot.txt"
    path.write_text("too short")
    with pytest.raises(ValueError, match="unexpectedly short"):
        load_bbh_fewshot_prefix(tmp_path, "date_understanding")


def test_synthetic_completion_is_reproducible_and_seeded() -> None:
    first = synthetic_completion(
        sample_id="item",
        gold="72",
        kind="math",
        fmt="cot_tagged",
        temp=0.7,
        seed=42,
    )
    second = synthetic_completion(
        sample_id="item",
        gold="72",
        kind="math",
        fmt="cot_tagged",
        temp=0.7,
        seed=42,
    )
    different = synthetic_completion(
        sample_id="item",
        gold="72",
        kind="math",
        fmt="cot_tagged",
        temp=0.7,
        seed=43,
    )
    assert first == second
    assert first != different
