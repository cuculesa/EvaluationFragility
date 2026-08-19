from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .datasets import make_samples
from .parsers import PARSERS
from .stub import synthetic_completion

TASK_VERSION = "1.0.0"


def _task_name(suite: str, fmt: str, temp: float, seed: int) -> str:
    temp_slug = str(temp).replace(".", "p")
    raw = f"evalfrag_{suite}_{fmt}_t{temp_slug}_s{seed}"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", raw)


def build_task(
    *,
    suite: str,
    fmt: str,
    temperature: float,
    generation_seed: int,
    n: int,
    dataset_seed: int,
    data_dir: Path,
    max_tokens: int,
    top_p: float,
    synthetic: bool,
    model: str = "",
) -> Any:
    """Build one Inspect Task. Imports Inspect lazily for testability."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig, ModelOutput
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
    from inspect_ai.solver import Generate, TaskState, generate, solver

    sample_rows, source = make_samples(
        suite=suite, fmt=fmt, n=n, dataset_seed=dataset_seed, data_dir=data_dir
    )
    samples = [Sample(**row) for row in sample_rows]
    kind = sample_rows[0]["metadata"]["kind"]

    def make_scorer(rule: str) -> Any:
        parser = PARSERS[kind][rule]

        @scorer(name=f"parse_{rule}", metrics=[accuracy(), stderr()])
        def parser_scorer() -> Any:
            async def score(state: TaskState, target: Target) -> Score:
                completion = state.output.completion if state.output else ""
                extracted = parser(completion)
                gold = target.text.strip()
                if kind == "math":
                    from .parsers import normalize_number

                    gold = normalize_number(gold)
                else:
                    letter = re.sub(r"[^A-Za-z]", "", gold).upper()
                    gold = f"({letter})"
                correct = extracted is not None and extracted == gold
                return Score(
                    value=CORRECT if correct else INCORRECT,
                    answer=extracted,
                    metadata={
                        "unparsed": extracted is None,
                        "gold": gold,
                        "parser": rule,
                        "parser_version": TASK_VERSION,
                    },
                )

            return score

        return parser_scorer()

    @solver(name="evalfrag_synthetic_generate")
    def synthetic_generate() -> Any:
        async def solve(state: TaskState, _: Generate) -> TaskState:
            metadata = state.metadata or {}
            text = synthetic_completion(
                sample_id=str(state.sample_id),
                gold=str(metadata["gold"]),
                kind=str(metadata["kind"]),
                fmt=fmt,
                temp=temperature,
                seed=generation_seed,
            )
            state.output = ModelOutput.from_content(model="offline-synthetic", content=text)
            return state

        return solve

    condition = {
        "suite": suite,
        "fmt": fmt,
        "temperature": temperature,
        "generation_seed": generation_seed,
        "dataset_seed": dataset_seed,
        "synthetic": synthetic,
        "task_version": TASK_VERSION,
    }
    return Task(
        name=_task_name(suite, fmt, temperature, generation_seed),
        display_name=f"{suite} · {fmt} · T={temperature} · seed={generation_seed}",
        version=TASK_VERSION,
        dataset=MemoryDataset(
            samples,
            name=suite,
            location=source,
            shuffled=True,
        ),
        solver=[synthetic_generate() if synthetic else generate()],
        scorer=[make_scorer(rule) for rule in ("strict", "flexible", "last_number")],
        # Anthropic's API rejects a request that sets both `temperature` and
        # `top_p` at once (400 invalid_request_error) -- it wants exactly one
        # sampling parameter, not both, even when top_p is the 1.0 no-op
        # default. Other providers (OpenAI, vLLM, etc.) accept both together.
        # So: always drive sampling via temperature (it's the one varied by
        # this harness's experiment grid), and only also send top_p when the
        # provider is known to accept the pair.
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=generation_seed,
            **({} if model.split("/", 1)[0].lower() == "anthropic"
               else {"top_p": top_p}),
        ),
        metadata={"evalfrag_condition": condition},
        tags=["evalfrag", "synthetic" if synthetic else "live"],
    )
