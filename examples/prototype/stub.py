"""
evalgrid.stub -- OFFLINE SYNTHETIC SOLVER. READ THIS BEFORE TRUSTING A NUMBER.

This sandbox cannot reach huggingface.co (blocked) and has no inference API
key, so no real model can be run here. This module stands in for one so that
the rest of the pipeline -- prompt rendering, answer extraction, scoring,
bootstrap CIs, paired tests, dashboard -- can be executed and verified
end to end.

What is REAL in a run driven by this stub
-----------------------------------------
  * the benchmark items and gold answers (GSM8K / BBH, fetched from source)
  * the prompt rendering for all four formats
  * every answer-extraction regex, and how it behaves on a given completion
  * all statistics, and the dashboard
  * therefore: the *parser-induced* score gaps are real consequences of the
    parser code plus the completion format, not invented numbers. If a
    completion says "the answer is 72" and the strict parser demands
    "#### 72", strict scores 0. That is arithmetic, not a guess.

What is FAKE
------------
  * the model's reasoning ability, i.e. every `p_correct` below
  * hence the ABSOLUTE accuracy of every condition
  * hence the size of the temperature effect

The knobs below are my stipulations, chosen to be roughly plausible for a
~7B open model. They are NOT measurements. Nothing in a stub run is evidence
about any real model. To get real numbers, run with `--live` (see run_grid.py),
which swaps this solver for Inspect's `generate()`.
"""

from __future__ import annotations

import hashlib
import random

from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, TaskState, solver

# ---- STIPULATED, NOT MEASURED --------------------------------------------
# p(underlying reasoning is correct) at temperature 0, per prompt format.
BASE_COMPETENCE = {
    "bare": 0.18,            # no room to reason -> weak on multi-step math
    "cot_zero_shot": 0.52,
    "cot_tagged": 0.54,
    "fewshot_tagged": 0.58,
}
# accuracy lost per unit temperature (sampling noise breaking long chains)
TEMP_PENALTY = {
    "bare": 0.05,
    "cot_zero_shot": 0.16,
    "cot_tagged": 0.15,
    "fewshot_tagged": 0.11,
}
# p(model honours the '#### x' output contract). Few-shot demonstrations pin
# formatting far harder than an instruction does, and instruction-following
# decays as temperature rises -- both well-attested, magnitudes stipulated.
TAG_COMPLIANCE = {
    "bare": (0.00, 0.00),           # (intercept, decay) - never emits a tag
    "cot_zero_shot": (0.00, 0.00),  # nothing asked for a tag
    "cot_tagged": (0.95, 0.38),
    "fewshot_tagged": (0.98, 0.10),
}

_MATH_STEPS = [
    "First, break the problem into parts.",
    "Compute the intermediate quantity.",
    "Combine the parts.",
]


def _rng(sample_id: str, fmt: str, temp: float, seed: int) -> random.Random:
    """Deterministic per (item, condition) so runs are reproducible."""
    h = hashlib.sha256(f"{sample_id}|{fmt}|{temp}|{seed}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _wrong(gold: str, r: random.Random, kind: str) -> str:
    if kind == "mc":
        letters = [c for c in "ABCDEF" if f"({c})" != gold]
        return f"({r.choice(letters)})"
    try:
        g = float(gold)
    except ValueError:
        return "0"
    off = r.choice([-3, -2, -1, 1, 2, 3, 10, -10])
    v = g + off if abs(g) > 1 else g * 2 + 1
    return str(int(v)) if v == int(v) else f"{v:.2f}"


def _synth_completion(
    gold: str, kind: str, fmt: str, temp: float, r: random.Random
) -> str:
    p_ok = max(0.02, BASE_COMPETENCE[fmt] - TEMP_PENALTY[fmt] * temp)
    ans = gold if r.random() < p_ok else _wrong(gold, r, kind)

    icept, decay = TAG_COMPLIANCE[fmt]
    tagged = r.random() < max(0.0, icept - decay * temp)

    if fmt == "bare":
        # Terse: often a naked answer, sometimes a one-liner.
        return ans if r.random() < 0.6 else f"The answer is {ans}."

    steps = "\n".join(r.sample(_MATH_STEPS, k=r.randint(2, 3)))
    if tagged:
        return f"{steps}\n#### {ans}"
    # Contract dropped -> the answer is still THERE, just not where a strict
    # parser looks. This is the failure mode the harness exists to expose.
    closer = r.choice(
        [f"So the answer is {ans}.", f"Therefore the total is {ans}.",
         f"The final answer: {ans}", f"Thus we get {ans}."]
    )
    return f"{steps}\n{closer}"


@solver
def stub_generate(seed: int = 0):
    """Drop-in replacement for inspect_ai.solver.generate() with no network."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        md = state.metadata
        fmt = md["fmt"]
        temp = float(md.get("temperature", 0.0))
        r = _rng(str(state.sample_id), fmt, temp, seed)
        text = _synth_completion(md["gold"], md["kind"], fmt, temp, r)
        state.output = ModelOutput.from_content(model="offline-stub", content=text)
        return state

    return solve
