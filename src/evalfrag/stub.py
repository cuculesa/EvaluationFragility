"""Synthetic development backend.

This module is never selected implicitly. The CLI requires config mode="synthetic"
and emits a permanent synthetic watermark in every result artifact.
"""

from __future__ import annotations

import hashlib
import random

BASE_COMPETENCE = {
    "bare": 0.18,
    "cot_zero_shot": 0.52,
    "cot_tagged": 0.54,
    "fewshot_tagged": 0.58,
}
TEMP_PENALTY = {
    "bare": 0.05,
    "cot_zero_shot": 0.16,
    "cot_tagged": 0.15,
    "fewshot_tagged": 0.11,
}
TAG_COMPLIANCE = {
    "bare": (0.00, 0.00),
    "cot_zero_shot": (0.00, 0.00),
    "cot_tagged": (0.95, 0.38),
    "fewshot_tagged": (0.98, 0.10),
}
_STEPS = [
    "First, break the problem into parts.",
    "Compute the intermediate quantity.",
    "Combine the parts.",
]


def _rng(sample_id: str, fmt: str, temp: float, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{sample_id}|{fmt}|{temp}|{seed}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _wrong(gold: str, rng: random.Random, kind: str) -> str:
    if kind == "mc":
        options = [f"({letter})" for letter in "ABCDEF" if f"({letter})" != gold]
        return rng.choice(options)
    try:
        numeric = float(gold.replace(",", ""))
    except ValueError:
        return "0"
    value = numeric + rng.choice([-10, -3, -2, -1, 1, 2, 3, 10])
    return str(int(value)) if value.is_integer() else str(value)


def synthetic_completion(
    *, sample_id: str, gold: str, kind: str, fmt: str, temp: float, seed: int
) -> str:
    rng = _rng(sample_id, fmt, temp, seed)
    p_correct = max(0.02, BASE_COMPETENCE[fmt] - TEMP_PENALTY[fmt] * temp)
    answer = gold if rng.random() < p_correct else _wrong(gold, rng, kind)
    intercept, decay = TAG_COMPLIANCE[fmt]
    tagged = rng.random() < max(0.0, intercept - decay * temp)
    if fmt == "bare":
        return answer if rng.random() < 0.6 else f"The answer is {answer}."
    steps = "\n".join(rng.sample(_STEPS, k=rng.randint(2, 3)))
    if tagged:
        return f"{steps}\n#### {answer}"
    return f"{steps}\nThe final answer is {answer}."
