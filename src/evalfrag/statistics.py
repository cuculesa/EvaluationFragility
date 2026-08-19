from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    # Common confidence levels are enough for this evaluation package.
    z_by_confidence = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    z = z_by_confidence.get(round(confidence, 2))
    if z is None:
        raise ValueError("supported confidence levels are 0.90, 0.95, and 0.99")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar(a: Sequence[int], b: Sequence[int]) -> tuple[int, int, float]:
    if len(a) != len(b):
        raise ValueError("paired vectors must have equal length")
    only_b = sum(x == 0 and y == 1 for x, y in zip(a, b, strict=True))
    only_a = sum(x == 1 and y == 0 for x, y in zip(a, b, strict=True))
    discordant = only_a + only_b
    if discordant == 0:
        return only_b, only_a, 1.0
    k = min(only_a, only_b)
    p = 2 * sum(math.comb(discordant, i) for i in range(k + 1)) / (2**discordant)
    return only_b, only_a, min(1.0, p)


def paired_bootstrap_delta(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    if len(a) != len(b) or not a:
        raise ValueError("paired non-empty vectors of equal length are required")
    rng = random.Random(seed)
    n = len(a)
    observed = mean([y - x for x, y in zip(a, b, strict=True)])
    deltas = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            total += b[i] - a[i]
        deltas.append(total / n)
    deltas.sort()
    alpha = 1 - confidence
    lo_index = max(0, min(resamples - 1, int((alpha / 2) * resamples)))
    hi_index = max(0, min(resamples - 1, int((1 - alpha / 2) * resamples) - 1))
    return observed, deltas[lo_index], deltas[hi_index]


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    values = list(p_values)
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])
    adjusted = [1.0] * m
    running = 1.0
    for rank_from_end, index in enumerate(reversed(order), start=1):
        rank = m - rank_from_end + 1
        running = min(running, values[index] * m / rank)
        adjusted[index] = min(1.0, running)
    return adjusted
