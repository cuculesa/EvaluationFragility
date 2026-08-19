from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schema import Cell, ConditionContrast, ParserContrast, Results
from .statistics import (
    benjamini_hochberg,
    exact_mcnemar,
    mean,
    paired_bootstrap_delta,
    wilson_interval,
)

PARSERS = ("parse_strict", "parse_flexible", "parse_last_number")


def _correct(record: dict[str, Any], parser: str) -> int:
    return int(bool(record["scores"][parser]["correct"]))


def _cluster_means(records: list[dict[str, Any]], parser: str) -> list[float]:
    by_item: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_item[str(record["sample_id"])].append(_correct(record, parser))
    return [mean(values) for values in by_item.values()]


def _bootstrap_mean(
    values: list[float], *, resamples: int, confidence: float, seed: int
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    zeros = [0.0] * len(values)
    _, lo, hi = paired_bootstrap_delta(
        zeros,
        values,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    return lo, hi


def aggregate_records(
    *,
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    confidence: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> Results:
    if not records:
        raise ValueError("cannot aggregate an empty record set")

    cells: list[Cell] = []
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    grouped_seed: dict[tuple[str, str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["suite"], record["fmt"], float(record["temp"]))
        grouped[key].append(record)
        grouped_seed[key + (int(record["seed"]),)].append(record)

    for (suite, fmt, temp, seed), group in sorted(grouped_seed.items()):
        for parser in PARSERS:
            values = [_correct(record, parser) for record in group]
            successes = sum(values)
            lo, hi = wilson_interval(successes, len(values), confidence)
            acc = successes / len(values)
            # Wilson's interval is mathematically guaranteed to contain the
            # point estimate, but floating-point rounding in the chain of
            # divisions can occasionally place ci_hi/ci_lo a hair outside
            # acc (e.g. 0.9999999999999998 instead of exactly 1.0) even
            # though the exact math says they should be equal. Clamp rather
            # than let a rounding artifact fail the schema's sanity check.
            lo, hi = min(lo, acc), max(hi, acc)
            cells.append(
                Cell(
                    suite=suite,
                    fmt=fmt,
                    temp=temp,
                    seed=seed,
                    parser=parser,
                    n=len(values),
                    unique_items=len({record["sample_id"] for record in group}),
                    acc=acc,
                    ci_lo=lo,
                    ci_hi=hi,
                    unparsed_rate=sum(
                        bool(record["scores"][parser]["unparsed"]) for record in group
                    )
                    / len(group),
                )
            )

    for (suite, fmt, temp), group in sorted(grouped.items()):
        for parser_index, parser in enumerate(PARSERS):
            item_means = _cluster_means(group, parser)
            lo, hi = _bootstrap_mean(
                item_means,
                resamples=bootstrap_resamples,
                confidence=confidence,
                seed=bootstrap_seed + parser_index,
            )
            acc = mean(item_means)
            # Unlike the Wilson interval above, a percentile bootstrap CI is
            # NOT guaranteed by construction to contain the point estimate,
            # especially at small n (e.g. this project's n=10 smoke-test
            # configs) -- observed directly: this is what actually tripped
            # the schema's containment check. Clamp so the reported interval
            # always brackets the accuracy it's describing.
            lo, hi = min(lo, acc), max(hi, acc)
            cells.append(
                Cell(
                    suite=suite,
                    fmt=fmt,
                    temp=temp,
                    seed=None,
                    parser=parser,
                    n=len(group),
                    unique_items=len(item_means),
                    acc=acc,
                    ci_lo=lo,
                    ci_hi=hi,
                    unparsed_rate=mean(
                        [
                            float(bool(record["scores"][parser]["unparsed"]))
                            for record in group
                        ]
                    ),
                )
            )

    parser_contrasts: list[ParserContrast] = []
    # Run exact McNemar tests within each generation seed. Pooling repeated
    # generations of the same benchmark item would treat correlated outcomes
    # as independent and make p-values look more precise than they are.
    for (suite, fmt, temp, seed), group in sorted(grouped_seed.items()):
        for parser_a, parser_b in zip(PARSERS[:-1], PARSERS[1:], strict=True):
            a = [_correct(record, parser_a) for record in group]
            b = [_correct(record, parser_b) for record in group]
            only_b, only_a, p_value = exact_mcnemar(a, b)
            parser_contrasts.append(
                ParserContrast(
                    suite=suite,
                    fmt=fmt,
                    temp=temp,
                    seed=seed,
                    parser_a=parser_a,
                    parser_b=parser_b,
                    acc_a=mean(a),
                    acc_b=mean(b),
                    delta=mean(b) - mean(a),
                    only_b=only_b,
                    only_a=only_a,
                    p_value=p_value,
                )
            )
    q_values = benjamini_hochberg([contrast.p_value for contrast in parser_contrasts])
    parser_contrasts = [
        contrast.model_copy(update={"q_value": q})
        for contrast, q in zip(parser_contrasts, q_values, strict=True)
    ]

    condition_contrasts: list[ConditionContrast] = []
    suites = sorted({record["suite"] for record in records})
    formats = list(meta["prompt_formats"])
    temperatures = sorted(float(t) for t in meta["temperatures"])

    def paired_condition(
        suite: str,
        parser: str,
        fmt_a: str,
        temp_a: float,
        fmt_b: str,
        temp_b: float,
        seed_offset: int,
    ) -> ConditionContrast | None:
        left = {
            (str(r["sample_id"]), int(r["seed"])): _correct(r, parser)
            for r in records
            if r["suite"] == suite and r["fmt"] == fmt_a and float(r["temp"]) == temp_a
        }
        right = {
            (str(r["sample_id"]), int(r["seed"])): _correct(r, parser)
            for r in records
            if r["suite"] == suite and r["fmt"] == fmt_b and float(r["temp"]) == temp_b
        }
        shared = sorted(set(left) & set(right))
        if not shared:
            return None
        # Average across generation seeds, then bootstrap over items. This treats
        # benchmark items as the resampling unit and conditions inference on the
        # configured finite seed set.
        by_item_a: dict[str, list[float]] = defaultdict(list)
        by_item_b: dict[str, list[float]] = defaultdict(list)
        for sample_id, seed in shared:
            by_item_a[sample_id].append(left[(sample_id, seed)])
            by_item_b[sample_id].append(right[(sample_id, seed)])
        item_ids = sorted(set(by_item_a) & set(by_item_b))
        a = [mean(by_item_a[item]) for item in item_ids]
        b = [mean(by_item_b[item]) for item in item_ids]
        delta, lo, hi = paired_bootstrap_delta(
            a,
            b,
            resamples=bootstrap_resamples,
            confidence=confidence,
            seed=bootstrap_seed + seed_offset,
        )
        return ConditionContrast(
            suite=suite,
            parser=parser,
            fmt_a=fmt_a,
            temp_a=temp_a,
            fmt_b=fmt_b,
            temp_b=temp_b,
            n_pairs=len(item_ids),
            delta=delta,
            ci_lo=lo,
            ci_hi=hi,
            inference_scope=(
                "paired item bootstrap; generation outcomes averaged across configured seeds"
            ),
        )

    offset = 100
    for suite in suites:
        for parser in PARSERS:
            for fmt in formats:
                if len(temperatures) > 1:
                    contrast = paired_condition(
                        suite,
                        parser,
                        fmt,
                        temperatures[0],
                        fmt,
                        temperatures[-1],
                        offset,
                    )
                    offset += 1
                    if contrast:
                        condition_contrasts.append(contrast)
            baseline_fmt = formats[0]
            baseline_temp = temperatures[0]
            for fmt in formats[1:]:
                contrast = paired_condition(
                    suite,
                    parser,
                    baseline_fmt,
                    baseline_temp,
                    fmt,
                    baseline_temp,
                    offset,
                )
                offset += 1
                if contrast:
                    condition_contrasts.append(contrast)

    warnings: list[str] = []
    if meta.get("synthetic"):
        warnings.append(
            "Synthetic backend: absolute scores and temperature effects are stipulated, "
            "not model measurements."
        )
    if len(meta.get("seeds", [])) < 3:
        warnings.append(
            "Fewer than three generation seeds limits claims about sampling variability."
        )
    if not meta.get("synthetic"):
        missing_identity = [
            field
            for field in ("model_revision", "serving_engine", "serving_engine_version")
            if not meta.get(field)
        ]
        if missing_identity:
            warnings.append(
                "Live run is missing deployment identity fields: "
                + ", ".join(missing_identity)
                + ". Capture them before publication."
            )
    if any(cell.seed is None and cell.unparsed_rate > 0.25 for cell in cells):
        warnings.append(
            "At least one condition has more than 25% unparsed outputs; report parser "
            "failure separately from accuracy."
        )
    warnings.append(
        "Condition intervals quantify benchmark-item uncertainty while conditioning on "
        "the configured model, provider, prompts, and finite seed set."
    )
    return Results(
        meta=meta,
        cells=cells,
        parser_contrasts=parser_contrasts,
        condition_contrasts=condition_contrasts,
        warnings=warnings,
    )