"""
Sweep prompt-format x temperature and score every completion three ways.

  python3 run_grid.py            # offline stub (no network, no key)
  python3 run_grid.py --live --model anthropic/claude-haiku-4-5-20251001
  python3 run_grid.py --live --model vllm/Qwen/Qwen2.5-7B-Instruct

--live is the only change needed to get real numbers; everything downstream
(parsers, stats, dashboard) is identical.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict

from inspect_ai import eval as ieval

from evalgrid.harness import PROMPT_FORMATS, bbh_task, gsm8k_task

TEMPS = (0.0, 0.7, 1.0)
PARSERS = ("parse_strict", "parse_flexible", "parse_last_number")


# ---------------------------------------------------------------- statistics
def bootstrap_ci(flags, n_boot=5000, seed=0):
    """Percentile bootstrap 95% CI over items."""
    if not flags:
        return (0.0, 0.0)
    r = random.Random(seed)
    n = len(flags)
    means = []
    for _ in range(n_boot):
        means.append(sum(flags[r.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def mcnemar(a, b):
    """Exact McNemar on paired binary vectors. Returns (b01, b10, p).

    Valid here because both parsers see the SAME completions -- the pairing is
    exact, so this isolates the parser effect with zero model variance."""
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b01, b10, min(1.0, p)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--model", default="mockllm/model")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    offline = not args.live

    suites = [("gsm8k", gsm8k_task, {}),
              ("bbh_date_understanding", bbh_task, {"name": "date_understanding"})]

    tasks, keys = [], []
    for suite, fn, extra in suites:
        for fmt in PROMPT_FORMATS:
            for t in TEMPS:
                tasks.append(fn(fmt=fmt, n=args.n, temperature=t,
                                offline=offline, **extra))
                keys.append((suite, fmt, t))

    print(f"running {len(tasks)} conditions x n={args.n} "
          f"({'OFFLINE STUB' if offline else args.model})")
    logs = ieval(tasks, model=args.model, log_dir="./logs",
                 display="none", log_level="error", max_tasks=8)

    # per-condition, per-parser item-level flags (aligned by sample id)
    cells, items = [], defaultdict(dict)
    for (suite, fmt, temp), lg in zip(keys, logs):
        if lg.status != "success":
            print(f"  FAILED {suite}/{fmt}/{temp}: {lg.error}")
            continue
        vec = {p: [] for p in PARSERS}
        unparsed = {p: 0 for p in PARSERS}
        ids = []
        for s in lg.samples:
            ids.append(s.id)
            for p in PARSERS:
                sc = s.scores[p]
                ok = 1 if sc.value == "C" else 0
                vec[p].append(ok)
                unparsed[p] += int(bool(sc.metadata.get("unparsed")))
        for p in PARSERS:
            lo, hi = bootstrap_ci(vec[p])
            cells.append({
                "suite": suite, "fmt": fmt, "temp": temp, "parser": p,
                "n": len(vec[p]),
                "acc": sum(vec[p]) / len(vec[p]),
                "ci_lo": lo, "ci_hi": hi,
                "unparsed_rate": unparsed[p] / len(vec[p]),
            })
        items[(suite, fmt, temp)] = {"ids": ids, **vec}

    # paired parser contrasts (same completions -> pure parser effect)
    contrasts = []
    for (suite, fmt, temp), v in items.items():
        for a, b in [("parse_strict", "parse_flexible"),
                     ("parse_flexible", "parse_last_number")]:
            b01, b10, p = mcnemar(v[a], v[b])
            contrasts.append({
                "suite": suite, "fmt": fmt, "temp": temp,
                "a": a, "b": b,
                "acc_a": sum(v[a]) / len(v[a]), "acc_b": sum(v[b]) / len(v[b]),
                "only_b": b01, "only_a": b10, "p": p,
            })

    out = {
        "meta": {
            "model": args.model,
            "offline_stub": offline,
            "n_per_condition": args.n,
            "temps": list(TEMPS),
            "formats": list(PROMPT_FORMATS),
            "parsers": list(PARSERS),
            "harness": "inspect_ai",
        },
        "cells": cells,
        "contrasts": contrasts,
    }
    json.dump(out, open(args.out, "w"), indent=1)

    # console summary: the headline spread
    print("\n=== accuracy by condition (GSM8K) ===")
    print(f"{'format':16}{'temp':>6}{'strict':>9}{'flexible':>10}"
          f"{'last_num':>10}{'unparsed':>10}")
    for fmt in PROMPT_FORMATS:
        for t in TEMPS:
            row = {c["parser"]: c for c in cells
                   if c["suite"] == "gsm8k" and c["fmt"] == fmt and c["temp"] == t}
            if not row:
                continue
            print(f"{fmt:16}{t:>6}"
                  f"{row['parse_strict']['acc']:>9.3f}"
                  f"{row['parse_flexible']['acc']:>10.3f}"
                  f"{row['parse_last_number']['acc']:>10.3f}"
                  f"{row['parse_strict']['unparsed_rate']:>10.1%}")
    g = [c for c in cells if c["suite"] == "gsm8k"]
    lo, hi = min(g, key=lambda c: c["acc"]), max(g, key=lambda c: c["acc"])
    print(f"\nsame model, same items. GSM8K spans "
          f"{lo['acc']:.1%} ({lo['fmt']}/T={lo['temp']}/{lo['parser'].split('_',1)[1]}) "
          f"-> {hi['acc']:.1%} ({hi['fmt']}/T={hi['temp']}/{hi['parser'].split('_',1)[1]})"
          f"  = {(hi['acc']-lo['acc'])*100:.1f} pp of pure methodology")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
