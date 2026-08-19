# Evaluation card

## Intended use

Measure sensitivity of exact-match reasoning scores to prompt format, decoder temperature, generation seed, and answer extraction.

## Not intended to establish

- broad model safety or capability;
- causal “reasoning loss” from temperature changes;
- superiority of one model over another unless both are run under the same registered design;
- population-level uncertainty beyond the benchmark items and finite generation seeds used.

## Experimental unit

A stable benchmark question ID. Generation seeds are repeated observations for each item and condition. Pooled confidence intervals resample item-level seed averages.

## Primary outputs

- exact-match accuracy by condition and parser;
- unparsed-output rate;
- paired parser contrasts on identical completions;
- paired temperature and prompt-format contrasts on identical benchmark items;
- raw Inspect logs for audit.

## Known limitations

- Exact-match parsers can both miss equivalent answers and accept accidental matches.
- Provider seed behavior varies. Multi-seed runs are rejected for providers not known to support seeds unless explicitly overridden.
- The permissive “last number/option” parser is intentionally vulnerable to trailing irrelevant text.
- Multiple testing correction is applied to parser contrasts, not to every possible exploratory dashboard comparison.
