# An Evaluation-Fragility Harness for LLM Benchmarks

EvalFrag is a production-oriented [Inspect AI](https://inspect.aisi.org.uk/) experiment for measuring how much a reasoning benchmark score moves when the **model and benchmark items stay fixed** but the evaluation method changes.

It varies:

- prompt format: bare, zero-shot chain-of-thought, explicit output contract, and few-shot with output contract;
- sampling temperature and generation seed;
- answer extraction: strict, flexible, and permissive.

The same completion is scored by all three parsers. That makes parser contrasts exact at the completion level and exposes a failure mode that aggregate accuracy usually hides: the model may contain the right answer while the evaluator fails to read it.

## Production properties

- **Live mode is the default.** Synthetic mode must be selected explicitly and permanently watermarks every artifact.
- **Retryable execution.** Runs use Inspect `eval_set`, sample retries, request retries, task checkpoints, and unique output directories.
- **Stable experimental units.** Sample IDs are hashes of source questions and remain identical across prompt and temperature conditions.
- **Dataset integrity.** Benchmark files are downloaded from their public source repositories, checksummed into `data/manifest.json`, and revalidated before every run.
- **Audit artifacts.** Each run contains the resolved configuration, Git/runtime provenance, raw Inspect logs, a bundled Inspect viewer, sample-level score records, validated aggregate results, failures, a state file, a cryptographic artifact manifest, and a self-contained dashboard.
- **Safer statistics.** Per-seed accuracy uses Wilson intervals. Pooled intervals bootstrap benchmark items after averaging over configured seeds. Temperature and prompt contrasts are paired by stable sample ID.
- **No causal overclaim.** The dashboard does not label flexible-parser movement as “reasoning loss.” It is reported as a score change under a less brittle extractor.
- **No external dashboard dependencies.** The HTML contains its own CSS, JavaScript, validated result payload, and a restrictive content-security policy.
- **Strict completeness gate.** Aggregation is blocked unless every configured condition has exactly the expected items, with no duplicate records or cross-condition item drift.
- **Separate publication gate.** `release-check` rejects synthetic, incomplete, or deployment-unidentified results even when exploratory execution succeeded.

## Install

Python 3.11, 3.12, and 3.13 are supported; 3.12 is recommended for deployment unless your platform standardizes on another supported version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The project pins `inspect-ai==0.3.251` and the evaluator code follows that version’s `eval_set()` `(success, logs)` contract. Update that pin deliberately, run the full test suite, and complete in-flight eval sets before upgrading because eval/task identity can change across Inspect releases.

## Prepare benchmark data

```bash
evalfrag prepare-data --data-dir data
evalfrag validate-data --data-dir data
```

This downloads:

- GSM8K test data from `openai/grade-school-math`;
- BBH date-understanding data and its official CoT prompt from `suzgunmirac/BIG-Bench-Hard`.

The downloaded data is not committed by default. See `DATASETS.md` for provenance and license notes.

## Run against a local vLLM server

Start a vLLM OpenAI-compatible endpoint separately, then:

```bash
export EVALFRAG_MODEL_BASE_URL=http://127.0.0.1:8000/v1
evalfrag run --config configs/default.toml
```

The configured model is `vllm/Qwen/Qwen2.5-7B-Instruct`. Change the model and concurrency limits in the TOML file to match your hardware. Fill in `model_revision`, `tokenizer_revision`, `serving_engine`, `serving_engine_version`, `quantization`, and `chat_template_sha256` before a publishable run. Inspect's current vLLM provider uses a local server architecture; the evaluator should not load the model in the same process.

## Run against a hosted Anthropic model

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
evalfrag run --config configs/live-claude.toml
```

Two provider-specific things to know before pointing a config at `anthropic/...`:

- **Do not set both `temperature` and `top_p`.** Anthropic's API rejects a
  request that specifies both (`400 invalid_request_error`), even when
  `top_p` is left at its nominal default of `1.0`. `tasks.py` already omits
  `top_p` automatically for any `model` starting with `anthropic/` — if you
  fork or rewrite that code path, keep that behavior, or every task in the
  grid will fail identically and immediately.
- **Anthropic is not in `SEED_SUPPORTED_PROVIDERS`** (see `config.py`), so a
  config with more than one entry in `seeds` will fail validation unless
  `allow_unseeded_provider = true` is set. Prefer a single seed for
  Anthropic runs rather than opting in to that flag, since a generation seed
  isn't confirmed to be honored the same way it is for OpenAI/vLLM/etc.

Fill in `model_revision`, `serving_engine`, etc. as empty strings for a
hosted API model — there is no local revision or serving engine to report,
and inventing values would misrepresent the deployment.

Before running the full grid against a live Anthropic model, run a small,
cheap smoke test first rather than committing straight to the full `n`. Copy
the live config and drop `n` down (e.g. `n=10`), keeping every other field
identical:

```bash
cp configs/live-claude.toml configs/live-claude-smoke.toml
# edit configs/live-claude-smoke.toml: set n = 10, and give it a distinct
# `name` (e.g. "evalfrag-live-claude-smoke") so its run directory doesn't
# get confused with a real full-size run

evalfrag run --config configs/live-claude-smoke.toml
```

At `n=10` this is 24 tasks x 10 samples = 240 real API calls instead of the
full grid's 4,800 — enough to confirm the model string, credentials,
provider-specific request shape (see the `temperature`/`top_p` note above),
and the full aggregation/dashboard pipeline all work end to end, for a small
fraction of the cost and time of the full run.

`evalfrag run --config configs/live-claude-smoke.toml` prints a live task
table (one row per prompt-format/temperature/seed condition, 24 rows for the
default suite/format/temperature grid), each ticking up from `0/10` toward
`10/10` as samples complete, and creates a fresh directory at
`runs/<name>-<timestamp>-<hash>/` — the exact name comes from the `name`
field in the config, so give the smoke config a distinct `name` (as above)
to keep its run directory from being confused with a real full-size run.
On success it ends with `Completed all tasks in '<run-dir>/inspect-logs'
successfully` and writes `results.json` and `dashboard.html` into that same
run directory automatically — no separate `evalfrag dashboard` call needed.
If it instead reports `N evaluation tasks failed`, check
`runs/<run-dir>/failures.json` first, and see "Recovering a partially failed
run" below rather than just re-running the same command, since a re-run
starts an entirely new grid rather than resuming.

Only move on to the full-size config once the smoke run produces a clean
`results.json` and `dashboard.html` with real (non-`n/a`) accuracy numbers:

```bash
open runs/<name>-<timestamp>-<hash>/dashboard.html
```

`dashboard.html` is generated automatically as the last step of a
successful `evalfrag run` — no separate command needed in the normal case.
If you need to rebuild it explicitly (e.g. after editing `dashboard.py`, or
if you're regenerating it from a `results.json` produced some other way,
such as by `scripts/resume_run.py`), use:

```bash
evalfrag dashboard \
  --results runs/<name>-<timestamp>-<hash>/results.json \
  --output runs/<name>-<timestamp>-<hash>/dashboard.html
```

## Synthetic pipeline verification

```bash
evalfrag run --config configs/synthetic.toml
```

Synthetic scores verify prompt rendering, parsing, statistics, aggregation, result validation, and dashboard generation. They are not evidence about any model.

## Run artifacts

Each run is written to `runs/<run-id>/`:

```text
config.resolved.json      exact configuration used
provenance.json           config hash, Git state, runtime, dataset hashes
failures.json             failed task summary; empty on success
run_state.json            running, complete, or failed terminal state
inspect-logs/             native Inspect .eval logs
inspect-viewer/           static Inspect log viewer bundle
records.jsonl             sample-level parser outcomes and output hashes
results.json              schema-v2 aggregate results
dashboard.html            self-contained methodology dashboard
artifacts.manifest.json    SHA-256 and byte size for every generated file
```

Completion text is omitted from `records.jsonl` by default, but it remains present in the access-controlled Inspect logs. Set `store_completion_text=true` only when the additional duplication is useful and approved.

> **`eval_set()` sample-loading caveat.** The log objects `eval_set()`
> returns in memory are not guaranteed to have their `samples` list fully
> populated even when `log.status == "success"` and the samples are present
> and correctly scored on disk. Trusting the in-memory object directly can
> silently produce zero records for a condition that actually succeeded,
> which then fails the completeness gate for the wrong reason (looks like a
> missing/failed condition; is actually a stale in-memory reference).
> `runner.py` re-reads every log fresh from disk with `read_eval_log()`
> immediately after `eval_set()` returns, specifically to guard against
> this. Any code path that consumes `eval_set()`'s return value directly
> (including ad hoc recovery scripts) should do the same rather than assume
> the returned objects are complete.

## Recovering a partially failed run

`evalfrag run` mints a new `run_id` (and therefore a new `inspect-logs`
directory) on every invocation. Re-running the same command after a partial
failure does **not** resume — it restarts the entire grid, re-attempting
conditions that already succeeded and re-billing every API call in a live
run, `runtime.checkpoint = true` notwithstanding (that flag only lets
Inspect resume samples *within* one `log_dir`; the CLI never points two
invocations at the same one).

To resume a specific failed run without re-paying for work that already
succeeded, use `scripts/resume_run.py`, which calls `eval_set()` directly
against the existing run's `inspect-logs/` directory and then completes the
same aggregation/validation/dashboard steps `evalfrag run` performs
internally:

```bash
python3 scripts/resume_run.py \
  --config configs/live-claude.toml \
  --run-dir runs/<the-run-id-to-resume>
```

It writes `records.jsonl`, `results.json`, and `dashboard.html` into that
same run directory once every condition succeeds, and refuses to aggregate
(printing the still-failing tasks instead) if anything remains incomplete.

## Validate or rebuild artifacts

```bash
evalfrag validate-results --results runs/<run-id>/results.json
evalfrag verify-artifacts --run-dir runs/<run-id>
evalfrag release-check --results runs/<run-id>/results.json

evalfrag dashboard \
  --results runs/<run-id>/results.json \
  --output runs/<run-id>/dashboard.html
```

## Migrating the original prototype

The uploaded prototype used a summary-only v1 JSON schema. It can be migrated for display, but it cannot recover sample-level paired intervals:

```bash
evalfrag migrate-v1 \
  --input examples/synthetic/results-v1.json \
  --output examples/synthetic/results.json

evalfrag dashboard \
  --results examples/synthetic/results.json \
  --output examples/synthetic/dashboard.html
```

## Tests and quality gates

```bash
make test
make lint
make check
```

CI runs parser, configuration, dataset-integrity, statistics, schema, runner-contract, release-gate, artifact-integrity, and dashboard tests on Python 3.11, 3.12, and 3.13. Live model calls are intentionally excluded from CI; production deployment must add a small endpoint smoke test using an approved model and credentials.

## Interpretation rules

1. Report the full method: model revision, provider, prompt format, decoder settings, seed set, parser, sample selection seed, and harness versions.
2. Report unparsed-output rate beside accuracy.
3. Treat permissive extraction as a sensitivity analysis, not automatically as the authoritative score.
4. Do not generalize item-bootstrap intervals to model families, providers, or prompts not included in the run.
5. Review raw completions behind large parser gaps before publication.
6. Do not compare a synthetic run to a live run.
7. Confidence intervals are clamped to always contain their point estimate.
   A Wilson interval is mathematically guaranteed to contain the accuracy it
   describes, but floating-point rounding in the formula can place a bound a
   fraction below/above it (observed directly: `ci_hi = 0.9999999999999999`
   for a perfect-accuracy `n=10` cell where `acc = 1.0`); a percentile
   bootstrap interval has no such guarantee at all, especially at small `n`.
   `aggregate.py` clamps both cases (`lo = min(lo, acc)`, `hi = max(hi, acc)`)
   so the schema's containment check reflects a real invariant rather than
   failing on rounding artifacts or occasionally-skewed bootstrap intervals.
   This matters most at small sample sizes (e.g. `n=10` smoke-test configs)
   and near-boundary accuracies (near 0.0 or 1.0).

## Security

See `SECURITY.md`, `PRODUCTION_READINESS.md`, and `VERIFICATION.md`. In particular, keep Inspect logs private unless they have been reviewed, because logs contain prompts and model completions even when `records.jsonl` stores only hashes.

## Container

Build with an approved Python base-image digest and release metadata:

```bash
docker build \
  --build-arg PYTHON_IMAGE=python:3.12-slim@sha256:<approved-digest> \
  --build-arg VCS_REF=$(git rev-parse HEAD) \
  --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -t evalfrag:1.0.0 .
```

Mount `/app/data` and `/app/runs` on persistent, access-controlled storage. The image runs as the unprivileged `evalfrag` user and contains the CLI, configuration templates, and runtime dependencies, but no model weights or credentials.
