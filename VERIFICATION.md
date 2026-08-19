# Verification record

Verified on 2026-07-30 in the artifact-build environment:

- `python -m compileall -q src tests`: passed.
- `PYTHONPATH=src pytest --cov=evalfrag --cov-fail-under=80 -q`: 45 tests passed; 84.74% total coverage.
- Legacy v1 migration and schema-v2 dashboard generation: passed.
- Schema-v2 JSON Schema synchronization check: passed.
- Original uploaded prototype copies match their source files byte-for-byte: passed.
- Self-contained dashboard JavaScript syntax check with Node.js 22: passed.
- Publication-gate and artifact-tamper tests: passed.
- PEP 517 wheel build: passed; two builds with the same `SOURCE_DATE_EPOCH` were byte-identical.
- Wheel-installed CLI migration, validation, dashboard, release-check failure-path, and artifact-verification smoke tests: passed.

Not executed in this environment:

- a live Inspect evaluation against a model endpoint;
- GPU/vLLM serving;
- Ruff and mypy, because those binaries were unavailable from the build environment’s package mirror.

The CI workflow runs Ruff, Ruff formatting, mypy, pytest with coverage, schema synchronization, dashboard smoke tests, and package installation in a normal GitHub-hosted environment. Deployment must additionally run a small approved live endpoint smoke test before the full grid.

## Independent re-check (2026-08-18)

This is a second, independent pass over the artifact, done in a sandbox with no network
access and neither `pydantic` nor `inspect-ai` installed. It should not be read as a
replacement for the checks above — it covers less — but everything listed here was actually
executed, not inferred from the source.

- `python -m compileall -q src tests`: passed.
- `parsers.py` and `statistics.py` have no dependency on `pydantic`/`inspect-ai`, so they were
  imported and exercised directly (strict/flexible/last-number math and multiple-choice
  parsing, `normalize_number`, `wilson_interval` against a known reference value, exact
  McNemar on both a null and a maximally discordant pair, and `benjamini_hochberg`
  monotonicity/ordering). All matched hand-computed expected values.
- Everything that imports `pydantic` (`config.py`, `schema.py`, `dashboard.py`, `runner.py`,
  `release.py`, `migration.py`, `cli.py`) could not be executed here, so the full pytest
  suite, `scripts/smoke_no_network.sh`, the wheel build, ruff, and mypy remain unverified in
  this pass. Re-run `make check` and `scripts/smoke_no_network.sh` in an environment with
  network access (or a populated wheel cache) before relying on this artifact.
- **Repo hygiene gap found and fixed:** `.gitignore` excluded `.venv/` but the delivered tree
  contained a stray `venv/` (a local, incompatible Python 3.9 virtualenv — this project
  requires >=3.11), an uncommitted-but-shipped `.idea/` directory, and `.DS_Store` files.
  `.idea/workspace.xml` also leaked a local absolute path (`/Users/ana/Downloads/...`),
  which is a minor but real information leak for a "production" artifact. `venv/`, `.idea/`,
  and `.DS_Store` were removed from this delivery, and `.gitignore` now excludes `venv/`,
  `.idea/`, `.DS_Store`, `dist/`, `build/`, and `*.egg-info/`.
- No other functional bugs were found in this pass. The source (config validation, dataset
  pinning/checksumming, statistics, schema invariants, dashboard escaping/CSP, artifact
  manifest tamper-checking, release gate) reads as carefully written and internally
  consistent; the claims in `PRODUCTION_READINESS.md` and `PROJECT_AUDIT.md` are plausible
  based on source review, but — per the point above — were not independently re-executed
  end-to-end here.
