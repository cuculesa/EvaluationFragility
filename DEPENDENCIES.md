# Dependency policy

## Runtime

- `inspect-ai==0.3.251` is pinned exactly because eval-set return values, retry semantics, log formats, and task identity are part of the evaluator contract.
- `pydantic>=2.10,<3` is constrained to the current major version for strict configuration and artifact schemas.
- Python is constrained to `>=3.11,<3.14`; the production container uses Python 3.12.

The wheel does not bundle model-serving software. A vLLM or other approved endpoint is deployed separately and recorded in the run metadata.

## Development

Ruff, mypy, pytest, and pytest-cov are constrained by major version in `pyproject.toml`. CI runs all quality gates on Python 3.11, 3.12, and 3.13.

## Release practice

- Build in a clean environment.
- Capture the resolved dependency inventory from the production image (`python -m pip freeze`).
- Scan the image and wheel with the organization’s approved SCA/vulnerability tooling.
- Pin the base image by digest in the release build, for example by passing `--build-arg PYTHON_IMAGE=python:3.12-slim@sha256:<approved-digest>`.
- Review and test any Inspect AI upgrade as an evaluator change, not routine dependency maintenance.
