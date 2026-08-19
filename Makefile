.PHONY: test lint typecheck check package synthetic-example offline-smoke clean

test:
	PYTHONPATH=src pytest --cov=evalfrag --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy src/evalfrag

package:
	rm -rf dist build
	python -m pip wheel . --no-deps -w dist

synthetic-example:
	PYTHONPATH=src python -m evalfrag.cli migrate-v1 --input examples/synthetic/results-v1.json --output examples/synthetic/results.json
	PYTHONPATH=src python -m evalfrag.cli dashboard --results examples/synthetic/results.json --output examples/synthetic/dashboard.html

offline-smoke:
	./scripts/smoke_no_network.sh

check: lint typecheck test package synthetic-example

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
