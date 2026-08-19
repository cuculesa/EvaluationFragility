#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src"

python -m compileall -q src tests
pytest --cov=evalfrag --cov-report=term-missing --cov-fail-under=80 -q

python -m evalfrag.cli migrate-v1 \
  --input examples/synthetic/results-v1.json \
  --output examples/synthetic/results.json
python -m evalfrag.cli validate-results \
  --results examples/synthetic/results.json
python -m evalfrag.cli dashboard \
  --results examples/synthetic/results.json \
  --output examples/synthetic/dashboard.html \
  --title "Synthetic pipeline verification"

# Synthetic evidence must never pass the publication gate.
set +e
python -m evalfrag.cli release-check \
  --results examples/synthetic/results.json >/tmp/evalfrag-release-check.log 2>&1
release_status=$?
set -e
if [[ "$release_status" -ne 3 ]]; then
  cat /tmp/evalfrag-release-check.log
  echo "expected synthetic release-check to exit 3, got $release_status" >&2
  exit 1
fi

# Keep the browser artifact self-contained and JavaScript-parseable.
if grep -Eiq '<script[^>]+src=|<link[^>]+href="https?://' examples/synthetic/dashboard.html; then
  echo "dashboard unexpectedly depends on an external script or stylesheet" >&2
  exit 1
fi
if command -v node >/dev/null 2>&1; then
  python - <<'PY'
from pathlib import Path
import re
html = Path("examples/synthetic/dashboard.html").read_text(encoding="utf-8")
scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", html, flags=re.I | re.S)
Path("/tmp/evalfrag-dashboard.js").write_text("\n".join(scripts), encoding="utf-8")
PY
  node --check /tmp/evalfrag-dashboard.js
fi

echo "offline smoke checks passed"
