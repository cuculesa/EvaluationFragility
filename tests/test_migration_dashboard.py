import json
from pathlib import Path

from evalfrag.dashboard import build_dashboard
from evalfrag.migration import migrate_v1
from evalfrag.schema import validate_results


def test_legacy_migration_and_dashboard_are_safe_and_self_contained(tmp_path: Path) -> None:
    migrated = tmp_path / "results.json"
    dashboard = tmp_path / "dashboard.html"
    migrate_v1(Path("examples/synthetic/results-v1.json"), migrated)
    payload = json.loads(migrated.read_text())
    validate_results(payload)
    build_dashboard(
        results_path=migrated,
        output_path=dashboard,
        title="</script><script>alert(1)</script>",
    )
    text = dashboard.read_text()
    assert "synthetic evaluation" in text
    assert "fonts.googleapis.com" not in text
    assert "https://" not in text
    assert "<script>alert(1)</script>" not in text
    assert "&lt;/script&gt;" in text
