import json
from pathlib import Path

from evalfrag.cli import main


def test_cli_migration_dashboard_and_release_failure(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    migrated = tmp_path / "results.json"
    dashboard = tmp_path / "dashboard.html"
    report = tmp_path / "release.json"

    assert (
        main(
            [
                "migrate-v1",
                "--input",
                str(root / "examples/synthetic/results-v1.json"),
                "--output",
                str(migrated),
            ]
        )
        == 0
    )
    assert main(["validate-results", "--results", str(migrated)]) == 0
    assert (
        main(
            [
                "dashboard",
                "--results",
                str(migrated),
                "--output",
                str(dashboard),
            ]
        )
        == 0
    )
    assert dashboard.is_file()
    assert (
        main(
            [
                "release-check",
                "--results",
                str(migrated),
                "--report",
                str(report),
            ]
        )
        == 3
    )
    assert json.loads(report.read_text())["ready"] is False


def test_cli_reports_bad_results(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert main(["validate-results", "--results", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err
