from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .dashboard import build_dashboard
from .datasets import prepare_data, validate_data
from .migration import migrate_v1
from .release import release_report, verify_artifact_manifest
from .runner import run_experiment
from .schema import validate_results
from .util import atomic_write_text, read_json_object


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evalfrag")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data", help="download and checksum benchmark data")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("validate-data", help="validate local datasets and manifest")
    p.add_argument("--data-dir", type=Path, default=Path("data"))

    p = sub.add_parser("run", help="execute the configured live or synthetic grid")
    p.add_argument("--config", type=Path, default=Path("configs/default.toml"))

    p = sub.add_parser("dashboard", help="build a self-contained dashboard")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("dashboard.html"))
    p.add_argument("--title", default="Evaluation methodology sensitivity")
    p.add_argument("--high-unparsed-threshold", type=float, default=0.10)

    p = sub.add_parser("validate-results", help="validate a results.json artifact")
    p.add_argument("--results", type=Path, required=True)

    p = sub.add_parser("migrate-v1", help="convert the legacy summary schema to v2")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("release-check", help="check whether live results are publication-ready")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--report", type=Path)

    p = sub.add_parser("verify-artifacts", help="verify every file in a completed run")
    p.add_argument("--run-dir", type=Path, required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare-data":
            manifest = prepare_data(args.data_dir.resolve(), overwrite=args.overwrite)
            print(json.dumps(manifest, indent=2))
        elif args.command == "validate-data":
            validate_data(args.data_dir.resolve())
            print("dataset validation passed")
        elif args.command == "run":
            config = load_config(args.config)
            run_dir = run_experiment(config, project_root=Path.cwd())
            print(run_dir)
        elif args.command == "dashboard":
            build_dashboard(
                results_path=args.results.resolve(),
                output_path=args.output.resolve(),
                title=args.title,
                high_unparsed_threshold=args.high_unparsed_threshold,
            )
            print(args.output.resolve())
        elif args.command == "validate-results":
            payload = read_json_object(args.results)
            validate_results(payload)
            print("results validation passed")
        elif args.command == "migrate-v1":
            migrate_v1(args.input.resolve(), args.output.resolve())
            print(args.output.resolve())
        elif args.command == "release-check":
            payload = read_json_object(args.results)
            report = release_report(payload)
            rendered = json.dumps(report, indent=2) + "\n"
            if args.report:
                atomic_write_text(args.report.resolve(), rendered)
            print(rendered, end="")
            return 0 if report["ready"] else 3
        elif args.command == "verify-artifacts":
            report = verify_artifact_manifest(args.run_dir.resolve())
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 3
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
