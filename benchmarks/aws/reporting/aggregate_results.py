#!/usr/bin/env python3
"""Merges raw JSON output from runner/aws_benchmark_runner.py (one file per
scale factor) and runner/scaling_test.py (one file per replica count) into
one combined dataset for generate_report.py/generate_dashboard.py.

Usage:
    python3 aggregate_results.py \\
        --benchmark-results results-sf100.json results-sf1000.json \\
        --scaling-results scaling-1replica.json scaling-2replicas.json scaling-4replicas.json scaling-8replicas.json \\
        --output aggregated.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-results", nargs="*", default=[])
    parser.add_argument("--scaling-results", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    benchmark_runs = []
    for path in args.benchmark_results:
        data = json.loads(Path(path).read_text())
        data["_source_file"] = path
        benchmark_runs.append(data)

    scaling_runs = []
    for path in args.scaling_results:
        data = json.loads(Path(path).read_text())
        data["_source_file"] = path
        scaling_runs.append(data)
    scaling_runs.sort(key=lambda r: r["replica_count"])

    if not benchmark_runs and not scaling_runs:
        print("WARNING: no input files given -- output will be empty.", file=sys.stderr)

    aggregated = {
        "benchmark_runs": benchmark_runs,
        "scaling_runs": scaling_runs,
    }
    Path(args.output).write_text(json.dumps(aggregated, indent=2))
    print(
        f"Wrote {args.output}: {len(benchmark_runs)} benchmark run(s), {len(scaling_runs)} scaling run(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
