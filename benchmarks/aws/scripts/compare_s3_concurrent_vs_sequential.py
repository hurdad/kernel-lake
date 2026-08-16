#!/usr/bin/env python3
"""Compares run_s3_throughput_all_instances.sh's two modes against each
other: a plain (sequential, one host at a time) run's
s3-throughput-<label>.json files vs. a --concurrent run's
s3-throughput-<label>-concurrent.json files, both pointed at the same
--output-dir.

Answers a real question raised about this benchmark's methodology: does
running the KernelLake/Spark/DuckDB legs' own S3 reads *at the same time*
see lower throughput than measuring each host in isolation -- i.e. do they
actually interfere with each other, or does each instance's independent
network allocation (and the VPC Gateway Endpoint's own lack of a shared
bandwidth cap) mean concurrent measurement is free? AWS's documented
per-instance-bandwidth/Gateway-Endpoint architecture says it should be
free; this makes that an empirical answer instead of an argument from
documentation.

Usage:
    # Two separate invocations into the same directory, one sequential
    # (default), one --concurrent:
    ../scripts/run_s3_throughput_all_instances.sh --scale-factor 1000 --output-dir "$RUN_DIR"
    ../scripts/run_s3_throughput_all_instances.sh --scale-factor 1000 --output-dir "$RUN_DIR" --concurrent

    python3 compare_s3_concurrent_vs_sequential.py --input-dir "$RUN_DIR"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Same 10% threshold measure_s3_throughput.sh's own plateau heuristic
# uses for "stopped meaningfully helping" -- reused here for "meaningfully
# worse under concurrent load", so both scripts agree on what counts as
# noise vs. a real effect.
INTERFERENCE_THRESHOLD = 0.10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    sequential_files = sorted(
        p for p in input_dir.glob("s3-throughput-*.json") if not p.stem.endswith("-concurrent")
    )
    if not sequential_files:
        print(f"No sequential s3-throughput-*.json files found in {input_dir} -- run "
              f"run_s3_throughput_all_instances.sh (without --concurrent) first.", file=sys.stderr)
        return 1

    rows = []
    missing_concurrent = []
    for seq_path in sequential_files:
        label = seq_path.stem.removeprefix("s3-throughput-")
        conc_path = input_dir / f"s3-throughput-{label}-concurrent.json"
        if not conc_path.exists():
            missing_concurrent.append(label)
            continue
        seq = json.loads(seq_path.read_text())
        conc = json.loads(conc_path.read_text())
        seq_mbps = seq["max_throughput_mbps"]
        conc_mbps = conc["max_throughput_mbps"]
        delta = (conc_mbps - seq_mbps) / seq_mbps
        rows.append({
            "label": label,
            "sequential_mbps": seq_mbps,
            "concurrent_mbps": conc_mbps,
            "delta_pct": delta * 100,
            "interference": delta < -INTERFERENCE_THRESHOLD,
        })

    if missing_concurrent:
        print(f"No matching --concurrent run found for: {', '.join(missing_concurrent)} -- run "
              f"run_s3_throughput_all_instances.sh --concurrent against the same --output-dir too.",
              file=sys.stderr)

    if not rows:
        print("No label has both a sequential and a --concurrent result to compare.", file=sys.stderr)
        return 1

    print(f"{'label':<15} {'sequential':>14} {'concurrent':>14} {'delta':>9}  verdict")
    print("-" * 70)
    any_interference = False
    for r in rows:
        verdict = "POSSIBLE INTERFERENCE" if r["interference"] else "no meaningful difference"
        any_interference = any_interference or r["interference"]
        print(f"{r['label']:<15} {r['sequential_mbps']:>10.1f} MB/s {r['concurrent_mbps']:>10.1f} MB/s "
              f"{r['delta_pct']:>+8.1f}%  {verdict}")

    print()
    if any_interference:
        print(f"At least one host's concurrent throughput dropped more than "
              f"{INTERFERENCE_THRESHOLD * 100:.0f}% below its sequential measurement -- real interference "
              f"between concurrently-benchmarked instances, not just AWS's documented per-instance "
              f"bandwidth architecture. Worth investigating (S3 per-prefix request-rate limits are the "
              f"most likely mechanism) before running the real KernelLake/Spark/DuckDB legs concurrently.")
        return 1
    print(f"No host's concurrent throughput dropped more than {INTERFERENCE_THRESHOLD * 100:.0f}% below "
          f"its sequential measurement -- consistent with each instance having its own independent network "
          f"allocation and no shared Gateway Endpoint bandwidth cap. Safe to run the real benchmark legs "
          f"concurrently to save wall-clock time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
