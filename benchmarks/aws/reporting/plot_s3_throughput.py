#!/usr/bin/env python3
"""Combines every run_s3_throughput_all_instances.sh output
(s3-throughput-<label>.json, one per instance -- see
scripts/measure_s3_throughput.sh) into a single chart: concurrency on the
x-axis, throughput on the y-axis, one line per instance/label. Answers the
question the per-instance JSON files can't answer alone -- how do the
KernelLake, Spark, and DuckDB hosts' real S3 read paths compare against
each other, not just against their own instance type's documented network
rating (see terraform/variables.tf's spark_instance_type comment: no exact
network-rating match exists in this price class, so this plot is the
empirical substitute the user chose over a closer spec match).

Requires matplotlib.

Usage:
    python3 plot_s3_throughput.py --input-dir /path/to/run/dir \\
        --output s3-throughput.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display available on a benchmark host/CI runner.
import matplotlib.pyplot as plt

DISCLAIMER = "UNOFFICIAL, TPC-H-derived benchmark. NOT a certified TPC-H result."

# Same per-engine palette as reporting/generate_pdf_report.py's
# add_latency_chart_page() -- keeps "green=KernelLake, red=Spark,
# purple=DuckDB" consistent across every chart in this benchmark's report,
# not a fresh color choice for this one plot. Matched by label *prefix*
# since KernelLake labels are "kernellake-0", "kernellake-1", ... (one per
# replica, see run_s3_throughput_all_instances.sh) rather than a single
# fixed string.
ENGINE_COLORS = {
    "kernellake": "#55A868",
    "spark": "#C44E52",
    "duckdb": "#8172B2",
}
FALLBACK_COLOR = "#4C72B0"

# Distinguishes multiple replicas of the same engine (currently only
# possible for kernellake, whose replica count is
# kernellake_instance_count -- see terraform/variables.tf) by shading the
# base engine color rather than assigning each replica an unrelated hue,
# so they still read as "the same engine" at a glance.
REPLICA_ALPHAS = (1.0, 0.65, 0.45, 0.3)


def color_for_label(label: str, replica_index: int) -> tuple[str, float]:
    prefix = label.split("-")[0]
    color = ENGINE_COLORS.get(prefix, FALLBACK_COLOR)
    alpha = REPLICA_ALPHAS[replica_index % len(REPLICA_ALPHAS)]
    return color, alpha


def load_runs(input_dir: Path) -> list[dict]:
    paths = sorted(input_dir.glob("s3-throughput-*.json"))
    if not paths:
        print(f"No s3-throughput-*.json files found under {input_dir}", file=sys.stderr)
        sys.exit(1)
    return [json.loads(p.read_text()) for p in paths]


def plot(runs: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))

    replica_index_by_prefix: dict[str, int] = {}
    for run in runs:
        label = run["label"]
        prefix = label.split("-")[0]
        replica_index = replica_index_by_prefix.get(prefix, 0)
        replica_index_by_prefix[prefix] = replica_index + 1
        color, alpha = color_for_label(label, replica_index)

        levels = sorted(run["levels"], key=lambda lv: lv["concurrency"])
        x = [lv["concurrency"] for lv in levels]
        y = [lv["throughput_mbps"] for lv in levels]
        ax.plot(x, y, marker="o", label=label, color=color, alpha=alpha, linewidth=2)

        if run.get("plateau_concurrency") is not None:
            plateau_y = next(
                lv["throughput_mbps"] for lv in levels if lv["concurrency"] == run["plateau_concurrency"]
            )
            ax.scatter(
                [run["plateau_concurrency"]], [plateau_y],
                color=color, alpha=alpha, s=90, zorder=5, edgecolors="black", linewidths=0.8,
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent GET requests")
    ax.set_ylabel("S3 read throughput (MB/s)")
    ax.set_title("Raw S3 GET throughput by instance")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.text(
        0.5, 0.01,
        "Circled points mark each line's measured plateau concurrency. " + DISCLAIMER,
        ha="center", fontsize=8, style="italic", color="firebrick",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Directory of s3-throughput-<label>.json files")
    parser.add_argument("--output", required=True, help="Output image path (e.g. s3-throughput.png)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    runs = load_runs(input_dir)
    plot(runs, Path(args.output))
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
