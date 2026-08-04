#!/usr/bin/env python3
"""Render a PDF report from one or more tools/benchmark_three_way.py --output JSON files.

Unofficial, TPC-H-*derived* benchmark report. Not a certified TPC-H result
-- every page carries this disclaimer explicitly, matching the same
caveat tools/benchmark_three_way.py's own printed output and
tools/generate_tpch.py already state.

Usage:
    python3 tools/generate_benchmark_report.py \
        --input sf0.01/report.json sf0.1/report.json sf1/report.json \
        --output benchmark_report.pdf

Each --input file is one tools/benchmark_three_way.py --output run (one
scale factor). Requires matplotlib -- see docker/Dockerfile's
`benchmark-gpu` stage, which installs it alongside pyspark specifically
for this script.
"""

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")  # No display available inside a container.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ENGINE_LABELS = {
    "kernellake-cpu": "KernelLake (CPU)",
    "kernellake-gpu": "KernelLake (GPU)",
    "pyspark": "PySpark",
}
ENGINE_COLORS = {
    "kernellake-cpu": "#4C72B0",
    "kernellake-gpu": "#55A868",
    "pyspark": "#C44E52",
}


def load_reports(paths: list) -> list:
    reports = []
    for path in paths:
        with open(path) as f:
            reports.append(json.load(f))
    return reports


def format_system_stats(system: dict) -> str:
    gpu = system.get("gpu") or {}
    lines = [
        f"CPU: {system.get('cpu_model', 'unknown')} ({system.get('cpu_logical_cores', '?')} logical cores)",
        f"RAM: {system.get('total_ram_gib', '?')} GiB",
        f"GPU: {gpu.get('name', 'none detected')}"
        + (f" ({gpu.get('memory_total', '?')}, driver {gpu.get('driver_version', '?')})" if gpu else ""),
        f"OS: {system.get('os', 'unknown')}",
        f"Python: {system.get('python_version', '?')}  PySpark: {system.get('pyspark_version', '?')}",
    ]
    return "\n".join(lines)


def add_title_page(pdf: PdfPages, reports: list) -> None:
    # verticalalignment="top" anchors each text block's *top* edge at `y`,
    # not its baseline -- much easier to stack multiple blocks without
    # overlap than reasoning about baseline-anchored multi-line text.
    # STATS_LINE_HEIGHT/BLOCK_GAP are deliberately generous fixed fractions
    # of this fixed 11-inch-tall figure (not computed from font metrics) --
    # good enough for the ~5-line system-stats block this always renders,
    # not meant to be a general-purpose text-layout engine.
    STATS_LINE_HEIGHT = 0.022
    BLOCK_GAP = 0.03

    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.92, "KernelLake Three-Way Benchmark", ha="center", va="top", fontsize=20, weight="bold")
    fig.text(
        0.5,
        0.87,
        "KernelLake-CPU vs. KernelLake-GPU vs. PySpark",
        ha="center",
        va="top",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.83,
        "UNOFFICIAL, TPC-H-derived benchmark. NOT a certified TPC-H result.",
        ha="center",
        va="top",
        fontsize=11,
        style="italic",
        color="firebrick",
    )

    y = 0.74
    for report in reports:
        sf = report.get("scale_factor", "?")
        fig.text(0.1, y, f"Scale factor {sf}", fontsize=12, weight="bold", va="top")
        y -= BLOCK_GAP
        stats_text = format_system_stats(report.get("system", {}))
        fig.text(0.1, y, stats_text, fontsize=9, family="monospace", va="top")
        y -= STATS_LINE_HEIGHT * len(stats_text.splitlines()) + BLOCK_GAP

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def add_summary_table_page(pdf: PdfPages, reports: list) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title("Median wall-clock time per query (seconds)", fontsize=13, weight="bold", pad=20)

    rows = []
    for report in reports:
        sf = report.get("scale_factor", "?")
        for result in report.get("results", []):
            if not result.get("validated"):
                rows.append([f"SF{sf}", f"Q{result['query']}", "NOT VALIDATED", "--", "--"])
                continue
            rows.append(
                [
                    f"SF{sf}",
                    f"Q{result['query']}",
                    f"{result['kernellake-cpu']['median_seconds']:.4f}",
                    f"{result['kernellake-gpu']['median_seconds']:.4f}",
                    f"{result['pyspark']['median_seconds']:.4f}",
                ]
            )

    table = ax.table(
        cellText=rows,
        colLabels=["Scale factor", "Query", "KernelLake-CPU", "KernelLake-GPU", "PySpark"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    pdf.savefig(fig)
    plt.close(fig)


def add_chart_page(pdf: PdfPages, reports: list, query_number: int) -> None:
    scale_factors = []
    timings = {engine: [] for engine in ENGINE_LABELS}
    for report in reports:
        sf = report.get("scale_factor", "?")
        result = next((r for r in report.get("results", []) if r.get("query") == query_number), None)
        if result is None or not result.get("validated"):
            continue
        scale_factors.append(f"SF{sf}")
        for engine in ENGINE_LABELS:
            timings[engine].append(result[engine]["median_seconds"])

    if not scale_factors:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6))
    x = range(len(scale_factors))
    width = 0.25
    for i, engine in enumerate(ENGINE_LABELS):
        offsets = [pos + (i - 1) * width for pos in x]
        ax.bar(offsets, timings[engine], width=width, label=ENGINE_LABELS[engine], color=ENGINE_COLORS[engine])
    ax.set_xticks(list(x))
    ax.set_xticklabels(scale_factors)
    ax.set_ylabel("Median wall-clock time (seconds)")
    ax.set_title(f"Q{query_number}: median time by scale factor")
    ax.legend()
    fig.text(
        0.5,
        0.01,
        "UNOFFICIAL, TPC-H-derived benchmark. NOT a certified TPC-H result.",
        ha="center",
        fontsize=8,
        style="italic",
        color="firebrick",
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="One or more benchmark_three_way.py --output files")
    parser.add_argument("--output", required=True, help="Path to write the PDF report")
    args = parser.parse_args()

    reports = load_reports(args.input)
    if not reports:
        print("no input reports given", file=sys.stderr)
        return 1

    query_numbers = sorted({r["query"] for report in reports for r in report.get("results", [])})

    with PdfPages(args.output) as pdf:
        add_title_page(pdf, reports)
        add_summary_table_page(pdf, reports)
        for query_number in query_numbers:
            add_chart_page(pdf, reports, query_number)

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
