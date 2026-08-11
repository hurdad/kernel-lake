#!/usr/bin/env python3
"""Renders a PDF version of the same AWS benchmark report generate_report.py
produces as Markdown/CSV -- title page, the written analysis/caveats (not
just raw tables), then the three headline tables (latency speedup, cost
per completed query, concurrency) plus the two supporting tables (scan
throughput, cost efficiency per query), each as its own page, plus a
latency-speedup bar chart per query.

Deliberately reuses generate_report.py's own table-building functions
(latency_speedup_table(), cost_per_query_table(), etc.) rather than
re-deriving the numbers -- this script is a rendering target, not a second
source of truth. A sibling to tools/generate_benchmark_report.py (that one
renders the *local* single-machine three-way benchmark's PDF) -- same
matplotlib PdfPages approach, different data shape, not a shared codepath.

Requires matplotlib.

Usage:
    python3 generate_pdf_report.py --input aggregated.json \\
        --cost-json cost.json [--duckdb-results duckdb-results.json] \\
        --output report.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display available on a benchmark host/CI runner.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from generate_report import (  # noqa: E402
    concurrency_table,
    cost_efficiency_table,
    cost_per_query_table,
    latency_speedup_table,
    merge_duckdb_results,
    merge_pyspark_results,
    scan_throughput_table,
)

DISCLAIMER = "UNOFFICIAL, TPC-H-derived benchmark. NOT a certified TPC-H result."


def add_title_page(pdf: PdfPages, data: dict, cost_data: dict | None) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.92, "KernelLake AWS Benchmark Report", ha="center", va="top", fontsize=20, weight="bold")
    fig.text(0.5, 0.87, DISCLAIMER, ha="center", va="top", fontsize=11, style="italic", color="firebrick")

    scale_factors = sorted({run["scale_factor"] for run in data["benchmark_runs"]})
    fig.text(
        0.5, 0.80,
        f"Scale factor(s): {', '.join(str(sf) for sf in scale_factors)}",
        ha="center", va="top", fontsize=12,
    )
    fig.text(
        0.5, 0.76,
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        ha="center", va="top", fontsize=10, color="dimgray",
    )
    if cost_data is not None:
        fig.text(
            0.5, 0.71,
            f"Real total run cost: ${cost_data.get('total_usd', 0):.2f}",
            ha="center", va="top", fontsize=11,
        )

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    """A page of written analysis/caveats, not a table -- the same prose
    generate_report.py's Markdown output carries, so a PDF-only reader
    gets the same "why do these numbers look like this" context, not just
    raw numbers."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, title, ha="center", va="top", fontsize=15, weight="bold")

    y = 0.88
    for para in paragraphs:
        # Naive fixed-width wrap (~95 chars/line at this font size/page
        # width) -- good enough for short caveat paragraphs, not a real
        # text-layout engine.
        wrapped = []
        for line in para.split("\n"):
            words = line.split(" ")
            current = ""
            for word in words:
                if len(current) + len(word) + 1 > 95:
                    wrapped.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()
            wrapped.append(current)
        text = "\n".join(wrapped)
        fig.text(0.08, y, text, fontsize=9.5, va="top", wrap=True)
        y -= 0.022 * (len(wrapped) + 1) + 0.03

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages, rows: list[dict], title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=13, weight="bold", pad=20)

    if not rows:
        fig.text(0.5, 0.5, "No data.", ha="center", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)
        return

    columns: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    def fmt(v) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    cell_text = [[fmt(row.get(c)) for c in columns] for row in rows]
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7 if len(columns) > 8 else 8.5)
    table.auto_set_column_width(col=list(range(len(columns))))
    table.scale(1, 1.4)
    pdf.savefig(fig)
    plt.close(fig)


def add_latency_chart_page(pdf: PdfPages, speedup_rows: list[dict], mode: str) -> None:
    rows = [r for r in speedup_rows if r["mode"] == mode]
    if not rows:
        return

    engines = ["kernellake_median_seconds", "pyspark_median_seconds", "duckdb_median_seconds"]
    labels = {"kernellake_median_seconds": "KernelLake", "pyspark_median_seconds": "PySpark", "duckdb_median_seconds": "DuckDB"}
    colors = {"kernellake_median_seconds": "#55A868", "pyspark_median_seconds": "#C44E52", "duckdb_median_seconds": "#8172B2"}
    present = [e for e in engines if any(r.get(e) is not None for r in rows)]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(rows))
    n = len(present)
    width = 0.8 / n
    for i, engine in enumerate(present):
        offset = (i - (n - 1) / 2) * width
        offsets = [pos + offset for pos in x]
        values = [r.get(engine) or 0.0 for r in rows]
        ax.bar(offsets, values, width=width, label=labels[engine], color=colors[engine])
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"SF{r['scale_factor']} Q{r['query']}" for r in rows], rotation=45, ha="right")
    ax.set_ylabel("Median wall-clock time (seconds)")
    ax.set_title(f"Latency by query ({mode})")
    ax.legend()
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=8, style="italic", color="firebrick")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="aggregate_results.py's output")
    parser.add_argument("--cost-json", default=None, help="runner/cost_model.py's --output")
    parser.add_argument("--duckdb-results", nargs="*", default=[], help="runner/duckdb_query_loop.py's --output file(s)")
    parser.add_argument("--pyspark-results", nargs="*", default=[], help="runner/pyspark_query_loop.py's --output file(s)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    cost_data = json.loads(Path(args.cost_json).read_text()) if args.cost_json else None
    duckdb_results = [json.loads(Path(p).read_text()) for p in args.duckdb_results]
    merge_duckdb_results(data["benchmark_runs"], duckdb_results)
    pyspark_results = [json.loads(Path(p).read_text()) for p in args.pyspark_results]
    merge_pyspark_results(data["benchmark_runs"], pyspark_results)

    speedup_rows = latency_speedup_table(data["benchmark_runs"])
    cost_rows = cost_per_query_table(data["benchmark_runs"], cost_data)
    concurrency_rows = concurrency_table(data["scaling_runs"])
    throughput_rows = scan_throughput_table(data["benchmark_runs"])
    efficiency_rows = cost_efficiency_table(data["benchmark_runs"], cost_data)

    unsupported = data["benchmark_runs"][0]["unsupported_queries"] if data["benchmark_runs"] else {}
    has_duckdb = any("duckdb_median_seconds" in row for row in speedup_rows)
    duckdb_has_cost = cost_data is not None and cost_data.get("duckdb_usd", 0) > 0

    coverage_text = "6 of TPC-H's 22 queries run today (Q1, Q3, Q6, Q12, Q14, Q19). The other 16 are blocked:\n" + "\n".join(
        f"- Q{q}: {reason}" for q, reason in sorted(unsupported.items())
    )
    analysis_paragraphs = [coverage_text]
    if has_duckdb:
        analysis_paragraphs.append(
            "DuckDB ran on its own dedicated, cost-tracked single-node host, included in the cost tables "
            "like KernelLake/PySpark."
            if duckdb_has_cost
            else "DuckDB is a single-node, CPU-only reference point -- not scored in the cost tables since "
            "this run has no dedicated per-instance $/hour to attribute a query's cost against."
        )
    if concurrency_rows:
        analysis_paragraphs.append(
            "Concurrency note: kernellake-server replicas are fully independent (no distributed execution, "
            "no coordinator) -- this measures concurrent-query throughput scaling, not a single query "
            "getting faster with more replicas."
        )
    analysis_paragraphs.append(
        "implied_cost_usd/tb_per_dollar in the cost-efficiency table are a normalized efficiency figure "
        "(this query's own median time x the engine's real on-demand $/hour), not a real EC2 invoice line "
        "item -- EC2 bills whole instance-hours regardless of which query is running."
    )

    with PdfPages(args.output) as pdf:
        add_title_page(pdf, data, cost_data)
        add_text_page(pdf, "Query coverage & analysis", analysis_paragraphs)
        add_table_page(pdf, speedup_rows, "Latency speedup ratio (headline #1)")
        for mode in sorted({r["mode"] for r in speedup_rows}):
            add_latency_chart_page(pdf, speedup_rows, mode)
        add_table_page(pdf, cost_rows, "Cost per completed query (headline #2)")
        if concurrency_rows:
            add_table_page(pdf, concurrency_rows, "Latency & throughput under concurrency (headline #3)")
        add_table_page(pdf, throughput_rows, "Scan throughput (supporting/mechanism detail)")
        add_table_page(pdf, efficiency_rows, "Cost efficiency per query (supporting/mechanism detail)")

    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
