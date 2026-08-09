#!/usr/bin/env python3
"""Generates the Markdown/CSV/JSON benchmark report from
aggregate_results.py's combined output. Leads with the three headline
numbers (latency speedup ratio, cost per completed query, latency/cost
under concurrency); scan throughput and cost/TB follow underneath as
supporting/mechanism detail -- see ../README.md's "What this measures"
section for why, and the plan's own "Primary goal" framing.

A sibling to tools/generate_benchmark_report.py (that one renders a PDF via
matplotlib, for the local single-machine benchmark) -- this one targets
Markdown/CSV/JSON specifically, since those are what generate_dashboard.py
and downstream documentation/blog-post inclusion need; not a literal
extension of its PDF-rendering internals.

Usage:
    python3 generate_report.py --input aggregated.json --cost-json cost-sf100.json \\
        --output-dir report/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "aws" / "runner"))

from cost_model import query_cost_efficiency  # noqa: E402


def engine_hourly_rates(cost_data: dict | None) -> dict[str, float | None]:
    """Real on-demand $/hour for each engine, derived from cost_model.py's
    --output JSON -- not looked up fresh here (this script has no AWS
    credentials/network dependency of its own; the rate is whatever
    compute_run_cost() already paid the Pricing API to find out).

    kernellake: detail.kernellake_hourly_rate_usd is already a real
    per-instance rate (compute_run_cost() -> on_demand_hourly_price()).
    pyspark: no single stored rate (the cluster can mix instance types --
    master vs. worker), so it's derived as spark_usd / total spark
    instance-hours -- the real blended $/hour the whole cluster cost
    during this run, appropriate here since a Spark query uses the whole
    cluster while running, not one node's worth of capacity.

    duckdb: deliberately absent, here and from cost_per_query_table()/
    cost_efficiency_table() below -- it runs in-process on whichever host
    ran the orchestrator (not on infra dedicated to it), so there is no
    real per-instance $/hour to attribute a query's cost against."""
    if cost_data is None:
        return {"kernellake": None, "pyspark": None}
    detail = cost_data.get("detail", {})
    kernellake_rate = detail.get("kernellake_hourly_rate_usd")
    spark_hours = sum(detail.get("spark_instance_hours", {}).values())
    spark_rate = (cost_data["spark_usd"] / spark_hours) if spark_hours > 0 else None
    return {"kernellake": kernellake_rate, "pyspark": spark_rate}


def latency_speedup_table(benchmark_runs: list) -> list[dict]:
    rows = []
    for run in benchmark_runs:
        sf = run["scale_factor"]
        for q in run["queries"]:
            for mode, entry in q["modes"].items():
                if "pyspark" not in entry and "duckdb" not in entry:
                    continue
                row = {
                    "scale_factor": sf,
                    "query": q["query"],
                    "mode": mode,
                    "kernellake_median_seconds": entry["kernellake"]["median_seconds"],
                }
                if "pyspark" in entry:
                    row["pyspark_median_seconds"] = entry["pyspark"]["median_seconds"]
                    row["latency_speedup_ratio"] = entry.get("latency_speedup_ratio")
                    row["results_match"] = entry.get("results_match")
                if "duckdb" in entry:
                    # DuckDB: single-node, CPU-only, in-process on the
                    # orchestrator host -- a reference point, not scored in
                    # any cost table (see new_duckdb_connection()'s comment
                    # in aws_benchmark_runner.py for why).
                    row["duckdb_median_seconds"] = entry["duckdb"]["median_seconds"]
                    row["latency_speedup_ratio_duckdb"] = entry.get("latency_speedup_ratio_duckdb")
                    row["results_match_duckdb"] = entry.get("results_match_duckdb")
                rows.append(row)
    return rows


def cost_per_query_table(benchmark_runs: list, cost_data: dict | None) -> list[dict]:
    rows = []
    for run in benchmark_runs:
        sf = run["scale_factor"]
        queries_completed_kl = sum(len(q["modes"]) for q in run["queries"])  # one completed run per query x mode
        queries_completed_spark = sum(1 for q in run["queries"] for m in q["modes"].values() if "pyspark" in m)
        if cost_data is None:
            rows.append(
                {
                    "scale_factor": sf,
                    "kernellake_cost_per_query_usd": None,
                    "pyspark_cost_per_query_usd": None,
                    "note": "no --cost-json given -- run runner/cost_model.py after this benchmark run to compute real cost",
                }
            )
            continue
        kl_cost = cost_data["kernellake_usd"] / queries_completed_kl if queries_completed_kl else None
        spark_cost = cost_data["spark_usd"] / queries_completed_spark if queries_completed_spark else None
        rows.append(
            {
                "scale_factor": sf,
                "kernellake_cost_per_query_usd": kl_cost,
                "pyspark_cost_per_query_usd": spark_cost,
                "cost_ratio_pyspark_over_kernellake": (spark_cost / kl_cost) if (kl_cost and spark_cost) else None,
            }
        )
    return rows


def concurrency_table(scaling_runs: list) -> list[dict]:
    return [
        {
            "replica_count": r["replica_count"],
            "concurrent_clients": r["concurrent_clients"],
            "queries_completed": r["queries_completed"],
            "queries_per_hour": r["queries_per_hour"],
            "latency_median_seconds": r["latency_median_seconds"],
            "latency_p95_seconds": r["latency_p95_seconds"],
            "latency_p99_seconds": r["latency_p99_seconds"],
        }
        for r in scaling_runs
    ]


def scan_throughput_table(benchmark_runs: list) -> list[dict]:
    """Supporting/mechanism metric -- see this module's own docstring.

    Uses q["bytes_scanned"]["compressed_bytes"] (real S3-object-size ground
    truth, computed by aws_benchmark_runner.py's table_bytes_scanned() from
    every table a query's SQL actually references -- not an estimate, and
    not the same thing as parquet_decoding_seconds, which would need
    server-side row-group-pruning-aware accounting this doesn't have) for
    *both* engines, since they scan the same physical S3 data -- gives an
    apples-to-apples throughput comparison even where Spark's own
    self-reported inputBytes (spark_scan_metrics, kept alongside for
    reference) might differ due to its own predicate/column pruning.
    """
    rows = []
    for run in benchmark_runs:
        sf = run["scale_factor"]
        for q in run["queries"]:
            bytes_scanned = q.get("bytes_scanned", {}).get("compressed_bytes")
            spark_scan = q.get("spark_scan_metrics", {})
            for mode, entry in q["modes"].items():
                row = {
                    "scale_factor": sf,
                    "query": q["query"],
                    "mode": mode,
                    "compressed_bytes_scanned": bytes_scanned,
                }
                for engine in ("kernellake", "pyspark", "duckdb"):
                    if engine not in entry or not bytes_scanned:
                        continue
                    seconds = entry[engine]["median_seconds"]
                    row[f"{engine}_throughput_gbps"] = (bytes_scanned / 1e9) / seconds if seconds > 0 else None
                if mode in spark_scan:
                    row["spark_self_reported_input_bytes"] = spark_scan[mode].get("input_bytes")
                rows.append(row)
    return rows


def cost_efficiency_table(benchmark_runs: list, cost_data: dict | None) -> list[dict]:
    """New KPI, complementary to cost_per_query_table() above -- see
    cost_model.py's QueryCostEfficiency docstring for the full definition
    and why it's a different question than that table's run-level cost.
    Per query/mode/engine: what would *this query alone* have cost at the
    engine's real on-demand $/hour, and how many TB/$ does that imply.
    Needs real bytes_scanned (scan_throughput_table's data) and a real
    hourly rate (--cost-json) -- rows/cells missing either are omitted
    rather than guessed.
    """
    if cost_data is None:
        return [{"note": "no --cost-json given -- run runner/cost_model.py after this benchmark run for real rates"}]
    rates = engine_hourly_rates(cost_data)
    rows = []
    for run in benchmark_runs:
        sf = run["scale_factor"]
        for q in run["queries"]:
            bytes_scanned = q.get("bytes_scanned", {}).get("compressed_bytes")
            if not bytes_scanned:
                continue
            for mode, entry in q["modes"].items():
                for engine in ("kernellake", "pyspark"):
                    rate = rates.get(engine)
                    if engine not in entry or rate is None:
                        continue
                    eff = query_cost_efficiency(entry[engine]["median_seconds"], bytes_scanned, rate)
                    rows.append(
                        {
                            "scale_factor": sf,
                            "query": q["query"],
                            "mode": mode,
                            "engine": engine,
                            "hourly_rate_usd": rate,
                            "median_seconds": entry[engine]["median_seconds"],
                            "tb_scanned": eff.tb_processed,
                            "implied_cost_usd": eff.implied_cost_usd,
                            "implied_cost_per_tb_usd": eff.implied_cost_per_tb_usd,
                            "tb_per_dollar": eff.tb_per_dollar,
                        }
                    )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[dict], title: str) -> str:
    if not rows:
        return f"## {title}\n\n_No data._\n\n"
    # Union of keys across every row, in first-seen order -- not just
    # rows[0]'s keys, since rows can be heterogeneous (e.g. a query/mode
    # with DuckDB data alongside one without, when --duckdb only covers
    # part of a run).
    columns: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    lines = [f"## {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="aggregate_results.py's output")
    parser.add_argument("--cost-json", default=None, help="runner/cost_model.py's --output, for real cost-per-query numbers")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    cost_data = json.loads(Path(args.cost_json).read_text()) if args.cost_json else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speedup_rows = latency_speedup_table(data["benchmark_runs"])
    cost_rows = cost_per_query_table(data["benchmark_runs"], cost_data)
    concurrency_rows = concurrency_table(data["scaling_runs"])
    throughput_rows = scan_throughput_table(data["benchmark_runs"])
    efficiency_rows = cost_efficiency_table(data["benchmark_runs"], cost_data)

    write_csv(speedup_rows, out_dir / "latency_speedup.csv")
    write_csv(cost_rows, out_dir / "cost_per_query.csv")
    write_csv(concurrency_rows, out_dir / "concurrency.csv")
    write_csv(throughput_rows, out_dir / "scan_throughput_supporting.csv")
    write_csv(efficiency_rows, out_dir / "cost_efficiency_per_query.csv")

    (out_dir / "raw_aggregated.json").write_text(json.dumps(data, indent=2))

    unsupported = data["benchmark_runs"][0]["unsupported_queries"] if data["benchmark_runs"] else {}
    md = [
        "# KernelLake AWS Benchmark Report",
        "",
        "**Unofficial, TPC-H-*derived* benchmark. Not a certified TPC-H result.** See ../README.md.",
        "",
        "## Query coverage",
        "",
        f"6 of TPC-H's 22 queries run today (Q1, Q3, Q6, Q12, Q14, Q19). The other 16 are blocked:",
        "",
    ]
    for q, reason in sorted(unsupported.items()):
        md.append(f"- **Q{q}**: {reason}")
    md.append("")
    has_duckdb = any("duckdb_median_seconds" in row for row in speedup_rows)
    speedup_title = (
        "Latency speedup ratio (headline #1) -- KernelLake vs. PySpark vs. DuckDB"
        if has_duckdb
        else "Latency speedup ratio (headline #1) -- KernelLake vs. PySpark"
    )
    md.append(markdown_table(speedup_rows, speedup_title))
    if has_duckdb:
        md.append(
            "_DuckDB is a single-node, CPU-only, in-process reference point (runs on the orchestrator "
            "host itself against the same real S3 data) -- not scored in the cost tables below, since "
            "it has no dedicated per-instance $/hour to attribute a query's cost against._"
        )
        md.append("")
    md.append(markdown_table(cost_rows, "Cost per completed query (headline #2)"))
    md.append(markdown_table(concurrency_rows, "Latency & throughput under concurrency (headline #3)"))
    md.append(
        "_Concurrency note: kernellake-server replicas above are fully independent (no distributed "
        "execution, no coordinator) -- this measures concurrent-query throughput scaling, not a single "
        "query getting faster with more replicas. See ../README.md._"
    )
    md.append("")
    md.append(markdown_table(throughput_rows, "Scan throughput (supporting/mechanism detail, not headline)"))
    md.append(
        markdown_table(
            efficiency_rows,
            "Cost efficiency per query -- TB scanned per $ at achieved query time "
            "(supporting/mechanism detail, not headline)",
        )
    )
    md.append(
        "_implied_cost_usd/tb_per_dollar above are a normalized efficiency figure (this query's own "
        "median time x the engine's real on-demand $/hour), not a real EC2 invoice line item -- EC2 "
        "bills whole instance-hours regardless of which query is running. See cost_model.py's "
        "QueryCostEfficiency for the full definition and how it differs from cost_per_query above._"
    )

    (out_dir / "report.md").write_text("\n".join(md))

    print(f"Wrote {out_dir}/report.md, *.csv, raw_aggregated.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
