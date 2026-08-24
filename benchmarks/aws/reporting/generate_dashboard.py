#!/usr/bin/env python3
"""Generates a single, self-contained interactive HTML dashboard from
generate_report.py's same input (aggregate_results.py's aggregated.json,
optionally runner/cost_model.py's cost JSON). Plotly.js is loaded from a
CDN (not embedded) -- this file is meant to be opened in a browser with
normal internet access, not air-gapped.

Top of page: the three headline charts (latency speedup ratio vs. scale
factor, cost per completed query vs. scale factor, latency/cost-per-query
vs. concurrency). Below that: supporting plots (scan throughput, aggregate
scaling efficiency).

Usage:
    python3 generate_dashboard.py --input aggregated.json --cost-json cost-sf100.json --output dashboard.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def build_traces(data: dict, cost_data: dict | None) -> dict:
    benchmark_runs = sorted(data["benchmark_runs"], key=lambda r: r["scale_factor"])
    scaling_runs = sorted(data["scaling_runs"], key=lambda r: r["replica_count"])

    # Headline #1: latency speedup ratio vs scale factor, one line per query.
    speedup_by_query: dict[int, dict[str, list]] = {}
    for run in benchmark_runs:
        sf = run["scale_factor"]
        for q in run["queries"]:
            entry = q["modes"].get("warm") or next(iter(q["modes"].values()), {})
            ratio = entry.get("latency_speedup_ratio")
            if ratio is None:
                continue
            bucket = speedup_by_query.setdefault(q["query"], {"x": [], "y": []})
            bucket["x"].append(sf)
            bucket["y"].append(ratio)

    # Headline #2: cost per completed query vs scale factor.
    cost_x, kl_cost_y, spark_cost_y = [], [], []
    if cost_data:
        for run in benchmark_runs:
            sf = run["scale_factor"]
            queries_completed_kl = sum(len(q["modes"]) for q in run["queries"])
            queries_completed_spark = sum(1 for q in run["queries"] for m in q["modes"].values() if "pyspark" in m)
            cost_x.append(sf)
            kl_cost_y.append(cost_data["kernellake_usd"] / queries_completed_kl if queries_completed_kl else None)
            spark_cost_y.append(cost_data["spark_usd"] / queries_completed_spark if queries_completed_spark else None)

    # Headline #3: latency & queries/hour vs concurrency.
    concurrency_x = [r["replica_count"] for r in scaling_runs]
    concurrency_latency_y = [r["latency_median_seconds"] for r in scaling_runs]
    concurrency_throughput_y = [r["queries_per_hour"] for r in scaling_runs]

    return {
        "speedup_by_query": speedup_by_query,
        "cost_x": cost_x,
        "kl_cost_y": kl_cost_y,
        "spark_cost_y": spark_cost_y,
        "concurrency_x": concurrency_x,
        "concurrency_latency_y": concurrency_latency_y,
        "concurrency_throughput_y": concurrency_throughput_y,
    }


def render_html(traces: dict, query_count: int) -> str:
    # A single json.dumps() of the whole payload, referenced from JS by
    # property access below -- not per-value Python repr() string
    # interpolation, which would silently emit invalid JavaScript for any
    # None/True/False (Python's None/True/False are not valid JS tokens;
    # json.dumps() correctly produces null/true/false). Caught by this
    # module's own smoke test once a fixture actually had a None value.
    traces_json = json.dumps(traces)

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>KernelLake AWS Benchmark Dashboard</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>
    body {{ font-family: -apple-system, sans-serif; margin: 2rem; max-width: 1100px; }}
    h1 {{ margin-bottom: 0; }}
    .caveat {{ color: #666; font-size: 0.9em; }}
    .chart {{ margin-bottom: 3rem; }}
    .supporting {{ opacity: 0.85; }}
  </style>
</head>
<body>
  <h1>KernelLake AWS Benchmark</h1>
  <p class="caveat">Unofficial, TPC-H-<em>derived</em> benchmark. Not a certified TPC-H result.
  Only {query_count} of TPC-H's 22 queries run today -- see the accompanying report.md for the full list of
  what's blocked and why.</p>

  <h2>Headline #1: Latency speedup ratio (KernelLake vs. PySpark)</h2>
  <div id="speedup" class="chart"></div>

  <h2>Headline #2: Cost per completed query</h2>
  <div id="cost" class="chart"></div>

  <h2>Headline #3: Latency &amp; throughput under concurrency</h2>
  <p class="caveat">kernellake-server replicas here are fully independent (no distributed execution,
  no coordinator) -- this measures concurrent-query throughput scaling, not a single query getting
  faster with more replicas.</p>
  <div id="concurrency-latency" class="chart"></div>
  <div id="concurrency-throughput" class="chart"></div>

  <h2 class="supporting">Supporting detail: scan throughput &amp; cost/TB</h2>
  <p class="caveat">Mechanism metrics explaining the headline numbers above, not top-line results
  themselves -- see report.md's own scan-throughput table for current data-availability caveats.</p>

  <script>
    const traces = {traces_json};

    const speedupTraces = Object.entries(traces.speedup_by_query).map(([q, bucket]) => ({{
      x: bucket.x, y: bucket.y, mode: "lines+markers", name: "Q" + q
    }}));
    Plotly.newPlot("speedup", speedupTraces, {{
      title: "Latency speedup ratio vs. scale factor",
      xaxis: {{title: "Scale factor", type: "log"}},
      yaxis: {{title: "PySpark / KernelLake latency"}}
    }});

    Plotly.newPlot("cost", [
      {{x: traces.cost_x, y: traces.kl_cost_y, mode: "lines+markers", name: "KernelLake"}},
      {{x: traces.cost_x, y: traces.spark_cost_y, mode: "lines+markers", name: "PySpark"}}
    ], {{
      title: "Cost per completed query vs. scale factor",
      xaxis: {{title: "Scale factor", type: "log"}},
      yaxis: {{title: "USD per query"}}
    }});

    Plotly.newPlot("concurrency-latency", [
      {{x: traces.concurrency_x, y: traces.concurrency_latency_y, mode: "lines+markers", name: "Median latency"}}
    ], {{
      title: "Median query latency vs. concurrent replicas",
      xaxis: {{title: "Replica count"}},
      yaxis: {{title: "Seconds"}}
    }});

    Plotly.newPlot("concurrency-throughput", [
      {{x: traces.concurrency_x, y: traces.concurrency_throughput_y, mode: "lines+markers", name: "Queries/hour"}}
    ], {{
      title: "Aggregate throughput vs. concurrent replicas",
      xaxis: {{title: "Replica count"}},
      yaxis: {{title: "Queries/hour"}}
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--cost-json", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    cost_data = json.loads(Path(args.cost_json).read_text()) if args.cost_json else None

    traces = build_traces(data, cost_data)
    # Derived from the real report data, not hardcoded -- see
    # generate_report.py's identical fix for why a hardcoded query count
    # here goes stale as soon as aws_benchmark_runner.py's own
    # ALL_QUERIES grows. Distinct query numbers across every benchmark
    # run (not just run 0) -- a dashboard can aggregate multiple scale
    # factors' runs, which should all have the same query set, but this
    # doesn't assume that.
    query_count = len({q["query"] for run in data["benchmark_runs"] for q in run["queries"]})
    Path(args.output).write_text(render_html(traces, query_count))
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
