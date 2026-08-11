"""AWS cost accounting for a benchmark run.

Headline output: **cost per completed query** -- run cost divided by
queries actually finished. cost/logical-TB, cost/physical-TB, cost/hour,
and cost/concurrent-user are also computed, as supporting/mechanism
detail (see ../../.claude context -- the plan's own "Primary goal"
framing: latency speedup + cost-per-query + concurrency behavior are the
headline story, scan throughput/cost-per-TB explain the mechanism behind
it, not the top-line result).

Pricing comes from the live AWS Pricing API (boto3 `pricing` client, same
approach as ../scripts/estimate_cost.py) rather than a hardcoded table.
Instance-hours come from real `DescribeInstances` LaunchTime/state, not a
caller-supplied guess -- see `instance_hours_from_aws()`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import boto3


REGION_PRICING_LOCATIONS = {
    "us-east-1": "US East (N. Virginia)",
}


def on_demand_hourly_price(pricing_client, instance_type: str, region: str) -> float:
    location = REGION_PRICING_LOCATIONS.get(region)
    if location is None:
        raise ValueError(f"No known Pricing-API location name for region '{region}' -- add it to REGION_PRICING_LOCATIONS.")
    response = pricing_client.get_products(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
            {"Type": "TERM_MATCH", "Field": "location", "Value": location},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
        ],
        MaxResults=1,
    )
    price_list = response.get("PriceList", [])
    if not price_list:
        raise RuntimeError(f"No Pricing API result for {instance_type} in {region}")
    product = json.loads(price_list[0])
    on_demand = product["terms"]["OnDemand"]
    (term_key,) = on_demand.keys()
    price_dimensions = on_demand[term_key]["priceDimensions"]
    (dim_key,) = price_dimensions.keys()
    return float(price_dimensions[dim_key]["pricePerUnit"]["USD"])


def instance_hours_from_aws(ec2_client, instance_ids: list[str], run_end: datetime | None = None) -> dict[str, float]:
    """Real instance-hours for `instance_ids`, from each instance's actual
    LaunchTime to `run_end` (default: now) -- not a caller-estimated
    duration. Billed EC2 time rounds up to the second (post-2017 billing
    granularity, 60s minimum) but this reports fractional hours as-is;
    round up yourself if you want a conservative (over-)estimate matching
    AWS's own invoice rounding.
    """
    run_end = run_end or datetime.now(timezone.utc)
    response = ec2_client.describe_instances(InstanceIds=instance_ids)
    hours: dict[str, float] = {}
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            launch_time = instance["LaunchTime"]
            hours[instance["InstanceId"]] = (run_end - launch_time).total_seconds() / 3600.0
    return hours


@dataclass
class RunCost:
    """Actual cost breakdown for one benchmark run window."""

    kernellake_usd: float = 0.0
    spark_usd: float = 0.0
    duckdb_usd: float = 0.0
    monitoring_usd: float = 0.0
    s3_storage_usd: float = 0.0
    detail: dict = field(default_factory=dict)

    @property
    def total_usd(self) -> float:
        return self.kernellake_usd + self.spark_usd + self.duckdb_usd + self.monitoring_usd + self.s3_storage_usd


def compute_run_cost(
    pricing_client,
    ec2_client,
    kernellake_instance_ids: list[str],
    kernellake_instance_type: str,
    spark_instance_ids: list[str],
    spark_instance_types: dict[str, str],  # instance_id -> type (a single instance today, see spark_cluster.tf; kept as a dict/list pair rather than a scalar in case a multi-node topology comes back)
    monitoring_instance_id: str,
    monitoring_instance_type: str,
    region: str,
    s3_bytes_stored: int = 0,
    duckdb_instance_id: str | None = None,
    duckdb_instance_type: str | None = None,
) -> RunCost:
    hours_kernellake = instance_hours_from_aws(ec2_client, kernellake_instance_ids)
    hours_spark = instance_hours_from_aws(ec2_client, spark_instance_ids)
    hours_monitoring = instance_hours_from_aws(ec2_client, [monitoring_instance_id])

    kernellake_rate = on_demand_hourly_price(pricing_client, kernellake_instance_type, region)
    monitoring_rate = on_demand_hourly_price(pricing_client, monitoring_instance_type, region)

    kernellake_usd = sum(hours_kernellake.values()) * kernellake_rate
    spark_usd = sum(
        hours_spark[iid] * on_demand_hourly_price(pricing_client, spark_instance_types[iid], region)
        for iid in spark_instance_ids
    )
    monitoring_usd = sum(hours_monitoring.values()) * monitoring_rate

    # DuckDB is optional: only present once terraform/duckdb_instance.tf's
    # dedicated host was actually used for this run (see
    # runner/duckdb_query_loop.py) -- an in-process-only run (the old
    # --duckdb flag on aws_benchmark_runner.py, still supported as a quick
    # option with no dedicated infra) has no instance to attribute cost
    # to, same as before this dedicated host existed.
    hours_duckdb: dict[str, float] = {}
    duckdb_usd = 0.0
    duckdb_rate: float | None = None
    if duckdb_instance_id and duckdb_instance_type:
        hours_duckdb = instance_hours_from_aws(ec2_client, [duckdb_instance_id])
        duckdb_rate = on_demand_hourly_price(pricing_client, duckdb_instance_type, region)
        duckdb_usd = sum(hours_duckdb.values()) * duckdb_rate

    # $0.023/GB-month, us-east-1 standard storage, prorated to the actual
    # run duration -- a small fraction of total cost next to GPU-instance
    # hours, not fetched live from the Pricing API's AmazonS3 service code
    # for that reason (documented approximation, not silently assumed
    # exact).
    run_hours = max(hours_kernellake.values(), default=0.0)
    s3_storage_usd = (s3_bytes_stored / 1e9) * 0.023 * (run_hours / (24 * 30))

    detail = {
        "kernellake_instance_hours": hours_kernellake,
        "spark_instance_hours": hours_spark,
        "monitoring_instance_hours": hours_monitoring,
        "kernellake_hourly_rate_usd": kernellake_rate,
        "monitoring_hourly_rate_usd": monitoring_rate,
    }
    if duckdb_rate is not None:
        detail["duckdb_instance_hours"] = hours_duckdb
        detail["duckdb_hourly_rate_usd"] = duckdb_rate

    return RunCost(
        kernellake_usd=kernellake_usd,
        spark_usd=spark_usd,
        duckdb_usd=duckdb_usd,
        monitoring_usd=monitoring_usd,
        s3_storage_usd=s3_storage_usd,
        detail=detail,
    )


def cost_per_completed_query(run_cost: RunCost, queries_completed: int, engine: str) -> float | None:
    """The headline cost metric. `engine` selects which cost bucket this
    query count actually ran against (KernelLake queries shouldn't be
    charged Spark's instance cost and vice versa, even though both run
    within the same overall benchmark window)."""
    if queries_completed <= 0:
        return None
    engine_cost = {
        "kernellake": run_cost.kernellake_usd,
        "pyspark": run_cost.spark_usd,
        "duckdb": run_cost.duckdb_usd,
    }.get(engine)
    if engine_cost is None:
        raise ValueError(f"Unknown engine '{engine}' -- expected 'kernellake', 'pyspark', or 'duckdb'")
    return engine_cost / queries_completed


def cost_per_tb(run_cost: RunCost, bytes_processed: int, engine: str) -> float | None:
    """Supporting/mechanism metric -- see this module's own docstring for
    why this isn't the headline number."""
    if bytes_processed <= 0:
        return None
    engine_cost = {
        "kernellake": run_cost.kernellake_usd,
        "pyspark": run_cost.spark_usd,
        "duckdb": run_cost.duckdb_usd,
    }.get(engine)
    if engine_cost is None:
        raise ValueError(f"Unknown engine '{engine}' -- expected 'kernellake', 'pyspark', or 'duckdb'")
    return engine_cost / (bytes_processed / 1e12)


@dataclass
class QueryCostEfficiency:
    """Per-query cost efficiency at an instance's on-demand rate --
    complementary to cost_per_tb() above, not a replacement for it. The
    difference matters:

    cost_per_tb() divides one *whole benchmark run's* real accrued cost
    (actual EC2 instance-hours, from compute_run_cost()) by bytes summed
    across *every* query in that run -- every query is charged the same
    average rate regardless of whether it was fast or slow, because the
    instance keeps billing by the hour no matter which query is currently
    running. It answers "what did this whole run cost, spread over the
    data it touched" -- correct for a real invoice, but it can't tell you
    Q1 was more cost-efficient than Q14.

    This answers a different question -- "at this instance's hourly rate,
    what would *this one query* have cost if billing could be metered to
    the second" -- by multiplying the instance's real on-demand $/hour
    (on_demand_hourly_price(), the same live-Pricing-API lookup
    compute_run_cost() itself uses) against just this query's own measured
    wall time. EC2 doesn't actually bill this way (whole-instance-hours,
    not per-query slices), so implied_cost_usd is never a real invoice
    line item -- it's a normalized efficiency figure for comparing queries
    or engines against each other, not a cost projection to budget from.
    Real budgeting should use cost_per_completed_query()/cost_per_tb()
    against actual instance-hours instead.
    """

    query_seconds: float
    bytes_processed: int
    hourly_rate_usd: float

    @property
    def implied_cost_usd(self) -> float:
        """What this query would have cost, billed by the second at
        hourly_rate_usd -- see class docstring for why this is a
        normalized comparison figure, not a real invoice line item."""
        return self.hourly_rate_usd * (self.query_seconds / 3600.0)

    @property
    def tb_processed(self) -> float:
        return self.bytes_processed / 1e12

    @property
    def implied_cost_per_tb_usd(self) -> float | None:
        if self.tb_processed <= 0:
            return None
        return self.implied_cost_usd / self.tb_processed

    @property
    def tb_per_dollar(self) -> float | None:
        """The inverse framing -- "how much data can this engine scan per
        dollar, at the rate it actually achieved this query." Whichever
        direction reads more naturally for a given report; both come from
        the same two numbers."""
        if self.implied_cost_usd <= 0:
            return None
        return self.tb_processed / self.implied_cost_usd


def query_cost_efficiency(query_seconds: float, bytes_processed: int, hourly_rate_usd: float) -> QueryCostEfficiency:
    """Builds a QueryCostEfficiency for one query -- see that class's own
    docstring for the metric definition and why it's distinct from
    cost_per_tb(). `hourly_rate_usd` is normally
    on_demand_hourly_price(pricing_client, instance_type, region) for
    whichever engine's instance actually ran this query (the KernelLake
    GPU host for a "kernellake" entry, a Spark worker/cluster-blended rate
    for a "pyspark" entry -- caller's choice which to pass, this function
    doesn't know or care which engine `query_seconds`/`bytes_processed`
    came from)."""
    return QueryCostEfficiency(
        query_seconds=query_seconds, bytes_processed=bytes_processed, hourly_rate_usd=hourly_rate_usd
    )


def _main() -> int:
    """CLI entry point: computes real cost for a run window (from actual
    EC2 instance launch times, not a caller-supplied guess) and writes it
    as JSON for ../reporting/generate_report.py to consume via its own
    --cost-json flag. Run this once per benchmark run, after the run has
    finished (so instance uptime reflects the real elapsed window), before
    tearing down.
    """
    import argparse
    import dataclasses
    import json as json_module

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--kernellake-instance-ids", required=True, help="Comma-separated")
    parser.add_argument("--kernellake-instance-type", default="g6.xlarge")
    parser.add_argument("--spark-instance-ids", required=True, help="Comma-separated")
    parser.add_argument("--spark-instance-types", required=True, help="Comma-separated, same order as --spark-instance-ids")
    parser.add_argument("--monitoring-instance-id", required=True)
    parser.add_argument("--monitoring-instance-type", default="t3.large")
    parser.add_argument("--duckdb-instance-id", default=None, help="terraform output duckdb_host_id -- omit if the run only used --duckdb in-process (no dedicated cost)")
    parser.add_argument("--duckdb-instance-type", default="m7i.xlarge")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--s3-bytes-stored", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spark_ids = args.spark_instance_ids.split(",")
    spark_types = args.spark_instance_types.split(",")
    if len(spark_ids) != len(spark_types):
        parser.error("--spark-instance-ids and --spark-instance-types must have the same length")

    pricing_client = boto3.client("pricing", region_name="us-east-1")  # Pricing API is us-east-1-only
    ec2_client = boto3.client("ec2", region_name=args.region)

    run_cost = compute_run_cost(
        pricing_client,
        ec2_client,
        kernellake_instance_ids=args.kernellake_instance_ids.split(","),
        kernellake_instance_type=args.kernellake_instance_type,
        spark_instance_ids=spark_ids,
        spark_instance_types=dict(zip(spark_ids, spark_types)),
        monitoring_instance_id=args.monitoring_instance_id,
        monitoring_instance_type=args.monitoring_instance_type,
        region=args.region,
        s3_bytes_stored=args.s3_bytes_stored,
        duckdb_instance_id=args.duckdb_instance_id,
        duckdb_instance_type=args.duckdb_instance_type,
    )
    from pathlib import Path

    Path(args.output).write_text(json_module.dumps(dataclasses.asdict(run_cost) | {"total_usd": run_cost.total_usd}, indent=2))
    duckdb_note = f", duckdb ${run_cost.duckdb_usd:.2f}" if args.duckdb_instance_id else ""
    print(f"Wrote {args.output}: total ${run_cost.total_usd:.2f} "
          f"(kernellake ${run_cost.kernellake_usd:.2f}, spark ${run_cost.spark_usd:.2f}{duckdb_note})")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
