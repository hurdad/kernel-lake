#!/usr/bin/env python3
"""Print a real, live-priced dollar estimate for a benchmark milestone
*before* any AWS resources are created or any data is generated -- called
out explicitly by every milestone in ../docs/RUNBOOK.md. This script only
prints; it never provisions, applies, or spends anything itself.

Uses the AWS Pricing API (boto3 `pricing` client, always queried against
us-east-1 regardless of --region -- it's the only region the Pricing API
endpoint is available in, though the returned prices are for whatever
--region you pass) for on-demand hourly rates, rather than a hardcoded
table that silently drifts out of date.

Usage:
    python3 estimate_cost.py --milestone m1
    python3 estimate_cost.py --milestone m3 --scale-factor 10000
    python3 estimate_cost.py --milestone m4 --replicas 8
"""
import argparse
import sys

import boto3

# Rough Parquet-on-S3 size per TPC-H scale factor, snappy-compressed --
# see tools/generate_tpch.py's own docstring for the row-count targets this
# mirrors (lineitem/part/orders/customer, TPC-H-shaped but synthetic).
# ~0.3-0.4 GB of compressed Parquet per scale-factor-unit is a reasonable
# real-world ratio for TPC-H-shaped data (compression ratio varies per
# column, this is a working estimate, not a promise) -- deliberately
# labeled as an estimate everywhere it's used below, never presented as
# exact.
ESTIMATED_GB_PER_SCALE_FACTOR = 0.35

INSTANCE_TYPES = {
    # g6.2xlarge, not the originally-planned g6.8xlarge: this account's
    # G/VT-family vCPU quota was only partially approved (8 vCPUs, not the
    # 32 requested -- see docs/COST_ESTIMATES.md), and g6.2xlarge (8
    # vCPUs, same 1x NVIDIA L4 GPU, less host RAM) is the largest g6 size
    # that fits. Update back to g6.8xlarge here if/when the quota appeal
    # succeeds.
    "kernellake": "g6.2xlarge",
    "spark_master": "m7i.xlarge",
    "spark_worker": "m7i.4xlarge",
    "monitoring": "t3.large",
    "data_gen": "c6i.8xlarge",
}

MILESTONES = {
    "m1": {"scale_factor": 100, "kernellake_replicas": 1, "spark_workers": 3, "hours": 4},
    "m2": {"scale_factor": 100, "kernellake_replicas": 1, "spark_workers": 3, "hours": 6},
    "m3-sf1000": {"scale_factor": 1000, "kernellake_replicas": 1, "spark_workers": 3, "hours": 8},
    "m3-sf3000": {"scale_factor": 3000, "kernellake_replicas": 1, "spark_workers": 3, "hours": 12},
    "m3-sf10000": {"scale_factor": 10000, "kernellake_replicas": 1, "spark_workers": 3, "hours": 24},
    "m4": {"scale_factor": 1000, "kernellake_replicas": 8, "spark_workers": 3, "hours": 3},
}


def on_demand_hourly_price(pricing_client, instance_type: str, region: str) -> float:
    region_names = {
        "us-east-1": "US East (N. Virginia)",
    }
    location = region_names.get(region)
    if location is None:
        print(
            f"WARNING: no known Pricing-API location name for region '{region}' -- "
            "add it to estimate_cost.py's region_names table. Returning 0.0 (estimate will be incomplete).",
            file=sys.stderr,
        )
        return 0.0

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
    import json

    price_list = response.get("PriceList", [])
    if not price_list:
        print(f"WARNING: no Pricing API result for {instance_type} in {region} -- treating as $0.0", file=sys.stderr)
        return 0.0
    product = json.loads(price_list[0])
    on_demand = product["terms"]["OnDemand"]
    (term_key,) = on_demand.keys()
    price_dimensions = on_demand[term_key]["priceDimensions"]
    (dim_key,) = price_dimensions.keys()
    return float(price_dimensions[dim_key]["pricePerUnit"]["USD"])


def estimate_s3_cost(scale_factor: int) -> dict:
    estimated_gb = scale_factor * ESTIMATED_GB_PER_SCALE_FACTOR
    # $0.023/GB-month standard storage (us-east-1, as of writing -- also
    # fetchable from the Pricing API's AmazonS3 service code, not done here
    # for brevity since S3 storage is a small fraction of total cost next
    # to GPU-instance-hours at these scale factors).
    storage_cost_per_month = estimated_gb * 0.023
    # PUT requests: generate_tpch.py writes one file per lineitem chunk
    # (--files) plus one each for part/orders/customer -- a handful of PUTs
    # regardless of scale factor, negligible ($0.005/1000 PUTs).
    return {
        "estimated_gb": estimated_gb,
        "storage_cost_per_month_usd": storage_cost_per_month,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--milestone", choices=sorted(MILESTONES), required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--scale-factor", type=int, default=None, help="Override the milestone's default scale factor")
    parser.add_argument("--replicas", type=int, default=None, help="Override the milestone's default kernellake replica count")
    parser.add_argument("--hours", type=float, default=None, help="Override the milestone's default estimated run duration")
    args = parser.parse_args()

    milestone = dict(MILESTONES[args.milestone])
    if args.scale_factor is not None:
        milestone["scale_factor"] = args.scale_factor
    if args.replicas is not None:
        milestone["kernellake_replicas"] = args.replicas
    if args.hours is not None:
        milestone["hours"] = args.hours

    pricing = boto3.client("pricing", region_name="us-east-1")  # Pricing API is us-east-1-only

    kernellake_rate = on_demand_hourly_price(pricing, INSTANCE_TYPES["kernellake"], args.region)
    spark_master_rate = on_demand_hourly_price(pricing, INSTANCE_TYPES["spark_master"], args.region)
    spark_worker_rate = on_demand_hourly_price(pricing, INSTANCE_TYPES["spark_worker"], args.region)
    monitoring_rate = on_demand_hourly_price(pricing, INSTANCE_TYPES["monitoring"], args.region)
    data_gen_rate = on_demand_hourly_price(pricing, INSTANCE_TYPES["data_gen"], args.region)

    hours = milestone["hours"]
    n_kernellake = milestone["kernellake_replicas"]
    n_workers = milestone["spark_workers"]

    kernellake_cost = kernellake_rate * n_kernellake * hours
    spark_cost = (spark_master_rate + spark_worker_rate * n_workers) * hours
    monitoring_cost = monitoring_rate * hours
    # Data generation runs once per scale factor, separately from the main
    # benchmark run -- estimated at a flat 2 hours for SF100/1000, scaling
    # up for the larger factors (a rough, documented guess, not measured
    # yet -- see docs/COST_ESTIMATES.md, which replaces this with real
    # observed durations after each milestone actually runs).
    data_gen_hours = {100: 1, 1000: 3, 3000: 8, 10000: 20}.get(milestone["scale_factor"], 2)
    data_gen_cost = data_gen_rate * data_gen_hours

    s3 = estimate_s3_cost(milestone["scale_factor"])
    total = kernellake_cost + spark_cost + monitoring_cost + data_gen_cost + s3["storage_cost_per_month_usd"]

    print(f"=== Cost estimate: {args.milestone} (region={args.region}) ===")
    print(f"Scale factor: SF{milestone['scale_factor']}")
    print(f"KernelLake replicas: {n_kernellake} x {INSTANCE_TYPES['kernellake']} @ ${kernellake_rate:.4f}/hr x {hours}h = ${kernellake_cost:.2f}")
    print(
        f"Spark cluster: 1x {INSTANCE_TYPES['spark_master']} + {n_workers}x {INSTANCE_TYPES['spark_worker']} "
        f"@ ${hours}h = ${spark_cost:.2f}"
    )
    print(f"Monitoring: 1x {INSTANCE_TYPES['monitoring']} @ {hours}h = ${monitoring_cost:.2f}")
    print(f"Data generation (one-time, SF{milestone['scale_factor']}): 1x {INSTANCE_TYPES['data_gen']} @ {data_gen_hours}h = ${data_gen_cost:.2f}")
    print(f"S3 storage (~{s3['estimated_gb']:.0f} GB estimated, first month): ${s3['storage_cost_per_month_usd']:.2f}")
    print("-" * 60)
    print(f"TOTAL ESTIMATE: ${total:.2f}")
    print()
    print("This is an ESTIMATE (on-demand rates, approximated data-gen duration/data")
    print("size, no data-transfer cost included). Real observed costs go in")
    print("../docs/COST_ESTIMATES.md after each milestone actually runs. Review before")
    print("running ../scripts/provision.sh --yes or ../scripts/generate_and_upload_data.sh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
