#!/usr/bin/env python3
"""Reads an already-generated flat TPC-H Parquet dataset from S3 (written by
generate_and_upload_data.sh) and writes it into real Iceberg tables via a
REST catalog (terraform/iceberg_catalog.tf's tabulario/iceberg-rest
instance) -- lineitem partitioned by years(l_shipdate), orders by
years(o_orderdate) (the two date columns every latency-headline query
(Q1/Q3/Q6/Q12/Q14) filters on), part/customer left unpartitioned (no
date-range predicate in any supported query touches them).

This exists specifically to exercise KernelLake's real, manifest-level
partition pruning (src/iceberg/partition_pruning.cpp) -- confirmed, before
writing this, that a plain Hive-partitioned directory layout does NOT do
this (only makes partition columns queryable, never skips files), and that
this script's exact API calls (schema conversion, partition spec
construction, append(), and the resulting prune-relevant scan.plan_files()
behavior) work as intended via a local pyiceberg smoke test against a
throwaway SqlCatalog before ever running this against real infrastructure.

Run on an ephemeral EC2 instance by generate_and_upload_iceberg_data.sh,
same pattern as generate_tpch.py/generate_and_upload_data.sh.

Usage:
    python3 generate_and_upload_iceberg_data.py \\
        --s3-bucket kernellake-bench-666052791151-ab12cd34 --scale-factor 100 \\
        --source-compression snappy \\
        --catalog-uri http://10.90.1.50:8181 --warehouse s3://.../warehouse/ \\
        --aws-region us-east-1
"""
from __future__ import annotations

import argparse
import sys

# table.append() only accepts a fully-materialized pa.Table (confirmed via
# Table.append()'s own signature -- df: pa.Table, no streaming/
# RecordBatchReader overload) -- but that doesn't mean the WHOLE table
# needs to be materialized at once: append() can be called repeatedly on
# the same Table object, each call adding a new snapshot's worth of data
# files. FILES_PER_BATCH below chunks the per-table source file list so
# peak memory stays bounded by one batch's decompressed size, not the
# whole table's. Learned the hard way: an earlier version read every
# source file for a table into one pa.Table before writing, which
# OOM-killed a 64GB r6i.2xlarge on lineitem alone at SF100 (600M rows) --
# "lineitem is a few GB decompressed" was a real underestimate, not a
# safe assumption.
TABLES = ("lineitem", "part", "orders", "customer")
FILES_PER_BATCH = 10

# (partition_column, transform_name) -- only date columns every supported
# query actually filters on get partitioned; see this file's own docstring.
PARTITION_COLUMNS = {
    "lineitem": "l_shipdate",
    "orders": "o_orderdate",
}


def s3_source_key_prefix(scale_factor: int, table: str, compression: str) -> str:
    # orders is multi-file too (generate_tpch.py batches it via
    # ORDERS_BATCH_ROWS the same way lineitem is batched via --files, e.g.
    # orders-00000.parquet..orders-00029.parquet at SF100) -- treating it
    # as a single "orders-00000.parquet" file silently read only the
    # first ~3% of the orders table. Found via a real FileNotFoundError
    # (pyarrow.dataset.dataset() doesn't shell-glob a "*" pattern when
    # given an explicit filesystem=, it needs real S3 listing instead --
    # see list_source_files() below) before this fix, not assumed.
    file_prefix_map = {
        "lineitem": "lineitem-",
        "part": "part-",
        "orders": "orders-",
        "customer": "customer-",
    }
    sf_dir = f"sf{scale_factor}" if compression == "snappy" else f"sf{scale_factor}-{compression}"
    return f"tpch-data/{sf_dir}/{file_prefix_map[table]}"


def list_source_files(s3_client, bucket: str, key_prefix: str) -> list[str]:
    """Real S3 listing, not a shell-glob string -- pyarrow.dataset.dataset()
    does not expand "*" wildcards itself when given an explicit
    filesystem= object (confirmed by a real FileNotFoundError before this
    was added); it needs an explicit list of paths."""
    paths = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                paths.append(f"{bucket}/{obj['Key']}")
    if not paths:
        raise FileNotFoundError(f"no .parquet files under s3://{bucket}/{key_prefix}")
    return paths


def create_iceberg_table(catalog, namespace: str, table_name: str, pa_schema):
    from pyiceberg.io.pyarrow import _ConvertToIcebergWithoutIDs, visit_pyarrow
    from pyiceberg.partitioning import PartitionField, PartitionSpec, UNPARTITIONED_PARTITION_SPEC
    from pyiceberg.schema import assign_fresh_schema_ids
    from pyiceberg.transforms import YearTransform

    identifier = f"{namespace}.{table_name}"
    try:
        catalog.drop_table(identifier)
        print(f"  (dropped pre-existing {identifier})", file=sys.stderr)
    except Exception:  # noqa: BLE001 -- table simply not present yet, the common case
        pass

    # _ConvertToIcebergWithoutIDs() assigns the SAME placeholder id (-1) to
    # every field -- fine for a schema with one field, actively wrong for
    # a real multi-column table: id-based lookups against it are
    # ambiguous, and silently resolve to whichever field an internal
    # id->name dict happened to process last (confirmed for real: this
    # previously resolved lineitem's partition source to l_comment
    # instead of l_shipdate, and Iceberg only caught it at table-creation
    # time via "Invalid source field l_comment ... for transform: year",
    # not at the point of the actual mistake). assign_fresh_schema_ids()
    # gives every field a real, unique, sequential id -- passing THAT
    # schema (a real pyiceberg.schema.Schema, not a bare pyarrow schema)
    # to create_table() also makes it skip its own internal re-conversion
    # entirely (Catalog._convert_schema_if_needed() returns early for an
    # already-Schema instance), so there is no second, independent
    # conversion that could disagree with the id this function computed.
    # Verified end-to-end (correct partition values, correct scan-time
    # file pruning, using a schema shaped like the real bug case -- shipdate
    # not last, a string column after it) before trusting this fix.
    schema = assign_fresh_schema_ids(visit_pyarrow(pa_schema, _ConvertToIcebergWithoutIDs()))

    partition_spec = UNPARTITIONED_PARTITION_SPEC
    if table_name in PARTITION_COLUMNS:
        source_col = PARTITION_COLUMNS[table_name]
        source_id = schema.find_field(source_col).field_id
        partition_spec = PartitionSpec(
            PartitionField(
                source_id=source_id,
                field_id=1000,
                transform=YearTransform(),
                name=f"{source_col}_year",
            )
        )

    print(f"  Creating {identifier} (partitioned by {PARTITION_COLUMNS.get(table_name, '<none>')})...", file=sys.stderr)
    return catalog.create_table(
        identifier=identifier,
        schema=schema,
        partition_spec=partition_spec,
        properties={"write.parquet.compression-codec": "zstd"},
    )


def write_table_batched(catalog, namespace: str, table_name: str, s3fs, paths: list[str]) -> None:
    import pyarrow.dataset as ds

    # Schema only -- reading one file's footer, not its data, to create
    # the table before the batched append loop below.
    schema = ds.dataset([paths[0]], filesystem=s3fs, format="parquet").schema
    table = create_iceberg_table(catalog, namespace, table_name, schema)

    total_rows = 0
    for i in range(0, len(paths), FILES_PER_BATCH):
        batch_paths = paths[i : i + FILES_PER_BATCH]
        batch_data = ds.dataset(batch_paths, filesystem=s3fs, format="parquet").to_table()
        table.append(batch_data)
        total_rows += batch_data.num_rows
        print(f"  {table_name}: batch {i // FILES_PER_BATCH + 1}/{-(-len(paths) // FILES_PER_BATCH)} "
              f"({len(batch_paths)} files, {batch_data.num_rows} rows) -- running total {total_rows}", file=sys.stderr)
    print(f"  {table_name}: {total_rows} rows written across {len(paths)} source files.", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--source-compression", default="snappy", choices=["snappy", "zstd"],
                         help="Which already-uploaded flat dataset to read from (see generate_and_upload_data.sh)")
    parser.add_argument("--catalog-uri", required=True, help="e.g. http://<iceberg-catalog-ip>:8181")
    parser.add_argument("--warehouse", required=True, help="e.g. s3://<bucket>/warehouse/")
    parser.add_argument("--catalog-name", default="bench")
    parser.add_argument("--aws-region", required=True)
    args = parser.parse_args()

    import boto3
    import pyarrow.fs as pafs
    from pyiceberg.catalog import load_catalog

    catalog = load_catalog(
        args.catalog_name,
        **{"uri": args.catalog_uri, "warehouse": args.warehouse, "s3.region": args.aws_region},
    )
    namespace = f"tpch_sf{args.scale_factor}"
    catalog.create_namespace_if_not_exists(namespace)

    # Default credential chain (this instance's own IAM role), same
    # convention as every other S3 access in this project.
    s3fs = pafs.S3FileSystem(region=args.aws_region)
    s3_client = boto3.client("s3", region_name=args.aws_region)

    for table_name in TABLES:
        key_prefix = s3_source_key_prefix(args.scale_factor, table_name, args.source_compression)
        paths = list_source_files(s3_client, args.s3_bucket, key_prefix)
        print(f"=== {table_name}: {len(paths)} file(s) under s3://{args.s3_bucket}/{key_prefix} ===", file=sys.stderr)
        write_table_batched(catalog, namespace, table_name, s3fs, paths)

    print(f"=== Done: {namespace} written to {args.catalog_name} ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
