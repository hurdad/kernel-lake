# KernelLake roadmap

Status as of the last update to this file. "Done" means built, compiled,
and covered by passing tests -- not merely designed or stubbed.

## Done

- Repository scaffolding, root CMake configuration, `CMakePresets.json`
  (`dev` / `release` / `gpu-dev`)
- Error hierarchy, strongly typed identifiers, minimal CLI skeleton
- YAML configuration loading and validation, spdlog-based logging
- Internal type system (`TypeId`/`DataType`/`Schema`) with Arrow adapters
- Typed expression tree (columns, literals, binary/unary ops, `CAST`,
  `BETWEEN`, aggregates)
- SQL parsing (vendored `hyrise/sql-parser`, MIT, pinned commit) with a
  `read_parquet(...)` syntax adapter and a parser-independent AST
- Binder / type checker: column resolution, aggregate/`GROUP BY`
  validation, safe implicit numeric casts
- Logical plan nodes with human-readable and JSON `EXPLAIN` output
- Rule-based optimizer: constant folding, boolean simplification, `BETWEEN`
  simplification, adjacent-filter combination, trivial-filter removal,
  redundant-projection removal, `LIMIT` pushdown, projection/predicate
  pushdown annotations
- Local object store (`LocalObjectStore`) and file discovery (single file,
  glob, directory; deterministic ordering; explicit errors on missing paths)
- Real Parquet metadata inspection (schema, row groups, column statistics)
  and row-group/file pruning against actual min/max statistics
- Physical plan nodes and the physical planner (`LogicalPlan` + pruning ->
  `PhysicalPlan`)
- `QueryEngine::explain_logical()` / `QueryEngine::explain()`, wired into
  `kernellake explain` and `kernellake inspect-parquet`
- GPU dependency vendoring (`cmake/ThirdPartyRapids.cmake`): libcudf, RMM,
  kvikio, nvcomp, and rapids-logger fetched from pinned PyPI wheels, no
  conda, verified with a real GPU-resident `cudf::column` allocation

## GPU dependencies resolved; GPU operators not yet implemented

The libcudf/RMM dependency question is solved: vendored via CMake
`FetchContent` from pinned RAPIDS PyPI wheels (no conda), see
[docs/architecture.md](docs/architecture.md#gpu-dependency-vendoring-no-conda)
and `cmake/ThirdPartyRapids.cmake`. Verified end-to-end with a real
GPU-resident `cudf::column` allocated and inspected on actual hardware via
the `gpu-dev` CMake preset. What's still not started:

- `ExecutionContext` and the `PhysicalOperator` streaming interface (both
  need `DeviceBatch`, wrapping `cudf::table`, to be a complete type)
- RAII CUDA device/stream/event/NVTX wrappers, RMM memory-pool
  configuration
- GPU operators: `ParquetScanOperator`, `FilterOperator`,
  `ProjectionOperator`, `ScalarAggregateOperator`, `HashAggregateOperator`,
  partial-aggregate merging, `LimitOperator`, `ArrowResultOperator`
- `QueryEngine::execute()` (currently throws `ExecutionError` explaining
  why) and the `kernellake query` CLI command
- Larger-than-GPU-memory batch iteration

## Not yet started (not GPU-blocked)

- `kernellake generate-data`: deterministic sample Parquet dataset generator
- DuckDB-based correctness validation (needs DuckDB installed; currently
  missing from this environment) and broader integration tests (multiple
  files/row-groups, nulls, dictionary-encoded strings)
- TPC-H tooling: `tools/generate_tpch.py`, `benchmarks/tpch/queries/*.sql`,
  `kernellake validate tpch`, `kernellake benchmark tpch` (Q6 first, then
  Q1; the query-file management and generator scaffolding don't strictly
  need GPU execution, but validation/benchmarking do)
- Docker images (`docker/Dockerfile.dev`, `docker/Dockerfile.runtime`)
- GitHub Actions CI (CPU-safe checks split from CUDA compile checks and GPU
  runtime tests)
- `NOTICE` / `THIRD_PARTY_LICENSES.md`, `clang-format`/`clang-tidy` config

## Explicit non-goals for the MVP

Distributed execution, multi-node/multi-GPU scheduling, Kubernetes
operators, full Iceberg catalog integration, Arrow Flight SQL, joins, all
22 TPC-H queries, cost-based optimization, fault-tolerant fragment
retries, query spilling, materialized views, transactions, data ingestion
(`INSERT`/`UPDATE`/`DELETE`), a proprietary storage format, and a web UI.
Interfaces may exist for some of these (see "Future architecture" in
`docs/architecture.md`) but none are implemented or exposed as supported
features.
