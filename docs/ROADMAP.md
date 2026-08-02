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
  `PhysicalPlan`), including remapping `ColumnExpression` indices above the
  scan to the narrowed (projection-pushed-down) physical schema
- GPU dependency vendoring (`cmake/ThirdPartyRapids.cmake`): libcudf, RMM,
  kvikio, nvcomp, and rapids-logger fetched from pinned PyPI wheels, no
  conda
- `ExecutionContext`, `DeviceBatch`, the Arrow<->cudf C-Data-Interface
  bridge, RAII CUDA device/stream wrappers, and RMM memory-pool/statistics/
  limit configuration (`kernellake_memory`, `kernellake_execution`)
- Every GPU operator the MVP query needs: `ParquetScanOperator`,
  `FilterOperator`, `ProjectionOperator`, `ScalarAggregateOperator`
  (cross-batch via `cudf::reduce`'s `init` param), `HashAggregateOperator`
  (cross-batch via `cudf::groupby::streaming_groupby`, bounded by
  `max_distinct_keys`), `LimitOperator`, `ArrowResultOperator` -- see
  "GPU operators" in `docs/ARCHITECTURE.md` for the two correctness
  bugs (STRING columns through `cudf::ast`, and scan column-index
  remapping) this surfaced and fixed
- `ORDER BY` execution: `SortOperator` (`cudf::stable_sorted_order` +
  `cudf::gather`, blocking), on both plain queries and after `GROUP BY`
  (scoped to a SELECT-list output name -- see `docs/ARCHITECTURE.md`).
  Surfaced and fixed two more latent physical-planner bugs in the process:
  `LogicalProjection`'s and `LogicalSort`'s conversion both unconditionally
  remapped their expressions against the scan schema, which is wrong for
  the aggregate-reprojection/post-aggregation-sort case (their expressions
  already reference the aggregate's own output schema) -- previously
  masked because every prior test happened to hit the case where the
  optimizer removes the redundant reprojection entirely. Verified against
  DuckDB for the surviving-reprojection case.
- `LIKE`/`NOT LIKE`, `IN`/`NOT IN` (desugared to `OR`/`AND` chains at bind
  time), `CASE WHEN ... END` (simple and searched, `SELECT` list and
  `GROUP BY` keys), and explicit `CAST` -- see "LIKE/IN/CASE/CAST
  implementation notes" in `docs/ARCHITECTURE.md` for scope limits, the
  discovery that `cudf::ast::compute_column` cannot produce STRING output
  even for a pure literal (worked around for `CASE`'s literal branches via
  a direct-scalar fast path), and the documented CAST-truncates-vs-DuckDB-
  rounds semantic difference. Also added `GROUP BY <alias>` resolution
  (base-table column first, `SELECT`-list alias fallback), needed to group
  by a computed `CASE` expression. All four surfaced and fixed real bugs:
  a `LikeExpression`-only `WHERE` column was silently dropped by the
  optimizer's required-columns pass, `GROUP BY <alias>` didn't work at all
  (pre-existing, not CASE-specific), `HashAggregateOperator` couldn't
  materialize a `CASE` group-by key, and a STRING `CASE` branch crashed the
  process (`std::terminate`) rather than failing cleanly. Verified against
  DuckDB. Also found and fixed, while debugging the STRING-CASE crash: no
  uncaught non-`KernelLakeError` exception (cudf/rmm/Arrow's own types)
  handler existed in the CLI, so such an exception crashed the whole
  process instead of printing a clean error -- fixed with a single
  top-level `try/catch` around command dispatch in `src/cli/main.cpp`.
- Fixed a real, latent GPU-memory-teardown race found while running the
  full `gpu-dev` test suite (not the task above's SQL work): `RmmEnvironment`'s
  destructor restored the previous CUDA device memory resource and freed
  its own pool without first synchronizing the device, so GPU work still
  in flight when one test's `RmmEnvironment` went out of scope could read
  or write memory that got freed and reissued to a *later*, unrelated
  test's pool -- manifesting as a driver-level segfault deep inside a much
  later test rather than a failure at the source. Every test passed
  individually or in most small combinations, since there was no
  still-in-flight work left to race against; reproduced deterministically
  with `ParquetScanOperatorTest` + `LimitOperator` + `SortOperator` run
  together, confirmed present before this session's changes (bisected to
  the last commit), and fixed with a `cudaDeviceSynchronize()` in
  `RmmEnvironment::~RmmEnvironment()` before releasing pool memory.
- `DECIMAL(p, s)` type support in GPU execution: columns, literals,
  comparisons, arithmetic, `SUM`/`MIN`/`MAX`, and explicit
  `CAST(... AS DECIMAL(p, s))` -- see "DECIMAL support" in
  `docs/ARCHITECTURE.md` for the full scope (implicit promotion only
  coerces literals, not columns; `AVG` over DECIMAL is not supported; CAST
  *to* DECIMAL is scoped like `CASE`, materialized outside `cudf::ast`
  since it has no `CAST_TO_DECIMAL*` operator). Verified against a real
  DECIMAL128 Parquet column (`tests/gpu/decimal_test.cpp`) and cross-checked
  against DuckDB. Surfaced and fixed two real bugs found via that testing:
  (1) `combine_binary`'s comparison path in binder.cpp treated any two
  DECIMALs as directly comparable by checking only `TypeId` equality, not
  precision/scale, letting two different DECIMAL types silently skip the
  mismatch check entirely (the identical bug, already present, was also
  just fixed in `BETWEEN`'s bound-unification); (2) none of this was
  DECIMAL-specific, but debugging it surfaced that `ORDER BY <alias>` for a
  computed expression only resolves after `GROUP BY`, not on a plain query
  -- already true and documented before this work, not a new limitation,
  but not obvious until a DECIMAL `CAST` test tripped over it.
- Groundwork for a future long-lived server (Phase 0 of a larger,
  in-progress epic -- see "Not yet started" below for the rest): split
  `QueryEngine::execute(sql)` into a reentrant `explain(sql)` (already
  existed) plus a new `execute(const PhysicalPlanPtr&, RmmEnvironment&)`
  overload taking an *externally owned* `RmmEnvironment`, so a future
  server can construct one `RmmEnvironment` at startup and reuse it across
  requests instead of racing construction/destruction of it per request
  (see "Concurrency" in `docs/ARCHITECTURE.md` for why that race is real,
  not hypothetical). Also added real per-operator timing: every operator
  the tree builds is now wrapped in a generic `InstrumentedOperator`
  (`operator_builder.cpp`) that records wall-clock `next()` time into a new
  `MetricsRegistry` (`ExecutionContext::metrics`, previously
  forward-declared but never implemented) and emits an NVTX range when
  `ProfilingSection::nvtx` is set (also previously an unused config bool).
  This finally populates `QueryResult`'s `metadata_inspection_seconds`,
  `parquet_decoding_seconds`, `gpu_execution_seconds`, and
  `device_to_host_seconds` fields (verified via `kernellake query --stats`
  against the real SF1 TPC-H data from the entry above -- e.g. a `GROUP BY
  l_returnflag` query showed `parquet_decoding_seconds: 0.133`,
  `gpu_execution_seconds: 0.187`, `elapsed_wall_seconds: 0.340`, internally
  consistent as expected). `host_to_device_seconds` stays a documented
  `nullopt` -- there is no separate host-to-device transfer phase in the
  current architecture to time, not a gap that was missed.
- Hash joins: a two-table `INNER JOIN ... ON <a.col = b.col>` (single
  equality key, both sides aliased `read_parquet(...)` sources) end to end --
  a rewritten multi-occurrence `read_parquet(...)` regex adapter (parser.cpp),
  qualified-column-aware binding with a combined `[left..., right...]`
  column-index space (new `LogicalJoin`/`HashJoinNode`/`HashJoinOperator`),
  and per-side required-column splitting in the optimizer. See "Hash joins"
  in `docs/ARCHITECTURE.md` for the full scope (INNER only, one equality
  key, no comma-style joins, no predicate pushdown across the join) and
  known limitations (same-named columns on both sides can't be referenced
  unqualified after the join). `HashJoinOperator` is the first operator
  whose *build* side is blocking (like `SortOperator`) while its *probe*
  side streams normally, built on `cudf::hash_join`. Verified end-to-end
  against real two-file Parquet fixtures and cross-checked against DuckDB
  (`tests/gpu/hash_join_test.cpp`, `tests/unit/binder_test.cpp`,
  `tests/unit/sql_parser_test.cpp`). Caught and fixed one real bug in the
  process: `operator_builder.cpp` would have built the join's two child
  operator trees as two arguments of the same constructor call, which is
  unspecified evaluation order in C++ and both recursive calls mutate a
  shared `next_id` counter as a side effect -- fixed by sequencing them as
  separate statements before either build() call actually ran.
- `QueryEngine::execute()`, wired to the real operator pipeline for
  `gpu-dev` builds (a CPU-only stub throws `ExecutionError` for `dev`
  builds), and the `kernellake query` CLI command (`--sql`/`--file`,
  `--format table|csv|jsonl|arrow`, `--output`, `--stats`) -- **the spec's
  required initial deliverable now runs end-to-end and has been verified
  on real GPU hardware**
- `kernellake generate-data`: deterministic synthetic dataset generator
  (`order_id`/`customer_id`/`region`/`amount`/`event_date`/`event_time`/
  `category`/`discount`), configurable row/file/row-group count,
  cardinality, null rate, skew, and seed; CPU-only, no CUDA dependency
- DuckDB cross-validation tooling (`tools/validate_against_duckdb.py`):
  runs the same SQL through `kernellake query --format arrow` and DuckDB
  reading the same Parquet files, compares results. Verified passing for
  filter/projection, scalar aggregates (SUM/COUNT/AVG/MIN/MAX), grouped
  aggregates (including high-cardinality `GROUP BY`), and `COUNT` over a
  nullable column

- TPC-H tooling (`lineitem`-only -- Q6 and Q1, both single-table; Q3/Q12/Q14
  need multi-table joins across TPC-H's actual schema, which goes beyond
  the two-table single-equality-key scope hash joins currently support --
  see "Hash joins" in `docs/ARCHITECTURE.md` -- so still not wired up):
  `tools/generate_tpch.py` (synthetic
  generator, not real `dbgen`), `benchmarks/tpch/queries/{q01,q06}.sql`,
  `tools/validate_tpch.py` (DuckDB cross-validation; the spec's `kernellake
  validate tpch` as a Python tool rather than a CLI subcommand, same choice
  as `validate_against_duckdb.py`), and `kernellake benchmark tpch`
  (cold/warm modes with median/mean/min/max/stddev over configurable
  iterations; `execution-only` mode is not implemented -- see
  `docs/TPCH.md`). Verified: Q1 and Q6 both match DuckDB at SF0.01, SF0.1,
  and real SF1 (6,000,000 generated `lineitem` rows, ~105 MiB zstd-compressed
  Parquet) -- see `docs/TPCH.md` for the SF1 correctness run and indicative
  benchmark numbers (Q6 ~75-100ms; Q1 ~0.2-2s, with real observed variance
  documented rather than smoothed over).

- `LICENSE` (Apache 2.0), `NOTICE`, `THIRD_PARTY_LICENSES.md` (every
  dependency actually declared in the build, verified against each
  package's own license metadata rather than assumed -- including flagging
  that the CUDA Toolkit and RAPIDS's vendored `nvcomp` component are
  NVIDIA proprietary SDK/EULA dependencies, not open source)
- `.clang-format` (the whole existing C++ tree has been reformatted to
  match and is currently clean); `.clang-tidy` config, spot-checked with a
  real `clang-tidy-18` run against a compiled source file (one real finding,
  34,609 non-user-code warnings correctly suppressed by `HeaderFilterRegex`)
  but not run across the whole tree and not wired into CI yet; optional
  `.pre-commit-config.yaml` (clang-format only)
- `docker/Dockerfile` (single file, multi-stage, two named targets:
  `dev` -- full CUDA devel image with the repo built inside it -- and
  `runtime` -- only the built binary plus its actual non-system runtime
  dependency closure, resolved via `ldd` rather than hard-coded vendored
  paths) and `.github/workflows/ci.yml`, one workflow with four jobs in
  dependency order: `format-check` and `cpu-build-test` (parallel,
  CPU-only build+test on the `dev` preset) -> `tpch-tooling-smoke`
  (`needs: cpu-build-test`; small-scale `generate_tpch.py` run plus
  `kernellake explain` -- not `query` -- against both query files) ->
  `docker-publish` (`needs:` all three, `if: github.event_name !=
  'pull_request'`; builds both `docker/Dockerfile` targets and pushes
  them to `ghcr.io/<owner>/<repo>:dev` and `:runtime`/`:latest` using
  `GITHUB_TOKEN`). **Confirmed on real GitHub Actions infrastructure**
  (run `30718829266`, 2026-08-01): all four jobs succeeded end to end,
  including `docker-publish` (~10 minutes, no runner-disk issues) --
  both images are live at `ghcr.io/hurdad/kernel-lake:dev`/`:runtime`/
  `:latest`. Cross-workflow `needs:` isn't a GitHub Actions feature, which
  is why this is one file rather than a separate docker-publish.yml
  gated some other way. GPU-dependent work (the `gpu-dev` preset,
  `tests/gpu/`, real query execution, DuckDB validation, TPC-H
  benchmarks) is intentionally not in this workflow -- standard
  GitHub-hosted runners have no GPU, and a skipped-GPU job must never be
  reported as passing; that needs a self-hosted GPU runner, not
  configured here (see "Not yet started")

- `tests/gpu/multi_batch_integration_test.cpp`: dictionary-encoded vs. plain
  string columns, two files with deliberately mismatched row-group layouts
  (3 row groups of 100 rows vs. 1 row group of 250 rows), and a forced
  multi-pass `ParquetScanOperator` (a 256-byte `pass_read_limit_bytes`
  against ~550 rows' worth of data) -- one test proves the scan genuinely
  emits more than 2 batches under these settings (not just 1-per-file),
  another runs the real `build_physical_plan` + `build_operator_tree`
  pipeline (the same one `QueryEngine::execute()` uses) end-to-end across
  those same conditions and checks the grouped sums/counts by hand

## Not yet started

- TPC-H `execution-only` benchmark mode (needs an operator-tree entry point
  that skips `ParquetScanOperator`); TPC-H at SF10+ scale (SF1 now verified,
  see "Done" above; SF10 untested); Q3/Q12/Q14 (need TPC-H's actual
  multi-way join schema wired up against hash joins' current two-table
  scope)
- A self-hosted GPU CI runner (would enable a `gpu-dev` build/test/
  benchmark/validate workflow to actually run in CI, rather than only
  locally) -- explicitly deferred; `hurdad/kernel-lake` is a public repo,
  and a self-hosted runner must never be reachable from `pull_request`
  events (only `push` to `main`, after review) or an outside contributor
  gets code execution on the runner's owner's hardware
- Running `clang-tidy` across the whole tree and wiring it into CI (config
  spot-checked, not exhaustively run)
- `docker run --gpus all` of the published `runtime` image against a real
  GPU (the image builds and pushes successfully in CI; actually running it
  hasn't been checked yet)
- **An Arrow Flight SQL server, otel-cpp observability, and a Helm chart**
  (a user-directed initiative beyond the original MVP scope -- see the
  "Explicit non-goals" note below on Flight SQL/Kubernetes). Phase 0
  (above) is done; still to build, in order: (1) the Flight SQL server
  itself (new `kernellake-server` binary, `libarrow-flight-sql-dev`/
  `libgrpc++-dev` from the same Apache Arrow apt repo already in use --
  confirmed available, no `FetchContent` vendoring needed for Flight
  itself), gated on a version-compatibility spike against Ubuntu 24.04's
  older system gRPC; (2) otel-cpp integration (vendored via `FetchContent`,
  building on Phase 0's `MetricsRegistry`/NVTX instrumentation points); (3)
  a Helm chart deploying the server as a Deployment+Service with GPU
  scheduling, blocked on (1). A related question -- bumping the Ubuntu
  baseline from 24.04 to 26.04 for apt-native otel-cpp and Ubuntu's own
  `nvidia-cuda-toolkit` package instead of the official `nvidia/cuda`
  Docker image -- was deliberately deferred: otel-cpp vendors cleanly on
  24.04 already, and swapping the CUDA toolchain source is a real,
  unverified risk with no Docker available in this project's dev
  environment to test it against.

## Explicit non-goals for the MVP

Distributed execution, multi-node/multi-GPU scheduling, a Kubernetes
*operator* (custom controller/CRDs -- the plain Helm chart in progress
above is a Deployment+Service, not an operator), full Iceberg catalog
integration, joins beyond a two-table `INNER JOIN` with a single equality
key (see "Hash joins" in `docs/ARCHITECTURE.md`), all 22 TPC-H queries,
cost-based optimization,
fault-tolerant fragment
retries, query spilling, materialized views, transactions, data ingestion
(`INSERT`/`UPDATE`/`DELETE`), a proprietary storage format, and a web UI.
Interfaces may exist for some of these (see "Future architecture" in
`docs/ARCHITECTURE.md`) but none are implemented or exposed as supported
features.
