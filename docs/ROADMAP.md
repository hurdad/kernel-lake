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
- A real CPU execution backend (Phase 1 of the CPU-backend/benchmark epic --
  see "Not yet started" below for Phase 2): `src/execution_cpu/` translates
  a `PhysicalPlanPtr` directly into an `arrow::acero::Declaration` tree and
  runs it via `arrow::acero::DeclarationToTable()` (Apache Arrow's own
  CPU-native streaming engine), rather than hand-rolled CPU operators.
  Always built, in both `dev` and `gpu-dev` presets, needing no CUDA at
  all; selected at runtime via `engine.backend: "cpu"` or `kernellake query
  --backend cpu`. Covers `ParquetScan`/`Filter`/`Projection` (arithmetic
  only)/`ScalarAggregate`/`HashAggregate`/`Sort`/`Limit` -- the same MVP
  scope the GPU engine started with; `LIKE`/`IN`/`CASE`/`CAST`-to-`DECIMAL`-
  or-`STRING`/`HashJoin` explicitly not yet supported (throws naming the
  construct). Two real bugs surfaced only by running actual queries, not by
  reading Arrow's headers: (1) Arrow Compute's kernel functions don't
  self-register when statically linked -- every query failed with "No
  function registered with name: ..." until `arrow::compute::Initialize()`
  was called explicitly; (2) `COUNT(*)` needs the dedicated `count_all`/
  `hash_count_all` functions, since `count`/`hash_count` reject an empty
  target with "accepts 2 arguments but 1 passed". Cross-backend parity is
  verified directly in `tests/gpu/query_engine_execute_test.cpp` (same SQL,
  same Parquet fixture, both backends, values compared) -- see "CPU
  execution backend" in `docs/ARCHITECTURE.md` for the full design,
  including why the parity check compares column values rather than full
  schema equality.
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
  match and is currently clean); `.clang-tidy` config, now run for real
  across every CPU/server/otel-buildable `src/*.cpp` (`dev`, `server-dev`,
  and `otel-dev` presets -- everything except `src/execution_gpu/`,
  `src/memory/rmm_environment.cpp`, and
  `src/api/query_engine_execute_gpu.cpp`, which need `gpu-dev`/real
  libcudf/RMM not available in this environment) via a real
  `run-clang-tidy-18` run, zero warnings after fixing every real finding
  (233 `readability-braces-around-statements` and friends auto-fixed;
  a genuine dead-code bug caught and fixed --
  `LogicalPlanNode::explain_text_recursive`'s `is_root` ternary evaluated
  to the same string on both branches, confirmed by comparing against
  `PhysicalPlanNode`'s equivalent, which has no such branch at all; several
  `bugprone-branch-clone`/`bugprone-optional-value-conversion` simplifications;
  `bugprone-unchecked-optional-access` findings resolved with documented
  NOLINTs where an invariant established elsewhere in the code already
  guarantees safety, not blanket-suppressed; a project-wide
  `CMAKE_CXX_SCAN_FOR_MODULES OFF` needed first, since CMake >= 3.28's
  Ninja module-dependency-scanning flags in `compile_commands.json`
  otherwise make every file unparseable by clang-tidy directly) and wired
  into both CI (a standalone `clang-tidy` job plus a scoped step each in
  `server-build-test`/`otel-build-test` for their own preset-only files)
  and `.pre-commit-config.yaml` (now clang-format **and** clang-tidy,
  scoped to `build/dev`'s own file set for the same reason)
- `docker/Dockerfile` (single file, multi-stage, publishing two runtime
  targets from shared build stages: `runtime-cpu` -- no CUDA/RAPIDS at all,
  456 MB, built and run for real in this session -- and `runtime-gpu` --
  full CUDA/RAPIDS closure, 2.17 GB (carried over from this same stage's
  pre-restructuring size; its own build steps are unchanged by this
  restructuring, just moved behind the new shared `dev-base` stage and
  renamed, so not independently re-measured here); the `dev-cpu`/`dev-gpu`
  build stages, 14.1 GB for the GPU one (same caveat), are
  intermediate-only and never published) and `.github/workflows/ci.yml`,
  one workflow with seven jobs in dependency order: `format-check` and
  `clang-tidy` (independent lint jobs) and `cpu-build-test` (CPU-only
  build+test on the `dev` preset) -> `tpch-tooling-smoke` (`needs:
  cpu-build-test`; small-scale `generate_tpch.py` run plus `kernellake
  explain` -- not `query` -- against both query files), `server-build-test`,
  `otel-build-test` -> `docker-publish` (`needs: [format-check,
  cpu-build-test, tpch-tooling-smoke]`, `if: github.event_name !=
  'pull_request'`; builds `runtime-cpu`/`runtime-gpu` and pushes them to
  `ghcr.io/<owner>/<repo>-cpu:latest` and `-gpu:latest` using
  `GITHUB_TOKEN`). `kernel-lake-cpu:latest` is a multi-arch
  (`linux/amd64`+`linux/arm64`) manifest via `docker buildx`
  (`docker/setup-qemu-action`) -- every `runtime-cpu` dependency is a plain
  apt package with a real arm64 build on Ubuntu 26.04, confirmed by a real
  `docker buildx build --platform linux/arm64 --target runtime-cpu`
  completing successfully (10m34s wall clock, apt installs plus ~90 C++
  translation units including Arrow/Parquet/Flight/gRPC/OpenTelemetry
  headers) via QEMU user-mode emulation (`tonistiigi/binfmt`) on this
  project's own amd64 development machine, with the resulting image then
  run for real too (`docker run --platform linux/arm64`, `--help` and
  `uname -m` both correct) -- no arm64 hardware or CI runner was used.
  `kernel-lake-gpu` stays `linux/amd64`-only: CUDA/
  RAPIDS' own arm64 support (nvidia-cuda-toolkit/libcufile-dev's arm64 apt
  packages, `cmake/ThirdPartyRapids.cmake`'s pinned `-cu12` wheels) hasn't
  been verified, and unlike the CPU image a build-only smoke test can't
  catch a real GPU-execution regression anyway -- that needs real arm64 GPU
  hardware (NVIDIA Grace/Jetson), not available here (see "Not yet
  started"). `runtime-cpu` was verified for real in this session (a real
  `docker build`, `generate-data`, and `query --backend cpu` against real
  generated data all produced correct output through the built image).
  **The previous, pre-clang-tidy/pre-cpu-gpu-split version of this workflow
  was separately confirmed on real GitHub Actions infrastructure** (run
  `30718829266`, 2026-08-01, all four of its jobs succeeded end to end,
  including `docker-publish` publishing `ghcr.io/hurdad/kernel-lake:dev`/
  `:runtime`/`:latest`, ~10 minutes, no runner-disk issues) -- re-confirming
  CI on real GitHub Actions with this session's clang-tidy/docker-split
  changes is still pending. Cross-workflow `needs:` isn't a GitHub Actions
  feature, which is why this is one file rather than a separate
  docker-publish.yml gated some other way. GPU-dependent work (the
  `gpu-dev` preset, `tests/gpu/`, real query execution, DuckDB validation,
  TPC-H benchmarks) is intentionally not in this workflow -- standard
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
- Bumped `docker/Dockerfile`'s baseline from Ubuntu 24.04/NVIDIA's official
  `nvidia/cuda` images to plain `ubuntu:26.04` with CUDA installed via
  apt's own `nvidia-cuda-toolkit` (12.4.1) -- resolves two real, confirmed
  blockers (Arrow Flight SQL doesn't link against Ubuntu 24.04's system
  Abseil; otel-cpp has no 24.04 apt package at all) without a CUDA
  major-version bump or RAPIDS re-vendor (12.4.1 vs. the previously-pinned
  12.6.3 are both CUDA 12.x, same `-cu12` wheels). Verified for real with
  Docker, not inferred from package metadata: `docker build --target dev`
  builds all 103 targets; `docker build --target runtime` produces a
  shared-library-only 2.17 GB image (down from `dev`'s 14.1 GB, no
  compiler/nvcc/headers); `docker run --gpus all` against a real GPU (RTX
  5060 Ti, a Blackwell/sm_120 card newer than CUDA 12.4's nvcc can target
  directly -- handled via embedded PTX + driver JIT) ran all 214 tests
  successfully, and a real query against real generated data through the
  `runtime` image alone produced correct GPU-executed results. See
  `docs/ARCHITECTURE.md`'s "Ubuntu 26.04 baseline" section for the full
  investigation, including the two extra apt packages this needed
  (`gnupg`, for the Arrow apt source's postinst; `libcufile-dev`, for
  kvikio's GPUDirect Storage dependency, not pulled in by
  `nvidia-cuda-toolkit` itself). This project's own non-container `dev`/
  `gpu-dev` presets and this sandbox's CUDA install are unaffected --
  still Ubuntu 24.04, unchanged.
- **`kernellake-server`: an Arrow Flight SQL server wired to `QueryEngine`**
  (Phase 1 of the Flight SQL/otel-cpp/Helm-chart epic below -- Phase 0,
  above, was already done; Phases 2-3 remain, see "Not yet started"). This
  session's own sandbox has itself since moved to Ubuntu 26.04 (`lsb_release`:
  `resolute`), correcting the prior entry above's assumption that the
  non-container dev environment would stay on 24.04 -- `server-dev` (new
  preset, `KERNELLAKE_BUILD_SERVER=ON`) builds and tests directly here, no
  Docker required, now that `libarrow-flight-dev`/`libarrow-flight-sql-dev`/
  `libgrpc++-dev` are all real installable apt packages on this host.
  `KernelLakeFlightSqlServer` (`src/server/flight_sql_server.cpp`)
  implements `GetFlightInfoStatement`/`DoGetStatement` -- execute-eagerly
  in the first RPC, stream-from-an-in-process-registry in the second, a
  deliberate Phase 1 simplification over a live cursor -- translating
  `KernelLakeError` subclasses to matching `arrow::Status` codes so no raw
  C++ exception crosses the gRPC boundary. Respects `engine.backend:
  gpu|cpu` exactly like the CLI's `--backend` flag; for `gpu`, a new
  `GpuExecutionCoordinator` owns the one long-lived `RmmEnvironment` a
  concurrent server needs (single-flight mutex), split gpu/stub like every
  other CUDA-conditional module so `server-dev` needs no CUDA at all. Also
  needed a real, non-obvious CMake fix: Arrow's `ArrowFlight`/`ArrowFlightSql`
  targets don't declare `gRPC::grpc++` as a dependency, and CMake's
  `$<LINK_GROUP:RESCAN,...>` genex turned out not to pull a grouped target's
  *own* transitive link dependencies inside the group boundary (confirmed by
  inspecting the generated link command) -- fixed with raw
  `-Wl,--start-group`/`--end-group` strings instead, which also sidesteps a
  false dependency-cycle CMake's LINK_GROUP analysis reported (from
  `Arrow::arrow_static` being shared across the whole `kernellake_*` tree
  outside any group). Verified for real: 147/147 tests pass under
  `server-dev` (including two new `tests/unit/flight_sql_server_test.cpp`
  cases -- a real `arrow::flight::sql::FlightSqlClient` round trip over
  gRPC, and an invalid-SQL case asserting a clean `Invalid` status rather
  than a dropped connection), plus a manual smoke test against a running
  server using an *independent* Python `adbc_driver_flightsql` client (not
  code from this project) running a real grouped aggregate against
  `generate-data`-produced Parquet and getting correct rows back. `dev`
  (145/145) and `gpu-dev` (214/214) both reconfirmed unaffected by any of
  this (`KERNELLAKE_BUILD_SERVER` defaults `OFF`). CI coverage for
  `KERNELLAKE_BUILD_SERVER=ON` is a separate, still-open item -- see "Not
  yet started".
- **OpenTelemetry observability** (Phase 2 of the Flight SQL/otel-cpp/
  Helm-chart epic -- Phase 0 and Phase 1, `kernellake-server`, both done
  above; Phase 3, a Helm chart, still open, see "Not yet started"). Built
  behind `KERNELLAKE_ENABLE_OTEL` (default `OFF`; new `otel-dev` preset,
  independent of `KERNELLAKE_BUILD_SERVER`), sourced from
  `opentelemetry-cpp-dev` 1.23.0. One span + one histogram observation per
  whole query (`kernellake query` and the server's
  `GetFlightInfoStatement`), plus every existing `spdlog` call bridged into
  OTel's Logs signal for free via a custom sink -- see
  `docs/ARCHITECTURE.md`'s "OpenTelemetry observability" section for the
  full design, the config schema (`observability.*` -- protocol choice
  (`grpc`/`http`), TLS incl. HTTP-only mTLS, per-signal processor/batch/
  sampler tuning exposing the underlying OTel SDK knobs directly), and five
  real bugs found and fixed while implementing it (an `nostd::shared_ptr`
  overload-resolution ambiguity, this apt package's ABI version lacking an
  assumed convenience overload, a yaml-cpp nested-node-indexing throw,
  OTLP/HTTP's per-signal path-suffix requirement, and the HTTP-vs-gRPC mTLS
  availability difference). Per-operator spans are explicitly deferred, not
  attempted. Verified for real: `otel-dev` (148/148 tests, including three
  deterministic in-memory-exporter tests -- no network involved) plus a
  manual smoke test against a real `docker run`'able collector
  (`jaegertracing/all-in-one`) covering *both* protocols -- real spans with
  full `QueryResult` attributes landed for both an `OK` and an `ERROR`
  query, over OTLP/gRPC and, once a real path-suffix bug was found and
  fixed, OTLP/HTTP too. `dev` (145/145), `gpu-dev` (214/214), and
  `server-dev` (147/147) all reconfirmed unaffected. CI coverage added
  (`otel-build-test` in `.github/workflows/ci.yml`, mirroring
  `server-build-test`'s `container: ubuntu:26.04` structure -- needed for
  the same reason: `opentelemetry-cpp-dev` has no Ubuntu 24.04 apt
  package either).
- **Docker image + Helm chart** (Phase 3 of the Flight SQL/otel-cpp/
  Helm-chart epic -- Phase 0, Phase 1 (`kernellake-server`), and Phase 2
  (OpenTelemetry observability) all done above). `docker/Dockerfile`'s
  `dev`/`runtime` images now build and ship `kernellake-server` with
  `KERNELLAKE_ENABLE_OTEL=ON`, alongside the existing `kernellake` CLI --
  see `docs/ARCHITECTURE.md`, "Docker image and Helm chart", for the apt
  package list, the `runtime-libs` `ldd`-closure change to cover both
  binaries, and a real pre-existing `.dockerignore` gap found and fixed
  along the way (the host's own local `build/` directory, with real
  `CMakeCache.txt`s, was being copied into the image and breaking
  `cmake --preset gpu-dev` inside the container). `charts/kernellake/` is
  a plain Deployment+Service (explicitly not an operator, see "Explicit
  non-goals" below), with a `backend: cpu|gpu` toggle mirroring
  `engine.backend` and an `observability.*` values surface mirroring
  `ObservabilitySection`'s top-level fields. Verified for real: `docker
  build --target dev`/`--target runtime` both complete and produce working
  binaries; a real ADBC Flight SQL round trip (`COUNT(*)`, `GROUP BY`)
  against a real 5000-row Parquet file through `kernellake-server` running
  inside the `runtime` image (CPU backend) returned correct results;
  `helm lint`/`helm template` (three value combinations) piped through
  `kubeconform -strict` all passed with no schema errors. GPU backend was
  also smoke-tested through the same container and returned an incorrect,
  silently-empty `COUNT(*)` -- root-caused as a column-pruning bug (see
  the new "Bare `COUNT(*)` returns 0 on the GPU backend" item below), not
  a Docker/CUDA-architecture issue and not caused by this phase's changes.
- **Cloud object storage (S3, GCS, Azure)** (Phase 4 of the Flight SQL/
  otel-cpp/Helm-chart/cloud-storage epic -- Phases 0-3 all done above).
  `read_parquet(...)` now accepts `s3://`, `gs://`/`gcs://`, and
  `abfs://`/`abfss://`/`az://` URIs alongside local paths, dispatched by
  scheme through a new `ObjectStoreRegistry` composite, transparently at
  every existing call site (CLI, `kernellake-server`, both GPU and CPU
  execution backends) -- see `docs/ARCHITECTURE.md`, "Cloud object storage
  (S3, GCS, Azure)", for the full design (config sections embedding
  Arrow's own `S3Options`/`GcsOptions`/`AzureOptions`/`HdfsOptions`
  directly rather than hand-copied field lists; the GPU path's new
  `ObjectStoreDatasource`, a `cudf::io::datasource` wrapper preserving
  bounded-memory pass-budgeted streaming for cloud sources; the CPU path's
  drop-in `ParquetFileReader::Open()` fix) and three real, non-obvious
  problems found and fixed while implementing it: GCS/Azure support
  needing far more of Arrow's bundled dependency closet than this project
  had ever linked before (~20 additional Abseil targets plus two genuinely
  new system dependencies, `libxml2-dev`/`uuid-dev`, fixed with a new
  shared CMake helper, `cmake/LinkArrowBundledCloudDeps.cmake`);
  `arrow::io::HdfsConnectionConfig::port` having no default member
  initializer, unlike every other field in that struct, discovered via a
  real `-fpermissive` compile error and fixed before it could read
  uninitialized memory as a config fallback; and a real CLI segfault on
  exit from a missing `arrow::fs::FinalizeS3()` shutdown call, fixed with
  an `S3ShutdownGuard` mirroring Phase 2's `ObservabilityShutdownGuard`.
  Verified for real: `dev` (148/148, +3 new tests), `gpu-dev` (217/217),
  `server-dev` (150/150), `otel-dev` (151/151) all pass with zero
  regressions; real end-to-end smoke tests against three real,
  locally-`docker run`'able emulators (MinIO, `fsouza/fake-gcs-server`,
  Azurite), each on both CPU and GPU execution backends (6 combinations
  total) -- a real 5000-row Parquet file uploaded via each provider's own
  independent Python SDK, then `kernellake query` producing a correct
  `GROUP BY region` and `SUM(order_id)` (`12497500`, exact) on every
  combination, all clean exit; `kernellake inspect-parquet` against a real
  `abfs://` URI verified separately too. GPU-backend checks deliberately
  used `GROUP BY`/`SUM` rather than bare `COUNT(*)`, to avoid conflating
  this phase's own verification with the separate, already-tracked
  bare-`COUNT(*)`-on-GPU bug (see the item just below) -- this phase's
  changes are unrelated to that bug. **HDFS** also has a real
  `HdfsObjectStore` (`src/storage/hdfs_object_store.cpp`), wrapping
  `arrow::fs::HadoopFileSystem` the same way the other three backends
  wrap their own Arrow filesystem -- it compiles and links cleanly with no
  Hadoop installed at all (`HadoopFileSystem` `dlopen()`s `libhdfs.so`
  lazily at runtime, not a build-time link dependency), confirmed by a
  real `hdfs://` CLI invocation failing cleanly with a clear, non-crashing
  `StorageError` ("Unable to load libjvm") rather than any crash. What
  this project genuinely cannot do is verify it against a *real* Hadoop
  cluster: unlike MinIO/fake-gcs-server/Azurite, there is no lightweight
  single-container emulator, and this development sandbox has no
  JDK/Hadoop installation either -- real connectivity/read-correctness
  against an actual namenode is untested, disclosed here rather than
  silently assumed.
- **Project-wide `-Wl,--no-as-needed`, added while verifying the above**
  (real, found by an actual Docker build combining
  `KERNELLAKE_BUILD_SERVER=ON` and `KERNELLAKE_ENABLE_OTEL=ON` together --
  a combination no local CMake preset tests on its own, since `server-dev`
  and `otel-dev` each only turn one on). `libgrpc.so` itself needs an
  Abseil symbol (`absl::debian9::ascii_internal::kPropertyBits`) that no
  CMake target declares as a dependency, same class of gap as
  ArrowFlight/ArrowFlightSql not declaring `gRPC::grpc++`. Wrapping the
  specific libraries involved in `-Wl,--no-as-needed`/`--as-needed`
  (tried first, matching the pattern already used for libxml2/libuuid)
  was **not reliable**: confirmed by two further real Docker build
  failures, each "fixing" one undefined symbol only for CMake's own
  link-line flattening to place the *next* one wrong too, because a
  library requested by several different `kernellake_*` targets across
  the whole dependency graph gets positioned by CMake's global
  topological sort, independent of where a raw linker flag was textually
  placed in any one target's own `target_link_libraries()` call. Fixed
  with a single project-wide `add_link_options(-Wl,--no-as-needed)` in
  the root `CMakeLists.txt` instead (see its own comment) -- this
  sidesteps the positional problem entirely, at the cost of a larger
  `DT_NEEDED` list, which has no real downside for this project's
  application binaries (the `runtime` Docker image's own shared-library
  closure is already computed from actual `ldd` output, not a minimal
  `DT_NEEDED` list, either way). Verified for real: a real `docker build
  --target dev` with both flags on (matching `docker/Dockerfile`'s own
  build exactly) linked all four binaries; `docker build --target runtime`
  copied both binaries' full `ldd` closure with no missing entries; a real
  `kernellake generate-data` + `kernellake query --backend cpu` for a bare
  `COUNT(*)` through the actual `runtime` container returned the correct
  row count. `dev` (148/148), `gpu-dev` (217/217), `server-dev` (150/150),
  and `otel-dev` (151/151) all reconfirmed unaffected.
- **Bare `COUNT(*)` returning 0 on the GPU backend, fixed** (found
  incidentally while smoke-testing Phase 3's Docker/Helm work on real
  hardware; root-caused, after an initial wrong CUDA-architecture
  hypothesis was tested and ruled out, as a column-pruning bug -- see
  `docs/ARCHITECTURE.md`'s "Ubuntu 26.04 baseline" section for the full
  root-cause writeup, and its "Cloud object storage" section for how it
  was confirmed as a pre-existing, unrelated issue while verifying Phase
  4). `SELECT COUNT(*) FROM read_parquet(...)` with no other column
  reference anywhere in the query legitimately produces an empty
  `LogicalScan::required_columns()` -- correct in principle, since
  `COUNT(*)` needs no column data -- but a `cudf::table` built from zero
  selected columns has no column to derive `num_rows()` from, unlike
  `arrow::RecordBatch` (which is why the CPU backend never had this bug).
  Fixed in `src/io/physical_planner.cpp`'s `convert_scan()`: when the
  narrowed column list would be empty, one arbitrary real column (the
  schema's first field) is kept selected purely to preserve row-count
  fidelity through `cudf::table` -- inert for every consumer except row
  counting, since nothing above the scan references it (that's exactly
  why `required_columns()` was empty). Verified for real: a new regression
  test, `QueryEngineExecuteTest.BareCountStarWithNoOtherColumnReferenceMatchesRealRowCount`
  in `tests/gpu/query_engine_execute_test.cpp` (the exact query shape no
  prior test covered); a real `kernellake query --backend gpu` for a bare
  `COUNT(*)` against a real 5000-row file now returns `5000` (was `0`);
  `dev` (148/148), `gpu-dev` (218/218, +1 new test), `server-dev`
  (150/150), and `otel-dev` (151/151) all pass with zero regressions.
- **Three-way benchmark: KernelLake-CPU vs. KernelLake-GPU vs. PySpark**
  (Phase 2 of the CPU-backend/benchmark epic; Phase 1, the CPU backend
  itself, was already done). `tools/benchmark_three_way.py` reuses
  `tools/generate_tpch.py`'s dataset for all three engines and validates
  pairwise agreement across all three *before* trusting any timing number
  (this project's existing rule from `tools/validate_tpch.py`, now applying
  to three engines instead of two) -- a query whose engines disagree is
  reported as a validation failure and excluded from timing, never silently
  timed anyway. Runs inside a new `docker/Dockerfile` stage,
  `benchmark-gpu` (`dev-gpu` + OpenJDK 21 + `pyspark`/`pyarrow`/`pandas`/
  `matplotlib`, never published), so it exercises a real
  `KERNELLAKE_WITH_CUDA=ON` build's `--backend gpu` alongside `--backend
  cpu` from the same binary. `tools/generate_benchmark_report.py` renders
  the `--output` JSON from one or more scale-factor runs into a PDF
  (system-stats page, summary table, a bar chart per query across scale
  factors) via matplotlib's `PdfPages`.

  A real, blocking bug surfaced by the first attempt to run Q1/Q6 through
  this: the CPU execution backend rejected *every* aggregate over a
  computed expression outright ("aggregating by a computed expression is
  not yet supported by the CPU execution backend") -- both TPC-H queries
  need exactly that (`SUM(l_extendedprice * l_discount)`,
  `SUM(l_extendedprice * (1 - l_discount))`, ...), so neither could run via
  `--backend cpu` at all before this. Root cause: Arrow Acero's
  `AggregateNodeOptions` can only target an already-existing column by
  `FieldRef`, unlike GROUP BY/ORDER BY keys (which really are always meant
  to be a plain column in this engine's own grammar) -- it has no way to
  evaluate an arbitrary expression itself. Fixed in
  `src/execution_cpu/acero_query_executor.cpp` by inserting an implicit
  "project" Declaration between the aggregate's child and the aggregate
  node itself, computing any non-plain-column aggregate argument under a
  synthetic column name first (pass-through, no new projection inserted,
  for the already-common all-plain-column case, e.g. bare `COUNT(*)`).
  Verified for real: matches DuckDB exactly for both Q1 and Q6 via
  `--backend cpu` (`tools/duckdb_compare.py`); two new regression tests in
  `tests/unit/query_engine_execute_cpu_test.cpp`; `dev` (171/171),
  `server-dev` (174/174), `otel-dev` (174/174) all pass with zero
  regressions.

  **Confirmed on real GPU hardware in this development session**: a real
  RTX 5060 Ti + i7-12700K + 15.49 GiB RAM, `docker run --gpus all` against
  a real `benchmark-gpu` image, three
  real TPC-H-derived datasets generated once on the host via
  `tools/generate_tpch.py` (SF0.01: 60K rows, SF0.1: 600K rows, SF1: 6M
  rows in 31s) and volume-mounted in (not regenerated per run). All three
  engines agreed on both Q1 and Q6's actual computed values at all three
  scale factors (`validated: true` for all 6 query/scale-factor
  combinations, not just a superficial pass/fail flag). Median wall-clock
  seconds (KernelLake-CPU / KernelLake-GPU / PySpark local[*]): SF0.01 Q1
  0.0545/0.3498/0.1607, Q6 0.0466/0.2962/0.0762; SF0.1 Q1
  0.0996/0.3640/0.1919, Q6 0.0623/0.3159/0.1005; SF1 Q1
  0.5604/0.4422/0.2849, Q6 0.1579/0.3317/0.1456. Notable real pattern, not
  smoothed over: the GPU backend is *slower* than both CPU and PySpark at
  SF0.01/SF0.1 (fixed CUDA kernel-launch/host-device-transfer overhead
  dominates at this data size) but its own cost stays roughly flat across
  scale factors while the CPU backend's grows with input size -- GPU
  overtakes CPU on Q1 by SF1. This is a single-machine, single-run
  (5 iterations, median reported) result on two trivial single-table scans,
  not a rigorous statistically-controlled study -- reported here as real,
  unsmoothed numbers per this project's own rule, not as a general
  CPU-vs-GPU-vs-Spark performance claim. Full report (system stats, summary
  table, per-query charts across all three scale factors) generated as a
  real PDF via `tools/generate_benchmark_report.py`; raw
  `benchmark_three_way.py --output` JSON and the rendered PDF are gitignored
  local artifacts (hardware-specific, point-in-time), not committed --
  the tooling that produces them is what's committed.

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
- `linux/arm64` support for the `kernel-lake-gpu` image (CUDA/RAPIDS'
  own arm64 apt/wheel availability unverified) -- needs real arm64 GPU
  hardware (NVIDIA Grace/Jetson) to verify the GPU execution path at all,
  not just a build-only smoke test, and none is available in this
  environment; `kernel-lake-cpu` already publishes a real multi-arch
  (`linux/amd64`+`linux/arm64`) manifest (see "Done" above)
- Extending `run-clang-tidy-18` to `src/execution_gpu/`,
  `src/memory/rmm_environment.cpp`, and
  `src/api/query_engine_execute_gpu.cpp` -- needs a real `gpu-dev` build
  (libcudf/RMM), not available in this environment; every other
  CPU/server/otel-buildable file is now covered (see "Done" above)
- Delta Lake read support (`read_delta(...)` alongside `read_parquet(...)`)
  -- explicitly deferred as its own follow-up plan, not forgotten. Reading
  `_delta_log/*.json` plus checkpoint Parquet files is required to
  correctly identify a Delta table's active file set (globbing `*.parquet`
  directly is wrong: it would include tombstoned/removed files and miss
  schema evolution); no mature native C++ Delta implementation exists
  (`delta-kernel-rs`'s C FFI, used by DuckDB's own Delta extension, is the
  closest) and there is no Rust toolchain in this project's dev environment
  yet, which makes this a genuinely new class of dependency deserving its
  own spike rather than a fourth line item bolted onto the two epics above.

## Explicit non-goals for the MVP

Distributed execution, multi-node/multi-GPU scheduling, a Kubernetes
*operator* (custom controller/CRDs -- the plain Helm chart above,
`charts/kernellake/`, is a Deployment+Service, not an operator), full
Iceberg catalog
integration, joins beyond a two-table `INNER JOIN` with a single equality
key (see "Hash joins" in `docs/ARCHITECTURE.md`), all 22 TPC-H queries,
cost-based optimization,
fault-tolerant fragment
retries, query spilling, materialized views, transactions, data ingestion
(`INSERT`/`UPDATE`/`DELETE`), a proprietary storage format, and a web UI.
Interfaces may exist for some of these (see "Future architecture" in
`docs/ARCHITECTURE.md`) but none are implemented or exposed as supported
features.
