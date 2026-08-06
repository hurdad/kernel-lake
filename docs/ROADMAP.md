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
- **Three-way benchmark extended to SF10 (60M rows) plus cold/warm modes,
  with two real bugs and one operational tuning issue found and fixed at
  scale.** `tools/benchmark_three_way.py` gained `--mode {cold,warm,both}`:
  cold evicts each engine's input file(s) from the page cache immediately
  before that engine reads them, via `posix_fadvise(fd, 0, 0,
  POSIX_FADV_DONTNEED)` -- the same no-root pattern already used by
  `src/cli/benchmark_tpch_command.cpp`, since this sandbox has no root
  access to `/proc/sys/vm/drop_caches`.

  Real bug #1: at SF10, the GPU backend failed with "Batch size (59619013)
  exceeds max_distinct_keys (10000000)" on Q1. Root cause, found by reading
  cudf's own vendored `groupby.hpp` doc comment inside the FetchContent
  source tree: `cudf::groupby::streaming_groupby`'s `max_distinct_keys`
  constrains two different things -- cumulative distinct keys across the
  object's lifetime *and* the row count of any single `aggregate()` call
  (an encoding-scheme constraint, unrelated to actual GROUP BY
  cardinality). `ParquetScanOperator`'s pass splitting is purely
  memory-based (`pass_read_limit_bytes`), so a single incoming batch can
  legitimately have far more rows than `max_distinct_keys` while still
  having very few actual distinct keys -- exactly what happened here (a
  59.6M-row batch, only ~6 real distinct keys). Fixed in
  `src/execution_gpu/hash_aggregate_operator.cpp`'s `process_batch()`: when
  a batch's row count exceeds `max_distinct_keys_`, slice it into
  `max_distinct_keys_`-sized row ranges via `cudf::slice()` and call
  `aggregate()` once per slice, instead of once for the whole batch.
  Verified with a new regression test,
  `HashAggregateOperator.SplitsSingleBatchExceedingMaxDistinctKeys` in
  `tests/gpu/hash_aggregate_operator_test.cpp` (a 5-row single batch, 2
  distinct keys, `max_distinct_keys=2`, forcing 3 `aggregate()` calls
  within one `process_batch()`); all 71 `gpu-dev` tests pass with zero
  regressions, and `dev`/`server-dev`/`otel-dev` (171/174/174) are
  unaffected since this file is GPU-only code not compiled by those
  presets.

  Real issue #2 (operational, not a code bug): after the above fix, SF10
  Q1 still failed with "Exceeded memory limit (failed to allocate 38.146973
  MiB)" from RMM -- the default `pool_max_bytes`/`query_memory_limit_bytes`
  (8 GiB) is undersized for a 60M-row hash-aggregate at this scale.
  Resolved by raising both to 12 GiB via `kernellake.yaml`'s existing
  config knobs (not a code change).

  Real bug #3: PySpark failed SF10 with `java.lang.OutOfMemoryError: Java
  heap space` reading a `lineitem` Parquet part file. Root cause:
  `benchmark_three_way.py`'s `SparkSession.builder` never set
  `spark.driver.memory`, so `local[*]` mode (driver and all executors share
  one JVM) ran 20 parallel executor threads against 60M rows under the
  default small heap. Fixed by adding a `--spark-driver-memory` flag
  (default `4g`; `8g` used for the actual SF10 run, which then succeeded).

  **Confirmed on real GPU hardware**: all 4 scale factors (SF0.01, SF0.1,
  SF1, SF10) x both queries x both modes (cold/warm) = 16 combinations, all
  `validated: true` (all three engines' actual computed values agreed).
  GPU overtakes KernelLake-CPU by SF10 (Q1: 5.23s CPU vs. 1.22s GPU warm;
  Q6: 1.20s CPU vs. 0.51s GPU warm) -- but PySpark stays fastest of all
  three engines at every single scale factor and mode, including SF10 (Q1
  warm: 1.06s PySpark vs. 1.22s GPU; Q6 warm: 0.39s PySpark vs. 0.51s GPU).
  Reported as a real, unsmoothed result, not explained away: these are
  both single-table scan+aggregate queries, so PySpark's `local[*]`
  parallelizes the Parquet scan itself across all 20 CPU cores concurrently,
  while the GPU path pays fixed CUDA-context/kernel-launch/host-device-
  transfer overhead per query that a 60M-row/~6GB dataset still isn't large
  enough to amortize against a well-parallelized JVM scan. Whether that
  crossover exists at a larger scale factor or wider query shape (multi-table
  join, higher compute-per-byte) is untested -- this project's own
  Q1/Q6-only, single-table TPC-H subset (see "Not yet started" below) can't
  answer that; would need SF30+ and/or the multi-table queries to find out.
  Single-machine, single-run (5 iterations, median reported) result, not a
  statistically-controlled study, per this project's own rule of reporting
  real numbers rather than a general performance claim.
- **Three-way benchmark extended to SF100 (600M rows), with a real GPU
  pass-sizing bug found and fixed.** `query_engine_execute_gpu.cpp`'s
  `pass_read_limit_bytes` (bounds how much a single Parquet-scan pass may
  read before `ParquetScanOperator` splits into another one) had two real
  bugs, both surfaced by the very first SF100 attempt at Q1 (GROUP BY
  `returnflag`/`linestatus` over an almost-unfiltered scan of the full
  table -- unlike Q6, whose date/discount/quantity filters keep its actual
  working set small at every scale factor tried so far): it read
  `memory.pool_max_bytes`, which is dead config whenever
  `memory.use_async_allocator` is `true` (the default -- see
  `rmm_environment.cpp`'s `build_base_resource()`; that field only sizes
  `rmm::mr::pool_memory_resource`, never constructed in the async-allocator
  case) instead of `engine.query_memory_limit_bytes`, the value
  `RmmEnvironment`'s `limiting_resource_adaptor` actually enforces; and its
  divisor (`/ 2`) left too little headroom for a query like Q1 that
  materializes two extra derived `DOUBLE` columns per pass
  (`SUM(extendedprice*(1-discount))`, `SUM(extendedprice*(1-discount)*
  (1+tax))`) on top of the 7 columns actually scanned. On this development
  session's real hardware (RTX 3070, 8 GiB VRAM -- smaller than the RTX
  5060 Ti used for the SF0.01-SF1 runs and the unspecified card used for
  SF10), Q1 threw a real `std::bad_alloc: ... RMM ... Exceeded memory
  limit` at both a 6.04 GiB and (retested smaller, to isolate whether it
  was a simple "raise the limit" fix like SF10's) a 3 GiB
  `engine.query_memory_limit_bytes` -- consistently needing about 1.2x the
  configured ceiling at both sizes, confirming this was the divisor, not
  the absolute value. Fixed by reading the correct field and lowering the
  divisor to `/ 4`; reverified against the real SF100 dataset and Q1 query
  with the fix, which completed correctly. Regression coverage: a new
  `QueryEngineExecuteGpuMemoryTest.GroupByWithDerivedAggregatesSucceedsUnderTightMemoryLimit`
  in `tests/gpu/query_engine_execute_test.cpp` -- see that test's own
  comment for why the real bug's exact scale (600M rows) isn't reproducible
  at unit-test speed, and why the pre-fix/post-fix *difference* is only
  demonstrated at the real SF100 scale above, not by this test in
  isolation (below roughly 1 GiB, cudf's chunked Parquet reader has its own
  fixed per-pass overhead, empirically ~76 MiB here, that dominates and
  masks the effect the divisor is meant to control). All 249 `gpu-dev`
  tests pass with zero regressions.

  **Confirmed on real GPU hardware**: both queries x both modes, all
  `validated: true`. Median wall-clock seconds (KernelLake-CPU /
  KernelLake-GPU / PySpark local[*]): Q1 cold 41.7954/10.2624/8.8723, warm
  40.0814/9.4430/7.9964; Q6 cold 10.2550/2.6891/3.2825, warm
  8.7350/1.9475/2.9804. Notable real pattern: the GPU backend now beats
  PySpark outright on Q6 at both modes (a reversal from SF0.01-SF10, where
  PySpark stayed fastest of all three engines at every scale factor tried)
  -- but PySpark still edges out GPU on Q1, though by a much smaller margin
  than at SF10. `--mode both`'s cold-vs-warm gap is modest for every engine
  at this scale (e.g. GPU Q1 10.26s cold vs. 9.44s warm), unlike smaller
  scale factors where it was closer to negligible -- consistent with disk
  I/O becoming a real, if still secondary, cost component once the working
  set (~11 GiB compressed, more decompressed) stops comfortably fitting in
  page cache. Single-machine, single-run (5 iterations, median reported)
  result on this project's Q1/Q6-only TPC-H subset, not a
  statistically-controlled study or a general performance claim, per this
  project's own reporting rule. Report (system stats, summary table, a
  bar chart per query across modes) rendered as `benchmark-results/
  sf100-report-v3.pdf` via `tools/generate_benchmark_report.py`; raw JSON
  and PDF are gitignored local artifacts, not committed.

  Q19 and Q12 were also attempted at SF100 once wired in (see the later
  "TPC-H Q19/Q12 wired into the three-way performance benchmark" entries
  below), surfacing two more real, separate resource limits at this scale
  -- both left out of the final SF100 numbers above rather than reported
  as false failures or silently retried away: **Q12 on GPU** OOM'd
  immediately (see the `HashJoinOperator` streaming gap in "Not yet
  started" below -- `orders`, the smaller table, ends up as the probe
  side and `lineitem`, the larger one, as the unconditionally-materialized
  build side, the opposite of what a real planner should choose). **Q19 on
  CPU** validated correctly and completed 2 of 5 cold iterations before
  the `kernellake` subprocess died with empty `stderr` -- the signature of
  a SIGKILL, not a normal exception (every other real failure in this
  session printed an actual error message) -- and host swap usage went
  from 0 B at the start of this session to 3 GiB, consistent with a real
  OOM. Root cause not fully isolated (candidates: `acero_query_executor
  .cpp`'s `read_scan_table()` materializing the full `lineitem`
  scan for a join into host RAM with no bound, already a documented MVP
  simplification in that file's own comment; PySpark's 64 GiB-capped
  driver JVM heap not being reclaimed between iterations; or both
  compounding) -- noted here as a real, reproduced-once symptom, not
  chased further this session. Both are left as open follow-ups (see "Not
  yet started" below) rather than blocking the Q1/Q6 numbers this entry
  reports.
- **CPU execution backend's `HashJoin`, fixed** (Phase 3 of the
  CPU-backend/benchmark epic -- scoping out which TPC-H queries beyond
  Q1/Q6 could be added next). The CPU backend rejected every join query
  outright ("physical plan node 'HashJoin' is not yet supported by the CPU
  execution backend"), a real CPU/GPU asymmetry: the parser/binder already
  accepted two-table `INNER JOIN ... ON` queries, and the GPU backend
  already executed them correctly (confirmed against DuckDB on a synthetic
  `lineitem` join `part`-shaped-table query matching TPC-H Q19's WHERE
  shape -- exact match). Fixed by mapping `HashJoinNode` to Arrow Acero's
  own `"hashjoin"` node (`HashJoinNodeOptions{JoinType::INNER, ...}`),
  which already implements the same two-table INNER equi-join; its default
  `output_all = true` (left columns then right) matches
  `HashJoinNode::build_schema()`'s convention exactly, needing no extra
  output-list bookkeeping. Verified for real: same Q19-shaped join query
  now matches DuckDB exactly on *both* backends; a new regression test,
  `QueryEngineExecuteCpuTest.TwoTableInnerJoinMatchesExpectedTotals`;
  `dev` (172/172, +1), `server-dev` (175/175), `otel-dev` (175/175) all
  pass with zero regressions. See `docs/ARCHITECTURE.md`'s "CPU execution
  backend" section for the full write-up.

  This also surfaced two real, separate gaps while checking what else the
  current grammar allows in a join query: `CASE` expressions are only
  supported in the `SELECT` list/`GROUP BY` keys, not inside `WHERE` or an
  aggregate argument (`SUM(CASE WHEN ... THEN 1 ELSE 0 END)`, needed by
  TPC-H Q12/Q14 and several others) -- confirmed on *both* backends via a
  real query against real data ("unrecognized expression type in
  {CPU,GPU} expression compiler"), not just inferred from the docs. Column-
  to-column comparisons (`WHERE l_commitdate < l_shipdate`, no `CASE`
  involved) already work correctly on both backends. **Update: the
  aggregate-argument half of this gap is fixed -- see the later "`CASE` in
  an aggregate argument, fixed on both backends" entry below; `CASE` in
  `WHERE` is still open on GPU only (CPU already supports it there too, as
  a side effect of that same fix).** See `docs/TPCH.md` for the current,
  up-to-date scope this leaves for TPC-H specifically.
- **TPC-H Q19 added** (2-table `lineitem`/`part` join, `OR` of `AND`s with
  `BETWEEN`, no `CASE` -- the one TPC-H query shape the current grammar and
  both execution backends already supported after the `HashJoin` fix
  above). `tools/generate_tpch.py` now also generates a real `part` table
  (`PART_ROWS_PER_SF = 200_000`, scaling directly with SF per TPC-H's own
  convention, unlike `lineitem`'s order-count-derived scaling); `l_partkey`
  is drawn from `[1, total_part_rows]` so every join finds a real `part`
  row, no dangling foreign keys. Added `benchmarks/tpch/queries/q19.sql`
  (the shared join key factored into the `JOIN ... ON` clause, common to
  all three `OR`-ed branches in canonical Q19; `p_container` narrowed to
  one representative value per branch, matching the generator's
  representative `CONTAINERS` subset rather than TPC-H's full domain).
  `tools/validate_tpch.py` and `kernellake benchmark tpch` both gained
  `--part-data` for this second table (substituting a `{part_data}`
  placeholder alongside the existing `{data}`), and `validate_tpch.py`
  gained `--backend cpu`/`--backend gpu` (previously always used the
  binary's own default, silently skipping the CPU backend entirely).
  Verified for real: Q19 matches DuckDB exactly at SF0.01 on *both*
  backends; Q1/Q6 still match DuckDB exactly after the generator change
  (the only change affecting them, `l_partkey`'s range, isn't referenced by
  either query); `dev` (172/172), `server-dev` (175/175), `otel-dev`
  (175/175), and a real `gpu-dev` Docker rebuild (249/249) all pass.
  Wiring Q19 into `tools/benchmark_three_way.py`'s three-way performance
  comparison is done separately -- see the next "Done" entry below.
- **TPC-H Q19 wired into the three-way performance benchmark.**
  `tools/benchmark_three_way.py` gained `--part-data` (mirroring
  `kernellake benchmark tpch`'s own flag from the entry above):
  `kernellake_sql()` substitutes `{part_data}` alongside `{data}`;
  `spark_sql()` rewrites `read_parquet('{part_data}')` to a second `part`
  temp view (`spark.read.parquet(args.part_data)`, registered once in
  `main()` alongside the existing `lineitem` view); both `run_kernellake_backend()`
  and `run_pyspark_query()` evict the `part` file(s) too in cold mode, not
  just `lineitem`. `--query all` only includes Q19 when `--part-data` is
  actually given -- omitted silently rather than attempted and reported as
  a validation failure, since a missing argument isn't a cross-engine
  disagreement (pass `--query 19` explicitly, without `--part-data`, to see
  that error instead). Verified for real on GPU hardware in a fresh
  `benchmark-gpu` Docker rebuild: Q19 validates `true` (all three engines
  agree) at SF0.01, with real median timings (KernelLake-CPU/GPU/PySpark:
  0.069s / 0.400s / 0.237s, warm) reported alongside Q1/Q6; `--query all`
  without `--part-data` still runs only Q1/Q6 exactly as before, confirming
  no regression to the existing single-table path.
- **`CASE` in an aggregate argument, fixed on both backends -- unblocking
  TPC-H Q12.** The CPU backend rejected `CASE` in *every* context
  ("unrecognized expression type in CPU expression compiler
  (LIKE/IN/CASE/DECIMAL are not yet supported...)"); the GPU backend
  already supported it in a *grouped* aggregate argument
  (`HashAggregateOperator` already had `CASE`-aware `compile_expr()`/
  `materialize()`) but not a *scalar* one (`ScalarAggregateOperator`
  compiled non-column arguments via the plain `cudf::ast`
  `ExpressionCompiler` directly, which has no `CASE` support at all).
  Fixed CPU by mapping `CaseExpression` to Arrow Compute's own
  `"case_when"` kernel in `compile_expression_cpu()` -- the one function
  shared by every context on this backend, so `WHERE`, `SELECT` list, and
  both grouped and scalar aggregate arguments all work as soon as it does.
  Fixed GPU by giving `ScalarAggregateOperator` the identical
  `CompiledExpr`/`CompiledCase`/`CompiledDecimalCast` machinery
  `HashAggregateOperator` already had (duplicated per-operator, matching
  this codebase's existing convention -- see `docs/ARCHITECTURE.md`'s
  "CASE expression implementation notes"). `CASE` inside `WHERE` on the GPU
  backend remains unsupported (`FilterOperator`'s own, separate gap;
  confirmed still failing after this fix) -- out of scope since neither
  Q12 nor Q14's `WHERE` clause needs it.

  Added TPC-H Q12 (2-table `orders`/`lineitem` join, `CASE` inside a
  grouped `SUM`) as the next query this unblocks. `tools/generate_tpch.py`
  now also generates an `orders` table (one row per distinct `l_orderkey`,
  per TPC-H's 1:N orders:lineitem relationship). `tools/validate_tpch.py`
  and `kernellake benchmark tpch` both gained `--orders-data` (mirroring
  `--part-data`'s existing pattern). Verified for real: new regression
  tests (`QueryEngineExecuteCpuTest.CaseInGroupedAggregateWhereAndScalarAggregateMatchesExpectedTotals`,
  `QueryEngineExecuteTest.CaseInScalarAggregateMatchesExpectedTotal`); Q12
  matches DuckDB exactly on both backends at SF0.01; Q1/Q6/Q19 still match
  after the generator change; `dev` (172/172), `server-dev` (175/175),
  `otel-dev` (175/175), and a real `gpu-dev` Docker rebuild (250/250) all
  pass with zero regressions.

  Q14 was left blocked at this point (needing `LIKE` inside a `CASE`
  branch, plus a separate gap found only once that one was fixed -- a
  `SELECT` item combining multiple aggregates arithmetically) -- both now
  fixed; see the later "TPC-H Q14 added" entry below for the full
  writeup.
- **TPC-H Q12 wired into the three-way performance benchmark**, the same
  way Q19 already is. `tools/benchmark_three_way.py` gained
  `--orders-data` (mirroring `--part-data`): `kernellake_sql()` substitutes
  `{orders_data}` alongside `{data}`/`{part_data}`; `spark_sql()` rewrites
  `read_parquet('{orders_data}')` to a third `orders` Spark temp view;
  cold-mode eviction covers the `orders` file(s) too. `--query all` only
  includes Q12 when `--orders-data` is given, same rationale as Q19's
  `--part-data` handling. Verified for real on GPU hardware in a fresh
  `benchmark-gpu` Docker rebuild: Q12 validates `true` at SF0.01 with real
  median timings (KernelLake-CPU/GPU/PySpark: 0.077s / 0.446s / 0.271s,
  warm); `--query all` with both `--part-data` and `--orders-data` given
  runs and validates all four queries (Q1, Q6, Q19, Q12) together.
- **Real SF1000 TPC-H run surfaces and fixes a 32-bit COUNT/AVG overflow in
  the GPU hash-aggregate path**, plus two smaller SF1000-scale
  infrastructure fixes needed to get there. `tools/generate_tpch.py`'s
  `orders` table generator materialized the *entire* table (Python lists
  sized to `max_orderkey`) before writing it -- fine through SF100, but
  SF1000's ~1.5B `orders` rows OOM-killed a real generation run at 127GB
  RSS; now batches into `ORDERS_BATCH_ROWS`-sized (5M row) chunks across
  multiple `orders-NNNNN.parquet` files, the same way `lineitem` already
  does via `--files`. The GPU backend then failed SF1000 Q1 immediately
  with kvikio's "Too many open files" -- SF1000's `lineitem` spans 1000
  Parquet files (10x SF100's), exceeding the default 1024 fd `ulimit`;
  fixed by raising `--ulimit nofile` on the benchmark's `docker run` (not a
  source change).

  With those cleared, a real SF1000 Q1 run surfaced a genuine correctness
  bug: `kernellake-gpu`'s `count_order` came back negative
  (`-1325530532`) for TPC-H's `A`/`R` `returnflag` groups, while `pyspark`
  reported the correct ~2.97 billion -- exactly `2969436764 - 2^32`, i.e.
  32-bit signed wraparound. Root cause: `HashAggregateOperator` requested
  cudf's native `COUNT`/`MEAN` groupby aggregations, both of which
  accumulate through an internal `cudf::size_type` (INT32) counter across
  every batch fed into the long-lived `streaming_groupby` -- fine at SF100
  (largest per-group count ~600M), but SF1000 pushes a single group past
  2^31 rows. The post-`finalize()` `int32->int64` cast the code already had
  couldn't recover the value; the overflow happened during accumulation,
  not at output. `AVG` shares the same root cause (`MEAN`'s internal
  division uses the identical 32-bit `COUNT_VALID`), confirmed by the same
  run's `avg_disc`/`avg_price`/`avg_qty` all being wrong for the same
  groups while every `SUM(...)` stayed correct.

  Fixed by never requesting cudf's `COUNT`/`MEAN` groupby aggregations at
  all: `COUNT`/`COUNT(*)` are now `SUM` over a synthesized INT64 "ones"
  column (all-valid for `COUNT(*)`; carrying the argument column's null
  mask for `COUNT(col)`), and `AVG` is decomposed into its own
  `SUM(argument)`/`COUNT(argument)` pair (the same trick) with the division
  done in `finalize()` over the two genuinely-INT64-accumulated results,
  instead of trusting cudf's internal division. See
  `HashAggregateOperator`'s `ValueColumnKind`/`AggregateOutputKind` in
  `hash_aggregate_operator.hpp`. Verified for real: same SF1000 Q1 run
  re-validated `kernellake-gpu`/`pyspark` agreement after the fix; all 73
  `gpu-dev` tests still pass (zero regressions).

  With both infrastructure fixes and the aggregate fix in place, ran the
  full SF1000 Q1/Q6 three-way benchmark (`--backends gpu,pyspark`,
  `kernellake-cpu` deliberately skipped -- see `--backends`'s own help text
  on the CPU backend's unbounded scan materialization, and this file's
  SF100 entry's Q19 CPU OOM above, neither of which had been re-verified
  safe at 10x the previously-tested scale). Both queries validated `true`.
  Median wall-clock seconds (KernelLake-GPU / PySpark local[*]): Q1 cold
  126.05/76.11, warm 104.49/73.43; Q6 cold 47.88/30.88, warm 28.56/24.34 --
  PySpark now leads GPU outright on both queries at both modes, a further
  shift from SF100 (where GPU had just overtaken PySpark on Q6) and SF10
  (PySpark led everywhere) -- consistent with the ~1000-file/107GiB
  -compressed `lineitem` scan's I/O and per-pass overhead scaling faster
  than compute at this size, not yet root-caused further. Q12/Q19 (joins)
  were not attempted at SF1000, matching SF100's own convention of not
  reporting a number for a backend/query combination with an already-known,
  unfixed resource limit at the tested scale (see this file's SF100 entry
  and "Not yet started" below) rather than risking another real OOM on this
  shared machine. Single-machine, single-run result, same caveats as the
  SF100 entry above. Report rendered as
  `benchmark-results/sf1000-report-v1.pdf`.

- **GPU `HashAggregateOperator` redesigned around plain per-pass `cudf::groupby`
  instead of `cudf::groupby::streaming_groupby`, cutting SF1000 Q1's GPU
  time by ~46%.** Profiling the SF1000 GPU-vs-PySpark reversal from the
  entry above (`kernellake query --stats`) found `parquet_decoding_seconds`
  was only 38.6s of Q1's 106.9s total, and a control comparison against Q6
  (identical scan/filter, `ScalarAggregateOperator` instead of
  `HashAggregateOperator`) showed only a 1.2s non-scan gap there vs. Q1's
  67.5s -- so the cost was specific to the GROUP BY path, not I/O or a lack
  of stream overlap. Root-caused (temporary call-count instrumentation, not
  kept) to `max_distinct_keys_` doubly bounding both `streaming_groupby`'s
  persistent hash-table capacity *and* the max row count a single
  `aggregate()` call accepts (an encoding-scheme constraint internal to
  cudf) -- so a ~33M-row-average scan pass had to be sliced into ~4 separate
  calls: 727 calls across 182 passes. Tried the obvious fix first --
  raising `max_distinct_keys_` so passes need fewer/no re-slicing -- and it
  didn't help: at 15,000,000 (545 calls, down from 727), total `aggregate()`
  time stayed ~60.6s, essentially unchanged; 20,000,000 and 40,000,000 both
  `std::bad_alloc`'d against the 8GB VRAM budget before finishing. So each
  `aggregate()` call's own cost scales up with `max_distinct_keys_` (bigger
  hash-table capacity = more expensive per call, inside cudf), roughly
  canceling out the benefit of needing fewer calls -- not a tunable-constant
  problem.

  Real fix: stopped using `cudf::groupby::streaming_groupby` entirely.
  `HashAggregateOperator` now runs a plain, one-shot `cudf::groupby::groupby`
  per incoming batch (cost scales with that batch's actual row count and
  actual distinct-key count, not a fixed capacity), then folds each batch's
  partial result into a running `accumulated_` table by concatenating
  `[accumulated_, this batch's partial]` and re-aggregating (cost scales
  with `accumulated_`'s actual size so far -- stays tiny for a
  low-cardinality GROUP BY no matter how many passes have gone by). This
  applies with no COUNT/AVG special-casing at all, because the SUM-of-ones
  fix above already reduced every physical value column this operator ever
  aggregates to a SUM, MIN, or MAX -- all associative/self-combinable, so
  re-aggregating already-partially-aggregated columns with the same
  aggregation is correct. `max_distinct_keys_` now means exactly what its
  name says (checked directly against `accumulated_->num_rows()` after
  every merge) instead of indirectly gating cudf's per-call row limit. See
  `HashAggregateOperator`'s class comment and `PhysicalAggKind` in
  `hash_aggregate_operator.hpp`/`.cpp`.

  Verified for real: 73/73 `gpu-dev` tests pass (zero regressions); the
  same real SF1000 Q1 query still validates against `pyspark` after the
  change, now at `gpu_execution_seconds` 57.3s (was 106.4s) with *lower*
  peak GPU memory too (6.04 GiB vs. 6.59 GiB) -- fewer, cheaper calls, not
  a memory/speed tradeoff. Re-ran the full SF1000 Q1/Q6 three-way benchmark
  (`--backends gpu,pyspark`, 2 iterations -- not 5, to keep this
  verification run short): Q1 median wall-clock seconds (KernelLake-GPU /
  PySpark local[*]) improved from cold 126.05/76.11 & warm 104.49/73.43 to
  cold **78.97/73.43** & warm **58.89/71.08** -- GPU now *beats* PySpark
  outright in warm mode and is close to parity in cold mode, a reversal of
  this file's previous "PySpark now leads GPU outright" finding. Q6
  (unaffected control) stayed flat (cold 48.07/30.74, warm 29.14/24.77 vs.
  the previous run's 47.88/30.88 & 28.56/24.34 -- within run-to-run noise),
  confirming the change is isolated to the hash-aggregate path with no
  side effects elsewhere. 2-iteration result, otherwise same caveats as the
  SF100 entry above. Report rendered as
  `benchmark-results/sf1000-report-v2.pdf`.
- **TPC-H Q14 added, fixing two real gaps found along the way.** Q14's own
  shape (`100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' THEN ... ELSE 0 END)
  / SUM(...)`, a two-table `lineitem`/`part` join) needed `LIKE` inside a
  `CASE` branch inside an aggregate argument, which neither backend
  supported at all (confirmed real: `LIKE` was previously rejected
  everywhere on the CPU backend, and the GPU backend's `CASE`-aware
  operators had no `LikeExpression` case either).

  Fixed CPU by mapping `LikeExpression` to Arrow Compute's own
  `"match_like"` kernel in `compile_expression_cpu()` -- the SQL `%`/`_`
  wildcard pattern needs no conversion, and since this is the one function
  shared by every context on this backend, `WHERE`, `SELECT` list, and
  `CASE` branches all gained `LIKE` support at once (mirroring exactly how
  the earlier `CASE` fix worked). Fixed GPU by giving `HashAggregateOperator`,
  `ScalarAggregateOperator`, and `ProjectionOperator` an identical
  `CompiledLike` fast path (mirroring their existing `CompiledCase`/
  `CompiledDecimalCast` pattern), each materializing via
  `cudf::strings::like()` -- the same primitive `FilterOperator` already
  uses for top-level `WHERE` `LIKE` conjuncts, just invoked from inside a
  `CASE` branch instead. `CASE` inside `WHERE` on GPU remains a separate,
  still-open gap (not needed by Q14's own `WHERE` clause, which has no
  `CASE` at all).

  Fixing `LIKE` surfaced a *third*, previously-undiscovered gap: Q14's
  outer `100.00 * SUM(...) / SUM(...)` combines two aggregates
  arithmetically, and `build_logical_plan()` only ever recognized a
  `SELECT` item as valid in an aggregate query if it *was* exactly an
  `AggregateExpression` or exactly matched a `GROUP BY` key -- anything
  else threw `"SELECT item '...' is neither an aggregate nor a GROUP BY
  column"`, confirmed via a minimal `SELECT 100.0 * SUM(x) / SUM(y)
  FROM ...` reproduction with no `CASE`/`LIKE`/join involved at all. Fixed
  by `rewrite_aggregate_refs()` in `logical_planner.cpp`: a recursive
  rewrite that finds every distinct aggregate subtree in a `SELECT` item
  (deduplicated by `to_string()`), replaces each with a `ColumnExpression`
  at its `LogicalAggregate` output slot, and short-circuits at any subtree
  matching a `GROUP BY` key (without recursing further -- essential for
  `GROUP BY <alias>` resolving to a computed expression whose own internals
  are deliberately exempt from the ungrouped-column check at bind time).
  This lives in shared, backend-agnostic logical-plan code, so it fixes
  both backends in one change. A bare top-level aggregate `SELECT` item
  still registers under its own alias (e.g. `AS revenue`), preserving the
  existing field-naming convention several tests already assert; any other
  (possibly deeply-nested) aggregate reference gets a synthetic
  `__kernellake_agg_N` name instead, since the outer `LogicalProjection`'s
  own alias is what actually determines the final output name regardless.

  Added `benchmarks/tpch/queries/q14.sql`, reusing the existing
  `--part-data` mechanism (same second table as Q19, `part`). Verified for
  real: Q14's full shape matches DuckDB exactly on *both* backends at
  SF0.01 (CPU and GPU both produced `16.89287984201223`, matching DuckDB's
  `16.892879842012228`); new regression tests
  (`QueryEngineExecuteCpuTest.LikeAndNotLikeInWhereMatchExpectedRows`,
  `QueryEngineExecuteCpuTest.ScalarAggregateArithmeticCombiningTwoAggregatesMatchesExpectedRatio`,
  `LogicalPlanner.AggregateArithmeticCombiningTwoAggregatesBuildsBothSlots`,
  `QueryEngineExecuteTest.CaseInScalarAggregateMatchesExpectedTotal` from
  the earlier fix now also validated end-to-end with the LIKE addition);
  `dev` (174/174), `server-dev` (177/177), `otel-dev` (177/177), and a real
  `gpu-dev` Docker rebuild (253/253, was 251) all pass with zero
  regressions. `--query all` (no `--part-data`/`--orders-data`) still runs
  only Q1/Q6, confirming no regression to the existing single-table path.
- **N-way (3+-table) joins, generalized from an original two-table-only
  design.** A prior session's investigation (see this file's own earlier
  entry, now superseded) found the underlying `hsql` SQL parser library
  already parses `A JOIN B JOIN C ON ...` correctly into a left-deep nested
  `TableRef` tree (`(A JOIN B) JOIN C`) -- this project's own AST
  conversion was the only thing rejecting that shape. Generalized across
  the whole pipeline: `sql::AstJoinClause` became a chain (`first` +
  `steps`) instead of a fixed `left`/`right` pair, with `parser.cpp`'s new
  `flatten_join_chain()` recursively unwinding hsql's nested tree (only
  ever recursing on `join->left`, since hsql's left-associative parsing
  guarantees `join->right` is always a plain leaf table); `Binder`
  (binder.cpp) generalized from its hardcoded dual-mode design
  (`input_schema_` vs. exactly-two `left_schema_`/`right_schema_`) to a
  single `std::vector<(alias, schema)>` list; `BoundJoin` became a chain of
  `BoundJoinStep`s, each `combined_key_index` an index into the
  *accumulated* schema of every source before that step (not just the
  immediately-preceding one); `build_logical_plan()` builds a left-deep
  chain of `LogicalJoin` nodes from that chain. The physical planner and
  execution layers (`HashJoinNode`/`HashJoinOperator` on GPU, Acero's
  `"hashjoin"` on CPU) needed **zero changes** -- both already recursed on
  arbitrary children, confirming the real complexity was entirely in the
  parser/binder/logical-plan layers, exactly as suspected going in. See
  `docs/ARCHITECTURE.md`'s "Hash joins" section for the full technical
  writeup.

  Also added a guard against a pathological join-chain length
  (`kMaxJoinSources = 12` in `parser.cpp`, well beyond any real query --
  TPC-H's own deepest join, Q8, needs 7), matching this project's existing
  convention of bounding parser input size/nesting depth.

  Verified for real against DuckDB (both backends), not just unit tests: a
  3-way join with a `GROUP BY` and an aggregate spanning all three sources;
  a third step's join condition referencing the *first* source directly
  (not the immediately-preceding second source), confirming
  `combined_key_index` resolves against the whole accumulated schema, not
  just adjacent sources; ambiguous-unqualified-column rejection across
  non-adjacent sources; qualified-column resolution and `SELECT *`
  expansion across 3 sources. New regression tests:
  `SqlParser.ParsesThreeTableInnerJoinChain`,
  `SqlParser.RejectsExcessiveJoinChainLength`,
  `Binder.ThreeWayJoinResolvesColumnsAcrossEveryStep`,
  `Binder.ThreeWayJoinRejectsAmbiguousUnqualifiedColumnFromNonAdjacentSources`,
  `LogicalPlanner.BuildsLeftDeepJoinChainForThreeTableJoin`,
  `QueryEngineExecuteCpuTest.ThreeTableInnerJoinMatchesExpectedTotals`,
  `HashJoinQueryTest.ThreeWayJoinGroupedSumMatchesExpectedTotals`. `dev`
  (179/179), `server-dev` (182/182), `otel-dev` (182/182), and a real
  `gpu-dev` Docker rebuild (259/259, was 253) all pass with zero
  regressions.
- **TPC-H Q3** (`customer JOIN orders JOIN lineitem`, `GROUP BY`, `ORDER BY
  revenue DESC, o_orderdate`, `LIMIT 10`), the concrete query the N-way
  join generalization above was building toward. `generate_tpch.py` gained
  `generate_customer_table()` (a fixed `CUSTOMER_ROWS = 150_000` rows
  regardless of scale factor, matching TPC-H's own SF1 customer count;
  `c_custkey`/`o_custkey` both drawn from the same `[1, CUSTOMER_ROWS]`
  range) and now writes `customer-00000.parquet` alongside `lineitem`/
  `part`/`orders`. New query file `benchmarks/tpch/queries/q03.sql`.
  `--customer-data` wired through every tool that already had
  `--part-data`/`--orders-data`: `tools/validate_tpch.py`,
  `kernellake benchmark tpch` (`src/cli/benchmark_tpch_command.cpp`), and
  `tools/benchmark_three_way.py`. Confirmed the two open questions from the
  "not yet started" entry this superseded: a multi-key `ORDER BY` after a
  3-way `JOIN` works correctly (initially assumed otherwise without
  testing it -- corrected after actually running it), and `ORDER BY ...
  LIMIT` after a `JOIN` needs a real `ORDER BY` to establish ordering
  before the `LIMIT`/`fetch` node (Q3 has one, so this isn't a concern for
  it). Verified against DuckDB at SF0.01 on the CPU backend: exact
  row-for-row, value-for-value match. Since then, also verified for real
  on the GPU backend via a `tools/benchmark_three_way.py` four-way run
  (real GPU hardware, RTX 5060 Ti, inside the `benchmark-gpu` Docker
  target) -- KernelLake-CPU, KernelLake-GPU, PySpark, and DuckDB all
  agreed on Q3's result (10 rows) at SF0.01, alongside Q1/Q6/Q19/Q12 all
  passing too.
- **DuckDB as a fourth benchmarking engine in `tools/benchmark_three_way.py`**
  (`ENGINES` gained `"duckdb"`, alongside `kernellake-cpu`/`kernellake-gpu`/
  `pyspark`). Unlike PySpark, which needs its own placeholder-to-temp-view
  SQL rewrite (`spark_sql()`), DuckDB natively supports
  `read_parquet('path') AS alias JOIN ...`, so it reuses the exact same
  substituted SQL `kernellake_sql()` already produces -- no new rewrite
  function needed, just a `run_duckdb_query()` wrapper around the existing
  `duckdb_compare.run_duckdb()` helper (already used by
  `validate_tpch.py`/`validate_against_duckdb.py`) with the same cold-mode
  page-cache-eviction handling every other engine gets. `--backends`
  default changed to `"cpu,gpu,pyspark,duckdb"`; `--query all`'s query-list
  logic extended to include Q3 once both `--orders-data` and
  `--customer-data` are passed. `docker/Dockerfile`'s `benchmark-gpu` stage
  (the image `tools/benchmark_three_way.py` actually runs inside, per its
  own docstring) installed `pyspark`/`pyarrow`/`pandas`/`matplotlib` but not
  `duckdb` -- added it there too, found and fixed before the first real run
  rather than after a container failure. Verified for real: a full
  `docker build --target benchmark-gpu` plus a `docker run --gpus all`
  four-way run (SF0.01, freshly generated with the `customer` table
  included) -- Q1/Q6/Q19/Q12/Q3 all validated across all four engines with
  zero disagreements.
- **CI fix: `tpch-tooling-smoke` job's query-file validation step used a
  bare `{data}` -> `*.parquet` glob**, which was safe when
  `generate_tpch.py` only wrote `lineitem` files but broke once
  `orders`/`part`/`customer` joined `lineitem` in the same output
  directory (starting at commit `2556abee`, from a concurrent session on
  another machine, then compounded by this session's own `customer`
  addition) -- the glob matched every table at once, so
  `kernellake explain` failed with a schema mismatch
  (`l_orderkey INT64 NOT NULL vs o_orderkey INT64 NOT NULL`) for every
  query file, five consecutive failed CI runs before being caught.
  `{part_data}`/`{orders_data}`/`{customer_data}` were also never
  substituted at all in this step, which would have surfaced as a
  *different* failure (file-not-found) for Q12/Q14/Q19/Q3 once the
  schema-mismatch bug was fixed, had that been missed too. Fixed
  `.github/workflows/ci.yml` to substitute `{data}` with
  `lineitem-*.parquet` specifically and added the three other
  placeholders' substitutions. Verified by reproducing the exact CI
  scenario locally (SF0.001, same generation command, same for-loop, same
  `sed` pattern): all 6 query files (`q01`/`q03`/`q06`/`q12`/`q14`/`q19`)
  now `explain` successfully with exit code 0. The same bare-glob mistake
  was also present in `docs/TPCH.md`'s own example commands (harmless
  there since they're illustrative, not executed, but corrected for
  consistency and to avoid readers copy-pasting a broken example).
- **Four-way (KernelLake-CPU/GPU, PySpark, DuckDB) crossover sweep at
  SF0.01, SF1, and SF10 -- all five queries (Q1/Q6/Q19/Q12/Q3), real GPU
  hardware (RTX 5060 Ti), 2 cold + 2 warm iterations each.** Answers "at
  what scale does GPU beat the others": it's query-shape-dependent, not a
  single crossover point.
  - **GPU vs. KernelLake-CPU**: GPU overtakes CPU on the join queries
    (Q12, Q3) by SF1 already, and on Q1 (single-table `GROUP BY`) by SF10.
  - **GPU vs. PySpark (`local[*]`)**: PySpark stays fastest at SF0.01
    across every query. By SF1, GPU already edges PySpark on Q12 and Q3.
    By SF10, GPU also overtakes PySpark on Q1, and Q3's margin widens
    substantially (GPU 1.38s vs. PySpark 2.57s warm) -- but PySpark still
    wins Q6 and Q19 at SF10 (both narrower, filter-heavy scans where
    `local[*]`'s 20-core-parallel Parquet read keeps an edge).
  - **GPU vs. DuckDB**: DuckDB wins every single query at every scale
    factor tested (SF0.01/SF1/SF10) by a wide margin (3-10x), being an
    in-process, no-subprocess, no-JVM engine with essentially zero
    per-query fixed overhead. No crossover found in this range; untested
    beyond SF10.
  - Full numbers (median wall-clock seconds, warm mode):

    | Query | SF | CPU | GPU | PySpark | DuckDB |
    |---|---|---|---|---|---|
    | Q1 | 0.01 | 0.059 | 0.311 | 0.140 | 0.0065 |
    | Q1 | 1 | 0.558 | 0.412 | 0.290 | 0.055 |
    | Q1 | 10 | -- | 0.843 | 1.014 | 0.261 |
    | Q6 | 0.01 | 0.047 | 0.291 | 0.082 | 0.0028 |
    | Q6 | 1 | 0.143 | 0.322 | 0.122 | 0.020 |
    | Q6 | 10 | -- | 0.522 | 0.382 | 0.096 |
    | Q19 | 0.01 | 0.056 | 0.298 | 0.170 | 0.0051 |
    | Q19 | 1 | 0.788 | 0.420 | 0.224 | 0.053 |
    | Q19 | 10 | -- | 1.102 | 0.638 | 0.294 |
    | Q12 | 0.01 | 0.063 | 0.332 | 0.169 | 0.0047 |
    | Q12 | 1 | 0.913 | 0.432 | 0.672 | 0.038 |
    | Q12 | 10 | -- | 0.904 | 1.222 | 0.148 |
    | Q3 | 0.01 | 0.094 | 0.395 | 0.187 | 0.010 |
    | Q3 | 1 | 0.826 | 0.502 | 0.513 | 0.065 |
    | Q3 | 10 | -- | 1.384 | 2.567 | 0.286 |

  Two real, reproducible resource limits surfaced along the way (neither a
  correctness bug -- both genuine capacity limits, documented per this
  project's own rule rather than silently retried away):
  1. **KernelLake-CPU backend crashed at SF10 with an empty-stderr,
     non-zero exit** when run as part of the four-way script specifically
     (a standalone repro of the exact same query/SQL/backend outside the
     script succeeded immediately). Root cause: the four-way script keeps
     one `SparkSession` (`--spark-driver-memory 8g`) resident for the
     *entire* run, and CPU's own `read_scan_table()` has no memory bound
     (see the "Not yet started" entry below) -- at SF10 the two together
     exceeded the container's 15.49 GiB RAM. Not a new bug (the
     unbounded-materialization limitation was already known and
     documented); worked around for this sweep by excluding
     `kernellake-cpu` from `--backends` at SF10, rather than reporting a
     false failure or lowering Spark's memory back into its own known
     SF10 OOM range.
  2. **GPU backend hit its configured `query_memory_limit_bytes` (8 GiB
     default in `config/kernellake.yaml`) on Q3 at SF10**: `std::bad_alloc:
     ... RMM ... Exceeded memory limit (failed to allocate 1.342852 GiB)`.
     This is a configured ceiling, not the RTX 5060 Ti's actual 16311 MiB
     VRAM -- raised to 12 GiB (`query_memory_limit_bytes`/`pool_max_bytes`
     both, mounted over the container's `config/kernellake.yaml`) and Q3
     then completed correctly, still with headroom under the physical
     card. All five queries subsequently validated and timed with this
     raised limit at SF10 (the table above uses it).
- **New "kernellake-gpu-server" engine in `tools/benchmark_three_way.py`**,
  answering a real question the four-way sweep above raised: how much of
  "kernellake-gpu"'s (the CLI-subprocess measurement) slowness is genuine
  GPU execution time vs. fixed per-query overhead the other engines don't
  pay? `run_kernellake()` (`duckdb_compare.py`) relaunches the `kernellake`
  CLI as a *fresh subprocess for every single query* -- for the GPU
  backend specifically, this means a brand new CUDA context + RMM memory
  pool (`GpuExecutionCoordinator`'s `RmmEnvironment`) every iteration.
  `kernellake-server` (Arrow Flight SQL, `KERNELLAKE_BUILD_SERVER` --
  already built by `docker/Dockerfile`'s `dev-gpu` stage, no new build
  work needed) constructs that same `RmmEnvironment` exactly **once**, at
  server startup (`src/server/flight_sql_server.cpp`'s constructor), and
  reuses it for every request after that. Added `start_kernellake_server()`
  (spawns the server against a temp config with just `server.port`
  overridden -- `engine.backend` already defaults to `"gpu"` -- and polls a
  raw TCP connect for readiness) and `run_kernellake_server_query()` (an
  `adbc-driver-flightsql` ADBC/DBAPI connection, opened once and reused for
  every query and iteration, `cursor.execute(sql)` +
  `cursor.fetch_arrow_table()`, timing only the round trip). New
  `--kernellake-server <path>` and `--server-port` flags; opt-in via
  `--backends ...,gpu-server` (not in the default set, since it needs an
  extra binary path and pip package). `docker/Dockerfile`'s `benchmark-gpu`
  stage gained the `adbc-driver-flightsql` pip package (pulls in
  `adbc-driver-manager` automatically).

  **Confirmed for real** (RTX 5060 Ti, `docker run --gpus all`, SF1 and
  SF10, all five queries): the server path is dramatically faster than the
  CLI-subprocess path, as the overhead hypothesis predicted --
  6-11x at SF1 (e.g. Q1 warm: 0.045s server vs. 0.383s CLI; Q6 warm:
  0.028s vs. 0.310s), and still a large, consistent margin at SF10 (Q3
  warm: 3.19s server vs. 9.29s CLI). At SF1, `kernellake-gpu-server` even
  **overtakes DuckDB** on Q1 (0.045s vs. 0.059s warm) -- the only engine
  configuration in this whole investigation that has beaten DuckDB on any
  query so far. That crossover doesn't hold at SF10, though: DuckDB pulls
  back ahead on every query there (0.11-0.31s vs. `gpu-server`'s
  0.17-3.6s) as real compute/transfer cost starts to dominate over the
  fixed overhead the server path eliminates. SF10 numbers came from only 2
  iterations each and show real variance worth flagging rather than
  smoothing over (e.g. Q12 `gpu-server` cold 2.66s vs. warm 0.83s, Q3
  `kernellake-gpu` CLI cold 5.73s vs. warm 9.29s -- warm slower than cold
  is not a stable trend at this sample size, not reported as one). More
  iterations and a wider scale-factor sweep would be needed to
  characterize the `gpu-server`-vs-DuckDB crossover precisely rather than
  bracket it between SF1 (server wins Q1) and SF10 (DuckDB wins
  everything).
- **`tools/benchmark_three_way.py` simplified from five engines down to
  three: KernelLake (via `kernellake-gpu-server`), PySpark, DuckDB.**
  Dropped `kernellake-cpu` and the CLI-subprocess `kernellake-gpu`
  measurement (`run_kernellake_backend()`, `--kernellake`) entirely, now
  that the server-based measurement above has shown the CLI-subprocess
  numbers were measuring mostly fixed per-query overhead rather than real
  execution time -- keeping both around no longer served a clear purpose
  for this script. `ENGINES` is now `("kernellake-gpu-server", "pyspark",
  "duckdb")`; `--backends` accordingly simplified to `gpu-server,pyspark,
  duckdb` (default: all three); `--kernellake` removed (no longer needed,
  since nothing invokes the CLI as a subprocess any more). This is a
  benchmarking-tool decision only -- the CPU and GPU-CLI execution paths
  in KernelLake itself (`kernellake query --backend cpu|gpu`) are
  untouched; only this comparison script's engine set changed.
  `tools/generate_benchmark_report.py` (the PDF report renderer) needed
  two real fixes to handle the wider five-engine data already collected
  before this simplification: its `ENGINE_LABELS`/`ENGINE_COLORS` dicts
  still only listed the original three engines (kernellake-cpu/-gpu/
  pyspark), so `duckdb` and `kernellake-gpu-server` were silently dropped
  from the table and charts entirely; and its bar-chart centering math
  (`(i - 1) * width`) assumed exactly 3 bars per group, which would
  misalign/overlap with a different count. Both fixed (labels/colors
  cover all five current engine names; centering now scales with however
  many engines are actually present). Confirmed for real: a rendered PDF
  across SF0.01/SF1/SF10 initially had garbled, overlapping column headers
  in the summary table from `ax.table()`'s equal-width column default not
  fitting 5 engine-name columns -- fixed with
  `table.auto_set_column_width()` and shorter labels (`KL-CPU`, `KL-GPU
  (CLI)`, `KL-GPU (server)`), re-rendered and visually confirmed clean.
- **`HashJoinOperator` build-side selection is now size-aware**, fixing
  the exact gap the "Not yet started" entry below (this superseded a
  version of it that flagged Q12 OOM'ing at SF100 specifically because
  `lineitem`, the *larger* table, ended up as the unconditionally-
  materialized build side purely because it was written second/`right`
  in the query, not because of any real size comparison).
  `src/io/physical_planner.cpp` gained `estimate_row_count()` (a
  `ParquetScanNode` reports the sum of its scanned files' whole-file row
  counts; a nested `HashJoinNode` reports `min(left, right)`; every
  single-child node above either just passes its child's estimate through
  via the generic `children()` accessor rather than needing a case per
  node type) and, in the `LogicalJoin` -> `HashJoinNode` conversion, swaps
  `left_child`/`right_child` (and their matching key indices) whenever the
  left side's estimate is smaller -- since `HashJoinOperator` always
  materializes its *right* child (see that class's own doc comment), this
  puts the actually-smaller table there instead of whichever side a query
  happened to write first. Confirmed safe to swap: every expression above
  a `HashJoinNode` already resolves columns by *name* against its
  `output_schema()` (see `find_scan_schema()`'s own comment), never by
  fixed position, so reordering is transparent throughout, including into
  an outer join in an N-way chain (which also resolves its own join key by
  name against whatever schema its already-converted child produced).

  New regression tests,
  `PhysicalPlannerJoinBuildSideTest.BuildsOnSmallerSideEvenWhenItIsWrittenFirstInTheQuery`
  and `.DoesNotSwapWhenTheSmallerSideIsAlreadyOnTheRight`. Verified for
  real against the actual TPC-H query files at SF10: `kernellake explain
  --format json` on Q12 now shows `lineitem` (8 files/64 row groups) as
  `left`/probe and `orders` (4 files/16 row groups) as `right`/build,
  reversed from before the fix; Q19 (where `part` was already the smaller
  side) shows no change, confirming the swap only fires when it should;
  Q3's 3-way chain shows the swap propagating sensibly through both join
  steps, ending with `customer` (smallest) innermost/build and `lineitem`
  (largest) outermost/probe -- a genuinely better join order, not just a
  2-table special case. `dev` (181/181, up from 179), `server-dev`
  (184/184, up from 182), `otel-dev` (184/184, up from 182), and a real
  `gpu-dev` Docker rebuild (261/261, up from 259) all pass with zero
  regressions. A real SF10 before/after `benchmark_three_way.py` run shows
  Q3 (the query the swap changes most, per the explain output above)
  ~18% faster warm (0.6626s -> 0.5448s); Q12 is flat at this scale (SF10's
  absolute data size isn't yet large enough for build-side choice to
  matter the way it did at the documented SF100 OOM) -- the real payoff
  is expected at the scale that originally surfaced the bug, not yet
  re-verified there.
- **Cost-per-TB-processed analysis added to `tools/benchmark_three_way.py`
  and `tools/generate_benchmark_report.py`**, for comparing engines on
  $/TB rather than just wall-clock time. New `--cost-per-hour
  engine=dollars,...` flag (e.g. `gpu-server=3.00,pyspark=1.00,
  duckdb=0.10`) -- deliberately has no built-in default rate for any
  engine, since real cost varies by cloud region, on-demand vs. reserved
  pricing, and on-prem amortization, and a fabricated default would
  misrepresent an actual dollar figure. `bytes_processed_for_query()`
  computes real on-disk (compressed Parquet) bytes for whichever tables a
  *specific* query's SQL actually references (checking each query file's
  own `{part_data}`/`{orders_data}`/`{customer_data}` placeholders, the
  same check `kernellake_sql()` already does), not every glob passed on
  the command line. `cost_per_tb_dollars()` = `rate * (median_seconds /
  3600) / (bytes_processed / 1e12)`, stored per engine per mode per query,
  alongside a new printed cost table (only shown when `--cost-per-hour`
  was given) and a new PDF page (table) plus one bar chart per query/mode
  in `generate_benchmark_report.py`, gated on `any_report_has_cost_data()`
  so older reports without cost data don't render empty cost pages.
  `add_chart_page()` generalized to take a `metric`/`ylabel`/`title` triple
  instead of hardcoding `median_seconds`, reused for both the existing
  timing charts and the new cost charts rather than duplicating the whole
  function. Verified for real at SF10 with illustrative example rates
  (`gpu-server=3.00,pyspark=1.00,duckdb=0.10` -- not real cloud pricing,
  just numbers to exercise the feature): DuckDB dominates $/TB at these
  rates as expected, but the GPU-server-vs-PySpark comparison flips by
  query -- GPU-server is cheaper on Q3 (0.33 vs. 0.53 $/TB) despite a 3x
  higher hourly rate, since it's fast enough to overcome that, but
  PySpark edges it on Q6 (0.113 vs. 0.125 $/TB), where GPU-server isn't
  fast enough to overcome the same rate gap -- a genuinely different
  conclusion than the raw-time table gives, which is the whole point of
  this metric. PDF re-rendered and visually confirmed (cost table shows
  `n/a` for the SF0.01/SF1 reports that were generated without
  `--cost-per-hour`, real figures for SF10).
- **CPU execution backend's Parquet scan is no longer unbounded** -- the
  documented gap (see the "Not yet started" entries this partially
  supersedes, below) where `acero_query_executor.cpp`'s `read_scan_table()`
  read every fragment's every selected row group fully into one in-memory
  `std::vector` before building a single `arrow::Table`, handed to Acero
  via `TableSourceNodeOptions`/`"table_source"` -- meaning this backend's
  memory footprint scaled with total input size, not whatever the pipeline
  actually had in flight. Replaced with `make_streaming_scan_reader()`: a
  lazy `arrow::RecordBatchReader` (built via `arrow::MakeFunctionIterator`
  + `RecordBatchReader::MakeFromIterator`) that opens and reads each
  fragment one batch at a time, handed to Acero via
  `RecordBatchReaderSourceNodeOptions`/`"record_batch_reader_source"`
  instead -- confirmed by a standalone test program that this factory name
  and node type actually work (not documented anywhere in the installed
  Arrow headers directly, only inferable from the `TableSourceNodeOptions`
  naming convention) before committing to it in the real fix.

  Two real bugs surfaced and fixed along the way, not just a mechanical
  swap: (1) a genuine **use-after-free/segfault** in the first draft --
  `parquet::arrow::FileReader::GetRecordBatchReader()`'s own docs say
  "FileReaders must outlive their RecordBatchReaders," but the first draft
  returned only the `RecordBatchReader` from a helper function, letting the
  `FileReader` local variable (which the returned reader depends on
  internally) be destroyed at that function's return -- crashed all 14
  `QueryEngineExecuteCpuTest` cases with a segfault, caught immediately by
  the existing test suite. Fixed by bundling both together in one
  `OpenFragment` struct with matched lifetimes. (2) Every exception the
  lazy reader's fragment-opening logic can throw (`StorageError`, etc.)
  must be caught *inside* the iterator callback itself and converted to an
  `arrow::Status` failure, not left to propagate as a C++ exception --
  `RecordBatchReaderSourceNodeOptions`'s own docs note each `ReadNext()`
  call runs as a task on Acero's I/O thread pool, where an uncaught
  exception would escape a thread-pool worker (likely `std::terminate()`)
  rather than reach `execute_physical_plan_cpu()`'s top-level try/catch the
  way every synchronous code path in this backend could previously assume.

  Verified for real: `dev`/`server-dev`/`otel-dev` all 181/184/184 (zero
  regressions, and this is exactly the suite that caught bug (1) above).
  Real SF10 CPU-backend run, Q1 (single-table, no join): peak RSS 130 MB
  (`/usr/bin/time -v`) while scanning ~1.08 GiB of compressed `lineitem`
  Parquet (and considerably more decompressed) -- genuinely bounded, not
  scaling with input size. Q12 (a join): peak RSS 3.2 GB, much higher than
  Q1's, but expected and unrelated to this fix -- Acero's own `"hashjoin"`
  node still fully materializes its build side in memory, an inherent
  property of hash joins (the same class of limitation already documented
  for the GPU `HashJoinOperator`, not something a scan-side fix changes).
  Both queries' results cross-checked against DuckDB and matched exactly
  (row-for-row, value-for-value) at SF10.

  Explicitly **not** addressed by this fix, still open: the adjacent "Not
  yet started" entry below about the CPU scan being single-threaded
  (`read_scan_table()`'s per-fragment reads ran sequentially, one core,
  entirely before Acero's own multi-threaded `ExecPlan` starts) -- this fix
  keeps that same sequential-per-fragment shape, it just no longer holds
  every fragment's data in memory simultaneously. Switching to Acero's
  native `"scan"`/`ScanNodeOptions` over an `arrow::dataset::Dataset`
  (which that entry already suggested) would address both memory *and*
  parallelism at once, but is a larger change than this session's fix.
- **`engine.query_memory_limit_bytes` now auto-detects from the GPU's free
  VRAM instead of a fixed 8 GiB default.** The fixed default was already a
  real source of friction this session -- Q3's 3-way join at SF10 needed
  it manually raised to 12 GiB on a 16 GiB card just to complete (see the
  four-way benchmark entries above). `0` (the new default, both in
  `EngineSection::query_memory_limit_bytes` and the checked-in
  `config/kernellake.yaml`/Helm chart `values.yaml`) now means
  "auto-detect"; a new `resolve_query_memory_limit_bytes()`
  (`kernellake/memory/rmm_environment.hpp`) resolves it via
  `cudaMemGetInfo()`, called by both `RmmEnvironment`'s constructor (the
  actually-enforced `limiting_resource_adaptor` ceiling) and
  `query_engine_execute_gpu.cpp`'s `pass_read_limit_bytes` sizing --
  deliberately the same function for both, since computing "auto" twice
  independently (even moments apart) could otherwise resolve to two
  different byte counts on a GPU with fluctuating external usage. An
  explicit non-zero config value always overrides auto-detection.

  **Three real, non-obvious findings from getting this right on actual
  hardware** (RTX 5060 Ti, WSL2, this session's own dev machine -- also
  used interactively, including for gaming, not a dedicated headless
  card), each one changing the design:
  1. Sizing off `cudaMemGetInfo()`'s *total* byte count (the card's full
     capacity) is unsafe -- confirmed by a real OOM at 75% of this card's
     16 GiB total, because several GiB were permanently or transiently
     held by something else entirely (the desktop compositor baseline,
     and separately, live GPU-memory usage while a game was running)
     unrelated to kernellake. Switched to the *free* byte count instead.
  2. 75% of *free* turned out too conservative -- it also OOM'd, needing
     more than that ceiling allowed, while a manually configured ~12 GiB
     ceiling succeeded even when free VRAM was reportedly *lower* than 12
     GiB at measurement time. This clarified what the ceiling actually is:
     RMM's `limiting_resource_adaptor` only *permits* allocation up to the
     configured amount, it doesn't reserve it upfront, and the real
     hardware ceiling is separately enforced by the CUDA allocator itself
     regardless of this config value -- so being too conservative here is
     the worse failure mode (it fails queries that could have actually
     fit), while being too generous just reproduces the same clean,
     already-handled "Exceeded memory limit" error a manually-misconfigured
     value would, in the rare case it's still not enough. Raised to 90% of
     free.
  3. **The actual root cause of every failed verification attempt during
     this work wasn't the resolution formula at all** -- the checked-in
     `config/kernellake.yaml` explicitly set `query_memory_limit_bytes:
     8589934592`, which (correctly, per this field's own "explicit value
     always overrides auto-detection" contract) meant `resolve_query_
     memory_limit_bytes()` was returning that fixed 8 GiB unchanged on
     every single test, regardless of which formula the code used --
     three different formula attempts, three identical "failed to
     allocate 779.213134 MiB" errors, byte-for-byte, was the tell.
     Fixed by changing the shipped config file's value (and the Helm
     chart's `values.yaml` default) to `0` too -- changing the C++
     default alone was not enough for anyone actually running the
     checked-in config, which is everyone using the CLI/server without
     passing `--config`.

  Verified for real once all three were fixed: `gpu-dev` 261/261 (zero
  regressions), and Q3's 3-way join at SF10 now succeeds using the
  **default config file, with zero manual overrides** -- the exact
  friction this feature set out to remove -- cross-checked against DuckDB,
  exact match (10 rows). `dev`/`server-dev`/`otel-dev` all still
  181/184/184 (this change is compiled everywhere, `config.cpp`/
  `config.hpp` are CPU-agnostic, but `resolve_query_memory_limit_bytes()`
  itself only compiles into GPU-enabled builds, matching where
  `RmmEnvironment` already lived).
- **Full code+docs audit surfaced a real drift bug in the memory
  auto-detection above, plus a doc-ordering slip and a defensive
  thread-safety gap, all from this same session's own work.**
  - **CLI-vs-server drift**: `query_engine_execute_gpu.cpp`'s
    `pass_read_limit_bytes` called `resolve_query_memory_limit_bytes(config_)`
    directly, the same function `RmmEnvironment`'s limiter uses -- fine for
    the CLI's one-shot `execute(sql)`, where both calls happen moments
    apart in the same query. But `kernellake-server` keeps **one**
    `RmmEnvironment` alive for its entire process lifetime
    (`GpuExecutionCoordinator`), reusing it across every request, while
    `pass_read_limit_bytes` still recomputed fresh on *every* query. Free
    VRAM at server startup (when the limiter's ceiling was actually fixed)
    and free VRAM hours later on some later query can genuinely differ --
    meaning pass-sizing could silently drift from the ceiling the limiter
    actually enforces, in either direction. Fixed by having
    `RmmEnvironment` store and expose the exact value it resolved at
    construction (`RmmEnvironment::query_memory_limit_bytes()`), and
    `pass_read_limit_bytes` reads that instead of re-resolving. Two new
    regression tests,
    `RmmEnvironment.QueryMemoryLimitBytesAccessorReflectsExplicitConfig`
    and `.QueryMemoryLimitBytesAccessorAutoDetectsWhenConfigIsZero`.
  - **Doc-comment misplacement**: `physical_planner.cpp`'s
    `estimate_row_count()` had been inserted between `find_scan_schema()`'s
    existing doc comment and `find_scan_schema()` itself, leaving
    `find_scan_schema()` with no comment directly above it and
    `estimate_row_count()`'s own comment sandwiched in between two
    unrelated comments. No behavior change, purely confusing to read --
    reordered so each function has its own comment directly above it.
  - **Defensive thread-safety**: `acero_query_executor.cpp`'s streaming
    scan reader mutates shared state (`ScanIterationState`) from a lambda
    Acero runs via `RecordBatchReaderSourceNodeOptions`, whose own docs say
    "each iteration... run on a new thread task" without stating whether
    those tasks are guaranteed serialized or could be pipelined
    concurrently -- not verified against every possible Acero plan shape,
    only a simple manual test. Added a `std::mutex` guarding the whole
    callback body as cheap insurance against a scheduling guarantee this
    code never actually confirmed, rather than relying on it.
  - **Resource leak in `tools/benchmark_three_way.py`**: `server_proc =
    start_kernellake_server(...)` sat *before* the `try`/`finally` block
    that stops PySpark, and PySpark's `SparkSession` was constructed
    *before* that call -- if kernellake-server startup failed for any
    reason (bad `--kernellake-server` path, GPU already busy, a slow GPU
    init exceeding the 30s startup timeout) after Spark had already started
    successfully, the exception propagated straight out of `main()` with
    the JVM never stopped, leaking a running Spark process. Fixed by
    moving both engines' startup inside the same `try` the query loop
    already used, so the one `finally` block covers whichever of
    Spark/kernellake-server actually started, regardless of which step
    failed. Verified for real: pointed `--kernellake-server` at a
    nonexistent path with `pyspark` also enabled -- Spark started (its own
    JVM traceback confirms `DataFrameReader.parquet()` ran), then the
    server-startup `FileNotFoundError` fired, then the process exited with
    a single clean traceback and no secondary exception during cleanup (a
    failing `spark.stop()` would have shown as a second, chained
    exception). Re-ran the same command with a valid path afterward to
    confirm the happy path is unaffected.

  Verified for real: `gpu-dev` 263/263 (up from 261), `dev`/`server-dev`/
  `otel-dev` still 181/184/184, zero regressions.
- **Continued the audit into modules not touched this session
  (`optimizer.cpp`, `binder.cpp`, GPU operators, cloud object stores) --
  found one more real bug, in `gcs_object_store.cpp`'s
  `parse_iso8601_utc()`.** `std::get_time(&tm, "%Y-%m-%dT%H:%M:%S")` only
  validates that specific format and leaves any trailing characters in the
  stream unconsumed without failing -- an offset-form timestamp like
  `"2026-01-01T00:00:00+05:00"` (a real ISO-8601 variant, just not the "Z"
  form this function's own comment says is the only one supported) would
  parse "successfully," silently discard the `+05:00`, and misinterpret
  the result as UTC -- producing an access-token expiration silently wrong
  by the offset amount, with no error at all. Fixed by capturing the
  remainder of the stream after the timestamp portion and requiring it be
  exactly `"Z"`, turning this into the same clear rejection every other
  malformed input already got. Three new regression tests
  (`GcsObjectStore.RejectsAccessTokenExpirationWithNonUtcOffset`,
  `.RejectsAccessTokenExpirationMissingUtcSuffix`,
  `.AcceptsValidUtcAccessTokenExpiration`) -- this module had no test file
  at all before. `optimizer.cpp`/`binder.cpp`/the GPU hash-join, hash-
  aggregate, and filter operators/S3 and Azure object stores were all
  read carefully and found already correct; HDFS's authority-stripping
  logic (`strip_authority()`, relies on a double `strip_scheme()`
  application) was traced through manually and confirmed correct, but
  left otherwise unverified -- consistent with this project's existing,
  already-documented position that HDFS has no real cluster or
  lightweight emulator available to test against. `dev` 184/184 (up
  from 181), `server-dev`/`otel-dev` unaffected at 184/184.
- **Continued the audit into the remaining GPU operators
  (`parquet_scan_operator.cpp`, `scalar_aggregate_operator.cpp`,
  `sort_operator.cpp`, `limit_operator.cpp`) -- found a real bug in
  `ScalarAggregateOperator::process_batch()`'s SUM/MIN/MAX/AVG
  accumulation.** The operator folds each batch's contribution into a
  running `cudf::scalar` across batches via `cudf::reduce()`'s `init`
  overload (`cudf::reduce(column, agg, output_type, init=running_value)`),
  needed since batches aren't retained. `cudf::reduce()`'s own header
  documents that empty or all-null input produces an invalid scalar, but
  doesn't say what happens when `init` is combined with such a column --
  empirically confirmed against a real GPU (RTX 5060 Ti, standalone
  `cudf::reduce` repro plus two new operator-level tests) that the
  init-based overload returns an *invalid* scalar whenever the **current
  batch** contributes zero valid values, regardless of whether `init`
  itself was valid, in either order (all-null batch first poisons a later
  valid batch's contribution; a valid running total followed by an
  all-null batch is wiped out too). Any nullable aggregate argument column
  with an all-NULL batch or pass anywhere in a multi-batch GPU scan (very
  plausible for sparse/clustered NULLs, or simply any pass on its own
  scanning a value that happens to be entirely NULL) silently turned the
  whole `SUM`/`MIN`/`MAX`/`AVG` result to NULL instead of the correct
  value from the other batches -- no error, wrong answer. Fixed by
  checking `column->size() == 0 || column->null_count() == column->size()`
  before ever touching `running_value`: a batch contributing nothing
  leaves the running value untouched; a first contribution is reduced
  without `init`; a later contribution folds in via `init` only once a
  running value already exists. Applied the same guard to `Avg`'s
  separate sum/count accumulation. Four new regression tests
  (`ScalarAggregateOperator.SumAcrossBatchesSurvivesAnEntirelyNullBatch`,
  `.SumAcrossBatchesWhereLaterBatchIsEntirelyNull`,
  `.AvgComputesMeanAcrossBatchesIncludingAnEntirelyNullBatch`, plus the
  standalone repro used to first pin down cudf's actual behavior before
  touching the real file). `parquet_scan_operator.cpp` (the all-local vs.
  `ObjectStoreDatasource`-backed chunked-reader branching, the
  empty-chunk-skipping loop in `next()`), `sort_operator.cpp` (full
  materialize-then-`stable_sorted_order`+`gather`, NULL ordering matching
  PostgreSQL's ASC-nulls-last/DESC-nulls-first convention), and
  `limit_operator.cpp` (truncation via `cudf::slice` plus a deep copy into
  an owned table, needed since the source batch's buffer is about to go
  out of scope) were all read carefully and found already correct.
  Verified for real: incremental `ninja`/`docker exec` rebuild against the
  existing `dev-gpu` image and a real RTX 5060 Ti (not a full image
  rebuild) -- the `kernellake_gpu_tests` binary went from 76/76 to 79/79,
  `kernellake_unit_tests` unaffected at 187/187. `dev`/`server-dev`/
  `otel-dev` untouched (this file is GPU-only, not linked into the
  CPU-only presets).
- **Continued the audit into the GPU and CPU expression compilers
  (`expression_compiler.cpp`, `expression_compiler_cpu.cpp`) -- found a
  real bug in `binder.cpp`'s numeric type promotion, shared by both
  execution backends.** `promote_numeric()` picks `UInt64` as the common
  type whenever either side of a comparison/arithmetic expression is
  `UInt64` (and `UInt32` similarly for a mixed `UInt32`/`Int32` pair), then
  `cast_if_needed()` casts the other (signed) side to match. Confirmed
  against a real GPU that this silently two's-complement-wraps a negative
  value instead of erroring: `CAST(-5 AS UINT64)` evaluates to
  `18446744073709551611` (`2^64 - 5`), not an error and not saturated to
  0 -- so e.g. `WHERE signed_col < unsigned_col` would silently produce
  the wrong answer for any negative `signed_col` (`UInt32`/`UInt64` are
  real, reachable KernelLake column types, mapped straight from Parquet's
  `UINT32`/`UINT64` logical types via `arrow_adapter.cpp`, not a
  theoretical corner case). Separately, `expression_compiler.cpp`'s
  `Negate` case synthesizes unary `-x` as `0 - x` in `x`'s own type (cudf's
  AST has no dedicated negation operator); for an unsigned `x` that wraps
  the same way -- confirmed `0u - 5u` (`UINT32`) evaluates to
  `4294967291` (`2^32 - 5`). Fixed both at bind time (shared by both
  backends, so the CPU backend's `expression_compiler_cpu.cpp` -- whose
  `Negate` case delegates straight to Arrow's own `"negate"` kernel -- is
  protected by the same guard without needing a separate fix there):
  `promote_numeric()` now throws a clear `BindingError` when mixing a
  signed and unsigned integer type (matching the existing rejection style
  for mismatched DECIMALs just above it in the same function), and
  `bind_node(AstUnary)` now rejects unary `-` on a `UInt32`/`UInt64`
  operand outright, since there is no correct unsigned negative result to
  produce. Six new regression tests: two GPU characterization tests
  pinning down the exact cudf wraparound values so a future cudf upgrade
  changing this behavior fails loudly rather than silently
  (`ExpressionCompiler.CastingNegativeInt64ToUInt64SilentlyWrapsAround`,
  `.UnaryNegateOnUnsignedColumnSilentlyWrapsAround`), and four
  binder-level rejection/acceptance tests
  (`Binder.MixingSignedAndUnsignedIntegerTypesInComparisonIsRejected`,
  `.MixingSignedAndUnsignedIntegerTypesInArithmeticIsRejected`,
  `.UnaryNegateOnUnsignedColumnIsRejected`,
  `.UnaryNegateOnSignedColumnStillWorks` -- the last confirming the fix
  doesn't overreach into signed operands). An explicit
  `CAST(unsigned_col AS BIGINT)` narrowing a large `UInt64` value was
  deliberately left as-is (not rejected): unlike the implicit-promotion
  case, an explicit user-written CAST losing precision on an
  out-of-range value is ordinary, widely-accepted SQL CAST semantics
  (the same way `CAST(3.9 AS INT)` truncates elsewhere in this project),
  not a hidden footgun the user never asked for. Verified for real:
  `dev` 188/188 (up from 184), `server-dev` 40/40 `Binder.*` tests
  passing (full-suite count unaffected by this change), `gpu-dev`
  191/191 `kernellake_unit_tests` (up from 187) and 78/78
  `kernellake_gpu_tests` (up from 76, both counts from this bug's own
  round -- the earlier `ScalarAggregateOperator` fix's container had
  already been torn down), via the same incremental `ninja`/`docker exec`
  rebuild against a real RTX 5060 Ti.
- **Continued the audit into the SQL parser (`parser.cpp`) -- found two
  more real, high-severity process-crash bugs, both confirmed with
  standalone repros before being fixed.** (1)
  `preprocess_from_read_parquet()`'s `read_parquet(...)` argument
  extraction used a `std::regex` pattern with a repeated group
  (`(?:'...'\s*,\s*)*'...'`). libstdc++'s `std::regex` recurses once per
  repetition of a `(...)*` group, so a single path argument long enough
  (confirmed empirically at ~35,000 characters -- well under this same
  file's own 1 MiB `kMaxSqlBytes` cap, which exists specifically to guard
  against this class of problem for hsql's own recursive-descent parser)
  drove it into a real C-stack overflow: `SELECT * FROM
  read_parquet('<~35,000-char path>')` segfaulted the whole process
  rather than raising a catchable `SqlError`. Since `kernellake-server`
  accepts arbitrary SQL text over Arrow Flight SQL from any client, this
  was a remotely-triggerable denial of service requiring nothing beyond
  the ability to send an ordinary-looking query. Fixed by replacing the
  regex entirely with a linear, non-recursive hand-written scanner
  (`try_parse_read_parquet_args`/`try_parse_quoted_string`) -- O(1) stack
  usage regardless of input size or shape, preserving the exact same
  matching semantics (case-insensitive `read_parquet`, verbatim
  passthrough of backslash-escaped characters inside a path, "no match"
  falling through to hsql's own parse error). (2) Separately, hsql's own
  recursive-descent parser builds a chained JOIN (`A JOIN B JOIN C JOIN
  ...`) as a left-deep TableRef tree and recurses once per level while
  doing so, with no depth limit of its own -- the same class of bug this
  file's pre-existing `kMaxParenDepth` guard already exists to catch for
  nested parentheses, just never extended to JOIN-chain depth. Confirmed
  a ~40,000-JOIN chain (comfortably under the 1 MiB SQL-length cap)
  segfaults *inside `hsql::SQLParser::parse()` itself*, before this
  project's own semantic `kMaxJoinSources` check (which only runs after
  hsql has already built its tree) ever gets a chance to reject it. Fixed
  by adding a second cheap pre-scan, `kMaxJoinKeywords` (64, generous
  above the semantic ceiling of 12 actual JOIN sources), counting
  case-insensitive `"join"` substring occurrences alongside the existing
  paren-depth scan in `check_sql_within_limits()` -- same "well before any
  parsing work, real or hsql's" philosophy as the existing guards. Both
  fixes verified by first reproducing the actual segfault (via a
  standalone program linked against the real `kernellake_sql`/`hsql`
  libraries, and separately by temporarily reverting just the
  `read_parquet` fix and re-running the new test, which crashed the whole
  gtest binary as expected) and then confirming the fix turns it into a
  clean `SqlError`. Six new regression tests: `AcceptsAVeryLong
  SinglePathArgumentWithoutCrashing`, `AcceptsManyCommaSeparated
  PathArgumentsWithoutCrashing`, `ReadParquetIsCaseInsensitive`,
  `ReadParquetPathArgumentPreservesEscapedQuoteVerbatim`,
  `RejectsExcessivelyLongJoinChainWithoutCrashing`, plus re-verifying
  `AcceptsManyCommaSeparatedPathArgumentsWithoutCrashing`'s companion
  path-count case. Separately stress-tested (no bug found, no fix
  needed): a 100,000-element `IN (...)` list parses fine with no
  recursion-depth issue, since hsql builds a flat list iteratively there
  rather than a nested tree the way JOIN chains and parenthesized
  expressions are built. Verified for real: `dev` 193/193 (up from 188),
  `server-dev` 25/25 `SqlParser.*` (full suite unaffected), `gpu-dev`
  199/199 `kernellake_unit_tests` (up from 198, this round's own count,
  bundled with every other fix from this same audit session applied
  together in one container) and 81/81 `kernellake_gpu_tests`.
- **Continued the audit into the Flight SQL server
  (`flight_sql_server.cpp`) -- found a real concurrency bug that defeats
  `max_pending_results`' documented cap.** `GetFlightInfoStatement`
  enforced the cap via a check-then-act pattern: lock, read
  `results_.size()`, unlock, run the query (potentially slow, and always
  unlocked), lock again, insert. Concurrent callers that all observe the
  cap not yet reached (since none of them has inserted yet) can all
  proceed to execute and all insert, growing `results_` well past the
  configured limit -- this isn't a contrived attack, just ordinary
  concurrent client usage (several queries in flight at once, which any
  real client doing more than one thing at a time will do). Confirmed
  empirically with a real gRPC stress test (20 concurrent
  `FlightSqlClient::Execute()` calls against a server configured with
  `max_pending_results=2`): 20 of 20 succeeded in one run, 13 of 20 in
  another, both far over the cap. Fixed by making the check-and-reserve
  atomic: a new `pending_count_` member is incremented in the *same*
  locked section as the cap check (reserving a slot before the query ever
  runs), then decremented on failure (both catch blocks) or folded into
  the real `results_` insert on success. One new regression test,
  `FlightSqlServerPendingResultsCapTest.CapIsEnforcedUnderConcurrentCallers`,
  run 10x in a row post-fix with zero failures (pre-fix it failed on the
  first attempt). `main.cpp` (config loading, S3/observability shutdown
  guards, error handling) and `gpu_execution_coordinator_{gpu,stub}.cpp`
  (the GPU backend's single-flight `RmmEnvironment` serialization) were
  also read and found already correct -- note the GPU coordinator's own
  mutex does *not* incidentally protect against this bug either, since it
  only serializes the query execution itself, not the surrounding
  check/insert in `results_` (confirmed by reasoning through the
  interleaving; the concurrency stress test above uses the CPU backend
  and reproduces the bug independent of that mutex regardless). Verified
  for real: `server-dev` 197/197 (up from 192), `gpu-dev` 188/188
  `kernellake_unit_tests`, both via a real gRPC server + real
  `FlightSqlClient` over TCP, the latter via Docker + a real RTX 5060 Ti.
- **Investigated broad Flight SQL client compatibility (JDBC/DBeaver-style
  tools, not just the Python ADBC path `benchmark_three_way.py` already
  uses) -- found kernellake-server couldn't run a single query for such a
  client, added prepared-statement support to fix it, and in the process
  found and fixed a severe, unrelated, pre-existing correctness bug that
  affects a huge fraction of ordinary queries on both backends.**
  Confirmed with a real `org.apache.arrow:flight-sql-jdbc-driver` (the
  actual driver DBeaver and most JDBC-based BI tools use) against a real
  running `kernellake-server`: even a plain `Statement.executeQuery()` --
  no `PreparedStatement` involved at all -- failed with
  `CreatePreparedStatement not implemented`, because the JDBC driver
  routes *every* query through the prepared-statement RPCs internally
  with no fallback to the plain-statement path. `GetSqlInfo` alone (the
  fix originally proposed before this was actually tested) would not have
  helped: no query could run at all, making metadata negotiation moot.
  Implemented `CreatePreparedStatement`/`ClosePreparedStatement`/
  `GetFlightInfoPreparedStatement` in `flight_sql_server.cpp`: since
  KernelLake has no bound-parameter ("?") support, a "prepared statement"
  is just a named, pre-bound `PhysicalPlanPtr` (parse/bind/plan happens at
  prepare time via the same `QueryEngine::explain()` `GetFlightInfoStatement`
  already used, just without executing yet, so a genuine syntax/binding
  error now surfaces at prepare time like a real prepared statement, and
  execution skips redundant parse/bind/plan work); `GetFlightInfoPreparedStatement`
  reuses the exact same eager-execute-and-buffer path (factored into a new
  `ExecuteAndBuffer` helper) and the exact same `CreateStatementQueryTicket()`
  ticket format `GetFlightInfoStatement` already used, so the existing
  `DoGetStatement` serves both kinds of query results with no separate
  `DoGetPreparedStatement` needed. `prepared_` (the un-executed-plan
  registry) gets the same unbounded-growth guard as `results_`
  (`max_pending_results`, since a client that never calls
  `ClosePreparedStatement` is the same class of concern as one that never
  calls `DoGet`). Verified for real with the actual JDBC driver end-to-end
  against a real `kernellake-server` (`PreparedStatement` prepare+execute
  returned the correct 3 rows over a real gRPC connection) and confirmed
  no regression in the existing RPC surface; `server-dev` `kernellake_unit_tests`
  and `gpu-dev` (Docker + real RTX 5060 Ti) both green throughout.
  **While building the JDBC repro dataset, found a severe, unrelated bug**
  in `optimizer.cpp`, present on both backends: `SELECT id, amount FROM t
  WHERE id < 3` (id, amount being the table's only two columns, selected
  in their original schema order) silently returned only the `id` column
  on both CPU and GPU, no error at all -- confirmed via the CLI directly
  (bypassing Flight SQL entirely) and via `explain`, which showed the scan
  itself had been pruned to `columns: [id]`, dropping `amount`. Root
  cause: `rewrite_plan()`'s `LogicalProjection` case elided a "no-op
  identity projection" (its items exactly reproduce the child schema, same
  order -- which `SELECT * ... WHERE ...` always desugars to) *before*
  `annotate_scan()`'s column-pruning pass ever walked the rewritten tree;
  with the projection gone, a `LogicalFilter` sitting under it only marked
  its own referenced columns (just the `WHERE id < 3` predicate's `id`) as
  required, silently losing all record that the query's actual output
  also needed `amount`. This exact shape (`SELECT *`/full-column-order
  `SELECT` plus any `WHERE`/`ORDER BY`/`GROUP BY`) is an extremely common
  query pattern, not a contrived edge case. Fixed by moving the
  elision from `optimizer.cpp` (before column pruning) to
  `physical_planner.cpp`'s `LogicalProjection` conversion (after it,
  once the scan's pruned schema and the projection's remapped items are
  both already known) -- the optimization itself (skip materializing a
  pass-through-only `ProjectionNode`) is still valid and still fires, it
  just now runs late enough to be safe. Regression tests added at both
  the newly-affected layers: `Optimizer.KeepsIdentityProjectionForColumnPruningToSeeLater`
  (replacing the old, now-relocated `RemovesRedundantIdentityProjection`),
  and `PhysicalPlannerTest.RegressionKeepsEveryColumnWhenSelectListMatchesSchemaOrder`
  plus `.ElidesIdentityProjectionAfterColumnPruning` (confirming the
  optimization still correctly fires once it's actually safe to).
  Discovering this correctly required first learning that `convert_scan()`
  in `physical_planner.cpp` preserves the scan's *original* field order
  when narrowing to `required_columns()` -- `required_columns()` itself is
  stored alphabetically sorted internally, which is a red herring for
  reasoning about the physical scan's actual output order; three existing
  `physical_planner_test.cpp` tests that expected a `ProjectionNode` to
  always survive needed updating once the (now-correct, now-later)
  elision started firing in more cases than before, and
  `QueryEngineTest.ExplainLogicalBindsAgainstRealParquetSchema` needed
  updating since `explain_logical()`'s optimized output for its exact
  query shape is now `LogicalProjection` at the top instead of
  `LogicalAggregate`. Verified for real end-to-end (not just structural
  plan-shape checks): re-ran the exact original failing query
  (`SELECT id, amount FROM read_parquet(...) WHERE id < 3` and
  `SELECT * FROM ... WHERE id < 3`) against a real GPU via the CLI on both
  backends post-fix -- both now correctly return every column with the
  right values. `dev` 195/195 (up from 193), `server-dev` 199/199,
  `gpu-dev` 190/190 `kernellake_unit_tests` and 76/76 `kernellake_gpu_tests`
  (via Docker + a real RTX 5060 Ti).
- **Hive-style partition discovery** (Phase 2 of the lakehouse roadmap --
  Unity Catalog as the metadata/governance layer, KernelLake staying a
  compute engine; see the roadmap plan for the full 5-phase sequencing:
  Parquet directories (done) -> Hive partitioning (this) -> Iceberg REST
  catalogs -> Delta Lake (via an extended `delta-txn-service`) -> Unity
  Catalog). `read_parquet('s3://bucket/table/')` now auto-detects a
  Hive-style directory layout (`region=US/date=2026-01-01/part-0.parquet`)
  with **no new SQL syntax** -- partition columns are discovered, type-
  inferred (integer/ISO-date/string), appended to the schema, and fully
  queryable (`SELECT`, `WHERE`, `GROUP BY`) on both backends, verified
  against a real GPU. A plain, non-partitioned source is completely
  unaffected (same code path, zero behavior change), confirmed by the full
  existing test suite passing unchanged throughout.
  - **New seam**: `resolve_table()`/`ResolvedTable`
    (`include/kernellake/io/table_resolution.hpp`,
    `src/io/table_resolution.cpp`) sits between file discovery and
    schema/physical-plan construction, replacing direct
    `discover_parquet_files()`/`inspect_parquet_file()` calls at both
    existing call sites (`query_engine.cpp`'s schema inspection,
    `physical_planner.cpp`'s `convert_scan()`). This is the one place
    every later phase (Iceberg/Delta/Unity Catalog) will plug into instead
    of reinventing name-to-file resolution.
  - **Recursive directory listing** (`ObjectStore::list_recursive()`) had
    to be added across all five backends first: the existing `list()`
    only lists a directory's *immediate* children
    (`fs::directory_iterator`, not recursive), so a Hive-partitioned
    source would otherwise resolve to zero files. `arrow::fs::FileSelector`
    already has a `recursive` flag, so this was close to free for the four
    Arrow-fs-backed stores (S3/GCS/Azure/HDFS) via one new shared helper,
    `generic_fs_list_recursive()`; only `LocalObjectStore` needed real new
    logic (`fs::recursive_directory_iterator`).
  - **Hive detection/parsing** (in `resolve_table()`): every discovered
    file's path must yield the exact same sequence of `key=value`
    directory segments immediately above the file, or none at all -- a mix
    (some files partitioned, some not, or different keys/depth) is
    rejected outright (`StorageError`) rather than guessed at, matching
    the project's existing "explicit errors over silent partial behavior"
    rule. A partition column colliding with an existing physical column
    name is also rejected. Type inference per column: integer if every
    observed value parses as one, else a valid ISO-8601 calendar date,
    else string.
  - **Planning-layer plumbing**: `PartitionColumn`/`PartitionTransform`
    live in a new shared header,
    `include/kernellake/types/partition_column.hpp` (in `kernellake_types`,
    not `kernellake_io`, specifically to avoid a circular dependency --
    `kernellake_io` already depends on `kernellake_planner`, which needed
    to reference these same types for `LogicalScan::partition_columns()`).
    `LogicalScan`, `build_logical_plan()` (both overloads, single-table and
    N-way join), `PhysicalFileFragment` (per-fragment partition values),
    and `ParquetScanNode` (which columns to physically read from the file
    vs. which are required-but-partition-derived) all thread this through;
    `physical_planner.cpp`'s `convert_scan()` splits a query's required
    columns into "physical" (handed to cudf's/Arrow's Parquet reader) and
    "partition" (never handed to either reader, since they don't exist in
    the file) sets.
  - **Execution-layer materialization** -- the actual "append a constant
    column per fragment" work, on both backends:
    - CPU (Acero, `acero_query_executor.cpp`): straightforward, since this
      backend already reads one fragment's `RecordBatchReader` at a time
      -- `append_partition_columns()` appends a constant-value `Array`
      (via `arrow::MakeArrayFromScalar`) built from a newly-exposed
      `literal_to_arrow_datum()` (moved out of
      `expression_compiler_cpu.cpp`'s anonymous namespace, reusing its
      existing literal-to-Arrow-scalar type mapping rather than
      duplicating it).
    - GPU (cudf, `parquet_scan_operator.cpp`) needed a real design
      decision, not just a port of the CPU approach: cudf's
      `chunked_parquet_reader`, when given every fragment at once (this
      operator's normal fast path), can legitimately batch rows from
      *multiple* source files into a single returned chunk when they fit
      within `pass_read_limit_bytes` together -- and there is no way to
      recover, after the fact, which of a chunk's rows came from which
      file, which a per-file constant partition value absolutely needs to
      know. Fixed by having the operator switch to a **per-fragment
      reading mode** (one `chunked_parquet_reader` per fragment,
      sequential -- exactly like the CPU backend) whenever
      `partition_columns` is non-empty, trading away cross-file pass
      batching specifically for partitioned scans (still fully
      pass-based/streaming *within* one large partition's own file) in
      exchange for provable per-row-range correctness, rather than
      guessing at a chunk-to-file boundary the API doesn't expose. The
      plain non-partitioned fast path is completely untouched.
  - New tests: `TableResolutionTest` (8 cases: plain/partitioned schema
    detection, int/date/string type inference, multi-level partitioning,
    three distinct rejection cases), `ParquetScanOperatorTest.
    MaterializesPartitionColumnsPerFragment` (two fragments sharing one
    underlying file with different assigned partition values, confirming
    no cross-fragment mix-up). Verified for real end-to-end on both
    backends against a real GPU (RTX 5060 Ti via Docker): `SELECT`,
    `SELECT *`, `WHERE`, and `GROUP BY` all correctly resolved/executed
    over real Hive-partitioned directories, including two-level
    (`region=.../yr=.../`) partitioning spanning multiple fragments with
    an aggregate query. `dev` 203/203, `server-dev` 207/207, `gpu-dev`
    200/200 `kernellake_unit_tests` and 77/77 `kernellake_gpu_tests`, zero
    regressions throughout.
  - Deferred (not needed for this phase, tracked for Iceberg/Unity
    Catalog): generalizing the SQL grammar/AST beyond `read_parquet(...)`
    for new named-source kinds, and the `ObjectStoreRegistry`
    dynamic-credential seam for vended cloud credentials -- see the
    lakehouse roadmap plan for why both are Phase 3+ concerns, not Phase 2
    ones.
- **Iceberg REST catalog config, client, manifest reading, schema
  translation, table resolution, and SQL surface** (Phase 3 of the
  lakehouse roadmap -- see "Hive-style partition discovery" above for the
  full 5-phase sequencing; see "read_iceberg(...) SQL surface" below for
  where this phase completes). `SELECT ... FROM
  read_iceberg('catalog.namespace.table')` now works end to end, real file
  resolution, row-group pruning and all.
  - **Config**: `iceberg.catalogs` (`include/kernellake/common/config.hpp`'s
    `IcebergSection`/`IcebergCatalogSection`) is a name-keyed map -- unlike
    `storage.{s3,gcs,azure,hdfs}`'s one-section-per-scheme shape, a real
    deployment commonly has more than one REST catalog (e.g.
    "prod"/"staging"), addressed by the leading component of a future
    `read_iceberg('catalog.namespace.table')`. Three `credentials_kind`s:
    `none`, `bearer_token` (a pre-obtained static token, the common case
    for Polaris/Nessie deployments fronting their own auth), and
    `oauth2_client_credentials` (the REST Catalog spec's own
    `POST /v1/oauth/tokens` flow). `validate_config()` checks `catalog_uri`
    is non-empty and each kind's required fields are present.
  - **Build**: `pkg_check_modules(CURL REQUIRED IMPORTED_TARGET libcurl)`
    (top-level `CMakeLists.txt`) and `cmake/ThirdPartyAvro.cmake`
    (hand-declared `Avro::avro` `IMPORTED` target via `find_library`/
    `find_path`, since avro-c ships neither a pkg-config file nor a CMake
    config package) are both unconditional, like the existing S3/GCS/
    Azure/HDFS deps -- not gated behind a build option.
  - **`IcebergRestCatalogClient`** (`include/kernellake/iceberg/
    rest_catalog_client.hpp`, `src/iceberg/rest_catalog_client.cpp`, new
    `kernellake_iceberg` static library): the first and only consumer of
    libcurl in the codebase so far, so it also establishes the
    RAII-wrapped-`CURL*`/`curl_slist*` pattern and write-callback
    convention future HTTP use can reuse. `load_table_metadata(namespace,
    table)` performs the REST Catalog spec's `GET .../namespaces/{ns}/
    tables/{table}` (multi-level namespaces joined by U+001F then
    URL-encoded as one path segment, per spec) and extracts just enough of
    the response -- `location`, `format-version`, `current-snapshot-id`,
    and each snapshot's `manifest-list` path -- for a future manifest
    reader to locate and read the current snapshot; schema/partition-spec
    translation into `kernellake::Schema` is intentionally deferred to that
    later integration. `oauth2_client_credentials` mode fetches and caches
    the bearer token (30s expiry safety margin, re-fetched on demand
    rather than proactively refreshed in the background).
  - New tests: `rest_catalog_client_test.cpp`, including a purpose-built
    single-connection-at-a-time loopback HTTP stub (raw sockets, not a
    mocking framework -- none exists in this test tree) exercising the
    real client against a real (local) HTTP server: bearer-token and
    oauth2-client-credentials auth, multi-level namespace encoding, the
    `prefix` config option, non-2xx responses, malformed/incomplete JSON,
    and connection failure, plus pure-unit coverage of
    `IcebergTableMetadata::current_manifest_list()`'s edge cases (no
    current snapshot, unknown snapshot id).
  - **`manifest_reader.cpp`** (`include/kernellake/iceberg/
    manifest_reader.hpp`, same `kernellake_iceberg` library): the first
    avro-c consumer, reading both manifest-list and manifest Avro Object
    Container Files off bytes an `ObjectStore` already fetched (no temp
    file: `fmemopen()` wraps the in-memory buffer as the `FILE*` avro-c's
    file-container reader requires) via avro-c's *generic* value API
    against the file's own embedded writer schema -- so it needs no
    hardcoded copy of the Iceberg spec's manifest schemas itself.
    `read_manifest_list()` extracts each entry's `manifest_path`/
    `manifest_length`/`content`/`added_snapshot_id` (per-manifest
    `partitions` summary stats not extracted, no pruning at that
    granularity yet). `read_manifest()` extracts each data/delete file
    entry's `status`/`file_path`/`file_format`/`record_count`/
    `file_size_in_bytes`, plus its `partition` struct's field values
    decoded generically by Avro type and *position* (int/long/string/null
    -- the only types Iceberg partition transforms produce at this layer;
    this reader has no partition-spec/schema of its own yet to interpret
    field *meaning* against, so a caller matches values back to spec
    fields by index). Column-level stats (`value_counts`,
    `lower_bounds`/`upper_bounds`, etc.) are decoded by avro-c along with
    everything else in each record but not extracted -- nothing consumes
    them yet.
  - New tests: `manifest_reader_test.cpp`, round-tripping through real
    Avro Object Container Files -- a small `AvroFixtureWriter` test helper
    uses avro-c's own writer API (simplified stand-in schemas: same field
    names/nesting this reader looks up, minus attributes it doesn't read)
    to produce real fixture bytes, covering ordered multi-entry
    manifest-lists, data-file entries with typed (string/int) and null
    partition values, malformed/empty input, and the `ObjectStore`-backed
    entry point end to end via a real `LocalObjectStore` + temp directory.
  - **Verification**: `dev` 227/227, `server-dev` 231/231
    `kernellake_unit_tests`, zero regressions; `gpu-dev` not re-verified
    this session (this environment's `gpu-dev` preset needs `nvcc` at
    `/usr/local/cuda/bin/nvcc`, not present in this shell -- pre-existing
    and unrelated to this change, which has no CUDA dependency of its
    own). Additionally run under ASan+UBSan+LeakSanitizer (a from-scratch
    build, not one of the checked-in presets): every test passes and the
    only leak found is inside system libavro-c 1.12.0 itself
    (`avro_file_reader_fp()`'s internal `avro_reader_memory()`, ~40 bytes),
    strictly on the malformed/truncated-input error path (2 allocations,
    matching the 2 tests that feed it corrupt bytes) -- never on a
    successful open, and with no handle in the public API to free it
    ourselves; documented at the call site
    (`src/iceberg/manifest_reader.cpp`'s `open_avro_reader()`) rather than
    worked around.
  - **Schema translation** (`include/kernellake/iceberg/
    schema_translation.hpp`, `src/iceberg/schema_translation.cpp`, same
    library): `iceberg_schema_to_kernellake_schema()` maps an Iceberg
    table's current-schema fields to a `kernellake::Schema` --
    `IcebergRestCatalogClient::load_table_metadata()` was extended to
    extract those fields in the first place (v2's `schemas` array +
    `current-schema-id`, falling back to v1's bare `schema` field for
    older/compat servers). Iceberg `required` becomes `nullable = false`;
    every supported primitive (boolean/int/long/float/double/date/
    timestamp/timestamptz/string/decimal(P,S)) maps directly, with
    `timestamp` and `timestamptz` both landing on kernellake's single
    Timestamp type since it has no timezone-aware/naive distinction of its
    own to map onto. Everything else -- time, uuid, fixed[N], binary, and
    every nested type (list/map/struct) -- throws rather than guessing at
    a lossy mapping; a struct/list/map field's raw JSON `type` (an object,
    not a string) is preserved via `dump()` on the way in specifically so
    that error message can show the caller what it actually saw.
  - New tests: `schema_translation_test.cpp` (every supported primitive,
    field order/naming preserved, required-vs-optional nullability,
    decimal precision/scale parsing, and rejection of time/uuid/nested
    types and malformed decimals) plus three new
    `rest_catalog_client_test.cpp` cases covering the v2 multi-schema
    selection, the v1 fallback, and an unmatched `current-schema-id`
    throwing. `dev` 239/239, `server-dev` 243/243 `kernellake_unit_tests`,
    zero regressions.
  - **`resolve_iceberg_table()`** (`include/kernellake/iceberg/
    iceberg_table_resolution.hpp`, `src/iceberg/iceberg_table_resolution.cpp`,
    same library): the piece that actually turns a catalog client +
    manifest reader + schema translation into a `ResolvedTable` -- the same
    shape `kernellake::resolve_table()` (plain/Hive Parquet,
    `kernellake/io/table_resolution.hpp`) produces, so both plug into the
    identical downstream seam (`convert_scan()` etc.) once a caller wires
    one in. Pipeline: `load_table_metadata()` -> current snapshot's
    manifest list -> each data manifest's entries -> live
    (status ADDED/EXISTING) data files -> `inspect_parquet_file()` per file
    for row-group statistics. Three real design calls, each documented at
    the call site rather than only here:
    - The table's *current* Iceberg schema (translated) is authoritative,
      not any individual file's own Parquet footer schema -- correct per
      Iceberg's schema-evolution model, but every live file's physical
      schema is required to match it *exactly* for now; genuine
      cross-snapshot schema evolution (added/renamed/widened columns)
      isn't reconciled yet and throws StorageError naming the offending
      file and field, rather than attempting a reconciliation this
      resolver doesn't do.
    - A live delete manifest (manifest-list `content == 1`) throws rather
      than being silently skipped: ignoring it would mean returning rows
      the snapshot says are deleted -- a correctness bug, not a missing
      feature.
    - `ResolvedTable::partition_columns` is always empty from this path
      (unlike the Hive-partitioning path): Iceberg's partition columns are
      already ordinary schema columns present in every data file (for the
      identity-transform case this targets first), not values that must be
      reconstructed from something absent in the file the way Hive's
      directory-encoded partitions are -- so no column materialization is
      needed for correctness. Partition *pruning* (skipping whole files by
      their manifest-recorded values without opening them) is a pure
      optimization, not yet implemented.
  - New tests: `iceberg_table_resolution_test.cpp` -- genuine end-to-end
    integration, not structural: a fake HTTP REST catalog (the same
    loopback-socket stub pattern as `rest_catalog_client_test.cpp`, real
    Avro manifest-list/manifest fixtures (avro-c's own writer API, same
    pattern as `manifest_reader_test.cpp`), and real Parquet data files
    (Arrow's Parquet writer, same pattern as `table_resolution_test.cpp`),
    all wired through a real `LocalObjectStore`. Covers: live files
    resolved with correct schema/row counts, DELETED-status entries
    correctly excluded, a live delete manifest throwing, a non-PARQUET
    data file throwing, a data file whose physical schema doesn't match
    the table's current schema throwing, and a table with no current
    snapshot resolving to zero files (a real, valid empty-table state, not
    an error).
  - **Verification**: `dev` 245/245, `server-dev` 249/249
    `kernellake_unit_tests`, zero regressions; also re-run under the same
    from-scratch ASan+UBSan+LeakSanitizer build used for the manifest
    reader -- all tests pass, and the only leak is the same
    already-diagnosed libavro-c 1.12.0 error-path leak from the manifest
    reader's own malformed-input tests, nothing new from this file's
    heavier avro-c/socket/Parquet-writer use.
  - **`read_iceberg('catalog.namespace.table')` SQL surface** -- the last
    piece of Phase 3, landing this session: `SELECT ... FROM
    read_iceberg('prod.db.orders')` now parses, binds, and physically plans
    end to end, including a two-table JOIN mixing a `read_parquet(...)` and
    a `read_iceberg(...)` source. The user-facing syntax was a deliberate
    choice, made after asking rather than unilaterally: mirrors
    `read_parquet(...)`'s existing function-call shape exactly, needing the
    smallest possible grammar surface (the alternative -- a bare
    `FROM prod.db.orders` identifier, closer to Trino/Spark -- would have
    needed real hyrise/sql-parser FROM-clause grammar changes to
    disambiguate from a future non-Iceberg named table).
    - **Zero AST/binder/`LogicalScan` changes.** `read_iceberg(...)`'s
      single string argument is re-encoded by the SQL preprocessor (see
      `preprocess_from_read_parquet()`'s now-generalized doc comment,
      `src/sql/parser.cpp`) as a single source path
      `"iceberg://catalog.namespace.table"` -- reusing `kernellake::Uri`'s
      existing scheme-dispatch idiom (the same one `ObjectStoreRegistry`
      already uses for S3/GCS/Azure/HDFS) instead of threading a new
      "source kind" concept through `AstParquetSource`, `BoundQuery`/
      `BoundJoin`, and `LogicalScan`, all of which stay exactly as they
      were before this phase.
    - **`TableSourceResolver`** (`kernellake/io/table_resolution.hpp`, new
      abstract interface, plus `resolve_table_or_delegate()`): the one real
      architectural addition, needed because `kernellake_iceberg` already
      depends on `kernellake_io` (for `inspect_parquet_file()`), so
      `kernellake_io` can't depend back on `kernellake_iceberg` without a
      cycle. The interface -- `can_resolve()`/`resolve()` -- lives in
      `kernellake_io` with no new dependency of its own;
      `kernellake::iceberg::IcebergSourceResolver`
      (`include/kernellake/iceberg/iceberg_source_resolver.hpp`) implements
      it, parsing the `iceberg://` marker, splitting
      `catalog.namespace.table` (>=3 non-empty dot-separated parts
      required), looking up the catalog by name in
      `EngineConfig::iceberg.catalogs`, and calling
      `resolve_iceberg_table()`. `QueryEngine` (`src/api/query_engine.cpp`)
      constructs one and passes it into both of `kernellake_io`'s own
      `resolve_table()` call sites -- `plan_logical()`'s schema-discovery
      step and `physical_planner.cpp`'s `convert_scan()` (now taking an
      `extra_resolver` parameter, threaded through its whole recursive
      `convert()` walk) -- via `resolve_table_or_delegate()`, so a source
      `resolve_table()` itself can't handle still resolves consistently at
      both places. Delta Lake/Unity Catalog integration (see above) is
      expected to plug in through this exact same seam.
      `IcebergRestCatalogClient` is constructed fresh per resolve call, not
      cached across queries -- simpler and correct, at the cost of
      repeating the `oauth2_client_credentials` handshake per query for
      that `credentials_kind`; a documented, deliberate MVP simplification,
      not a permanent design decision.
    - New tests: `sql_parser_test.cpp` (parses `read_iceberg(...)`, rejects
      more than one argument, a mixed `read_parquet`/`read_iceberg` JOIN),
      `iceberg_source_resolver_test.cpp` (scheme dispatch, unknown catalog,
      malformed qualified names), and the capstone --
      `query_engine_iceberg_test.cpp` -- a real `QueryEngine` (constructed
      exactly as the CLI/Flight SQL server would) running
      `explain_logical()`/`explain()` over `read_iceberg(...)` SQL text
      end to end against a fake HTTP REST catalog and real Avro/Parquet
      fixtures, verifying the resulting logical schema and that a real
      `ParquetScanNode` comes out the other end.
    - **Verification**: `dev` 256/256, `server-dev` 260/260
      `kernellake_unit_tests`, zero regressions (confirmed the existing
      `read_parquet(...)`-only suite is entirely unaffected -- the whole
      point of the zero-AST-change design); `gpu-dev` not re-verified this
      session (same pre-existing `nvcc` path gap noted earlier in this
      phase, unrelated to this change). Re-run under the same from-scratch
      ASan+UBSan+LeakSanitizer build used earlier in this phase: all 256
      tests pass, only the same already-diagnosed libavro-c 1.12.0
      error-path leak, nothing new.
    - Not yet done, tracked as a later phase: interpreting partition
      values against named, transform-aware partition-spec fields
      (bucket/truncate/year/month/day/hour) for partition-level pruning
      (a pure optimization, not a correctness gap -- see
      `resolve_iceberg_table()`'s own comment on why partition columns
      aren't needed for correct results today), and caching
      `IcebergRestCatalogClient`/its OAuth2 token across queries against
      the same catalog.

  **Phase 3 (Iceberg REST catalogs) is now functionally complete for the
  MVP's scope**: config, REST catalog client (bearer/OAuth2 auth), Avro
  manifest reading, Iceberg-to-`kernellake::Schema` translation, real file
  resolution with row-group pruning, and a working `read_iceberg(...)` SQL
  surface, all tested end to end against real fixtures. Row-level deletes,
  schema evolution across files, and partition-spec-aware pruning remain
  explicit, documented non-goals for now -- see the entries above for
  exactly where each one throws instead of guessing.

- **Delta Lake support, started (gRPC client to delta-txn-service)**
  (Phase 4 of the lakehouse roadmap). Supersedes this file's own earlier
  "Not yet started" entry for Delta, which assumed the only path in was
  vendoring `delta-kernel-rs` (or `delta-rs`, what it turned out to use)
  directly into this C++ project via a Rust toolchain -- the actual path
  taken avoids that entirely: KernelLake talks to a separate, standalone
  Rust gRPC service, `delta-txn-service`
  (a sibling repo, not part of this one), purely as a *client*, the same
  way it's already a client of S3/GCS/Azure/HDFS/Iceberg REST catalogs.
  No Rust toolchain, no `delta-kernel-rs`/`delta-rs` dependency, anywhere
  in this repo's own build.
  - **Two RPCs consumed**: `GetTable` (version/schema/protocol) and the
    new `ListActiveFiles` (server-streaming active-file listing -- didn't
    exist before this session; added to delta-txn-service specifically
    for this, since it previously only supported writes). Both added to
    delta-txn-service's own proto and Rust implementation as part of this
    work, not pre-existing.
  - **Config**: `DeltaSection` (`include/kernellake/common/config.hpp`) --
    a single section, not a name-keyed map like `IcebergSection`:
    delta-txn-service is architecturally "a centralized coordinator," one
    deployment per environment, and a Delta table is addressed directly
    by its own storage URI with no catalog/namespace concept to key a map
    by. `grpc_endpoint` empty means not configured.
  - **Proto vendoring + C++ codegen**: `proto/delta_txn.proto` is a
    hand-copied (not submodule/fetch-linked) copy of delta-txn-service's
    own proto file -- see that file's own header comment for the sync
    convention. `cmake/ThirdPartyDeltaTxnProto.cmake` runs `protoc` +
    `grpc_cpp_plugin` at build time to generate real C++ client stubs
    into the build directory (never checked in). `gRPC`/`Protobuf`
    `find_package` calls moved from `KERNELLAKE_BUILD_SERVER`-gated to
    unconditional (a Delta *client* is ordinary query-engine
    functionality, not server-only, unlike the Flight SQL server that
    previously was gRPC's only consumer here).
  - **`DeltaTxnClient`** (`include/kernellake/delta/delta_txn_client.hpp`,
    `src/delta/delta_txn_client.cpp`, new `kernellake_delta` static
    library): pimpl'd (matches `IcebergRestCatalogClient`'s own
    convention) so consumers never see a grpc++/generated-proto type.
    `get_table()`/`list_active_files()` translate straight into plain
    `DeltaTableInfo`/`DeltaActiveFile` structs -- schema translation
    (Delta's own JSON schema string -> `kernellake::Schema`) and the
    actual `resolve_delta_table()`/`read_delta(...)` SQL-surface
    integration (the Iceberg-phase equivalent of
    `resolve_iceberg_table()`/`IcebergSourceResolver`) are **not**
    part of this slice -- this client can fetch real table state and
    file lists end to end, but nothing in the SQL grammar/binder/
    physical planner calls it yet. That's the next piece.
  - **Distributed tracing, both sides**: `kernellake::observability`
    gained `ClientSpan` (a new, more general sibling to the existing
    whole-query `QuerySpan`) -- creates a real child span per outbound
    call and injects its W3C `traceparent`/`tracestate` context into gRPC
    metadata, working in both the stub (`KERNELLAKE_ENABLE_OTEL=OFF`,
    the default -- `inject()` is a no-op) and real
    (`KERNELLAKE_ENABLE_OTEL=ON`) builds; `QuerySpan` was also changed to
    attach itself as the ambient `RuntimeContext`, so a `ClientSpan`
    started during query execution correctly nests under it. On
    delta-txn-service's own side, a real gap was found and fixed while
    wiring this up: the service already had `tracing_opentelemetry`
    wired in, but never actually created a span around any request, so
    there was nothing to export regardless -- a new `TraceContextLayer`
    (mirroring the existing `GrpcMetricsLayer`'s tower-`Layer` shape) now
    extracts an incoming `traceparent` and creates the actual per-request
    span everything else runs inside.
  - **A real concurrency bug found and fixed in delta-txn-service**
    (prompted by an explicit request to audit for races): `TableLockManager
    ::remove_if_unused()`'s ref-count-then-remove was two separate,
    non-atomic steps -- a concurrent `lock_for()` for the same table_uri
    could resurrect an entry moments before it was removed anyway,
    letting two concurrent commits to the same table end up on two
    different mutexes with no mutual exclusion between them at all. Not a
    data-corruption risk (Delta's own atomic-conditional-put commit
    protocol is the real backstop), but it defeated the whole reason
    that lock exists. Fixed via `DashMap`'s `Entry` API (one atomic
    check-and-remove under the same shard lock `lock_for()` itself uses)
    and backed by two new `#[tokio::test(flavor = "multi_thread")]`
    stress tests that both reliably failed against the old
    implementation.
  - **A full audit of delta-txn-service's own pre-existing code**
    surfaced two more real, documented-but-not-yet-fixed findings (see
    each file's own doc comment for the full detail, and README.md's
    "Metrics"/"Concurrency model" sections for the operator-facing
    summary): `commit.rs` always uses `DeltaOperation::Write` for
    delta-rs's own conflict-checking regardless of what `CommitOperation`
    a client actually requested (affects conflict-checking precision for
    non-Write operations, not correctness); `telemetry/metrics.rs`'s
    `grpc.server.errors` counter only reliably catches pre-flight
    rejections, not real application-level failures, since gRPC's actual
    status arrives in HTTP/2 trailers this middleware never inspects.
    Every Rust source file plus both proto files (delta-txn-service's own
    and KernelLake's vendored copy) were also given substantially more
    explanatory comments as part of this same audit pass.
  - **Verification**: every delta-txn-service change was verified via a
    real `docker build` (the repo's own Dockerfile already runs
    `cargo test --release` as a build step) -- not just written and
    assumed correct, since this sandbox has no Rust toolchain of its own
    to compile against directly; two early API-usage mistakes (a method
    that turned out to be `pub(crate)` not `pub`, a missing extension-
    trait import) were each caught this way and fixed before landing.
    `dev` 267/267, `server-dev` and `otel-dev` equally green (including a
    new `DeltaTxnClient` test suite exercising a real in-process gRPC
    server, and a trace-context test proving a real, well-formed
    `traceparent` header end to end), zero regressions on the KernelLake
    side.
  - Not yet done: `resolve_delta_table()`, Delta-schema-string ->
    `kernellake::Schema` translation, a `DeltaSourceResolver`
    (`TableSourceResolver` implementation, same seam Iceberg's own
    resolver plugs into), and the `read_delta(...)` SQL surface itself --
    this phase currently ends at "can fetch real table state and file
    lists from delta-txn-service," not yet "can query a Delta table."

## Not yet started

- **GPU Parquet scan's cold-vs-warm gap is explained by cuFile running in
  compatibility mode on this dev machine, not a kernellake code issue.**
  Following up the `HashAggregateOperator` redesign above, `Not yet
  started`'s to-do to profile the scan path: `parquet_decoding_seconds` is
  now the dominant cost for both Q1 (58% of GPU total) and Q6 (96%), so it
  was next up. Controlled measurement (evicting page cache via
  `posix_fadvise DONTNEED`, same technique `benchmark_three_way.py` already
  uses for "cold"): a real SF1000 Q6 scan (107GiB compressed `lineitem`)
  takes 45.8s cold (~2.3 GB/s) vs. 26.7s immediately after, warm (~4.0
  GB/s) -- a real, repeatable 72% gap, not noise (earlier `--stats` numbers
  quoted elsewhere in this file weren't cache-controlled and shouldn't be
  compared against each other for this reason). A naive parallel `dd
  iflag=direct` read of the same files only reached ~1.9 GB/s, so cuFile's
  cold-path read is already *faster* than a naive raw-I/O baseline --
  ruling out "kernellake/cuFile isn't parallelizing reads enough" as the
  explanation.

  Root cause: `/usr/local/cuda-12.6/gds/tools/gdscheck -p` on this machine
  reports `properties.use_compat_mode : true` and `NVMe : Unsupported`
  under `DRIVER CONFIGURATION` -- the `nvidia-fs` kernel module (required
  for true GPU-Direct Storage, DMA straight from NVMe to GPU memory,
  bypassing the host entirely) isn't loaded (`lsmod | grep nvidia_fs`:
  empty), so every cuFile read on this box goes disk -> host page cache ->
  cuFile's internal POSIX bounce-buffer pool -> GPU, not a direct path.
  That's exactly why warm (page cache already populated) is so much faster
  than cold (forces a real disk read) -- and it's an environment/driver gap
  on this specific dev machine, not something fixable in kernellake's own
  source.

  Investigated what installing `nvidia-fs` would take, and explicitly
  decided not to attempt it -- three independent reasons, not just one:
  (1) NVIDIA's own official GDS support matrix requires a Tesla- or
  Quadro-class GPU (Pascal/Volta/Turing/Ampere); this machine's RTX 3070 is
  Ampere but GeForce (consumer), not Tesla/Quadro, so it's outside official
  support regardless of driver/module versions. (2) `gdscheck`'s own
  driver-compatibility line says the installed GPU driver (610.57.04) only
  supports `nvidia-fs` <= 2.17.4, but every version available via the
  already-configured CUDA apt repo is >= 2.20.6 -- no version this system
  can actually install is one `gdscheck` itself claims is compatible.
  (3) `gdscheck` separately warns GDS "is not guaranteed to work
  functionally or in a performant way" under this system's current IOMMU
  passthrough setting. (Mechanically, an install attempt would likely at
  least *build*: matching kernel headers for the running `7.0.0-28-generic`
  are installed, DKMS already works here -- the NVIDIA driver itself is
  DKMS-built for this exact kernel -- and Secure Boot is disabled, so no
  signing blocker. That just means it probably wouldn't fail loudly, not
  that it would fix anything.) Net: real root/kernel-module risk on a
  shared dev machine, stacked against three separate signals that it
  likely wouldn't resolve the compat-mode gap even if it installed cleanly
  -- not attempted. If a future session has access to actual
  Tesla/Quadro-class hardware, this would be worth reopening; on this
  machine it's a dead end.
- **Q6's remaining GPU-vs-PySpark gap isn't explained by cuFile I/O thread
  count either, and the actual cause is still open.** Follow-up to the
  compat-mode finding above: tried raising `execution.max_io_threads`
  (cuFile's host-thread-per-GPU count for parallel I/O in compat mode)
  from its default of 4 to 16 via a custom `cufile.json` bind-mounted into
  the container. No effect -- 45.76s vs. the 45.84-45.95s baseline, within
  run-to-run noise. `mpstat` sampled during a real cold-cache scan explains
  why: of 20 logical cores, only ~4-5 show any activity at all (matching
  the default thread count), and those show substantial `%iowait`
  (up to ~53%) rather than `%usr`/`%sys` -- i.e. those threads are mostly
  blocked waiting on the disk, not CPU-bound on decompression, so adding
  more threads had nothing to parallelize against. Consistent with this: a
  raw `dd iflag=direct` sequential read of the same files tops out around
  1.8-1.9 GB/s regardless of parallelism (1-way vs. 8-way concurrent `dd`
  processes made almost no difference either), while cuFile's actual
  cold-scan throughput is ~2.3 GB/s -- already *faster* than the naive
  baseline, suggesting kernellake's cold-mode scan is close to a real
  hardware/access-pattern ceiling for this NVMe, not something more
  threads or config tuning fixes.

  Open puzzle, not yet root-caused: PySpark's cold read of the same data
  hits ~3.5 GB/s -- faster than *both* kernellake's cuFile scan (2.3 GB/s)
  *and* the raw `dd` baseline (1.9 GB/s) on the identical disk. Thread
  count and a hardware ceiling don't explain this, since PySpark exceeds
  what raw sequential `dd` reads achieve on the same hardware. Leading
  candidate: an access-pattern difference, not a parallelism one -- cudf's
  chunked Parquet reader reads specific column-chunk byte ranges (likely a
  more scattered read pattern within each file even though it reads less
  total data per row group), whereas PySpark's reader may read more
  contiguously per file. Confirming this would need syscall-level I/O
  tracing (comparing actual `read()` offsets/sizes between the two
  engines' access to the same files) -- not done this session. Like the
  GDS finding above, this increasingly looks like it may not have a fix
  within kernellake's own code either way, but that isn't confirmed yet.
- TPC-H `execution-only` benchmark mode (needs an operator-tree entry point
  that skips `ParquetScanOperator`); TPC-H beyond SF10 (SF0.01/0.1/1/10 all
  verified, see "Done" above)
- Re-running the three-way benchmark's full cold/warm x SF0.01-SF10 sweep
  (see the earlier "Confirmed on real GPU hardware" entry above) with Q19
  and Q12 included, now that both are wired in -- the only way to test
  whether GPU vs. PySpark's crossover point (PySpark stays fastest through
  SF10 on the Q1/Q6-only sweep already run) looks different for a
  join-shaped query; not yet done at any scale factor beyond the SF0.01
  spot-checks above
- Wiring TPC-H Q14 into `tools/benchmark_three_way.py`'s three-way
  performance comparison, the same way Q19 already is (it needs the same
  `--part-data` mechanism Q19 uses, so this should be a small, mostly
  mechanical follow-up, not a new fix) -- not yet done
- `CASE` inside `WHERE` on the GPU backend (`FilterOperator`'s own gap,
  separate from the aggregate-argument fix above; the CPU backend already
  supports this via its one shared expression compiler) -- not needed by
  any TPC-H query added so far, so not yet prioritized
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
- CPU execution backend's Parquet scan is single-threaded, unlike Acero
  itself. `acero_query_executor.cpp`'s `read_scan_table()` (every
  `ParquetScanNode` is translated to a `"table_source"` Declaration fed by
  this function, not Acero's own dataset-aware `"scan"` node) reads each
  fragment with `parquet::arrow::FileReader::Make()` (no `use_threads` set,
  defaults off) in a plain sequential loop across files, entirely before
  Acero's own multi-threaded `ExecPlan` (`DeclarationToTable()`'s default
  `ExecContext`, backed by `arrow::internal::GetCpuThreadPool()`) ever
  starts -- so the scan itself, likely the dominant cost for a
  near-unfiltered query like Q1, never uses more than one core. Noticed
  investigating why KernelLake-CPU's SF100 Q1 (42.05s median) is
  disproportionately slower than both KernelLake-GPU (10.27s) and
  PySpark's `local[*]` (9.02s, which explicitly parallelizes the scan
  across every core) -- not yet root-caused further or fixed, just
  identified for a later session. Likely fix shape: either read fragments
  concurrently (e.g. one thread per file, matching this project's own
  20-core dev hardware) or switch to Acero's native `"scan"`
  (`arrow::acero::ScanNodeOptions` over an `arrow::dataset::Dataset`)
  instead of pre-materializing a `table_source`, which would also pick up
  Acero's own fragment-level parallelism for free.
- GPU `HashJoinOperator` still has no bounded-memory/streaming design --
  unlike `ParquetScanOperator`'s pass-based reading or
  `HashAggregateOperator`'s `max_distinct_keys` batching, `open()`
  unconditionally drains the *entire* right (build) side into device
  memory (`while (right_->next()) { right_batches.push_back(...) }`, then
  `cudf::concatenate()` and one `cudf::hash_join` over all of it) before
  the probe side is ever touched. The physical planner now chooses the
  smaller side to build on (see "Done" above), which narrows the real
  SF100 Q12 OOM this entry originally documented (`lineitem`, 600M rows,
  used to be forced onto the build side purely by clause order), but
  doesn't close it: building on `orders` (150M rows) instead would still
  likely exceed a real 8-12 GiB VRAM budget at that scale. Genuinely
  bounded memory needs a streaming/partitioned (grace) hash join, a
  substantially bigger feature than the build-side fix. Not yet
  re-attempted at SF100 with the fix in place to confirm exactly how much
  it narrows the gap in practice.
- Q19 on the **CPU** backend got killed partway through SF100 timed
  iterations (correctness validated fine first; died on cold iteration
  3/5) -- a real, reproduced-once resource failure, root cause not yet
  isolated. The `kernellake` subprocess's `stderr` was empty on failure
  (every other real crash this session printed an actual message), and
  host swap usage went from unused to 3 GiB over the session, both
  consistent with the OS OOM-killing the process rather than it throwing
  a normal exception. Candidate causes, none confirmed yet: (1)
  `acero_query_executor.cpp`'s `read_scan_table()` materializes an
  entire join's input tables into host RAM with no bound (a documented
  MVP simplification, see that file's own comment -- for Q19 that's all
  of `lineitem`'s needed columns, ~600M rows); (2) PySpark's
  `spark.driver.memory=64g` JVM heap not being reclaimed between the
  three-way loop's repeated iterations; (3) both compounding under `both`
  mode's repeated cold-then-warm cycles. A retry of the same run
  immediately after (with `--orders-data` dropped, Q19 kept) completed
  Q1/Q6 cleanly through all 20 iterations with no recurrence, so this
  isn't a hard, deterministic block at this scale -- but also not
  something to treat as a one-off fluke without further investigation
  before trusting a full Q1/Q6/Q19 SF100 (or larger) run's numbers.

  Candidate cause (1) above (`read_scan_table()`'s unbounded
  materialization) has since been fixed -- see "Done" above -- though
  whether that was actually *this* crash's root cause was never confirmed
  either way, so this entry is left as an open, not-fully-explained
  historical incident rather than marked resolved.

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
