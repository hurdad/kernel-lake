# syntax=docker/dockerfile:1

# Single place to bump the GPU build's CUDA version -- overridable at
# `docker build` time via `--build-arg CUDA_VERSION=...` without editing
# this file. Must be a real published `nvidia/cuda:<version>-devel-
# ubuntu26.04` tag (NVIDIA only started publishing ubuntu26.04-tagged
# images at 13.3 -- see docs/ARCHITECTURE.md's "Ubuntu 26.04 baseline"
# section). Declared before the first FROM so it's in scope for
# gpu-build-base's own FROM line below; not re-declared in any later
# stage since nothing past gpu-build-base needs the raw version string
# again.
ARG CUDA_VERSION=13.3.1
# Explicit frontend pin (added 2026-08-17, alongside this file's first use
# of `RUN --mount=type=cache` in cpu-release/gpu-release below): modern
# Docker/buildx defaults to a recent-enough frontend that this likely
# wasn't strictly required, but a `RUN --mount` syntax failure would be a
# real, if unlikely, way for docker-publish-cpu/-gpu to break -- pin
# explicitly rather than rely on whatever a given runner's Docker install
# happens to default to.
#
# Single multi-stage Dockerfile publishing two runtime images from one
# source of truth: `runtime-cpu` (CPU-only, no CUDA/RAPIDS -- built from the
# `cpu-release` preset) and `runtime-gpu` (full CUDA/RAPIDS build -- built
# from the `gpu-release` preset). Both published images are optimized
# (RelWithDebInfo) builds, not the Debug builds this project's own
# `cpu-dev`/`gpu-dev` presets produce for local iteration -- see
# CMakePresets.json. Both ship the same two binaries (kernellake CLI,
# kernellake-server with OpenTelemetry export) plus exactly their own
# non-base-system shared-library closure -- no compiler, no CUDA headers,
# no nvcc, no static libs, no source tree. The intermediate `cpu-release`/
# `gpu-release` build stages (full toolchain, repo built inside them) are
# never tagged/pushed on their own -- only `--target runtime-cpu`/
# `runtime-gpu` are published (see .github/workflows/ci.yml's docker-publish
# job); build either stage directly with --target if you want a full build
# environment to iterate in. A third, also-never-published stage,
# `benchmark-gpu`, adds PySpark on top of `gpu-release` for
# tools/benchmark_three_way.py's ad-hoc KernelLake-CPU vs. KernelLake-GPU
# vs. PySpark comparison -- see that stage's own comment below.
#
# runtime-cpu already publishes as a real multi-arch (linux/amd64+
# linux/arm64) manifest -- see .github/workflows/ci.yml's docker-publish
# job (docker buildx + docker/setup-qemu-action) and docs/ROADMAP.md for
# how that was verified. runtime-gpu stays linux/amd64-only: NVIDIA's arm64
# CUDA support (Jetson/Grace-class hardware) hasn't been verified, and
# unlike runtime-cpu a build-only smoke test can't catch a real
# GPU-execution regression -- that needs real arm64 GPU hardware, not
# available yet (see docs/ROADMAP.md's "Not yet started" section).
#
#   docker build --target cpu-release -f docker/Dockerfile -t kernellake/kernellake:cpu-release .
#   docker build --target gpu-release -f docker/Dockerfile -t kernellake/kernellake:gpu-release .
#   docker build --target runtime-cpu -f docker/Dockerfile -t kernellake/kernellake:runtime-cpu .
#   docker build --target runtime-gpu -f docker/Dockerfile -t kernellake/kernellake:runtime-gpu .
#   docker build                      -f docker/Dockerfile -t kernellake/kernellake:runtime-cpu . # runtime-cpu is the default (last) stage
#
# Run either runtime image the same way (the spec's own usage example, GPU
# variant):
#   docker run --rm --gpus all -v /tmp/kernellake-sales:/data:ro \
#     kernellake/kernellake:runtime-gpu query --backend gpu --sql "SELECT region, SUM(amount) FROM read_parquet('/data/*.parquet') GROUP BY region"
#
# The CPU image needs no --gpus flag and defaults to the CPU backend:
#   docker run --rm -v /tmp/kernellake-sales:/data:ro \
#     kernellake/kernellake:runtime-cpu query --sql "SELECT region, SUM(amount) FROM read_parquet('/data/*.parquet') GROUP BY region"
#
# benchmark-gpu (needs --gpus all -- it exercises the real GPU backend as
# one of the three engines compared). Mount a directory already populated
# by tools/generate_tpch.py (run once on the host, outside the container)
# rather than regenerating the dataset on every benchmark run:
#   docker build --target benchmark-gpu -f docker/Dockerfile -t kernellake/kernellake:benchmark-gpu .
#   docker run --rm --gpus all -v /tmp/kernellake-tpch:/data:ro \
#     kernellake/kernellake:benchmark-gpu --kernellake build/gpu-release/src/cli/kernellake \
#       --data '/data/*.parquet' --query all --iterations 5 --output /data/report.json
#   docker run --rm -v /tmp/kernellake-tpch:/data --entrypoint python3 \
#     kernellake/kernellake:benchmark-gpu tools/generate_benchmark_report.py \
#       --input /data/sf0.01/report.json /data/sf0.1/report.json /data/sf1/report.json \
#       --output /data/benchmark_report.pdf
#
# Base: plain `ubuntu:26.04` for the CPU-only build path (`build-base`,
# `cpu-release`), NVIDIA's official `nvidia/cuda:${CUDA_VERSION}-devel-
# ubuntu26.04` image for the GPU build path (`gpu-build-base`, built
# straight from that image rather than from `build-base` -- see that
# stage's own comment). This project spent one earlier session avoiding
# NVIDIA's official image in favor of Ubuntu's own apt `nvidia-cuda-
# toolkit` package specifically to dodge the CUDA 12->13 major-version
# bump that switching would have forced (NVIDIA only publishes
# `ubuntu26.04`-tagged images starting at CUDA 13.3) -- see
# docs/ARCHITECTURE.md's "Ubuntu 26.04 baseline" section for the full
# history. Revisited once RAPIDS libcudf/RMM/kvikio/nvcomp turned out to
# publish `-cu13` wheels at the exact same versions already pinned
# (`cmake/ThirdPartyRapids.cmake`), making the re-vendor a clean
# CUDA-suffix swap rather than also a RAPIDS version bump. Both CUDA
# major versions stay supported there (selected automatically from
# whatever `nvcc` a given configure actually finds, not a second
# hardcoded setting) so a non-Docker CUDA 12.x dev environment keeps
# working unchanged.
#
# `docs/ARCHITECTURE.md`'s other two Ubuntu-26.04-specific facts (Arrow
# Flight SQL doesn't link against Ubuntu 24.04's older system Abseil;
# otel-cpp has no 24.04 apt package at all) are unrelated to this CUDA
# sub-decision and still apply unchanged -- both build paths stay on
# Ubuntu 26.04 regardless of where CUDA itself comes from.
#
# Phase 3 (docs/ROADMAP.md): also builds and ships kernellake-server
# (KERNELLAKE_BUILD_SERVER=ON) with OpenTelemetry export enabled
# (KERNELLAKE_ENABLE_OTEL=ON), alongside the kernellake CLI -- both
# binaries, one image, same scope in runtime-cpu and runtime-gpu (they
# differ only in whether the GPU execution backend is compiled in). See
# charts/kernellake/ for the Helm chart deploying kernellake-server as a
# Deployment+Service on top of these images (it picks runtime-cpu or
# runtime-gpu based on the chart's own `backend` value).

# ---------------------------------------------------------------------------
# build-base: everything shared between the CPU-only and GPU build
# environments. Not a runnable image on its own (no COPY of the repo, no
# build). Named generically (not tied to a specific CMake preset) since
# both downstream build stages now use the optimized `cpu-release`/
# `gpu-release` presets, not a `dev`-named one.
# ---------------------------------------------------------------------------
FROM ubuntu:26.04 AS build-base

ENV DEBIAN_FRONTEND=noninteractive

# One layer for everything build-base needs, in dependency order:
#   0. `apt-get upgrade -y` right after `update` -- patches whatever's
#      already in the `ubuntu:26.04` base layer (glibc, openssl, etc.) to
#      whatever the archive currently serves, not just the packages this
#      step goes on to install fresh. Standard practice for not silently
#      building on a stale, potentially-vulnerable base-image snapshot.
#   1. wget/ca-certificates/lsb-release/gnupg -- prerequisites for step 2
#      itself (fetching and registering the Arrow apt source).
#   2. The Apache Arrow apt source (resolves to the `resolute` suite
#      automatically via lsb_release), then a second `apt-get update` to
#      pick it up.
#   3. Everything else in one apt-get install: core toolchain; Arrow/
#      Parquet/Flight/Flight-SQL (pinned to =25.0.1-1, the exact apt
#      package version string incl. Debian revision -- every other real
#      third-party dependency here is pinned already, RAPIDS by
#      URL+SHA-256 in cmake/ThirdPartyRapids.cmake, so Arrow floating to
#      whatever "-latest-" served was the one real exception; if this
#      revision ever disappears from the repo, bump it to whatever
#      `apt-cache policy libarrow-dev` reports as available then, don't
#      unpin); libxml2-dev/uuid-dev (Azure SDK's bundled XML parsing/
#      request-ID generation -- config.hpp unconditionally links Arrow's
#      bundled cloud-storage code, see docs/ARCHITECTURE.md's "Cloud
#      object storage" section); libavro-dev/libcurl4-openssl-dev
#      (Iceberg manifest reading and REST/OAuth2 calls); gRPC/
#      opentelemetry-cpp-dev (kernellake-server + OTel export, needed in
#      both CPU and GPU images now that both ship kernellake-server);
#      clang-format-18/clang-tidy-18 (same pinned major version CI's
#      format-check/clang-tidy jobs and .pre-commit-config.yaml already
#      use -- lets any image built from this stage run either lint
#      locally, matching what .pre-commit-config.yaml's own comment
#      already documents as the correct way to verify GPU-only files
#      clang-tidy's CI job can't reach, e.g. `docker run --entrypoint
#      clang-tidy-18 kernellake:gpu-release -p build/gpu-release ...`
#      against this same image's own already-built compile_commands.json,
#      no separate install step needed).
#   4. gcovr (pip, not apt -- Ubuntu's own gcovr package lags upstream) --
#      used by .github/workflows/ci.yml's coverage/gpu-coverage jobs, both
#      of which `docker run` a stage built from this one rather than
#      installing gcovr fresh in a separate CI step every run.
#   5. ccache -- every CI job that actually compiles (cpu-build-test,
#      gpu-build-test, coverage, gpu-coverage, server-build-test,
#      otel-build-test, clang-tidy, and cpu-release/gpu-release below)
#      previously started stone cold every run, recompiling the entire
#      tree from scratch on every push -- Docker's own layer cache doesn't
#      help here even for the two in-Dockerfile release builds, since
#      `COPY . .` (and therefore the `RUN cmake --build` layer after it)
#      invalidates on *any* file change anywhere in the repo, which is
#      every real push. Installed here so it's baked into the image these
#      jobs already build/cache instead of an apt-get that'd run fresh
#      every time; each job passes -DCMAKE_<LANG>_COMPILER_LAUNCHER=ccache
#      at configure time and either mounts a host-persisted, GHA-cached
#      ccache directory into its own `docker run` invocation (the six
#      jobs in .github/workflows/ci.yml that build outside the Dockerfile
#      -- see each one's own `Restore ccache` step), or uses a BuildKit
#      `RUN --mount=type=cache` (cpu-release/gpu-release below, which
#      build *inside* the Dockerfile itself -- see those stages' own
#      comments).
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
      wget ca-certificates lsb-release gnupg \
    && wget -O /tmp/arrow-apt-source.deb \
      "https://packages.apache.org/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb" \
    && apt-get install -y -V /tmp/arrow-apt-source.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential cmake ninja-build git pkg-config ccache \
         python3 python3-pip \
         libgtest-dev libbenchmark-dev libspdlog-dev nlohmann-json3-dev libyaml-cpp-dev \
         libxml2-dev uuid-dev libavro-dev libcurl4-openssl-dev \
         libgrpc++-dev protobuf-compiler-grpc opentelemetry-cpp-dev \
         libarrow-dev=25.0.1-1 libparquet-dev=25.0.1-1 libarrow-dataset-dev=25.0.1-1 \
         libarrow-flight-dev=25.0.1-1 libarrow-flight-sql-dev=25.0.1-1 \
         clang-format-18 clang-tidy-18 \
    && rm -rf /var/lib/apt/lists/* /tmp/arrow-apt-source.deb \
    && pip install --break-system-packages --no-cache-dir gcovr

# ---------------------------------------------------------------------------
# cpu-release: full CPU-only build environment, repo built inside it
# (`cpu-release` preset -- optimized, RelWithDebInfo -- + BUILD_SERVER/
# ENABLE_OTEL, no CUDA at all). Also a complete, runnable image on its own
# (`--target cpu-release`).
# ---------------------------------------------------------------------------
FROM build-base AS cpu-release

WORKDIR /workspace
COPY . .

# ctest runs here (unlike gpu-release below): tests/unit/ is CPU-only and
# needs no GPU device, so it can run at `docker build` time same as
# CI's cpu-build-test/server-build-test/otel-build-test jobs do outside
# Docker -- a broken build never makes it into runtime-cpu.
#
# --mount=type=cache,target=/root/.ccache: a BuildKit cache mount, not
# Docker's regular layer cache -- this RUN layer's own cache key is
# invalidated by the `COPY . .` layer above on every real push (any file
# change anywhere in the repo invalidates it), so without this the build
# below would recompile the entire tree from scratch every single build
# regardless of how little actually changed. A cache mount persists
# across builds independent of layer invalidation (backed by
# docker/build-push-action@v7's own cache-to/cache-from GHA scope, same
# mechanism the rest of this Dockerfile already relies on for layer
# caching). id=ccache-cpu-release keeps this stage's cache namespace
# separate from gpu-release's own below, since both may build in the same
# CI run.
RUN --mount=type=cache,target=/root/.ccache,id=ccache-cpu-release \
    cmake --preset cpu-release \
      -DKERNELLAKE_BUILD_SERVER=ON \
      -DKERNELLAKE_ENABLE_OTEL=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    && cmake --build --preset cpu-release \
    && ctest --preset cpu-release --output-on-failure

ENTRYPOINT ["/workspace/build/cpu-release/src/cli/kernellake"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# gpu-build-base: everything a GPU build needs *before* the repo is copied
# in -- the full toolchain (mirroring build-base's own package list below,
# duplicated rather than inherited -- see comment on that), plus CUDA and
# GPUDirect Storage headers from NVIDIA's own image instead of apt. Not a
# runnable image on its own. Split into its own stage (not inlined into
# gpu-release) so .github/workflows/ci.yml's gpu-coverage job can build/
# cache this exact stage and `docker run` its own configure/build/test/
# coverage steps against a live checkout, without also paying for
# gpu-release's own baked-in build every time.
#
# Built FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu26.04, not FROM
# build-base -- see the top-of-file "Base:" comment for the full history
# of this choice. Real, accepted tradeoff: this duplicates build-base's
# package-install RUN block below (same list, same order, same comment
# numbering) rather than sharing it -- Docker has no clean way to
# parameterize a single stage's base image and share a RUN layer across
# two different FROMs in the same build. If build-base's own package list
# above ever changes, this block needs the identical change made here too
# -- nothing enforces that automatically.
# ---------------------------------------------------------------------------
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu26.04 AS gpu-build-base

ENV DEBIAN_FRONTEND=noninteractive

# Identical to build-base's own RUN block above (see that stage's numbered
# comment for the full per-package rationale -- not repeated here), minus
# `nvidia-cuda-toolkit`/`libcufile-dev`: this base image already ships CUDA
# itself plus GDS/cuFile dev headers pre-installed (confirmed for real --
# `dpkg -l` on a bare `nvidia/cuda:13.3.1-devel-ubuntu26.04` container
# already lists `libcufile-13-3`/`libcufile-dev-13-3` and
# `/usr/local/cuda-13.3/targets/x86_64-linux/include/cufile.h` exists),
# so there is nothing left to apt-install for either. Deliberately NOT
# `apt-get install libcufile-dev` (the bare, unsuffixed package name) as a
# defensive extra step: that resolves to Ubuntu's own multiverse package
# (`1.9.1.3~12.4.1-8`, CUDA-12.4-targeted, confirmed via `apt-cache
# policy`), not NVIDIA's CUDA-13-matched `libcufile-dev-13-3` already
# present -- installing it would silently downgrade/mismatch the GDS
# headers this image already has right.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
      wget ca-certificates lsb-release gnupg \
    && wget -O /tmp/arrow-apt-source.deb \
      "https://packages.apache.org/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb" \
    && apt-get install -y -V /tmp/arrow-apt-source.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential cmake ninja-build git pkg-config ccache \
         python3 python3-pip \
         libgtest-dev libbenchmark-dev libspdlog-dev nlohmann-json3-dev libyaml-cpp-dev \
         libxml2-dev uuid-dev libavro-dev libcurl4-openssl-dev \
         libgrpc++-dev protobuf-compiler-grpc opentelemetry-cpp-dev \
         libarrow-dev=25.0.1-1 libparquet-dev=25.0.1-1 libarrow-dataset-dev=25.0.1-1 \
         libarrow-flight-dev=25.0.1-1 libarrow-flight-sql-dev=25.0.1-1 \
         clang-format-18 clang-tidy-18 \
    && rm -rf /var/lib/apt/lists/* /tmp/arrow-apt-source.deb \
    && pip install --break-system-packages --no-cache-dir gcovr

# ---------------------------------------------------------------------------
# gpu-release: full GPU build environment, repo built inside it
# (`gpu-release` preset -- optimized, RelWithDebInfo -- + BUILD_SERVER/
# ENABLE_OTEL). Also a complete, runnable image on its own (`--target
# gpu-release`). Built FROM gpu-build-base (not the NVIDIA CUDA image
# directly) so that stage's whole toolchain+CUDA+gcovr layer is shared
# with, not duplicated for, .github/workflows/ci.yml's gpu-coverage job --
# see gpu-build-base's own comment.
# ---------------------------------------------------------------------------
FROM gpu-build-base AS gpu-release

WORKDIR /workspace
COPY . .

# RAPIDS libcudf/RMM/kvikio/nvcomp/rapids-logger are vendored automatically
# by CMake FetchContent on first configure (cmake/ThirdPartyRapids.cmake,
# pinned PyPI wheels, no conda) -- no separate install step needed here.
# That file auto-detects the real installed CUDA major version
# (`CUDAToolkit_VERSION_MAJOR`) and picks matching `-cu12`/`-cu13` wheels
# accordingly -- nothing to configure here either.
#
# CMAKE_CUDA_COMPILER needs no override here (unlike when this stage
# installed CUDA via apt's nvidia-cuda-toolkit, which put nvcc at the
# Debian-convention /usr/bin/nvcc): gpu-build-base's nvidia/cuda image puts
# it at /usr/local/cuda/bin/nvcc, the NVIDIA installer/Docker-image
# convention CMakePresets.json's gpu-dev/gpu-release presets already
# default to for this project's own non-container dev environment -- both
# environments' toolchains now agree without either one needing an
# explicit override.
#
# CMAKE_CUDA_ARCHITECTURES: no GPU is visible during `docker build` (unlike
# `docker run --gpus all`), so the top-level CMakeLists.txt's default of
# "native" (query the actual device) cannot be used here -- it needs a real
# device to probe. ARG CUDA_ARCHITECTURES (default below, overridable via
# `docker build --build-arg CUDA_ARCHITECTURES=...`) instead: Turing through
# Blackwell (75/80/86/89/90/100) get real compiled code; Blackwell's entry
# is also left without the "-real" suffix so its PTX is embedded too,
# letting the driver JIT-compile for newer architectures it doesn't have
# native code for.
#
# The old default here (Volta through Hopper, 70/75/80/86/89) stopped
# compiling outright once this stage switched to CUDA 13.3 -- confirmed for
# real (`nvcc fatal: Unsupported gpu architecture 'compute_70'`), not a
# guess: CUDA 13's nvcc dropped Volta support entirely. Real, current
# supported range re-checked directly against this exact image
# (`nvcc --list-gpu-arch` inside `nvidia/cuda:13.3.1-devel-ubuntu26.04`):
# compute_75/80/86/87/88/89/90/100/103/110/120/121 -- Turing is now the
# floor, and Blackwell (100/103/110/120/121) is natively supported for the
# first time (previously only reachable via Hopper's PTX forward-JIT, which
# is exactly what let the RTX 5060 Ti used to verify this image, a
# Blackwell/sm_120 card, run all 214 tests successfully under CUDA 12.4's
# nvcc despite having no native sm_120 support at all).
#
# No `ctest` here (unlike cpu-release above): tests/gpu/ needs a real GPU
# device to run, which `docker build` never has (only `docker run --gpus
# all` does) -- per this project's own rule, a skipped-GPU test must never
# be reported as passing, so it isn't run here at all rather than faked.
# Verify gpu-release's tests with a real `docker run --gpus all` against
# this image built with -DKERNELLAKE_BUILD_TESTS=ON instead (see
# README.md/docs/ARCHITECTURE.md).
#
# -DKERNELLAKE_BUILD_TESTS=OFF: the gpu-release preset (CMakePresets.json)
# defaults this ON, which without ctest ever running here (see above) just
# means compiling the entire tests/gpu/ + tests/unit/ suites for nothing --
# real, wasted time sitting directly in docker-publish-gpu's critical path,
# confirmed for real on a live SF1000 benchmarking run whose retry kept
# stalling waiting on this job. cpu-release above leaves it ON since ctest
# genuinely runs there as a build-time correctness gate.
#
# --mount=type=cache: same BuildKit cache-mount reasoning as cpu-release's
# own identical comment above -- id=ccache-gpu-release keeps this stage's
# ccache namespace separate from cpu-release's. ccache does cache nvcc
# invocations too (not just gcc/g++), so this covers the CUDA translation
# units, not just the plain C++ ones.
ARG CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120"
RUN --mount=type=cache,target=/root/.ccache,id=ccache-gpu-release \
    cmake --preset gpu-release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -DKERNELLAKE_BUILD_SERVER=ON \
      -DKERNELLAKE_ENABLE_OTEL=ON \
      -DKERNELLAKE_BUILD_TESTS=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    && cmake --build --preset gpu-release

ENTRYPOINT ["/workspace/build/gpu-release/src/cli/kernellake"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# benchmark-gpu: gpu-release plus a JVM, PySpark, and DuckDB, for
# tools/benchmark_three_way.py's unofficial KernelLake-CPU vs.
# KernelLake-GPU vs. PySpark vs. DuckDB comparison (see docs/ROADMAP.md).
# Not a published image (never built with --target by CI) -- an ad-hoc
# local tool for comparing all four engines against the same generated
# dataset on the same hardware inside one container (`docker run --gpus
# all`), not a deployment artifact. Built from gpu-release specifically
# (not cpu-release) because it needs a real KERNELLAKE_WITH_CUDA=ON build
# to exercise `kernellake query --backend gpu` -- the CPU-backend
# comparison point comes from the same binary's `--backend cpu` flag, not
# a separate build.
#
# NOTE: this used to build FROM the Debug `dev-gpu` stage (now renamed/
# replaced by the optimized `gpu-release`) -- every KernelLake-CPU/-GPU
# timing number recorded in docs/ROADMAP.md's three-way-benchmark entries
# so far was therefore measured against a Debug build, not release-
# optimized code. That's a real, meaningful change for anyone re-running
# these benchmarks after this switch, not just a rename: expect different
# (likely faster) KernelLake timings than the historical numbers on file,
# and don't directly compare a post-switch run against a pre-switch one.
# ---------------------------------------------------------------------------
FROM gpu-release AS benchmark-gpu

# openjdk-21-jdk-headless: PySpark needs a real JVM on PATH, not vendored by
# the pyspark pip package itself. -headless (no X11/desktop deps) since this
# only ever runs inside a container.
RUN apt-get update && apt-get install -y --no-install-recommends \
      openjdk-21-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# --break-system-packages: this stage is a standalone benchmarking tool
# image, not a Python-development environment other tooling shares with
# apt-managed Python packages -- there is no conflicting apt-installed
# package for pyspark/pyarrow/pandas/matplotlib/duckdb to clash with here.
# pandas: PySpark's own DataFrame.toPandas() (used to bridge a Spark SQL
# result into a pa.Table for cross-engine comparison) requires it directly
# -- confirmed by a real "[PACKAGE_NOT_INSTALLED] Pandas >= 2.2.0 must be
# installed" failure before this was added. matplotlib is for
# tools/generate_benchmark_report.py (renders the multi-scale-factor PDF
# report from benchmark_three_way.py's --output JSON files), not the
# benchmark script itself. duckdb: benchmark_three_way.py's third engine
# (tools/duckdb_compare.py's run_duckdb()), same package validate_tpch.py
# already depends on. adbc-driver-flightsql (pulls in adbc-driver-manager
# as a dependency automatically): benchmark_three_way.py's KernelLake
# engine, an ADBC Flight SQL client hitting a long-lived `kernellake-server`
# process -- an earlier version of the script relaunched the `kernellake`
# CLI as a fresh subprocess per query instead, which paid CUDA context/RMM
# pool init on every single query and was dropped for that reason (see
# run_kernellake_server_query() in benchmark_three_way.py).
RUN pip install --break-system-packages --no-cache-dir \
      pyspark pyarrow pandas matplotlib duckdb adbc-driver-flightsql

ENTRYPOINT ["python3", "tools/benchmark_three_way.py"]

# ---------------------------------------------------------------------------
# runtime-libs-{cpu,gpu}: not published images (never built with --target
# directly by CI) -- just resolves both binaries' actual runtime
# shared-library closure (their union) so the runtime-{cpu,gpu} stages
# below can copy exactly those files, without hard-coding vendored library
# names/paths that would go stale on a RAPIDS/Arrow/gRPC/otel version bump.
# Only a short, fixed list of glibc/libgcc basenames is excluded -- those
# are guaranteed present, at an identical version, on any ubuntu:26.04
# image (every stage here builds FROM one), so copying them would be
# redundant, not a correctness concern either way.
# ---------------------------------------------------------------------------
FROM cpu-release AS runtime-libs-cpu

RUN mkdir -p /opt/kernellake-libs && \
    { ldd build/cpu-release/src/cli/kernellake; \
      ldd build/cpu-release/src/server/kernellake-server; } | \
    awk '{print $3}' | grep '^/' | sort -u | \
    while read -r lib; do \
      case "$(basename "$lib")" in \
        ld-linux-x86-64.so.2|libc.so.6|libm.so.6|libpthread.so.0|libdl.so.2|librt.so.1|libgcc_s.so.1|libstdc++.so.6) ;; \
        *) cp -Ln "$lib" /opt/kernellake-libs/ 2>/dev/null || true ;; \
      esac; \
    done

FROM gpu-release AS runtime-libs-gpu

RUN mkdir -p /opt/kernellake-libs && \
    { ldd build/gpu-release/src/cli/kernellake; \
      ldd build/gpu-release/src/server/kernellake-server; } | \
    awk '{print $3}' | grep '^/' | sort -u | \
    while read -r lib; do \
      case "$(basename "$lib")" in \
        ld-linux-x86-64.so.2|libc.so.6|libm.so.6|libpthread.so.0|libdl.so.2|librt.so.1|libgcc_s.so.1|libstdc++.so.6) ;; \
        *) cp -Ln "$lib" /opt/kernellake-libs/ 2>/dev/null || true ;; \
      esac; \
    done

# ---------------------------------------------------------------------------
# runtime-gpu: only what's needed to run the built binaries -- plain
# ubuntu:26.04 (the same base gpu-release used, so the excluded glibc/
# libgcc/libstdc++ libraries above are guaranteed to match), the two
# compiled binaries (kernellake, kernellake-server), and exactly their
# combined non-base-system shared library closure (CUDA runtime .so's,
# Arrow/Parquet, RAPIDS, gRPC, opentelemetry-cpp, and every other vendored
# dependency). No compiler, no nvcc, no CUDA headers, no static libraries,
# no source tree, no `-dev` apt packages -- this stage never installs any
# (only `ca-certificates`/`libstdc++6`, both plain runtime packages), and
# starts fresh `FROM ubuntu:26.04` rather than `FROM gpu-release`, so
# nothing from the build environment leaks in by inheritance either.
#
# ENTRYPOINT stays the kernellake CLI, unchanged -- the README's own `docker
# run` example still works as documented. charts/kernellake/'s Deployment
# overrides `command` to run kernellake-server instead; one image serving
# two purposes via command override, not two separate images.
# ---------------------------------------------------------------------------
FROM ubuntu:26.04 AS runtime-gpu

# apt-get upgrade -y: this stage starts fresh FROM ubuntu:26.04 (see above),
# so it doesn't inherit build-base's own upgrade -- this is the actual
# published image, so patching whatever's in the base layer matters most
# here, not just in the build stages.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
      ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Runs as a fixed-UID non-root user, not the root this stage's own base
# layer defaults to -- neither binary needs root (no privileged ports, no
# host-level access), and a container escape from a root process is a
# strictly worse outcome than from this one. UID/GID 10001 chosen so it's
# stable across rebuilds and so charts/kernellake/'s Deployment can pin a
# matching `securityContext.runAsUser` if it ever needs to; --no-create-home
# because this user never needs a home directory or shell.
RUN groupadd --gid 10001 kernellake \
    && useradd --uid 10001 --gid kernellake --no-create-home --shell /usr/sbin/nologin kernellake

RUN mkdir -p /opt/kernellake/bin /opt/kernellake/lib
COPY --from=gpu-release --chown=kernellake:kernellake /workspace/build/gpu-release/src/cli/kernellake /opt/kernellake/bin/kernellake
COPY --from=gpu-release --chown=kernellake:kernellake /workspace/build/gpu-release/src/server/kernellake-server /opt/kernellake/bin/kernellake-server
COPY --from=runtime-libs-gpu --chown=kernellake:kernellake /opt/kernellake-libs/ /opt/kernellake/lib/

USER kernellake
ENV LD_LIBRARY_PATH=/opt/kernellake/lib
ENTRYPOINT ["/opt/kernellake/bin/kernellake"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# runtime-cpu: same as runtime-gpu above but from cpu-release/
# runtime-libs-cpu -- no CUDA/RAPIDS shared libraries in the closure at
# all, and the same "no compiler/headers/static libs/source tree, starts
# fresh FROM ubuntu:26.04" minimality applies here too. This is the
# default (last) stage, matching the rest of this project treating the CPU
# backend as the default/primary one (KERNELLAKE_WITH_CUDA is OFF by
# default -- see root CMakeLists.txt).
# ---------------------------------------------------------------------------
FROM ubuntu:26.04 AS runtime-cpu

# apt-get upgrade -y: same reasoning as runtime-gpu's own identical step --
# this stage starts fresh FROM ubuntu:26.04 too, so it needs its own.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
      ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# See runtime-gpu's identical block above for why this exists and why UID/
# GID 10001 specifically.
RUN groupadd --gid 10001 kernellake \
    && useradd --uid 10001 --gid kernellake --no-create-home --shell /usr/sbin/nologin kernellake

RUN mkdir -p /opt/kernellake/bin /opt/kernellake/lib
COPY --from=cpu-release --chown=kernellake:kernellake /workspace/build/cpu-release/src/cli/kernellake /opt/kernellake/bin/kernellake
COPY --from=cpu-release --chown=kernellake:kernellake /workspace/build/cpu-release/src/server/kernellake-server /opt/kernellake/bin/kernellake-server
COPY --from=runtime-libs-cpu --chown=kernellake:kernellake /opt/kernellake-libs/ /opt/kernellake/lib/

USER kernellake
ENV LD_LIBRARY_PATH=/opt/kernellake/lib
ENTRYPOINT ["/opt/kernellake/bin/kernellake"]
CMD ["--help"]
