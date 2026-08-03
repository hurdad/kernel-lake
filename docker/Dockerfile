# Single multi-stage Dockerfile publishing two images from one source of
# truth: `dev` (full CUDA toolkit, everything needed to iterate on
# KernelLake) and `runtime` (only the built binary plus its actual runtime
# shared-library closure -- no compiler, no CUDA headers, no nvcc, no static
# libs). Build whichever you need with --target:
#
#   docker build --target dev     -f docker/Dockerfile -t kernellake/kernellake:dev .
#   docker build --target runtime -f docker/Dockerfile -t kernellake/kernellake:runtime .
#   docker build                  -f docker/Dockerfile -t kernellake/kernellake:runtime .   # runtime is the default (last) stage
#
# Run either the same way (the spec's own usage example):
#   docker run --rm --gpus all -v /tmp/kernellake-sales:/data:ro \
#     kernellake/kernellake:runtime query --sql "SELECT region, SUM(amount) FROM read_parquet('/data/*.parquet') GROUP BY region"
#
# Base: plain `ubuntu:26.04`, with CUDA installed via Ubuntu's own apt
# `nvidia-cuda-toolkit` package (12.4.1) rather than NVIDIA's official
# `nvidia/cuda` devel/runtime images -- verified by a real `docker build`
# and `docker run --gpus all` (a full 214/214 test pass on real hardware)
# in this project's own development session. See docs/ARCHITECTURE.md's
# "Ubuntu 26.04 baseline" section for why: NVIDIA's apt Flight SQL package
# doesn't link against Ubuntu 24.04's older system Abseil, otel-cpp has no
# 24.04 apt package at all, and both work cleanly on 26.04. RAPIDS
# libcudf/RMM stay on the same `-cu12` pinned wheels (cmake/ThirdPartyRapids.cmake)
# -- CUDA 12.4 is a minor step down from the previously-pinned 12.6.3, not a
# major-version change, and needs no re-vendoring.
#
# Phase 3 (docs/ROADMAP.md): also builds and ships kernellake-server
# (KERNELLAKE_BUILD_SERVER=ON) with OpenTelemetry export enabled
# (KERNELLAKE_ENABLE_OTEL=ON), alongside the kernellake CLI -- both
# binaries, one image. See charts/kernellake/ for the Helm chart deploying
# kernellake-server as a Deployment+Service on top of this image.

# ---------------------------------------------------------------------------
# dev: full build environment, with the repository built inside it. Also a
# complete, runnable image on its own (`--target dev`).
# ---------------------------------------------------------------------------
FROM ubuntu:26.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive

# cmake comes straight from apt here: Ubuntu 26.04 ships 4.2.3, already
# above the >= 3.30.4 libcudf's cudf-config.cmake requires -- unlike Ubuntu
# 24.04 (3.28.3), no Kitware tarball download needed any more.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git pkg-config wget ca-certificates lsb-release gnupg \
      python3 python3-pip \
      libgtest-dev libbenchmark-dev libspdlog-dev nlohmann-json3-dev libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Apache Arrow / Parquet C++ (official Apache Arrow apt repo; resolves to
# the `resolute` suite automatically via lsb_release)
RUN wget -O /tmp/arrow-apt-source.deb \
      "https://packages.apache.org/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb" \
    && apt-get install -y -V /tmp/arrow-apt-source.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends libarrow-dev libparquet-dev libarrow-dataset-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/arrow-apt-source.deb

# Phase 4 (docs/ROADMAP.md): cloud object storage (S3/GCS/Azure, see
# docs/ARCHITECTURE.md's "Cloud object storage" section). config.hpp
# unconditionally includes <arrow/filesystem/{s3,gcs,azure,hdfs}fs.h>, so
# every build (not just KERNELLAKE_BUILD_SERVER/KERNELLAKE_ENABLE_OTEL)
# now needs libarrow_bundled_dependencies.a's bundled google-cloud-cpp
# (GCS) and Azure SDK C++ (Azure) code to actually link -- which pulls in
# two genuinely new system dependencies this project never needed before:
# libxml2 (Azure SDK's bundled XML request/response parsing) and libuuid
# (Azure SDK's request-ID generation). Confirmed by a real link failure
# without these two packages installed.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libxml2-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# CUDA toolkit + GPUDirect Storage headers, both from Ubuntu's own apt
# repos. libcufile-dev is not pulled in by nvidia-cuda-toolkit itself but is
# required by kvikio (a libcudf dependency) -- confirmed by a real
# configure failure ("Compiled with cuFile support but cuFile not found")
# before this line was added.
RUN apt-get update && apt-get install -y --no-install-recommends \
      nvidia-cuda-toolkit libcufile-dev \
    && rm -rf /var/lib/apt/lists/*

# kernellake-server (KERNELLAKE_BUILD_SERVER) + OpenTelemetry export
# (KERNELLAKE_ENABLE_OTEL) dependencies -- the exact same package lists
# already verified in .github/workflows/ci.yml's server-build-test/
# otel-build-test jobs, which run inside this same ubuntu:26.04 base.
# libarrow-flight-dev/libarrow-flight-sql-dev come from the Arrow apt repo
# configured above; libgrpc++-dev/protobuf-compiler-grpc/
# opentelemetry-cpp-dev are plain Ubuntu universe packages, same as
# nvidia-cuda-toolkit itself -- no additional apt source needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libarrow-flight-dev libarrow-flight-sql-dev \
      libgrpc++-dev protobuf-compiler-grpc \
      opentelemetry-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

# RAPIDS libcudf/RMM/kvikio/nvcomp/rapids-logger are vendored automatically
# by CMake FetchContent on first configure (cmake/ThirdPartyRapids.cmake,
# pinned PyPI wheels, no conda) -- no separate install step needed here.
#
# CMAKE_CUDA_COMPILER: apt's nvidia-cuda-toolkit installs nvcc at
# /usr/bin/nvcc (Debian packaging convention), not
# /usr/local/cuda/bin/nvcc (NVIDIA installer/Docker-image convention,
# still used by CMakePresets.json's gpu-dev preset for this project's own
# non-container dev environment) -- overridden here, not in the preset
# itself, so the two environments' actual toolchains each get their own
# correct path without either one needing to guess the other's layout.
#
# CMAKE_CUDA_ARCHITECTURES: no GPU is visible during `docker build` (unlike
# `docker run --gpus all`), so the top-level CMakeLists.txt's default of
# "native" (query the actual device) cannot be used here -- it needs a real
# device to probe. Volta through Hopper (70/75/80/86/89) get real compiled
# code; Hopper's entry is also left without the "-real" suffix so its PTX
# is embedded too, letting the driver JIT-compile for newer architectures
# it doesn't have native code for (this is exactly what let the RTX 5060 Ti
# used to verify this image, a Blackwell/sm_120 card newer than anything
# CUDA 12.4's nvcc can target directly, run all 214 tests successfully).
# KERNELLAKE_BUILD_SERVER/KERNELLAKE_ENABLE_OTEL are independent CMake
# options, not tied to the gpu-dev preset's own inherited variables -- both
# compose cleanly as extra cache-variable overrides here, no new preset
# needed. Produces build/gpu-dev/src/server/kernellake-server alongside the
# existing build/gpu-dev/src/cli/kernellake.
RUN cmake --preset gpu-dev \
      -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="70-real;75-real;80-real;86-real;89" \
      -DKERNELLAKE_BUILD_SERVER=ON \
      -DKERNELLAKE_ENABLE_OTEL=ON \
    && cmake --build --preset gpu-dev

ENTRYPOINT ["/workspace/build/gpu-dev/src/cli/kernellake"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# runtime-libs: not a published image (never built with --target) -- just
# resolves both binaries' actual runtime shared-library closure (their
# union) so the runtime stage below can copy exactly those files, without
# hard-coding vendored library names/paths that would go stale on a
# RAPIDS/Arrow/CUDA/gRPC/otel version bump. Only a short, fixed list of
# glibc/libgcc basenames is excluded -- those are guaranteed present, at an
# identical version, on any ubuntu:26.04 image (dev and runtime both build
# FROM one), so copying them would be redundant, not a correctness concern
# either way.
# ---------------------------------------------------------------------------
FROM dev AS runtime-libs

RUN mkdir -p /opt/kernellake-libs && \
    { ldd build/gpu-dev/src/cli/kernellake; \
      ldd build/gpu-dev/src/server/kernellake-server; } | \
    awk '{print $3}' | grep '^/' | sort -u | \
    while read -r lib; do \
      case "$(basename "$lib")" in \
        ld-linux-x86-64.so.2|libc.so.6|libm.so.6|libpthread.so.0|libdl.so.2|librt.so.1|libgcc_s.so.1|libstdc++.so.6) ;; \
        *) cp -Ln "$lib" /opt/kernellake-libs/ 2>/dev/null || true ;; \
      esac; \
    done

# ---------------------------------------------------------------------------
# runtime: only what's needed to run the built binaries -- plain
# ubuntu:26.04 (the same base `dev` used, so the excluded glibc/libgcc/
# libstdc++ libraries above are guaranteed to match), the two compiled
# binaries (kernellake, kernellake-server), and exactly their combined
# non-base-system shared library closure (CUDA runtime .so's, Arrow/
# Parquet, RAPIDS, gRPC, opentelemetry-cpp, and every other vendored
# dependency). No compiler, no nvcc, no CUDA headers, no static libraries,
# no source tree. This is the default (last) stage.
#
# ENTRYPOINT stays the kernellake CLI, unchanged -- the README's own `docker
# run` example still works as documented. charts/kernellake/'s Deployment
# overrides `command` to run kernellake-server instead; one image serving
# two purposes via command override, not two separate images.
# ---------------------------------------------------------------------------
FROM ubuntu:26.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/kernellake/bin /opt/kernellake/lib
COPY --from=dev /workspace/build/gpu-dev/src/cli/kernellake /opt/kernellake/bin/kernellake
COPY --from=dev /workspace/build/gpu-dev/src/server/kernellake-server /opt/kernellake/bin/kernellake-server
COPY --from=runtime-libs /opt/kernellake-libs/ /opt/kernellake/lib/

ENV LD_LIBRARY_PATH=/opt/kernellake/lib
ENTRYPOINT ["/opt/kernellake/bin/kernellake"]
CMD ["--help"]
