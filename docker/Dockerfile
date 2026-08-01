# Single multi-stage Dockerfile publishing two images from one source of
# truth: `dev` (full CUDA devel toolkit, everything needed to iterate on
# KernelLake) and `runtime` (only the built binary plus its actual runtime
# dependency closure). Build whichever you need with --target:
#
#   docker build --target dev     -f docker/Dockerfile -t kernellake/kernellake:dev .
#   docker build --target runtime -f docker/Dockerfile -t kernellake/kernellake:runtime .
#   docker build                  -f docker/Dockerfile -t kernellake/kernellake:runtime .   # runtime is the default (last) stage
#
# Run either the same way (the spec's own usage example):
#   docker run --rm --gpus all -v /tmp/kernellake-sales:/data:ro \
#     kernellake/kernellake:runtime query --sql "SELECT region, SUM(amount) FROM read_parquet('/data/*.parquet') GROUP BY region"
#
# NOT verified by an actual `docker build`/`docker run --gpus all` in this
# repository's own development session -- Docker was not available there
# (see docs/ROADMAP.md). Review and test before relying on either image.

# ---------------------------------------------------------------------------
# dev: full build environment, with the repository built inside it. Also a
# complete, runnable image on its own (`--target dev`).
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ninja-build git pkg-config wget ca-certificates lsb-release \
      python3 python3-pip \
      libgtest-dev libbenchmark-dev libspdlog-dev nlohmann-json3-dev libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# CMake >= 3.30.4 is required by libcudf's own cudf-config.cmake, newer
# than Ubuntu 24.04's apt package (3.28.3) -- same Kitware binary tarball
# approach documented in README.md's Requirements section.
RUN wget -qO /tmp/cmake.tar.gz \
      https://github.com/Kitware/CMake/releases/download/v3.30.4/cmake-3.30.4-linux-x86_64.tar.gz \
    && tar -xzf /tmp/cmake.tar.gz -C /opt \
    && ln -s /opt/cmake-3.30.4-linux-x86_64/bin/* /usr/local/bin/ \
    && rm /tmp/cmake.tar.gz

# Apache Arrow / Parquet C++ (official Apache Arrow apt repo)
RUN wget -O /tmp/arrow-apt-source.deb \
      "https://packages.apache.org/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb" \
    && apt-get install -y -V /tmp/arrow-apt-source.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends libarrow-dev libparquet-dev libarrow-dataset-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/arrow-apt-source.deb

WORKDIR /workspace
COPY . .

# RAPIDS libcudf/RMM/kvikio/nvcomp/rapids-logger are vendored automatically
# by CMake FetchContent on first configure (cmake/ThirdPartyRapids.cmake,
# pinned PyPI wheels, no conda) -- no separate install step needed here.
# Do not hard-code a host-specific CUDA path: CUDAToolkit is located via
# find_package(CUDAToolkit REQUIRED), which finds this image's own toolkit
# install without any absolute path baked into CMakeLists.txt.
RUN cmake --preset gpu-dev && cmake --build --preset gpu-dev

ENTRYPOINT ["/workspace/build/gpu-dev/src/cli/kernellake"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# runtime-libs: not a published image (never built with --target) -- just
# resolves `dev`'s binary's actual non-system runtime dependency closure so
# the runtime stage below can copy exactly those files, without hard-coding
# vendored library names/paths that would go stale on a RAPIDS version bump.
# ---------------------------------------------------------------------------
FROM dev AS runtime-libs

RUN mkdir -p /opt/kernellake-libs && \
    ldd build/gpu-dev/src/cli/kernellake | awk '{print $3}' | grep '^/' | sort -u | \
    while read -r lib; do \
      case "$lib" in \
        /lib/*|/lib64/*|/usr/lib/*) ;; \
        *) cp -Ln "$lib" /opt/kernellake-libs/ 2>/dev/null || true ;; \
      esac; \
    done

# ---------------------------------------------------------------------------
# runtime: only what's needed to run a built `kernellake` binary -- the CUDA
# *runtime* (not devel/compiler) base image, the compiled binary, and
# exactly the non-system shared libraries it needs. No compiler, no build
# tools, no source tree. This is the default (last) stage.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS runtime

RUN mkdir -p /opt/kernellake/bin /opt/kernellake/lib
COPY --from=dev /workspace/build/gpu-dev/src/cli/kernellake /opt/kernellake/bin/kernellake
COPY --from=runtime-libs /opt/kernellake-libs/ /opt/kernellake/lib/

ENV LD_LIBRARY_PATH=/opt/kernellake/lib
ENTRYPOINT ["/opt/kernellake/bin/kernellake"]
CMD ["--help"]
