# Third-party licenses

KernelLake itself is licensed under the Apache License 2.0 (see `LICENSE`).
This file lists every third-party dependency actually declared in the
build (verified against `CMakeLists.txt`, `cmake/*.cmake`, and each
package's own license metadata as installed in the development
environment this was compiled in -- not assumed from general knowledge),
plus the Python tools used only for development/validation and never
linked into the `kernellake` binary.

**Two entries below are not open-source licenses** -- the CUDA Toolkit and
RAPIDS's vendored `nvcomp` component are distributed under NVIDIA's own
proprietary SDK/EULA terms. Read their "Redistribution" notes before
redistributing a GPU-enabled build; KernelLake's own Apache-2.0 license
does not extend to them.

## Linked into the `kernellake` binary (both presets)

| Dependency | License | Source | How it's consumed |
| --- | --- | --- | --- |
| Apache Arrow C++ | Apache-2.0 | `dev@arrow.apache.org` / Apache Arrow project | System package (`libarrow-dev`, official Apache Arrow apt repo) |
| Apache Parquet C++ | Apache-2.0 | Same Apache Arrow release as above | System package (`libparquet-dev`) |
| spdlog | MIT ("Expat") | github.com/gabime/spdlog | System package (`libspdlog-dev`) |
| yaml-cpp | MIT-style ("X11") | github.com/jbeder/yaml-cpp | System package (`libyaml-cpp-dev`) |
| nlohmann/json | MIT ("Expat") | github.com/nlohmann/json | System package (`nlohmann-json3-dev`) |
| hyrise/sql-parser | MIT | github.com/hyrise/sql-parser, pinned commit | Vendored via CMake `FetchContent` (`cmake/ThirdPartySqlParser.cmake`); KernelLake wraps it with a `read_parquet(...)` syntax adapter and its own AST -- see `docs/ARCHITECTURE.md` |
| libxml2 | MIT-style ("MIT-1") | xmlsoft.org | System package (`libxml2-dev`), needed by `libarrow-dev`'s bundled Azure SDK C++ code (XML request/response parsing) -- see below and `docs/ARCHITECTURE.md`'s "Cloud object storage" section |
| libuuid (util-linux) | BSD-3-Clause | kernel.org/pub/linux/utils/util-linux | System package (`uuid-dev`), needed by `libarrow-dev`'s bundled Azure SDK C++ code (request-ID generation) |
| google-cloud-cpp | Apache-2.0 | github.com/googleapis/google-cloud-cpp | Bundled *inside* `libarrow-dev`'s `libarrow_bundled_dependencies.a` (Arrow's own vendored copy, used by `arrow::fs::GcsFileSystem`) -- license taken from the upstream project directly; unlike every other entry in this file, not independently verified against a local installed `/usr/share/doc/<pkg>/copyright` file, since it isn't its own apt package |
| Azure SDK for C++ | MIT ("Expat") | github.com/Azure/azure-sdk-for-cpp | Bundled *inside* `libarrow-dev`'s `libarrow_bundled_dependencies.a` (used by `arrow::fs::AzureFileSystem`) -- same caveat as google-cloud-cpp above |

## Linked into the `kernellake` binary (`gpu-dev` / `KERNELLAKE_WITH_CUDA=ON` only)

| Dependency | License | Source | How it's consumed |
| --- | --- | --- | --- |
| RAPIDS libcudf | Apache-2.0 | `libcudf-cu12` PyPI wheel, pinned version+SHA-256 | Vendored via CMake `FetchContent` (`cmake/ThirdPartyRapids.cmake`), no conda |
| RAPIDS RMM | Apache-2.0 | `librmm-cu12` PyPI wheel | Same as above |
| RAPIDS kvikio | Apache-2.0 | `libkvikio-cu12` PyPI wheel | Same as above |
| rapids-logger | Apache-2.0 | `rapids_logger` PyPI wheel | Same as above |
| NVIDIA CCCL (Thrust/CUB/libcudacxx) | Apache-2.0 with LLVM exception upstream | Bundled with the CUDA Toolkit's `cuda-cccl` apt package in this build | Transitive dependency of RAPIDS libcudf; **as consumed here** (the apt-packaged CUDA Toolkit, not the standalone `NVIDIA/cccl` GitHub release), it is distributed under the CUDA Toolkit EULA umbrella (see below), not a standalone open-source license grant |
| **NVIDIA nvCOMP** | **Proprietary -- "NVIDIA Software Development Kit" EULA, not open source** | `nvidia-libnvcomp-cu12` PyPI wheel (supplies `libnvcomp.so.5`, an undeclared transitive dependency of libcudf's own build) | Vendored via `FetchContent`; symlinked into libcudf's own runtime directory (see `cmake/ThirdPartyRapids.cmake`) so `libcudf.so`'s `$ORIGIN` runpath resolves it |
| **NVIDIA CUDA Toolkit** (`CUDAToolkit`: `cudart`, etc.) | **Proprietary -- NVIDIA CUDA Toolkit End User License Agreement, not open source** | Installed separately by the developer (`sudo apt-get install` the NVIDIA CUDA repo's toolkit package); not vendored by KernelLake's build | `find_package(CUDAToolkit REQUIRED)`, `CUDA::cudart` |

### nvCOMP redistribution note

nvCOMP's EULA (`nvidia_libnvcomp_cu12`'s `LICENSE` file) permits
distributing its binary as part of an application under specific
conditions (SDK section 1.2, "Distribution Requirements"): the
distributable component may only be accessed by your application, your
distribution terms must remain consistent with NVIDIA's license grant and
IP-protection terms, and it may not be used in a way that would subject it
to an open-source license (SDK section 2.6). This is normal for
NVIDIA SDK components and is the same category of dependency as the CUDA
Toolkit itself -- it does not affect KernelLake's own Apache-2.0
licensing, but it does mean a `gpu-dev` build is not 100%
open-source-licensed end to end, and anyone redistributing such a build
should read nvCOMP's and the CUDA Toolkit's EULAs directly rather than
relying on this summary.

## Linked into `kernellake-server` (`KERNELLAKE_BUILD_SERVER=ON`) / OpenTelemetry export (`KERNELLAKE_ENABLE_OTEL=ON`) only

Both default `OFF` in the local `dev`/`gpu-dev` presets, but the published
`docker/Dockerfile` `runtime` image turns both on (see
`docs/ARCHITECTURE.md`'s "Docker image: kernellake-server + OpenTelemetry"
section) -- these are genuinely shipped in the default distributed
artifact, verified against each package's own installed
`/usr/share/doc/<pkg>/copyright`, not assumed from general knowledge.

| Dependency | License | Source | How it's consumed |
| --- | --- | --- | --- |
| Apache Arrow Flight / Flight SQL C++ | Apache-2.0 | Same Apache Arrow release as the base Arrow entry above | System package (`libarrow-flight-dev`, `libarrow-flight-sql-dev`, official Apache Arrow apt repo) |
| gRPC | Apache-2.0 | github.com/grpc/grpc | System package (`libgrpc++-dev`, `protobuf-compiler-grpc`) |
| Abseil | Apache-2.0 | github.com/abseil/abseil-cpp | System package, transitive dependency of gRPC and (on Ubuntu 26.04) Arrow itself -- see `docs/ARCHITECTURE.md`'s "Ubuntu 26.04 baseline" section |
| Protocol Buffers | BSD-3-Clause | github.com/protocolbuffers/protobuf | System package (`libprotobuf-dev`), transitive dependency of gRPC and Arrow Flight |
| opentelemetry-cpp | Apache-2.0 | github.com/open-telemetry/opentelemetry-cpp | System package (`opentelemetry-cpp-dev`, apt-native on Ubuntu 26.04 only -- see `docs/ARCHITECTURE.md`) |

## Declared but not yet actually used

| Dependency | License | Notes |
| --- | --- | --- |
| GoogleTest | BSD-3-Clause | Test-only (`libgtest-dev`), never linked into the `kernellake` binary itself |
| Google Benchmark | Apache-2.0 | `find_package(benchmark REQUIRED)` when microbenchmarks are enabled, but `benchmarks/CMakeLists.txt` has no targets yet -- see `docs/ROADMAP.md` |

## Development/validation tooling (never linked into the `kernellake` binary)

| Tool | License | Notes |
| --- | --- | --- |
| DuckDB (Python package) | MIT | Used only by `tools/validate_against_duckdb.py` and `tools/validate_tpch.py` as an out-of-process correctness oracle, invoked via `pip install duckdb`; never linked into or distributed with the `kernellake` binary |
| PyArrow | Apache-2.0 | Used by the same validation tools and by `tools/generate_tpch.py`, same out-of-process relationship |

## Compatibility

All open-source licenses above (Apache-2.0, MIT/Expat, MIT-style/X11,
BSD-3-Clause) are mutually compatible with KernelLake's own Apache
License 2.0 and with each other; none impose copyleft or
source-disclosure obligations on KernelLake's own code. No source code
has been copied from any of these dependencies into KernelLake's own
source tree -- they are consumed exclusively as system packages, CMake
`FetchContent` vendored builds, or (for DuckDB/PyArrow) separate
out-of-process tools.

## TPC-H

KernelLake's TPC-H-derived benchmark suite (`benchmarks/tpch/`,
`tools/generate_tpch.py`, `tools/validate_tpch.py`, `kernellake benchmark
tpch`) is an unofficial, TPC-H-*derived* suite: it is not licensed,
audited, or certified by the Transaction Processing Performance Council.
See `docs/TPCH.md`.
