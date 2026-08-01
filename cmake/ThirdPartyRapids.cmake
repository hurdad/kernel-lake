# Vendors the RAPIDS C++ SDK pieces KernelLake's GPU layer needs --
# rapids-logger, librmm, libkvikio, and libcudf -- as prebuilt
# CMake-consumable packages, without requiring conda/mamba. Verified
# end-to-end on real hardware: a standalone smoke-test CMake project
# linking cudf::cudf successfully allocated and inspected a GPU-resident
# cudf::column via this exact vendoring approach.
#
# RAPIDS has no Ubuntu apt packaging, and the conda distribution is the
# usual recommended path, but each of these projects also publishes
# self-contained "lib*" wheels on PyPI: a plain zip containing C++ headers,
# a compiled shared library, and a full CMake package config
# (lib64/cmake/<name>/<name>-config.cmake), with no Python runtime
# dependency at all (the wheels are tagged py3-none, i.e. ABI-agnostic).
# We fetch those wheel archives the same way ThirdPartySqlParser.cmake
# vendors hyrise/sql-parser: pinned by URL + SHA-256, extracted by
# FetchContent, consumed via find_package(... CONFIG).
#
# This file is only included when KERNELLAKE_WITH_CUDA is ON.
#
# Verified requirement (from libcudf's own cudf-config.cmake): CMake >=
# 3.30.4. If configuration fails with a "CMake 3.30.4 or higher is
# required" error, your `cmake` is too old -- see docs/architecture.md for
# how to get a newer one without root (a portable Kitware release tarball,
# since Ubuntu 24.04's apt cmake is 3.28 and pip installs are blocked by
# PEP 668 on a system Python without --break-system-packages).
#
# libcudf.so also has two *undeclared* (not expressed via cudf-config.cmake
# or cudf-dependencies.cmake) transitive link dependencies that must be
# resolved separately or the final link fails with "undefined reference":
#   - libnvcomp.so.5 (NVIDIA's compression library, used for Parquet
#     codecs). NOT provided by the `nvidia-nvcomp-cu12` wheel -- that one
#     only ships Python-version-specific extension modules. The actual
#     plain .so.5 comes from `nvidia-libnvcomp-cu12`.
#   - libkvikio.so (RAPIDS's GPU-direct storage I/O library), which is a
#     normal find_package-able RAPIDS component.
include(FetchContent)

set(_kernellake_rapids_version "26.6.0")
set(_kernellake_rapids_logger_version "0.2.3")
set(_kernellake_nvcomp_version "5.3.0.16")

FetchContent_Declare(
  rapids_logger
  URL "https://files.pythonhosted.org/packages/69/b6/139d9df6d0f7bd289a9a6286cecfff999e41c36865515d7fdb56b7b32a14/rapids_logger-0.2.3-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
  URL_HASH SHA256=7fe67ef4049c5d8ba6154746325dcf7cc0f327f0efa8f2611fc8f64e67510f60
  DOWNLOAD_NAME rapids_logger.zip
)

FetchContent_Declare(
  librmm_cu12
  URL "https://files.pythonhosted.org/packages/d4/cd/4886deecdb10781b8c18ca69f4c2a4dab81e2e2b4480d46d032e1bd29a53/librmm_cu12-26.6.0-py3-none-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
  URL_HASH SHA256=be0e1879633992b57c4e8c9c8df82419bc5a557029ce5b2aecad5d68743c7fe4
  DOWNLOAD_NAME librmm_cu12.zip
)

FetchContent_Declare(
  libkvikio_cu12
  URL "https://files.pythonhosted.org/packages/e2/1a/f840df152aaae112b28b3a6db3c0ef9da4ff552ab95337e1e014c0281d8e/libkvikio_cu12-26.6.0-py3-none-manylinux_2_28_x86_64.whl"
  URL_HASH SHA256=934b1f8cc1ef11e967f6d93b835d8bbdd24bba30366a3ba51cf200a9b8a711f7
  DOWNLOAD_NAME libkvikio_cu12.zip
)

FetchContent_Declare(
  nvidia_libnvcomp_cu12
  URL "https://files.pythonhosted.org/packages/b7/20/e025aa3fa2ee33ee6a4871aa5a9e10b05472913adaff649ba21f2c98291d/nvidia_libnvcomp_cu12-5.3.0.16-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
  URL_HASH SHA256=535944be2e37ead56ad1ef80e9dd58a3126fc2ccf238ddffc9d1cb7a48f5300a
  DOWNLOAD_NAME nvidia_libnvcomp_cu12.zip
)

FetchContent_Declare(
  libcudf_cu12
  URL "https://files.pythonhosted.org/packages/d1/81/16870639cfea5e8d922642099d04fefd6c29fc037e79b2a78de9f1f4fe85/libcudf_cu12-26.6.0-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
  URL_HASH SHA256=d6f9f88bd51562de7d731931ef357fb81e18c5d7af05eab8e5a0c85e1cead1dc
  DOWNLOAD_NAME libcudf_cu12.zip
)

# None of these archives have a top-level CMakeLists.txt, so
# FetchContent_MakeAvailable only downloads/extracts them -- it does not
# add_subdirectory() anything (same behavior relied on in
# ThirdPartySqlParser.cmake).
FetchContent_MakeAvailable(rapids_logger librmm_cu12 libkvikio_cu12 nvidia_libnvcomp_cu12 libcudf_cu12)

# The actual package trees are nested one level below each archive's root
# (mirroring how these are laid out under Python's site-packages).
# CCCL's own cccl-config.cmake locates its Thrust/CUB/libcudacxx siblings
# via paths relative to itself, so it's enough to put each package's own
# lib64/cmake and lib64/rapids/cmake directories on the prefix path -- we
# don't need to hunt down every vendored transitive config individually.
list(APPEND CMAKE_PREFIX_PATH
  "${rapids_logger_SOURCE_DIR}/rapids_logger/lib64/cmake"
  "${librmm_cu12_SOURCE_DIR}/librmm/lib64/cmake"
  "${librmm_cu12_SOURCE_DIR}/librmm/lib64/rapids/cmake"
  "${libkvikio_cu12_SOURCE_DIR}/libkvikio/lib64/cmake"
  "${libcudf_cu12_SOURCE_DIR}/libcudf/lib64/cmake"
  "${libcudf_cu12_SOURCE_DIR}/libcudf/lib64/rapids/cmake"
)

find_package(rapids_logger ${_kernellake_rapids_logger_version} CONFIG REQUIRED)
find_package(rmm CONFIG REQUIRED)
find_package(kvikio CONFIG REQUIRED)
find_package(cudf CONFIG REQUIRED)

# libcudf.so has two undeclared transitive shared-library dependencies,
# libnvcomp.so.5 and libkvikio.so, that are awkward to resolve from the
# consuming executable's side: libcudf.so carries its own DT_RUNPATH
# ("$ORIGIN", i.e. only its own directory), and per ELF/ld.so semantics an
# object with its own DT_RUNPATH resolves *its* NEEDED entries using that
# runpath (plus LD_LIBRARY_PATH / ld.so.cache / default paths) -- never the
# loading executable's rpath. Making them direct link dependencies of a
# consumer doesn't reliably help either: the linker's --as-needed
# optimization (default on this toolchain) drops libkvikio.so again since
# nothing in KernelLake calls its symbols directly.
#
# The robust fix, independent of any consumer's link flags: symlink both
# libraries directly into libcudf.so's own directory, so its "$ORIGIN"
# runpath finds them without any help from whoever links cudf::cudf.
# (librmm.so and librapids_logger.so don't need this treatment -- our own
# targets link them directly and unconditionally use their symbols, so
# --as-needed keeps them and they're already loaded by the time libcudf.so
# needs them.)
set(_kernellake_cudf_lib_dir "${libcudf_cu12_SOURCE_DIR}/libcudf/lib64")
file(CREATE_LINK
     "${nvidia_libnvcomp_cu12_SOURCE_DIR}/nvidia/libnvcomp/lib64/libnvcomp.so.5"
     "${_kernellake_cudf_lib_dir}/libnvcomp.so.5"
     SYMBOLIC)
file(CREATE_LINK
     "${libkvikio_cu12_SOURCE_DIR}/libkvikio/lib64/libkvikio.so"
     "${_kernellake_cudf_lib_dir}/libkvikio.so"
     SYMBOLIC)
