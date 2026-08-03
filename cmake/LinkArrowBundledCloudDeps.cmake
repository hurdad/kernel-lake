# config.hpp unconditionally includes <arrow/filesystem/{s3,gcs,azure,hdfs}fs.h>
# (see include/kernellake/common/config.hpp), so every executable that links
# kernellake_common transitively needs libarrow_bundled_dependencies.a's
# bundled google-cloud-cpp (GCS) and Azure SDK C++ (Azure) code to resolve.
# google-cloud-cpp needs many more Abseil symbols
# (absl::debian9::StrCat/StrAppend/crc_internal::*/etc. -- "debian9" is this
# Ubuntu 26.04 apt package's own inline-namespace name for Abseil, an
# ABI-stability convention, not a version number) than gRPC alone already
# pulled in for Flight SQL, and gRPC's own OTel exporter usage of Abseil
# (e.g. absl::debian9::ascii_internal::kPropertyBits, needed by libgrpc.so
# itself) isn't declared as an explicit dependency by any CMake target
# either. The project-wide `add_link_options(-Wl,--no-as-needed)` in the
# root CMakeLists.txt (see its own comment for why -- this is not solvable
# by wrapping specific libraries in --no-as-needed/--as-needed locally,
# confirmed by repeated real link failures each "fixing" one symbol only
# for CMake's link-line flattening to place the next one wrong too) makes
# every one of these libraries available for symbol resolution regardless
# of where CMake's own topological sort places them, so this function only
# needs to list them as plain, ungrouped dependencies -- no
# --start-group/--end-group or --no-as-needed/--as-needed wrapping needed
# here specifically.
#
# The Azure SDK's bundled XML parsing (its REST API request/response
# handling) needs libxml2, and its request-ID generation needs libuuid --
# neither was a dependency of this project before, confirmed by a real
# link failure without them installed (see docker/Dockerfile's own
# comment on the apt packages this needs, libxml2-dev/uuid-dev).
function(kernellake_link_arrow_bundled_cloud_deps target)
  target_link_libraries(${target} PRIVATE
    Arrow::arrow_static
    absl::strings
    absl::strings_internal
    absl::str_format
    absl::str_format_internal
    absl::time
    absl::time_zone
    absl::base
    absl::synchronization
    absl::crc32c
    absl::crc_internal
    absl::crc_cord_state
    absl::crc_cpu_detect
    absl::cord
    absl::cord_internal
    absl::civil_time
    absl::city
    absl::int128
    absl::throw_delegate
    absl::raw_logging_internal
    absl::status
    absl::statusor
    LibXml2::LibXml2
    PkgConfig::UUID
  )
endfunction()
