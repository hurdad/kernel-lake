# config.hpp unconditionally includes <arrow/filesystem/{s3,gcs,azure,hdfs}fs.h>
# (see include/kernellake/common/config.hpp), so every executable that links
# kernellake_common transitively needs libarrow_bundled_dependencies.a's
# bundled google-cloud-cpp (GCS) and Azure SDK C++ (Azure) code to resolve.
# Two real, confirmed-by-an-actual-link-failure problems, both worked around
# here rather than at each individual executable target:
#
# 1. google-cloud-cpp needs many more Abseil symbols
#    (absl::debian9::StrCat/StrAppend/crc_internal::*/etc. -- "debian9" is
#    this Ubuntu 26.04 apt package's own inline-namespace name for Abseil,
#    an ABI-stability convention, not a version number) than gRPC alone
#    already pulled in for Flight SQL. A plain link leaves these
#    unresolved; wrapping Arrow::arrow_static together with the relevant
#    absl:: targets in a --start-group/--end-group rescan fixes it, same
#    class of issue (and same fix) as ArrowFlight/ArrowFlightSql vs.
#    gRPC::grpc++ in src/server/CMakeLists.txt -- see that file's own
#    comment for why this uses raw -Wl,--start-group/--end-group strings
#    rather than CMake's $<LINK_GROUP:RESCAN,...> genex (false-positives a
#    dependency cycle, since Arrow::arrow_static is used throughout the
#    rest of the kernellake_* tree outside any group too). Note: CMake's
#    own link-line flattening may relocate some of these libraries outside
#    the textual group boundary if other targets also request them plainly
#    elsewhere in the graph -- that turned out to be fine in practice
#    (confirmed by an actual successful link): enough copies end up
#    somewhere after libarrow_bundled_dependencies.a's own natural
#    position for the linker's final undefined-symbol pass to find them.
# 2. The Azure SDK's bundled XML parsing (its REST API request/response
#    handling) needs libxml2, and its request-ID generation needs libuuid
#    -- neither was a dependency of this project before. Both are shared
#    libraries with no other consumer in this project's own dependency
#    graph, so the linker's default --as-needed drops them right where
#    they're declared (nothing needs their symbols yet at that point) and
#    never revisits them once libarrow_bundled_dependencies.a's object code
#    actually needs them later in the link line -- confirmed by an actual
#    link failure without the explicit --no-as-needed/--as-needed pair
#    below, which forces the linker to keep them live for its final pass.
function(kernellake_link_arrow_bundled_cloud_deps target)
  target_link_libraries(${target} PRIVATE
    -Wl,--start-group
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
    -Wl,--end-group
    -Wl,--no-as-needed
    LibXml2::LibXml2
    PkgConfig::UUID
    -Wl,--as-needed
  )
endfunction()
