# Only built when KERNELLAKE_BUILD_SERVER is ON (see src/CMakeLists.txt).
# Requires libarrow-flight-dev, libarrow-flight-sql-dev, and gRPC -- see
# docs/ARCHITECTURE.md's "Ubuntu 26.04 baseline" section for why Arrow
# Flight SQL specifically needs 26.04, not 24.04.
add_library(kernellake_flight_sql_server STATIC
  flight_sql_server.cpp
  $<IF:$<BOOL:${KERNELLAKE_WITH_CUDA}>,gpu_execution_coordinator_gpu.cpp,gpu_execution_coordinator_stub.cpp>
)

target_include_directories(kernellake_flight_sql_server PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_flight_sql_server PUBLIC cxx_std_20)

target_link_libraries(kernellake_flight_sql_server PUBLIC
  kernellake_api
  kernellake_common
  ArrowFlightSql::arrow_flight_sql_static
  ArrowFlight::arrow_flight_static
  Arrow::arrow_static
)

if(KERNELLAKE_WITH_CUDA)
  target_link_libraries(kernellake_flight_sql_server PUBLIC kernellake_memory)
endif()

add_executable(kernellake-server main.cpp)

# Arrow's own ArrowFlightSql/ArrowFlight CMake targets don't declare
# gRPC::grpc++ as a dependency, so a plain link fails with undefined
# references into libarrow_flight.a's grpc-transport object code (e.g.
# grpc::g_core_codegen_interface, absl::debian9::Mutex::lock/unlock) --
# libarrow_flight_sql.a/libarrow_flight.a and libgrpc++.so were never
# adjacent on the link line at all without this. Verified for real
# (docs/ARCHITECTURE.md's "Ubuntu 26.04 baseline" section) that wrapping
# them together in a --start-group/--end-group rescan fixes it.
#
# Deliberately uses raw -Wl,--start-group/--end-group strings rather than
# CMake's $<LINK_GROUP:RESCAN,...> genex: LINK_GROUP's own CMake-level
# dependency-cycle detection false-positives here, because Arrow::arrow_static
# (which this group transitively needs) is also used throughout the rest of
# the kernellake_* tree outside any group -- CMake sees that as the group
# depending on itself. Plain strings aren't analyzed as target dependencies
# at all, so they sidestep that check entirely while emitting the identical
# --start-group/--end-group linker directives.
target_link_libraries(kernellake-server PRIVATE
  -Wl,--start-group
  kernellake_flight_sql_server
  ArrowFlightSql::arrow_flight_sql_static
  ArrowFlight::arrow_flight_static
  gRPC::grpc++
  -Wl,--end-group
)
