# Always built, in every preset -- both the kernellake CLI
# (query_command.cpp) and kernellake_flight_sql_server (flight_sql_server.cpp)
# call this module's API unconditionally. Exactly one of
# query_tracing_otel.cpp/query_tracing_stub.cpp provides the implementation,
# selected by KERNELLAKE_ENABLE_OTEL, mirroring src/api/CMakeLists.txt's
# KERNELLAKE_WITH_CUDA split -- never both, to avoid an ODR conflict.
# query_tracing.hpp itself never includes an opentelemetry-cpp header either
# way, so an OFF build's call sites never see one, and
# query_tracing_stub.cpp doesn't either -- only query_tracing_otel.cpp does,
# and it (plus query_tracing_test_support.cpp, its test-only sibling -- see
# internal.hpp's comment for why it's a separate .o) is entirely absent from
# the source list (never compiled, never linked) when
# KERNELLAKE_ENABLE_OTEL=OFF.
add_library(kernellake_observability STATIC
  $<IF:$<BOOL:${KERNELLAKE_ENABLE_OTEL}>,query_tracing_otel.cpp,query_tracing_stub.cpp>
  $<$<BOOL:${KERNELLAKE_ENABLE_OTEL}>:query_tracing_test_support.cpp>
)

target_include_directories(kernellake_observability PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_observability PUBLIC cxx_std_20)

target_link_libraries(kernellake_observability PUBLIC kernellake_common)

if(KERNELLAKE_ENABLE_OTEL)
  # "exporters_otlp_grpc"/"exporters_otlp_http" (requested in the root
  # find_package call) are find_package *component* names, not CMake
  # targets -- the actual targets they provide are these six, which
  # transitively pull in opentelemetry-cpp::{trace,metrics,logs,resources}
  # via otlp_recordable (verified by inspecting the installed
  # *-target.cmake files' own INTERFACE_LINK_LIBRARIES, not assumed).
  target_link_libraries(kernellake_observability PRIVATE
    opentelemetry-cpp::sdk
    opentelemetry-cpp::otlp_grpc_exporter
    opentelemetry-cpp::otlp_grpc_metrics_exporter
    opentelemetry-cpp::otlp_grpc_log_record_exporter
    opentelemetry-cpp::otlp_http_exporter
    opentelemetry-cpp::otlp_http_metric_exporter
    opentelemetry-cpp::otlp_http_log_record_exporter
  )
endif()
