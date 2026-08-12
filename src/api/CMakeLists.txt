add_library(kernellake_api STATIC
  query_engine.cpp
  # Always built, in both presets -- see the comment at the top of the file.
  query_engine_execute_cpu.cpp
  # Exactly one of these provides QueryEngine::execute() -- see the comment
  # at the top of each file. Never both, to avoid an ODR conflict.
  $<IF:$<BOOL:${KERNELLAKE_WITH_CUDA}>,query_engine_execute_gpu.cpp,query_engine_execute_stub.cpp>
)

target_include_directories(kernellake_api PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_api PUBLIC cxx_std_20)

target_link_libraries(kernellake_api PUBLIC
  kernellake_common
  kernellake_planner
  kernellake_optimizer
  kernellake_storage
  kernellake_io
  kernellake_iceberg
  kernellake_unitycatalog
  kernellake_delta
  kernellake_execution_cpu
  Arrow::arrow_static
)

if(KERNELLAKE_WITH_CUDA)
  target_link_libraries(kernellake_api PUBLIC kernellake_execution_gpu kernellake_memory)
endif()
