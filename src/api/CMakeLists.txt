add_library(kernellake_api STATIC
  query_engine.cpp
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
  Arrow::arrow_static
)

if(KERNELLAKE_WITH_CUDA)
  target_link_libraries(kernellake_api PUBLIC kernellake_execution kernellake_memory)
endif()
