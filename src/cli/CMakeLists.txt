add_executable(kernellake
  main.cpp
  inspect_parquet_command.cpp
  explain_command.cpp
)

target_link_libraries(kernellake PRIVATE
  kernellake_common
  kernellake_storage
  kernellake_io
  kernellake_api
)
