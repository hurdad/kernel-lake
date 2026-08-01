add_executable(kernellake
  main.cpp
  inspect_parquet_command.cpp
  explain_command.cpp
  query_command.cpp
  result_formatter.cpp
  generate_data_command.cpp
)

target_link_libraries(kernellake PRIVATE
  kernellake_common
  kernellake_storage
  kernellake_io
  kernellake_api
  kernellake_generator
  Arrow::arrow_static
)
