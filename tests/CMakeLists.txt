add_subdirectory(unit)

if(KERNELLAKE_WITH_CUDA)
  add_subdirectory(gpu)
endif()
