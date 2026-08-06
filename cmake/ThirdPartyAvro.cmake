# Apache Avro's C library (avro-c, package `libavro-dev` on Debian/Ubuntu)
# is used to read Iceberg manifest-list/manifest files (Avro Object
# Container Files) -- see src/iceberg/manifest_reader.cpp. Unlike every
# other third-party dependency in this project, avro-c ships neither a
# pkg-config file nor a CMake config package, so it's found the manual way
# (find_library/find_path) and wrapped in a hand-declared IMPORTED target,
# the same shape pkg_check_modules(... IMPORTED_TARGET ...) would produce
# for a library that *did* have a .pc file (see CMakeLists.txt's own
# `pkg_check_modules(UUID ...)` for that simpler case).
find_library(AVRO_LIBRARY NAMES avro)
find_path(AVRO_INCLUDE_DIR NAMES avro.h)

if(NOT AVRO_LIBRARY OR NOT AVRO_INCLUDE_DIR)
  message(FATAL_ERROR
    "Apache Avro C library (avro-c) not found -- install libavro-dev "
    "(Debian/Ubuntu) or the equivalent avro-c development package for your "
    "distribution. Needed for Iceberg manifest-list/manifest reading (see "
    "src/iceberg/manifest_reader.cpp).")
endif()

if(NOT TARGET Avro::avro)
  add_library(Avro::avro UNKNOWN IMPORTED)
  set_target_properties(Avro::avro PROPERTIES
    IMPORTED_LOCATION "${AVRO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${AVRO_INCLUDE_DIR}"
  )
endif()
