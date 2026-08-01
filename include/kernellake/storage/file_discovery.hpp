#pragma once

#include <string>
#include <vector>

#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Resolves the path arguments of `read_parquet('path' [, 'path2', ...])`
// into a deterministic, deduplicated, sorted list of concrete Parquet files,
// using `store` to do the actual listing (so this works the same way
// against LocalObjectStore today and a future S3ObjectStore later).
//
// Throws StorageError if any source lists zero files (via the ObjectStore
// itself) or if any listed object is not a ".parquet" file -- KernelLake
// does not silently skip or reinterpret non-Parquet input.
[[nodiscard]] std::vector<ObjectInfo> discover_parquet_files(ObjectStore& store,
                                                              const std::vector<std::string>& sources);

}  // namespace kernellake
