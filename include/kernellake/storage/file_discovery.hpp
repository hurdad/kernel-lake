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

// Like discover_parquet_files(), but for each source with no glob
// characters, tries ObjectStore::list_recursive() first -- finding files
// nested arbitrarily deep, e.g. under Hive-style partition directories
// (`region=US/date=2026-01-01/part-0.parquet`) -- falling back to the
// plain list() behavior (exact single file, or a flat/glob directory
// listing) if the source isn't actually a directory. A source with no
// nested files (a flat directory, or an exact file) resolves identically
// either way, since a recursive listing of a real directory is always a
// superset of its flat listing. Used by
// kernellake::resolve_table() (kernellake/io/table_resolution.hpp) to
// auto-detect Hive partitioning with no new SQL syntax.
[[nodiscard]] std::vector<ObjectInfo> discover_parquet_files_recursive(
    ObjectStore& store, const std::vector<std::string>& sources);

}  // namespace kernellake
