#pragma once

#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Local-filesystem-backed ObjectStore. `list(prefix)` supports:
//   - a single file path                    ("/data/sales.parquet")
//   - a glob pattern in the final component  ("/data/sales/*.parquet")
//   - a directory (lists its immediate *.parquet children)
// Results are always sorted lexicographically by path for deterministic
// ordering. Throws StorageError (naming the offending path) when a
// component does not exist or a glob/directory matches nothing.
class LocalObjectStore final : public ObjectStore {
 public:
  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override;
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override;
};

}  // namespace kernellake
