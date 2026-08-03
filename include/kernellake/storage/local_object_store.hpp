#pragma once

#include <string>

#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Local-filesystem-backed ObjectStore. `list(prefix)` supports:
//   - a single file path                    ("/data/sales.parquet")
//   - a glob pattern in the final component  ("/data/sales/*.parquet")
//   - a directory (lists its immediate *.parquet children)
// Results are always sorted lexicographically by path for deterministic
// ordering. Throws StorageError (naming the offending path) when a
// component does not exist or a glob/directory matches nothing.
//
// Every path handed to list()/open() is confined to `local_root` (default
// "/", i.e. no confinement): resolved via std::filesystem::weakly_canonical
// (so both ".."-segments and symlinks are followed) and rejected with
// StorageError if the result falls outside local_root. This is the only
// enforcement of StorageSection::local_root (kernellake/common/config.hpp)
// anywhere in the codebase -- without it, a query's read_parquet(...)
// argument (or the CLI's --path/--output) could read any file the process
// has access to regardless of what an operator configured as the intended
// root, which matters once kernellake-server's Flight SQL port is reachable
// by less-trusted clients than the operator running it.
class LocalObjectStore final : public ObjectStore {
 public:
  explicit LocalObjectStore(std::string local_root = "/") : local_root_(std::move(local_root)) {}

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override;
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override;

 private:
  std::string local_root_;
};

}  // namespace kernellake
