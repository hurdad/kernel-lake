#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "kernellake/storage/object_store.hpp"

namespace kernellake::iceberg {

// Reads a position-delete file's *full content* (its own `file_path`/`pos`
// columns -- see the Iceberg spec's "Position Delete Files" section --
// not just its Parquet footer, unlike inspect_parquet_file()) and returns,
// for each data file path it references, how many distinct row positions
// this one delete file marks as deleted for it.
//
// This exists specifically to support resolve_iceberg_table()'s
// whole-file-deletion detection: a data file is dropped from the read set
// entirely when the *total* deleted-position count across every position
// delete file referencing it (there can be more than one, e.g. from
// separate compaction passes) reaches its own `record_count` -- see that
// function's own comment. Reading actual delete positions and applying
// them to individual rows of a *partially* deleted file (real row-level
// filtering) is not implemented; resolve_iceberg_table() throws in that
// case rather than silently returning deleted rows.
//
// The delete file's own path is joined against nothing -- like every
// other file path this codebase reads via manifests, it's already the
// exact string a data/delete file's own `file_path` field records
// (Iceberg's position-delete `file_path` column always quotes a
// referenced data file's `file_path` verbatim, so the map keys returned
// here already match data file paths exactly with no normalization
// needed).
[[nodiscard]] std::unordered_map<std::string, std::int64_t> read_position_delete_counts(ObjectStore& store,
                                                                                        const Uri& uri);

}  // namespace kernellake::iceberg
