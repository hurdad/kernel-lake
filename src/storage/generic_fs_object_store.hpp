#pragma once

// Private, src-local shared implementation for the three cloud ObjectStore
// backends (S3/GCS/Azure) -- each is a thin arrow::fs::FileSystem wrapper
// with identical list()/open() logic (LocalObjectStore's own glob/directory/
// exact-path branching, StorageError-on-missing contract), differing only
// in which concrete arrow::fs::FileSystem constructs the bytes. Not
// installed; included only by s3_object_store.cpp/gcs_object_store.cpp/
// azure_object_store.cpp.

#include <arrow/filesystem/filesystem.h>

#include <memory>
#include <string>
#include <vector>

#include "kernellake/storage/object_store.hpp"

namespace kernellake::detail {

// Classic iterative wildcard matcher: '*' matches any sequence (including
// empty), '?' matches exactly one character. Shared by LocalObjectStore and
// the three cloud backends, which all support the same glob-in-final-
// component convention for list().
[[nodiscard]] bool glob_match(std::string_view pattern, std::string_view text);
[[nodiscard]] bool has_glob_chars(std::string_view text);

// Strips a URI's "<scheme>://" prefix, leaving the bucket/key path Arrow's
// filesystem implementations expect (they are constructed directly via
// each Options struct's own Make(), not via arrow::fs::FileSystemFromUri(),
// so they know nothing about the scheme prefix themselves).
[[nodiscard]] std::string strip_scheme(const Uri& uri);

// Shared list()/open() implementation, parameterized by the already-
// constructed FileSystem and a human-readable backend label (used in
// StorageError messages, e.g. "s3", "gcs", "azure").
[[nodiscard]] std::vector<ObjectInfo> generic_fs_list(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                      std::string_view backend_label, const Uri& prefix);

// Like generic_fs_list(), but `prefix` must be a directory and every
// Parquet file anywhere in its subtree is returned, via
// arrow::fs::FileSelector::recursive -- see ObjectStore::list_recursive()'s
// own doc comment.
[[nodiscard]] std::vector<ObjectInfo> generic_fs_list_recursive(
    const std::shared_ptr<arrow::fs::FileSystem>& fs, std::string_view backend_label, const Uri& prefix);

[[nodiscard]] std::unique_ptr<RandomAccessObject> generic_fs_open(
    const std::shared_ptr<arrow::fs::FileSystem>& fs, std::string_view backend_label, const Uri& uri);

// Strips an "hdfs://namenode:port" authority from `uri`, leaving just the
// path portion (still "hdfs://"-prefixed) -- e.g.
// "hdfs://namenode:port/path" -> "hdfs:///path". Declared here (rather than
// staying file-local to hdfs_object_store.cpp) purely so it's directly unit
// testable as pure string parsing, the same reason strip_scheme() above is
// exposed. See hdfs_object_store.cpp's own comment on why HDFS alone (unlike
// S3/GCS/Azure) needs this extra step before delegating to generic_fs_list()/
// generic_fs_open().
[[nodiscard]] Uri strip_hdfs_authority(const Uri& uri);

}  // namespace kernellake::detail
