#pragma once

#include "kernellake/common/config.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// S3-backed ObjectStore ("s3://bucket/key"), wrapping arrow::fs::S3FileSystem.
// Same list()/open() contract as LocalObjectStore (glob-in-final-component,
// directory listing, StorageError on anything missing -- never a silent
// empty result). Constructing this calls arrow::fs::EnsureS3Initialized()
// (idempotent, AWS SDK global init) the first time; callers only need to
// make sure arrow::fs::FinalizeS3() runs once at process shutdown (see
// src/cli/main.cpp / src/server/main.cpp).
class S3ObjectStore final : public ObjectStore {
 public:
  explicit S3ObjectStore(const S3Section& config);

  // Constructs an S3ObjectStore from already-obtained, short-lived
  // credentials (e.g. Unity Catalog's vended temporary-table-credentials --
  // see kernellake::unitycatalog::UnityCatalogSourceResolver) rather than
  // this process's static storage.s3 config. `base_options` supplies
  // everything else (region, endpoint_override, TLS, proxy) -- an operator
  // already configures these for any S3 access, so no new S3-related
  // config fields are needed just to support vended credentials. Calls
  // `options.ConfigureAccessKey(...)` directly, the same Arrow call
  // S3Section's own "explicit" credentials_kind uses (see
  // s3_object_store.cpp's make_s3_filesystem()), just fed vended values
  // instead of environment variables. Intended to be constructed fresh per
  // use and discarded, never cached past the credentials' own TTL.
  S3ObjectStore(const arrow::fs::S3Options& base_options, const std::string& access_key_id,
                const std::string& secret_access_key, const std::string& session_token);

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override;
  [[nodiscard]] std::vector<ObjectInfo> list_recursive(const Uri& prefix) override;
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override;

 private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
};

}  // namespace kernellake
