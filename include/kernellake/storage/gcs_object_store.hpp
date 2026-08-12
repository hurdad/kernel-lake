#pragma once

#include "kernellake/common/config.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// GCS-backed ObjectStore ("gs://bucket/key"), wrapping arrow::fs::GcsFileSystem.
// Same list()/open() contract as LocalObjectStore.
class GcsObjectStore final : public ObjectStore {
 public:
  explicit GcsObjectStore(const GcsSection& config);

  // Constructs a GcsObjectStore from an already-obtained OAuth2 access
  // token (e.g. Unity Catalog's vended "gcp_oauth_token" -- see
  // kernellake::unitycatalog::UnityCatalogSourceResolver) rather than this
  // process's static storage.gcs config. `base_options` supplies
  // everything else (endpoint_override, scheme, project_id, etc.) --
  // mirrors S3ObjectStore's own vended-credentials constructor exactly,
  // see that class's comment for the full rationale. Since Unity
  // Catalog's real response shape for this credential kind wasn't
  // independently verified against a live server (see
  // UnityCatalogTemporaryCredentials's own comment), and Arrow's
  // GcsOptions::FromAccessToken() requires an expiration timestamp UC's
  // response may or may not actually carry, this constructor uses a
  // fixed, generous 1-hour expiration rather than parsing one -- the
  // token is used immediately for one query's worth of requests, never
  // cached past this object's own lifetime, so the real expiration
  // doesn't matter as long as it's not already in the past.
  GcsObjectStore(const arrow::fs::GcsOptions& base_options, const std::string& access_token);

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override;
  [[nodiscard]] std::vector<ObjectInfo> list_recursive(const Uri& prefix) override;
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override;

 private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
};

}  // namespace kernellake
