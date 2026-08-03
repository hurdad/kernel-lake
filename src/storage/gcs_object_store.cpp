#include "kernellake/storage/gcs_object_store.hpp"

#include <arrow/filesystem/gcsfs.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "generic_fs_object_store.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

// Parses an ISO-8601 UTC timestamp ("2026-01-01T00:00:00Z") into Arrow's
// TimePoint, for GcsOptions::FromAccessToken's expiration argument. Kept
// minimal (UTC "Z" suffix only, no offset parsing) since this only matters
// for the narrow "access_token" credentials_kind path.
arrow::fs::TimePoint parse_iso8601_utc(const std::string& text) {
  std::tm tm{};
  std::istringstream stream(text);
  stream >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
  if (stream.fail()) {
    throw StorageError("gcs: storage.gcs.access_token_expiration '" + text +
                       "' is not a valid ISO-8601 UTC timestamp (expected e.g. "
                       "'2026-01-01T00:00:00Z')");
  }
  const std::time_t time = timegm(&tm);
  return std::chrono::time_point_cast<arrow::fs::TimePoint::duration>(std::chrono::system_clock::from_time_t(time));
}

// GcsOptions' credential-selecting factories (Defaults()/Anonymous()/
// FromAccessToken()/FromServiceAccountCredentials()) each construct a fresh
// GcsOptions, discarding whatever plain fields (endpoint_override, scheme,
// etc.) the caller already set -- unlike S3Options/AzureOptions, GcsOptions
// has no instance-level Configure*() mutator. So: build credentials via the
// right factory first, then copy the caller's plain fields on top.
arrow::fs::GcsOptions build_options(const GcsSection& config) {
  arrow::fs::GcsOptions options;
  if (config.credentials_kind == "anonymous") {
    options = arrow::fs::GcsOptions::Anonymous();
  } else if (config.credentials_kind == "access_token") {
    if (config.access_token.empty()) {
      throw StorageError(
          "gcs: storage.gcs.credentials_kind is 'access_token' but storage.gcs.access_token is empty");
    }
    options = arrow::fs::GcsOptions::FromAccessToken(config.access_token,
                                                      parse_iso8601_utc(config.access_token_expiration));
  } else if (config.credentials_kind == "service_account_json") {
    if (config.json_credentials.empty()) {
      throw StorageError("gcs: storage.gcs.credentials_kind is 'service_account_json' but "
                         "storage.gcs.json_credentials is empty");
    }
    options = arrow::fs::GcsOptions::FromServiceAccountCredentials(config.json_credentials);
  } else {
    options = arrow::fs::GcsOptions::Defaults();
  }

  options.endpoint_override = config.options.endpoint_override;
  options.scheme = config.options.scheme;
  options.default_bucket_location = config.options.default_bucket_location;
  options.retry_limit_seconds = config.options.retry_limit_seconds;
  options.project_id = config.options.project_id;
  return options;
}

std::shared_ptr<arrow::fs::FileSystem> make_gcs_filesystem(const GcsSection& config) {
  const arrow::Result<std::shared_ptr<arrow::fs::GcsFileSystem>> result =
      arrow::fs::GcsFileSystem::Make(build_options(config));
  if (!result.ok()) {
    throw StorageError("gcs: failed to construct filesystem: " + result.status().ToString());
  }
  return *result;
}

}  // namespace

GcsObjectStore::GcsObjectStore(const GcsSection& config) : fs_(make_gcs_filesystem(config)) {}

std::vector<ObjectInfo> GcsObjectStore::list(const Uri& prefix) {
  return detail::generic_fs_list(fs_, "gcs", prefix);
}

std::unique_ptr<RandomAccessObject> GcsObjectStore::open(const Uri& uri) {
  return detail::generic_fs_open(fs_, "gcs", uri);
}

}  // namespace kernellake
