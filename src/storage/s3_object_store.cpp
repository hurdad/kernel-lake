#include "kernellake/storage/s3_object_store.hpp"

#include <arrow/filesystem/s3fs.h>

#include <cstdlib>
#include <mutex>

#include "generic_fs_object_store.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

void ensure_s3_initialized() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    const arrow::Status status = arrow::fs::EnsureS3Initialized();
    if (!status.ok()) {
      throw StorageError("s3: failed to initialize AWS SDK: " + status.ToString());
    }
  });
}

std::shared_ptr<arrow::fs::FileSystem> make_s3_filesystem(const S3Section& config) {
  ensure_s3_initialized();

  // S3Options' credentials_kind selection is which factory/Configure*()
  // method is called, not stored state on the options struct itself (see
  // config.hpp's own comment on S3Section) -- start from the caller's
  // already-configured plain fields (region, endpoint_override, scheme,
  // proxy_options, TLS, etc.) and mutate only the credential portion in
  // place, preserving everything else.
  arrow::fs::S3Options options = config.options;
  if (config.credentials_kind == "anonymous") {
    options.ConfigureAnonymousCredentials();
  } else if (config.credentials_kind == "explicit") {
    const char* access_key = std::getenv("AWS_ACCESS_KEY_ID");
    const char* secret_key = std::getenv("AWS_SECRET_ACCESS_KEY");
    if (access_key == nullptr || secret_key == nullptr) {
      throw StorageError(
          "s3: storage.s3.credentials_kind is 'explicit' but AWS_ACCESS_KEY_ID/"
          "AWS_SECRET_ACCESS_KEY are not both set in the environment");
    }
    const char* session_token = std::getenv("AWS_SESSION_TOKEN");
    options.ConfigureAccessKey(access_key, secret_key, session_token != nullptr ? session_token : "");
  } else if (config.credentials_kind == "role") {
    options.ConfigureAssumeRoleCredentials(options.role_arn, options.session_name, options.external_id,
                                           options.load_frequency);
  } else if (config.credentials_kind == "web_identity") {
    options.ConfigureAssumeRoleWithWebIdentityCredentials();
  } else {
    options.ConfigureDefaultCredentials();
  }

  const arrow::Result<std::shared_ptr<arrow::fs::S3FileSystem>> result = arrow::fs::S3FileSystem::Make(options);
  if (!result.ok()) {
    throw StorageError("s3: failed to construct filesystem: " + result.status().ToString());
  }
  return *result;
}

}  // namespace

S3ObjectStore::S3ObjectStore(const S3Section& config) : fs_(make_s3_filesystem(config)) {}

std::vector<ObjectInfo> S3ObjectStore::list(const Uri& prefix) {
  return detail::generic_fs_list(fs_, "s3", prefix);
}

std::unique_ptr<RandomAccessObject> S3ObjectStore::open(const Uri& uri) {
  return detail::generic_fs_open(fs_, "s3", uri);
}

}  // namespace kernellake
