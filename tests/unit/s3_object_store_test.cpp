#include <arrow/filesystem/s3fs.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/storage/s3_object_store.hpp"

namespace kernellake {
namespace {

// Saves/restores AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY around a test
// that needs them deliberately unset -- S3ObjectStore's "explicit"
// credentials_kind reads them straight from the environment (see
// s3_object_store.cpp's make_s3_filesystem()), so a real value already set
// in whatever environment runs this suite must not leak into (or be
// clobbered by) the "unset" test case below.
class ExplicitCredentialsEnvGuard {
 public:
  ExplicitCredentialsEnvGuard() {
    if (const char* value = std::getenv("AWS_ACCESS_KEY_ID"); value != nullptr) {
      saved_access_key_ = std::string(value);
    }
    if (const char* value = std::getenv("AWS_SECRET_ACCESS_KEY"); value != nullptr) {
      saved_secret_key_ = std::string(value);
    }
    ::unsetenv("AWS_ACCESS_KEY_ID");
    ::unsetenv("AWS_SECRET_ACCESS_KEY");
  }

  ~ExplicitCredentialsEnvGuard() {
    if (saved_access_key_) {
      ::setenv("AWS_ACCESS_KEY_ID", saved_access_key_->c_str(), 1);
    }
    if (saved_secret_key_) {
      ::setenv("AWS_SECRET_ACCESS_KEY", saved_secret_key_->c_str(), 1);
    }
  }

 private:
  std::optional<std::string> saved_access_key_;
  std::optional<std::string> saved_secret_key_;
};

// make_s3_filesystem()'s "explicit" branch is a constructor-time config
// validation check (env vars present before ever touching the network), same
// shape as GcsObjectStore's access_token_expiration validation -- safe to
// test with no real AWS access.
TEST(S3ObjectStore, ExplicitCredentialsWithoutEnvVarsThrowsStorageError) {
  const ExplicitCredentialsEnvGuard guard;

  S3Section config;
  config.credentials_kind = "explicit";
  EXPECT_THROW((void)(S3ObjectStore(config)), StorageError);
}

// arrow::fs::S3FileSystem::Make() doesn't touch the network at construction
// time (same as GcsFileSystem::Make()/AzureFileSystem::Make()), so anonymous
// credentials construction is safe to run with no real AWS endpoint.
TEST(S3ObjectStore, AnonymousCredentialsConstructorDoesNotThrow) {
  S3Section config;
  config.credentials_kind = "anonymous";
  config.options.region = "us-east-1";
  EXPECT_NO_THROW((void)(S3ObjectStore(config)));
}

// "default" (S3Section's own default credentials_kind, ConfigureDefault
// Credentials()) also only builds an AWS credentials provider chain at
// construction time -- no network round trip until list()/open() actually
// runs.
TEST(S3ObjectStore, DefaultCredentialsConstructorDoesNotThrow) {
  S3Section config;
  config.options.region = "us-east-1";
  EXPECT_NO_THROW((void)(S3ObjectStore(config)));
}

}  // namespace
}  // namespace kernellake
