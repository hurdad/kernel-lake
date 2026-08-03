#include "kernellake/storage/azure_object_store.hpp"

#include <arrow/filesystem/azurefs.h>

#include "generic_fs_object_store.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

std::shared_ptr<arrow::fs::FileSystem> make_azure_filesystem(const AzureSection& config) {
  // AzureOptions' credential fields are private, settable only through
  // these instance-level Configure*() mutators (see config.hpp's own
  // comment on AzureSection) -- apply on top of the caller's already-
  // configured plain fields (account_name, authorities, schemes,
  // background_writes), preserving everything else.
  arrow::fs::AzureOptions options = config.options;
  arrow::Status status;
  if (config.credentials_kind == "anonymous") {
    status = options.ConfigureAnonymousCredential();
  } else if (config.credentials_kind == "storage_shared_key") {
    if (config.storage_shared_key.empty()) {
      throw StorageError("azure: storage.azure.credentials_kind is 'storage_shared_key' but "
                         "storage.azure.storage_shared_key is empty");
    }
    status = options.ConfigureAccountKeyCredential(config.storage_shared_key);
  } else if (config.credentials_kind == "sas_token") {
    if (config.sas_token.empty()) {
      throw StorageError(
          "azure: storage.azure.credentials_kind is 'sas_token' but storage.azure.sas_token is empty");
    }
    status = options.ConfigureSASCredential(config.sas_token);
  } else if (config.credentials_kind == "client_secret") {
    status = options.ConfigureClientSecretCredential(config.tenant_id, config.client_id, config.client_secret);
  } else if (config.credentials_kind == "managed_identity") {
    status = options.ConfigureManagedIdentityCredential(config.client_id);
  } else if (config.credentials_kind == "cli") {
    status = options.ConfigureCLICredential();
  } else if (config.credentials_kind == "workload_identity") {
    status = options.ConfigureWorkloadIdentityCredential();
  } else if (config.credentials_kind == "environment") {
    status = options.ConfigureEnvironmentCredential();
  } else {
    status = options.ConfigureDefaultCredential();
  }
  if (!status.ok()) {
    throw StorageError("azure: failed to configure '" + config.credentials_kind +
                       "' credentials: " + status.ToString());
  }

  const arrow::Result<std::shared_ptr<arrow::fs::AzureFileSystem>> result =
      arrow::fs::AzureFileSystem::Make(options);
  if (!result.ok()) {
    throw StorageError("azure: failed to construct filesystem: " + result.status().ToString());
  }
  return *result;
}

}  // namespace

AzureObjectStore::AzureObjectStore(const AzureSection& config) : fs_(make_azure_filesystem(config)) {}

std::vector<ObjectInfo> AzureObjectStore::list(const Uri& prefix) {
  return detail::generic_fs_list(fs_, "azure", prefix);
}

std::unique_ptr<RandomAccessObject> AzureObjectStore::open(const Uri& uri) {
  return detail::generic_fs_open(fs_, "azure", uri);
}

}  // namespace kernellake
