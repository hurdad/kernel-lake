#pragma once

#include "kernellake/common/config.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Azure Blob-backed ObjectStore ("abfs://container/key"), wrapping
// arrow::fs::AzureFileSystem. Same list()/open() contract as
// LocalObjectStore.
class AzureObjectStore final : public ObjectStore {
 public:
  explicit AzureObjectStore(const AzureSection& config);

  // Constructs an AzureObjectStore from an already-obtained user-delegation
  // SAS token (e.g. Unity Catalog's vended "azure_user_delegation_sas" --
  // see kernellake::unitycatalog::UnityCatalogSourceResolver) rather than
  // this process's static storage.azure config. `base_options` supplies
  // everything else (account_name, authorities, schemes) -- mirrors
  // S3ObjectStore's own vended-credentials constructor exactly, see that
  // class's comment for the full rationale. Unity Catalog's real response
  // shape for this credential kind wasn't independently verified against
  // a live server (see UnityCatalogTemporaryCredentials's own comment) --
  // a SAS token self-encodes its own expiration, so (unlike the GCS
  // vended constructor) nothing here needs a placeholder expiry.
  AzureObjectStore(const arrow::fs::AzureOptions& base_options, const std::string& sas_token);

  [[nodiscard]] std::vector<ObjectInfo> list(const Uri& prefix) override;
  [[nodiscard]] std::vector<ObjectInfo> list_recursive(const Uri& prefix) override;
  [[nodiscard]] std::unique_ptr<RandomAccessObject> open(const Uri& uri) override;

 private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
};

}  // namespace kernellake
