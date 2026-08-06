#include "kernellake/iceberg/rest_catalog_client.hpp"

#include <curl/curl.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <memory>
#include <mutex>
#include <utility>

#include "kernellake/common/errors.hpp"

namespace kernellake::iceberg {

namespace {

// libcurl requires exactly one curl_global_init()/curl_global_cleanup() pair
// per process, not per handle -- std::call_once makes the first call (from
// however many threads construct a client) do it exactly once.
// Intentionally never cleaned up: curl_global_cleanup() would need to run
// after every other thread has stopped using libcurl, which this class has
// no way to know; leaving it for process-exit teardown is fine for a
// process-lifetime global like this.
std::once_flag curl_global_init_flag;
void ensure_curl_global_init() {
  std::call_once(curl_global_init_flag, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

struct CurlEasyDeleter {
  void operator()(CURL* handle) const { curl_easy_cleanup(handle); }
};
using CurlEasyPtr = std::unique_ptr<CURL, CurlEasyDeleter>;

struct CurlSlistDeleter {
  void operator()(curl_slist* list) const { curl_slist_free_all(list); }
};
using CurlSlistPtr = std::unique_ptr<curl_slist, CurlSlistDeleter>;

CurlEasyPtr make_curl_easy() {
  ensure_curl_global_init();
  CurlEasyPtr handle(curl_easy_init());
  if (!handle) {
    throw StorageError("iceberg rest catalog: curl_easy_init() failed");
  }
  return handle;
}

size_t write_to_string(char* data, size_t size, size_t nmemb, void* user_data) {
  auto* out = static_cast<std::string*>(user_data);
  out->append(data, size * nmemb);
  return size * nmemb;
}

std::string url_encode(CURL* handle, const std::string& value) {
  char* escaped = curl_easy_escape(handle, value.c_str(), static_cast<int>(value.size()));
  if (escaped == nullptr) {
    throw StorageError(fmt::format("iceberg rest catalog: failed to URL-encode '{}'", value));
  }
  std::string result(escaped);
  curl_free(escaped);
  return result;
}

// Multi-level namespaces are addressed by the REST Catalog spec as their
// parts joined with U+001F (unit separator), the whole joined string then
// URL-encoded as a single path segment -- not each part encoded and joined
// by literal slashes, which would be indistinguishable from a deeper path.
std::string encode_namespace(CURL* handle, const std::vector<std::string>& namespace_parts) {
  std::string joined;
  for (size_t i = 0; i < namespace_parts.size(); ++i) {
    if (i > 0) {
      joined.push_back('\x1F');
    }
    joined += namespace_parts[i];
  }
  return url_encode(handle, joined);
}

// GET when post_body is null, POST (form-encoded) otherwise -- covers both
// requests this client makes (table-metadata GET, oauth2 token POST).
std::string http_request(const std::string& url, const std::string& bearer_token,
                         const std::string* post_body) {
  CurlEasyPtr handle = make_curl_easy();
  std::string response_body;

  curl_easy_setopt(handle.get(), CURLOPT_URL, url.c_str());
  curl_easy_setopt(handle.get(), CURLOPT_WRITEFUNCTION, write_to_string);
  curl_easy_setopt(handle.get(), CURLOPT_WRITEDATA, &response_body);
  curl_easy_setopt(handle.get(), CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(handle.get(), CURLOPT_TIMEOUT, 30L);
  curl_easy_setopt(handle.get(), CURLOPT_CONNECTTIMEOUT, 10L);
  // Avoid libcurl's default use of signals/alarm() for timeouts, which is
  // not safe from a multi-threaded query engine.
  curl_easy_setopt(handle.get(), CURLOPT_NOSIGNAL, 1L);

  curl_slist* raw_headers = curl_slist_append(nullptr, "Accept: application/json");
  if (!bearer_token.empty()) {
    const std::string auth_header = fmt::format("Authorization: Bearer {}", bearer_token);
    raw_headers = curl_slist_append(raw_headers, auth_header.c_str());
  }
  if (post_body != nullptr) {
    raw_headers = curl_slist_append(raw_headers, "Content-Type: application/x-www-form-urlencoded");
    curl_easy_setopt(handle.get(), CURLOPT_POSTFIELDS, post_body->c_str());
    curl_easy_setopt(handle.get(), CURLOPT_POSTFIELDSIZE, static_cast<long>(post_body->size()));
  }
  const CurlSlistPtr headers(raw_headers);
  curl_easy_setopt(handle.get(), CURLOPT_HTTPHEADER, headers.get());

  const CURLcode result = curl_easy_perform(handle.get());
  if (result != CURLE_OK) {
    throw StorageError(
        fmt::format("iceberg rest catalog: request to '{}' failed: {}", url, curl_easy_strerror(result)));
  }

  long status_code = 0;
  curl_easy_getinfo(handle.get(), CURLINFO_RESPONSE_CODE, &status_code);
  if (status_code < 200 || status_code >= 300) {
    throw StorageError(fmt::format("iceberg rest catalog: request to '{}' returned HTTP {}: {}", url,
                                   status_code, response_body));
  }
  return response_body;
}

nlohmann::json parse_json_response(const std::string& url, const std::string& body) {
  try {
    return nlohmann::json::parse(body);
  } catch (const nlohmann::json::exception& e) {
    throw StorageError(
        fmt::format("iceberg rest catalog: response from '{}' is not valid JSON: {}", url, e.what()));
  }
}

std::vector<IcebergSchemaField> parse_schema_fields(const nlohmann::json& schema_json,
                                                    const std::string& url) {
  std::vector<IcebergSchemaField> fields;
  if (!schema_json.contains("fields")) {
    throw StorageError(fmt::format("iceberg rest catalog: schema from '{}' is missing 'fields'", url));
  }
  for (const nlohmann::json& field_json : schema_json.at("fields")) {
    try {
      IcebergSchemaField field;
      field.id = field_json.at("id").get<int32_t>();
      field.name = field_json.at("name").get<std::string>();
      field.required = field_json.value("required", false);
      // A struct/list/map field's "type" is a nested JSON object, not a
      // string -- captured via dump() so schema_translation.hpp's
      // unsupported-type error can show the caller what it actually saw,
      // rather than this failing on a bad get<std::string>() cast first.
      const nlohmann::json& type_json = field_json.at("type");
      field.type = type_json.is_string() ? type_json.get<std::string>() : type_json.dump();
      fields.push_back(std::move(field));
    } catch (const nlohmann::json::exception& e) {
      throw StorageError(
          fmt::format("iceberg rest catalog: a schema field from '{}' is missing a required attribute: {}",
                      url, e.what()));
    }
  }
  return fields;
}

}  // namespace

std::optional<std::string> IcebergTableMetadata::current_manifest_list() const {
  if (!current_snapshot_id.has_value()) {
    return std::nullopt;
  }
  for (const IcebergSnapshot& snapshot : snapshots) {
    if (snapshot.snapshot_id == *current_snapshot_id) {
      return snapshot.manifest_list;
    }
  }
  return std::nullopt;
}

IcebergRestCatalogClient::IcebergRestCatalogClient(IcebergCatalogSection config)
    : config_(std::move(config)) {}

std::string IcebergRestCatalogClient::fetch_oauth2_token() {
  const double now =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  // 30s safety margin so a cached token doesn't expire mid-flight between
  // this check and the request it's about to authorize.
  if (!cached_oauth2_token_.empty() && now < oauth2_token_expiry_unix_seconds_ - 30.0) {
    return cached_oauth2_token_;
  }

  std::string body;
  {
    const CurlEasyPtr encoder = make_curl_easy();
    body = fmt::format("grant_type=client_credentials&client_id={}&client_secret={}",
                       url_encode(encoder.get(), config_.oauth2_client_id),
                       url_encode(encoder.get(), config_.oauth2_client_secret));
    if (!config_.oauth2_scope.empty()) {
      body += fmt::format("&scope={}", url_encode(encoder.get(), config_.oauth2_scope));
    }
  }

  const std::string url = fmt::format("{}/v1/oauth/tokens", config_.catalog_uri);
  const nlohmann::json response = parse_json_response(url, http_request(url, /*bearer_token=*/"", &body));

  if (!response.contains("access_token")) {
    throw StorageError(
        fmt::format("iceberg rest catalog: oauth2 token response from '{}' is missing 'access_token'", url));
  }
  cached_oauth2_token_ = response.at("access_token").get<std::string>();
  const double expires_in_seconds = response.value("expires_in", 3600.0);
  oauth2_token_expiry_unix_seconds_ = now + expires_in_seconds;
  return cached_oauth2_token_;
}

std::string IcebergRestCatalogClient::bearer_token_for_request() {
  if (config_.credentials_kind == "bearer_token") {
    return config_.bearer_token;
  }
  if (config_.credentials_kind == "oauth2_client_credentials") {
    return fetch_oauth2_token();
  }
  return "";
}

IcebergTableMetadata IcebergRestCatalogClient::load_table_metadata(
    const std::vector<std::string>& namespace_parts, const std::string& table) {
  std::string encoded_namespace;
  std::string encoded_table;
  {
    const CurlEasyPtr encoder = make_curl_easy();
    encoded_namespace = encode_namespace(encoder.get(), namespace_parts);
    encoded_table = url_encode(encoder.get(), table);
  }

  const std::string url = config_.prefix.empty()
                              ? fmt::format("{}/v1/namespaces/{}/tables/{}", config_.catalog_uri,
                                            encoded_namespace, encoded_table)
                              : fmt::format("{}/v1/{}/namespaces/{}/tables/{}", config_.catalog_uri,
                                            config_.prefix, encoded_namespace, encoded_table);

  const nlohmann::json response =
      parse_json_response(url, http_request(url, bearer_token_for_request(), /*post_body=*/nullptr));

  if (!response.contains("metadata")) {
    throw StorageError(
        fmt::format("iceberg rest catalog: LoadTableResult from '{}' is missing 'metadata'", url));
  }
  const nlohmann::json& metadata_json = response.at("metadata");

  IcebergTableMetadata metadata;
  try {
    metadata.location = metadata_json.at("location").get<std::string>();
    metadata.format_version = metadata_json.at("format-version").get<int32_t>();
  } catch (const nlohmann::json::exception& e) {
    throw StorageError(fmt::format(
        "iceberg rest catalog: table metadata from '{}' is missing a required field: {}", url, e.what()));
  }

  if (metadata_json.contains("current-snapshot-id") && !metadata_json.at("current-snapshot-id").is_null()) {
    metadata.current_snapshot_id = metadata_json.at("current-snapshot-id").get<int64_t>();
  }

  if (metadata_json.contains("snapshots")) {
    for (const nlohmann::json& snapshot_json : metadata_json.at("snapshots")) {
      try {
        IcebergSnapshot snapshot;
        snapshot.snapshot_id = snapshot_json.at("snapshot-id").get<int64_t>();
        snapshot.manifest_list = snapshot_json.at("manifest-list").get<std::string>();
        metadata.snapshots.push_back(std::move(snapshot));
      } catch (const nlohmann::json::exception& e) {
        throw StorageError(
            fmt::format("iceberg rest catalog: a 'snapshots' entry from '{}' is missing a required field: {}",
                        url, e.what()));
      }
    }
  }

  // v2 table metadata carries every historical schema in "schemas", with
  // "current-schema-id" picking the live one; v1 (deprecated but still
  // seen from older/compat servers) carries only a single "schema" with
  // no id-selection needed.
  if (metadata_json.contains("schemas") && !metadata_json.at("schemas").is_null()) {
    const int32_t current_schema_id = metadata_json.value("current-schema-id", 0);
    bool found_current_schema = false;
    for (const nlohmann::json& schema_json : metadata_json.at("schemas")) {
      if (schema_json.value("schema-id", -1) == current_schema_id) {
        metadata.schema_fields = parse_schema_fields(schema_json, url);
        found_current_schema = true;
        break;
      }
    }
    if (!found_current_schema) {
      throw StorageError(fmt::format(
          "iceberg rest catalog: table metadata from '{}' has no schema matching current-schema-id {}", url,
          current_schema_id));
    }
  } else if (metadata_json.contains("schema")) {
    metadata.schema_fields = parse_schema_fields(metadata_json.at("schema"), url);
  }

  return metadata;
}

}  // namespace kernellake::iceberg
