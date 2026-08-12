#include "kernellake/unitycatalog/unity_catalog_client.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <utility>

#include "kernellake/common/errors.hpp"
#include "kernellake/common/http_client.hpp"

namespace kernellake::unitycatalog {

namespace {

constexpr const char* kErrorPrefix = "unity catalog";

nlohmann::json parse_json_response(const std::string& url, const std::string& body) {
  try {
    return nlohmann::json::parse(body);
  } catch (const nlohmann::json::exception& e) {
    throw StorageError(fmt::format("{}: response from '{}' is not valid JSON: {}", kErrorPrefix, url, e.what()));
  }
}

std::vector<UnityCatalogColumn> parse_columns(const nlohmann::json& table_json, const std::string& url) {
  std::vector<UnityCatalogColumn> columns;
  if (!table_json.contains("columns")) {
    return columns;
  }
  for (const nlohmann::json& column_json : table_json.at("columns")) {
    try {
      UnityCatalogColumn column;
      column.name = column_json.at("name").get<std::string>();
      column.type_name = column_json.value("type_name", "");
      // A struct/array/map column's "type_json" is itself a JSON-encoded
      // string in UC's response (unlike Iceberg's raw nested-object "type"
      // -- see IcebergSchemaField's own comment for that different shape),
      // so it's always already a plain string here, captured verbatim.
      column.type_json = column_json.value("type_json", "");
      column.nullable = column_json.value("nullable", true);
      column.position = column_json.value("position", 0);
      columns.push_back(std::move(column));
    } catch (const nlohmann::json::exception& e) {
      throw StorageError(
          fmt::format("{}: a 'columns' entry from '{}' is missing a required attribute: {}", kErrorPrefix, url,
                      e.what()));
    }
  }
  return columns;
}

}  // namespace

UnityCatalogClient::UnityCatalogClient(UnityCatalogInstanceSection config) : config_(std::move(config)) {}

std::string UnityCatalogClient::fetch_oauth2_token() {
  const double now =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  // 30s safety margin so a cached token doesn't expire mid-flight between
  // this check and the request it's about to authorize -- same margin
  // IcebergRestCatalogClient::fetch_oauth2_token() uses.
  if (!cached_oauth2_token_.empty() && now < oauth2_token_expiry_unix_seconds_ - 30.0) {
    return cached_oauth2_token_;
  }

  std::string body;
  {
    const CurlEasyPtr encoder = make_curl_easy();
    body = fmt::format("grant_type=client_credentials&client_id={}&client_secret={}",
                       url_encode(encoder.get(), config_.oauth2_client_id, kErrorPrefix),
                       url_encode(encoder.get(), config_.oauth2_client_secret, kErrorPrefix));
    if (!config_.oauth2_scope.empty()) {
      body += fmt::format("&scope={}", url_encode(encoder.get(), config_.oauth2_scope, kErrorPrefix));
    }
  }

  const std::vector<std::string> headers = {"Accept: application/json",
                                            "Content-Type: application/x-www-form-urlencoded"};
  const nlohmann::json response = parse_json_response(
      config_.oauth2_token_endpoint, http_request(config_.oauth2_token_endpoint, headers, &body, kErrorPrefix));

  if (!response.contains("access_token")) {
    throw StorageError(fmt::format("{}: oauth2 token response from '{}' is missing 'access_token'", kErrorPrefix,
                                   config_.oauth2_token_endpoint));
  }
  cached_oauth2_token_ = response.at("access_token").get<std::string>();
  const double expires_in_seconds = response.value("expires_in", 3600.0);
  oauth2_token_expiry_unix_seconds_ = now + expires_in_seconds;
  return cached_oauth2_token_;
}

std::string UnityCatalogClient::bearer_token_for_request() {
  if (config_.credentials_kind == "bearer_token") {
    return config_.bearer_token;
  }
  if (config_.credentials_kind == "oauth2_client_credentials") {
    return fetch_oauth2_token();
  }
  return "";
}

UnityCatalogTableInfo UnityCatalogClient::get_table(const std::string& catalog, const std::string& schema,
                                                    const std::string& table) {
  const std::string full_name = fmt::format("{}.{}.{}", catalog, schema, table);
  std::string encoded_full_name;
  {
    const CurlEasyPtr encoder = make_curl_easy();
    encoded_full_name = url_encode(encoder.get(), full_name, kErrorPrefix);
  }

  const std::string url = fmt::format("{}/tables/{}", config_.uc_url, encoded_full_name);
  const std::string bearer_token = bearer_token_for_request();
  std::vector<std::string> headers = {"Accept: application/json"};
  if (!bearer_token.empty()) {
    headers.push_back(fmt::format("Authorization: Bearer {}", bearer_token));
  }
  const nlohmann::json response =
      parse_json_response(url, http_request(url, headers, /*post_body=*/nullptr, kErrorPrefix));

  UnityCatalogTableInfo info;
  try {
    info.table_id = response.at("table_id").get<std::string>();
    info.table_type = response.at("table_type").get<std::string>();
    info.data_source_format = response.value("data_source_format", "");
    info.storage_location = response.at("storage_location").get<std::string>();
  } catch (const nlohmann::json::exception& e) {
    throw StorageError(
        fmt::format("{}: table info from '{}' is missing a required field: {}", kErrorPrefix, url, e.what()));
  }
  info.columns = parse_columns(response, url);
  return info;
}

UnityCatalogTemporaryCredentials UnityCatalogClient::get_temporary_table_credentials(
    const std::string& table_id, const std::string& operation) {
  const std::string url = fmt::format("{}/temporary-table-credentials", config_.uc_url);
  const nlohmann::json request_body = {{"table_id", table_id}, {"operation", operation}};
  const std::string body = request_body.dump();

  const std::string bearer_token = bearer_token_for_request();
  std::vector<std::string> headers = {"Accept: application/json", "Content-Type: application/json"};
  if (!bearer_token.empty()) {
    headers.push_back(fmt::format("Authorization: Bearer {}", bearer_token));
  }
  const nlohmann::json response =
      parse_json_response(url, http_request(url, headers, &body, kErrorPrefix));

  if (!response.contains("aws_temp_credentials")) {
    throw StorageError(fmt::format(
        "{}: temporary-table-credentials response from '{}' is missing 'aws_temp_credentials' -- only AWS "
        "S3 vended credentials are supported (see docs/ROADMAP.md)",
        kErrorPrefix, url));
  }
  const nlohmann::json& aws_json = response.at("aws_temp_credentials");

  UnityCatalogTemporaryCredentials credentials;
  try {
    credentials.access_key_id = aws_json.at("access_key_id").get<std::string>();
    credentials.secret_access_key = aws_json.at("secret_access_key").get<std::string>();
    credentials.session_token = aws_json.at("session_token").get<std::string>();
  } catch (const nlohmann::json::exception& e) {
    throw StorageError(fmt::format(
        "{}: 'aws_temp_credentials' from '{}' is missing a required field: {}", kErrorPrefix, url, e.what()));
  }
  return credentials;
}

}  // namespace kernellake::unitycatalog
