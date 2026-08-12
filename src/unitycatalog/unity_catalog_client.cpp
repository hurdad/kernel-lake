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

// Like `json.value(key, "")`, but also treats an explicitly-present `null`
// as absent -- `.value()` alone only covers the key being missing
// entirely, and throws a type_error trying to convert `null` to
// std::string otherwise. Confirmed against a real unitycatalog/unitycatalog
// server, not assumed: "next_page_token" (the field this matters most
// for -- every list_*() method's pagination loop termination depends on
// it) comes back as an explicit `null`, not an absent key, once the last
// page is reached.
std::string string_or_empty(const nlohmann::json& json, const char* key) {
  if (!json.contains(key) || json.at(key).is_null()) {
    return "";
  }
  return json.at(key).get<std::string>();
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
      column.type_name = string_or_empty(column_json, "type_name");
      // A struct/array/map column's "type_json" is itself a JSON-encoded
      // string in UC's response (unlike Iceberg's raw nested-object "type"
      // -- see IcebergSchemaField's own comment for that different shape),
      // so it's always already a plain string here, captured verbatim.
      column.type_json = string_or_empty(column_json, "type_json");
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

// Shared by get_table() (a single TableInfo object) and list_tables() (one
// element of a "tables" array) -- both are the exact same JSON shape.
UnityCatalogTableInfo parse_table_info(const nlohmann::json& table_json, const std::string& url) {
  UnityCatalogTableInfo info;
  try {
    info.table_id = table_json.at("table_id").get<std::string>();
    info.table_type = table_json.at("table_type").get<std::string>();
    info.data_source_format = string_or_empty(table_json, "data_source_format");
    info.storage_location = table_json.at("storage_location").get<std::string>();
  } catch (const nlohmann::json::exception& e) {
    throw StorageError(
        fmt::format("{}: table info from '{}' is missing a required field: {}", kErrorPrefix, url, e.what()));
  }
  info.columns = parse_columns(table_json, url);
  return info;
}

}  // namespace

UnityCatalogClient::UnityCatalogClient(UnityCatalogInstanceSection config, const UnityCatalogTokenCache* token_cache)
    : config_(std::move(config)), token_cache_(token_cache) {}

std::string UnityCatalogClient::fetch_oauth2_token() {
  const double now =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  // 30s safety margin so a cached token doesn't expire mid-flight between
  // this check and the request it's about to authorize -- same margin
  // IcebergRestCatalogClient::fetch_oauth2_token() uses.
  if (!cached_oauth2_token_.empty() && now < oauth2_token_expiry_unix_seconds_ - 30.0) {
    return cached_oauth2_token_;
  }

  // This instance's own local cache (above) is empty or stale -- before
  // making a real network round trip, check the caller-shared cache (if
  // any), which may already hold a still-valid token fetched by a
  // *different* UnityCatalogClient instance (a previous resolve() call,
  // possibly for a previous query entirely -- see
  // UnityCatalogTokenCache's own class comment for why this exists).
  // config_.uc_url uniquely identifies which configured instance this is,
  // the same identity UnityCatalogSection::instances is itself keyed by.
  if (token_cache_ != nullptr) {
    if (const std::optional<std::string> shared = token_cache_->try_get(config_.uc_url, now)) {
      cached_oauth2_token_ = *shared;
      // oauth2_token_expiry_unix_seconds_ deliberately left at its default
      // (0.0): this instance doesn't know the real expiry the token was
      // originally fetched with, only that the shared cache says it's
      // still valid *right now*. Leaving it at 0.0 means this instance's
      // own fast-path check above always treats it as stale on any next
      // call, falling back to re-checking the (cheap, mutex-guarded)
      // shared cache rather than risking a use-past-expiry from a stale
      // local guess.
      return cached_oauth2_token_;
    }
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
  if (token_cache_ != nullptr) {
    // Margin-adjusted the same way this instance's own fast-path check
    // above is (see that comment), so a future UnityCatalogClient reading
    // this shared entry never treats a token as valid right up to the
    // literal expiry instant the server gave.
    token_cache_->store(config_.uc_url, cached_oauth2_token_, oauth2_token_expiry_unix_seconds_ - 30.0);
  }
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

nlohmann::json UnityCatalogClient::authenticated_get_json(const std::string& url) {
  const std::string bearer_token = bearer_token_for_request();
  std::vector<std::string> headers = {"Accept: application/json"};
  if (!bearer_token.empty()) {
    headers.push_back(fmt::format("Authorization: Bearer {}", bearer_token));
  }
  return parse_json_response(url, http_request(url, headers, /*post_body=*/nullptr, kErrorPrefix));
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
  return parse_table_info(authenticated_get_json(url), url);
}

std::vector<UnityCatalogCatalogInfo> UnityCatalogClient::list_catalogs() {
  std::vector<UnityCatalogCatalogInfo> catalogs;
  std::string page_token;
  do {
    std::string url = fmt::format("{}/catalogs", config_.uc_url);
    if (!page_token.empty()) {
      const CurlEasyPtr encoder = make_curl_easy();
      url += fmt::format("?page_token={}", url_encode(encoder.get(), page_token, kErrorPrefix));
    }
    const nlohmann::json response = authenticated_get_json(url);
    if (!response.contains("catalogs")) {
      throw StorageError(fmt::format("{}: list response from '{}' is missing 'catalogs'", kErrorPrefix, url));
    }
    for (const nlohmann::json& catalog_json : response.at("catalogs")) {
      try {
        UnityCatalogCatalogInfo info;
        info.name = catalog_json.at("name").get<std::string>();
        info.comment = string_or_empty(catalog_json, "comment");
        catalogs.push_back(std::move(info));
      } catch (const nlohmann::json::exception& e) {
        throw StorageError(fmt::format("{}: a 'catalogs' entry from '{}' is missing a required field: {}",
                                       kErrorPrefix, url, e.what()));
      }
    }
    page_token = string_or_empty(response, "next_page_token");
  } while (!page_token.empty());
  return catalogs;
}

std::vector<UnityCatalogSchemaInfo> UnityCatalogClient::list_schemas(const std::string& catalog) {
  std::vector<UnityCatalogSchemaInfo> schemas;
  std::string page_token;
  do {
    std::string url;
    {
      const CurlEasyPtr encoder = make_curl_easy();
      url = fmt::format("{}/schemas?catalog_name={}", config_.uc_url,
                        url_encode(encoder.get(), catalog, kErrorPrefix));
      if (!page_token.empty()) {
        url += fmt::format("&page_token={}", url_encode(encoder.get(), page_token, kErrorPrefix));
      }
    }
    const nlohmann::json response = authenticated_get_json(url);
    if (!response.contains("schemas")) {
      throw StorageError(fmt::format("{}: list response from '{}' is missing 'schemas'", kErrorPrefix, url));
    }
    for (const nlohmann::json& schema_json : response.at("schemas")) {
      try {
        UnityCatalogSchemaInfo info;
        info.name = schema_json.at("name").get<std::string>();
        info.catalog_name = string_or_empty(schema_json, "catalog_name");
        info.full_name = string_or_empty(schema_json, "full_name");
        schemas.push_back(std::move(info));
      } catch (const nlohmann::json::exception& e) {
        throw StorageError(fmt::format("{}: a 'schemas' entry from '{}' is missing a required field: {}",
                                       kErrorPrefix, url, e.what()));
      }
    }
    page_token = string_or_empty(response, "next_page_token");
  } while (!page_token.empty());
  return schemas;
}

std::vector<UnityCatalogTableInfo> UnityCatalogClient::list_tables(const std::string& catalog,
                                                                    const std::string& schema) {
  std::vector<UnityCatalogTableInfo> tables;
  std::string page_token;
  do {
    std::string url;
    {
      const CurlEasyPtr encoder = make_curl_easy();
      url = fmt::format("{}/tables?catalog_name={}&schema_name={}", config_.uc_url,
                        url_encode(encoder.get(), catalog, kErrorPrefix),
                        url_encode(encoder.get(), schema, kErrorPrefix));
      if (!page_token.empty()) {
        url += fmt::format("&page_token={}", url_encode(encoder.get(), page_token, kErrorPrefix));
      }
    }
    const nlohmann::json response = authenticated_get_json(url);
    if (!response.contains("tables")) {
      throw StorageError(fmt::format("{}: list response from '{}' is missing 'tables'", kErrorPrefix, url));
    }
    for (const nlohmann::json& table_json : response.at("tables")) {
      tables.push_back(parse_table_info(table_json, url));
    }
    page_token = string_or_empty(response, "next_page_token");
  } while (!page_token.empty());
  return tables;
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

  UnityCatalogTemporaryCredentials credentials;
  if (response.contains("aws_temp_credentials") && !response.at("aws_temp_credentials").is_null()) {
    const nlohmann::json& aws_json = response.at("aws_temp_credentials");
    try {
      credentials.access_key_id = aws_json.at("access_key_id").get<std::string>();
      credentials.secret_access_key = aws_json.at("secret_access_key").get<std::string>();
      credentials.session_token = aws_json.at("session_token").get<std::string>();
    } catch (const nlohmann::json::exception& e) {
      throw StorageError(fmt::format("{}: 'aws_temp_credentials' from '{}' is missing a required field: {}",
                                     kErrorPrefix, url, e.what()));
    }
    return credentials;
  }
  if (response.contains("gcp_oauth_token") && !response.at("gcp_oauth_token").is_null()) {
    try {
      credentials.gcp_oauth_token = response.at("gcp_oauth_token").at("oauth_token").get<std::string>();
    } catch (const nlohmann::json::exception& e) {
      throw StorageError(fmt::format("{}: 'gcp_oauth_token' from '{}' is missing a required field: {}",
                                     kErrorPrefix, url, e.what()));
    }
    return credentials;
  }
  if (response.contains("azure_user_delegation_sas") && !response.at("azure_user_delegation_sas").is_null()) {
    try {
      credentials.azure_sas_token =
          response.at("azure_user_delegation_sas").at("sas_token").get<std::string>();
    } catch (const nlohmann::json::exception& e) {
      throw StorageError(fmt::format(
          "{}: 'azure_user_delegation_sas' from '{}' is missing a required field: {}", kErrorPrefix, url,
          e.what()));
    }
    return credentials;
  }
  throw StorageError(fmt::format(
      "{}: temporary-table-credentials response from '{}' has none of 'aws_temp_credentials'/"
      "'gcp_oauth_token'/'azure_user_delegation_sas' -- unrecognized or unsupported cloud",
      kErrorPrefix, url));
}

}  // namespace kernellake::unitycatalog
