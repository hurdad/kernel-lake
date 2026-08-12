#include "kernellake/common/http_client.hpp"

#include <fmt/format.h>

#include <mutex>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {
std::once_flag curl_global_init_flag;

size_t write_to_string(char* data, size_t size, size_t nmemb, void* user_data) {
  auto* out = static_cast<std::string*>(user_data);
  out->append(data, size * nmemb);
  return size * nmemb;
}
}  // namespace

void CurlEasyDeleter::operator()(CURL* handle) const noexcept {
  curl_easy_cleanup(handle);
}

void CurlSlistDeleter::operator()(curl_slist* list) const noexcept {
  curl_slist_free_all(list);
}

void ensure_curl_global_init() {
  std::call_once(curl_global_init_flag, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

CurlEasyPtr make_curl_easy() {
  ensure_curl_global_init();
  CurlEasyPtr handle(curl_easy_init());
  if (!handle) {
    throw StorageError("curl_easy_init() failed");
  }
  return handle;
}

std::string url_encode(CURL* handle, const std::string& value, const std::string& error_prefix) {
  char* escaped = curl_easy_escape(handle, value.c_str(), static_cast<int>(value.size()));
  if (escaped == nullptr) {
    throw StorageError(fmt::format("{}: failed to URL-encode '{}'", error_prefix, value));
  }
  std::string result(escaped);
  curl_free(escaped);
  return result;
}

std::string http_request(const std::string& url, const std::vector<std::string>& headers,
                         const std::string* post_body, const std::string& error_prefix) {
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

  curl_slist* raw_headers = nullptr;
  for (const std::string& header : headers) {
    raw_headers = curl_slist_append(raw_headers, header.c_str());
  }
  if (post_body != nullptr) {
    curl_easy_setopt(handle.get(), CURLOPT_POSTFIELDS, post_body->c_str());
    curl_easy_setopt(handle.get(), CURLOPT_POSTFIELDSIZE, static_cast<long>(post_body->size()));
  }
  const CurlSlistPtr owned_headers(raw_headers);
  if (raw_headers != nullptr) {
    curl_easy_setopt(handle.get(), CURLOPT_HTTPHEADER, raw_headers);
  }

  const CURLcode result = curl_easy_perform(handle.get());
  if (result != CURLE_OK) {
    throw StorageError(
        fmt::format("{}: request to '{}' failed: {}", error_prefix, url, curl_easy_strerror(result)));
  }

  long status_code = 0;
  curl_easy_getinfo(handle.get(), CURLINFO_RESPONSE_CODE, &status_code);
  if (status_code < 200 || status_code >= 300) {
    throw StorageError(fmt::format("{}: request to '{}' returned HTTP {}: {}", error_prefix, url, status_code,
                                   response_body));
  }
  return response_body;
}

}  // namespace kernellake
