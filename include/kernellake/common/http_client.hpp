#pragma once

#include <curl/curl.h>

#include <memory>
#include <string>
#include <vector>

namespace kernellake {

// RAII wrapper for a CURL* easy handle -- curl_easy_cleanup() on destruction.
struct CurlEasyDeleter {
  void operator()(CURL* handle) const noexcept;
};
using CurlEasyPtr = std::unique_ptr<CURL, CurlEasyDeleter>;

// RAII wrapper for a curl_slist* header list -- curl_slist_free_all() on
// destruction.
struct CurlSlistDeleter {
  void operator()(curl_slist* list) const noexcept;
};
using CurlSlistPtr = std::unique_ptr<curl_slist, CurlSlistDeleter>;

// Ensures curl_global_init() has run exactly once for this process, however
// many callers (across however many REST/HTTP clients) construct a handle.
// Idempotent to call repeatedly. Intentionally never paired with
// curl_global_cleanup(): that would need to run only after every other
// thread has stopped using libcurl, which no caller here has any way to
// know -- leaving it for process-exit teardown is fine for a
// process-lifetime global like this.
void ensure_curl_global_init();

// Constructs a fresh CURL easy handle, calling ensure_curl_global_init()
// first. Throws StorageError if curl_easy_init() fails.
[[nodiscard]] CurlEasyPtr make_curl_easy();

// URL-percent-encodes `value` using `handle`'s own escaping context. Throws
// StorageError (prefixed with `error_prefix`) if curl_easy_escape() fails.
[[nodiscard]] std::string url_encode(CURL* handle, const std::string& value, const std::string& error_prefix);

// GET when post_body is null, POST otherwise. `headers` are sent verbatim
// (e.g. "Accept: application/json", "Authorization: Bearer ...",
// "Content-Type: application/json") -- callers pick exactly which headers a
// given request needs, since different REST APIs disagree on POST body
// encoding (the Iceberg REST spec's OAuth2 token endpoint is form-encoded;
// Unity Catalog's JSON endpoints are not). `error_prefix` (e.g. "iceberg
// rest catalog", "unity catalog") is prepended to every thrown
// StorageError's message so a caller's errors stay identifiable regardless
// of which REST client produced them. Throws StorageError on any transport
// failure or non-2xx HTTP response.
[[nodiscard]] std::string http_request(const std::string& url, const std::vector<std::string>& headers,
                                       const std::string* post_body, const std::string& error_prefix);

}  // namespace kernellake
