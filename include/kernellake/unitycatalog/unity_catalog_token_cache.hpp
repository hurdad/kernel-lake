#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace kernellake::unitycatalog {

// A thread-safe cache of OAuth2 access tokens, keyed by caller-chosen
// string (UnityCatalogClient uses one configured instance's `uc_url`, see
// its own fetch_oauth2_token()) so a query engine can share one real token
// per Unity Catalog instance across every UnityCatalogClient constructed
// during its lifetime -- not just within one client's own local
// cached_oauth2_token_/oauth2_token_expiry_unix_seconds_, which only helps
// the get_table()+get_temporary_table_credentials() pair inside a single
// resolve() call, since a fresh UnityCatalogClient (and thus a fresh local
// cache) is otherwise constructed on every resolve() call and every
// separate query (see UnityCatalogSourceResolver's own comment on why:
// the same deliberate MVP simplification IcebergSourceResolver's
// IcebergRestCatalogClient and DeltaSourceResolver's DeltaTxnClient still
// use, unaffected by this cache).
//
// Deliberately narrow: caches only the OAuth2 access token itself (a
// short string plus its own expiry), never a whole UnityCatalogClient or
// its table-lookup results -- get_table()/list_*() must always hit the
// real server to reflect current catalog state, only the *authentication*
// step is worth skipping when the previous token is still valid.
//
// Every public method is const and internally synchronized (mutable
// mutex_/entries_) so this can be a plain (non-mutable) QueryEngine member
// safely touched from QueryEngine's const methods -- the same "safe
// shared mutable state behind its own synchronization" pattern
// QueryEngine's own `mutable ObjectStoreRegistry store_` member already
// relies on (see query_engine.hpp). This keeps explain()'s documented
// "touches no shared mutable state that isn't already safe under
// concurrency" reentrancy claim (docs/ARCHITECTURE.md's Concurrency
// notes) true in spirit: every access is mutex-guarded, so concurrent
// callers (e.g. the Flight SQL server's concurrently-planned queries) see
// no data race, only real cache sharing.
class UnityCatalogTokenCache {
 public:
  // std::nullopt if `key` has no entry, or its entry's expiry is at or
  // past `now_unix_seconds` (the caller's own margin already applied --
  // see UnityCatalogClient::fetch_oauth2_token()'s identical 30s-margin
  // comment; this cache stores whatever expiry the caller already
  // margin-adjusted, rather than re-deriving its own).
  [[nodiscard]] std::optional<std::string> try_get(const std::string& key, double now_unix_seconds) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(key);
    if (it == entries_.end() || now_unix_seconds >= it->second.expiry_unix_seconds) {
      return std::nullopt;
    }
    return it->second.token;
  }

  void store(const std::string& key, std::string token, double expiry_unix_seconds) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    entries_[key] = Entry{std::move(token), expiry_unix_seconds};
  }

 private:
  struct Entry {
    std::string token;
    double expiry_unix_seconds;
  };

  mutable std::mutex mutex_;
  mutable std::unordered_map<std::string, Entry> entries_;
};

}  // namespace kernellake::unitycatalog
