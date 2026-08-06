#include "kernellake/iceberg/iceberg_source_resolver.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <string>
#include <vector>

#include "kernellake/common/errors.hpp"
#include "kernellake/iceberg/iceberg_table_resolution.hpp"
#include "kernellake/iceberg/rest_catalog_client.hpp"

namespace kernellake::iceberg {

namespace {

constexpr std::string_view kScheme = "iceberg";

struct QualifiedName {
  std::string catalog;
  std::vector<std::string> namespace_parts;
  std::string table;
};

// "catalog.ns1.ns2....table" -> {catalog, {ns1, ns2, ...}, table}. Requires
// at least 3 non-empty, dot-separated parts (a catalog name, one or more
// namespace levels, and a table name) -- an Iceberg namespace is never
// empty, so "catalog.table" (2 parts) is rejected outright rather than
// guessed at as an empty namespace.
QualifiedName parse_qualified_name(const std::string& text) {
  std::vector<std::string> parts;
  std::string current;
  for (const char c : text) {
    if (c == '.') {
      parts.push_back(current);
      current.clear();
    } else {
      current += c;
    }
  }
  parts.push_back(current);

  const bool has_empty_part = std::any_of(parts.begin(), parts.end(), [](const std::string& p) { return p.empty(); });
  if (parts.size() < 3 || has_empty_part) {
    throw StorageError(fmt::format(
        "read_iceberg(...): '{}' isn't a valid catalog.namespace.table reference (need at least 3 "
        "non-empty, dot-separated parts: a catalog name, one or more namespace levels, and a table name)",
        text));
  }
  QualifiedName name;
  name.catalog = parts.front();
  name.table = parts.back();
  name.namespace_parts.assign(parts.begin() + 1, parts.end() - 1);
  return name;
}

}  // namespace

bool IcebergSourceResolver::can_resolve(const std::vector<std::string>& sources) const {
  return sources.size() == 1 && Uri(sources[0]).scheme() == kScheme;
}

ResolvedTable IcebergSourceResolver::resolve(ObjectStore& store, const std::vector<std::string>& sources) {
  const std::string& source = sources.at(0);
  const std::size_t scheme_end = source.find("://");
  const QualifiedName name = parse_qualified_name(source.substr(scheme_end + 3));

  const auto it = catalogs_.catalogs.find(name.catalog);
  if (it == catalogs_.catalogs.end()) {
    throw ConfigurationError(
        fmt::format("read_iceberg(...): no iceberg.catalogs entry named '{}' is configured", name.catalog));
  }

  // Constructed fresh per call rather than cached across queries -- simpler
  // and correct, at the cost of repeating the oauth2_client_credentials
  // handshake (when that's the configured credentials_kind) on every query
  // against this catalog. A deliberate MVP simplification, not a permanent
  // design decision -- see docs/ROADMAP.md.
  IcebergRestCatalogClient client(it->second);
  return resolve_iceberg_table(store, client, name.namespace_parts, name.table);
}

}  // namespace kernellake::iceberg
