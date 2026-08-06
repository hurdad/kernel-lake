#include "kernellake/delta/delta_source_resolver.hpp"

#include <fmt/format.h>

#include <string>
#include <vector>

#include "kernellake/common/errors.hpp"
#include "kernellake/delta/delta_table_resolution.hpp"
#include "kernellake/delta/delta_txn_client.hpp"

namespace kernellake::delta {

namespace {
constexpr std::string_view kScheme = "delta";
}  // namespace

bool DeltaSourceResolver::can_resolve(const std::vector<std::string>& sources) const {
  return sources.size() == 1 && Uri(sources[0]).scheme() == kScheme;
}

ResolvedTable DeltaSourceResolver::resolve(ObjectStore& store, const std::vector<std::string>& sources) {
  const std::string& source = sources.at(0);
  const std::size_t scheme_end = source.find("://");
  const std::string table_uri = source.substr(scheme_end + 3);

  if (config_.grpc_endpoint.empty()) {
    throw ConfigurationError(
        "read_delta(...): no delta.grpc_endpoint is configured (see DeltaSection in config.hpp)");
  }

  // Constructed fresh per call rather than cached across queries -- same
  // deliberate MVP simplification as IcebergSourceResolver's own
  // per-call IcebergRestCatalogClient, at the cost of a new gRPC channel
  // per query. See docs/ROADMAP.md.
  DeltaTxnClient client(config_);
  return resolve_delta_table(store, client, table_uri);
}

}  // namespace kernellake::delta
