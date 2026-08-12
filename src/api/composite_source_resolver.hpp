#pragma once
// Private, src/-local header shared between query_engine.cpp (planning/
// explain) and query_engine_execute_stub.cpp/query_engine_execute_gpu.cpp
// (real execution) -- never installed, never included from
// include/kernellake/. Only kernellake_api needs it.
#include <vector>

#include "kernellake/io/table_resolution.hpp"

namespace kernellake {

// resolve_table_or_delegate() (kernellake/io/table_resolution.hpp) takes a
// single `extra_resolver` slot; this combines three TableSourceResolvers
// (IcebergSourceResolver, DeltaSourceResolver, UnityCatalogSourceResolver,
// today) behind that one slot so every source kind works in the same
// query/JOIN without any resolver needing to know the others exist. Safe
// because their can_resolve() checks are mutually exclusive (distinct
// Uri schemes -- "iceberg", "delta", "unitycatalog") -- first match wins,
// but there's never more than one real match for a given source.
class CompositeSourceResolver final : public TableSourceResolver {
 public:
  CompositeSourceResolver(TableSourceResolver& first, TableSourceResolver& second, TableSourceResolver& third)
      : first_(first), second_(second), third_(third) {}

  [[nodiscard]] bool can_resolve(const std::vector<std::string>& sources) const override {
    return first_.can_resolve(sources) || second_.can_resolve(sources) || third_.can_resolve(sources);
  }

  [[nodiscard]] ResolvedTable resolve(ObjectStore& store, const std::vector<std::string>& sources,
                                      const std::vector<PushablePredicate>& predicates) override {
    if (first_.can_resolve(sources)) {
      return first_.resolve(store, sources, predicates);
    }
    if (second_.can_resolve(sources)) {
      return second_.resolve(store, sources, predicates);
    }
    return third_.resolve(store, sources, predicates);
  }

 private:
  TableSourceResolver& first_;
  TableSourceResolver& second_;
  TableSourceResolver& third_;
};

}  // namespace kernellake
