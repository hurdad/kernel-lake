#pragma once

#include <string>
#include <string_view>
#include <unordered_map>

namespace kernellake {

// Accumulates per-operator wall-clock time across a single query's
// execution, keyed by PhysicalOperator::name() (see operator_builder.cpp's
// InstrumentedOperator, which populates this generically for every node in
// the tree -- no individual operator needs its own instrumentation code).
//
// Each recorded total is *inclusive* of that operator's children: since
// InstrumentedOperator times a plain wall-clock wrapper around next(), and
// a parent's next() call (e.g. FilterOperator's) naturally invokes its
// child's already-separately-instrumented next() internally, a leaf
// operator's (e.g. ParquetScanOperator's) total is its true self time, but
// a non-leaf operator's total also contains everything below it. This is
// simple flame-graph-style accounting, not exclusive self-time -- documented
// here so a caller doesn't sum every operator's total expecting it to equal
// the whole query's execution time (it will double-count).
//
// One instance per query (constructed fresh per QueryEngine::execute()
// call), not process-wide -- matches ExecutionContext's own "no
// process-wide current-query state" design. Not thread-safe: today's
// execution model runs exactly one operator tree at a time per
// ExecutionContext (see docs/ARCHITECTURE.md's Concurrency notes), so no
// locking is needed; revisit if that changes.
class MetricsRegistry {
 public:
  void record(std::string_view operator_name, double seconds) {
    totals_[std::string(operator_name)] += seconds;
  }

  [[nodiscard]] double total_seconds(std::string_view operator_name) const {
    const auto it = totals_.find(std::string(operator_name));
    return it != totals_.end() ? it->second : 0.0;
  }

  [[nodiscard]] const std::unordered_map<std::string, double>& per_operator_seconds() const noexcept {
    return totals_;
  }

 private:
  std::unordered_map<std::string, double> totals_;
};

}  // namespace kernellake
