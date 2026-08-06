#pragma once

#include <vector>

#include "kernellake/iceberg/manifest_reader.hpp"
#include "kernellake/iceberg/rest_catalog_client.hpp"
#include "kernellake/planner/logical_plan.hpp"

namespace kernellake::iceberg {

// Returns true if `predicates` prove that no row in a data file with these
// partition values (as recorded in its own manifest entry --
// ManifestDataFileEntry::partition_values, positionally matching
// `spec.fields`) could possibly match -- i.e. the file can be skipped
// entirely, without ever calling inspect_parquet_file() on it. This is the
// file-level analog of kernellake::evaluate_pruning()'s row-group
// min/max pruning (kernellake/io/parquet_pruning.hpp), applied one level
// earlier in the pipeline.
//
// Only ever returns true when a predicate can be *proven* impossible;
// missing spec/field/value data, an unrecognized transform, or an
// incomparable literal always falls back to false ("must scan") --
// correctness over aggressive pruning, same rule row-group pruning
// follows.
//
// Supported transforms: identity, year, month, day, hour -- all monotonic
// (non-decreasing) functions of their source column, so both equality and
// range predicates (=, <, <=, >, >=) prune correctly by comparing the
// file's own (already-transformed) partition value against
// transform(literal) as if it were a degenerate single-point [V, V] range,
// reusing the same per-operator logic evaluate_pruning() uses for a real
// [min, max] range. `!=` only prunes for `identity`: a coarsening
// transform's partition value being equal to transform(literal) doesn't
// mean every row's *source* value equals literal (e.g. a `day`-partitioned
// timestamp column has many distinct timestamps sharing one partition
// value), so `!=` can't be proven empty from the transform value alone --
// only `identity` (no coarsening: partition value *is* the source value)
// preserves that guarantee.
//
// `bucket[N]` and `truncate[W]` are not evaluated at all (a predicate
// against either always falls through to "must scan"): unlike the four
// transforms above, correctly computing Iceberg's bucket hash (a specific
// murmur3 variant over a type-specific byte encoding) or truncate's
// per-type truncation rules needs its own careful, independently-verified
// implementation -- getting either wrong would silently skip files that
// still contain matching rows, a real correctness bug, not just a missed
// optimization. Left as an explicit, documented non-goal until that
// verification work happens (see docs/ROADMAP.md).
[[nodiscard]] bool partition_values_prove_empty(const IcebergTableMetadata& table_metadata,
                                                const IcebergPartitionSpec& spec,
                                                const std::vector<PartitionFieldValue>& partition_values,
                                                const std::vector<PushablePredicate>& predicates);

}  // namespace kernellake::iceberg
