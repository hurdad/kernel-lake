#pragma once

#include <cstdint>
#include <string>

#include "kernellake/types/schema.hpp"

namespace kernellake {

// Reserved for future table-format phases (Iceberg's richer partition
// transforms -- bucket/truncate/year/month/day, see docs/ROADMAP.md's
// lakehouse roadmap); Hive-style discovery (kernellake/io/table_resolution.hpp)
// only ever produces Identity, since a raw `key=value` path segment has no
// transform of its own.
enum class PartitionTransform : std::uint8_t {
  Identity,
};

// A column whose values are derived from a file's location (a Hive-style
// `key=value` directory segment today; an Iceberg partition spec entry in a
// later phase) rather than being physically present in the file itself.
// Shared by kernellake::LogicalScan (kernellake/planner/logical_plan.hpp) and
// kernellake::ParquetScanNode/PhysicalFileFragment
// (kernellake/planner/physical_plan.hpp) -- kept in kernellake_types (which
// both kernellake_planner and kernellake_io already depend on) rather than
// kernellake_io itself, since kernellake_io depends on kernellake_planner
// and a reverse dependency here would be circular.
struct PartitionColumn {
  std::string name;
  DataType type;
  PartitionTransform transform = PartitionTransform::Identity;
};

}  // namespace kernellake
