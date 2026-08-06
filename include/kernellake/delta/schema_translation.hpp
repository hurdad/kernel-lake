#pragma once

#include <string>

#include "kernellake/types/schema.hpp"

namespace kernellake::delta {

// Translates a Delta table's `schema_string` (DeltaTableInfo::schema_string,
// see delta_txn_client.hpp) -- Delta's own JSON schema encoding, a
// Spark-schema-compatible top-level {"type":"struct","fields":[...]}
// object -- into a kernellake::Schema, in the same field order the JSON
// itself lists. Includes every column, partition and physical alike,
// exactly as Delta's own schema does -- see delta_table_resolution.hpp for
// how partition columns are subsequently split out from the physical-file
// schema used to validate each Parquet file's own footer.
//
// Supported Delta primitive types: boolean, byte/short/integer (all widen
// to kernellake's single Int32 -- this codebase has no narrower integer
// type of its own), long, float, double, date, timestamp, timestamp_ntz
// (both map to kernellake's single Timestamp type, matching how
// kernellake::iceberg::iceberg_schema_to_kernellake_schema already
// collapses timestamp/timestamptz), string, and decimal(P,S). Everything
// else -- binary, void, and every nested type (struct, array, map) --
// throws StorageError rather than guessing at a lossy mapping, the same
// "explicit errors over silent partial behavior" rule the Iceberg
// translation follows; support is deferred until something actually needs
// to read a column of one of those types.
//
// A field's own JSON `nullable` key maps directly to DataType::nullable
// (unlike Iceberg's inverted `required` key).
[[nodiscard]] Schema delta_schema_to_kernellake_schema(const std::string& schema_json);

}  // namespace kernellake::delta
