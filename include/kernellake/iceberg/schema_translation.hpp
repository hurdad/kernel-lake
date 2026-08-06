#pragma once

#include "kernellake/iceberg/rest_catalog_client.hpp"
#include "kernellake/types/schema.hpp"

namespace kernellake::iceberg {

// Translates an Iceberg table schema's fields (as returned by
// IcebergRestCatalogClient::load_table_metadata(), in column order) into a
// kernellake::Schema. Iceberg's `required` (true) becomes DataType's
// `nullable = false`; every other field is nullable.
//
// Supported Iceberg primitive types: boolean, int, long, float, double,
// date, timestamp, timestamptz (both map to kernellake's single Timestamp
// type, which -- like the rest of this codebase's type system -- has no
// timezone-aware/naive distinction of its own), string, and
// decimal(P,S). Everything else -- time, uuid, fixed[N], binary, and every
// nested type (list, map, struct) -- throws StorageError rather than
// guessing at a lossy mapping, matching this project's "explicit errors
// over silent partial behavior" rule; support is deferred until something
// actually needs to read a column of one of those types.
[[nodiscard]] Schema iceberg_schema_to_kernellake_schema(const std::vector<IcebergSchemaField>& fields);

}  // namespace kernellake::iceberg
