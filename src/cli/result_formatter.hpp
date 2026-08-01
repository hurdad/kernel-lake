#pragma once

#include <optional>
#include <string>

#include "kernellake/api/query_engine.hpp"

namespace kernellake::cli {

enum class ResultFormat {
  Table,  // Aligned terminal table.
  Csv,
  JsonLines,
  ArrowIpc,  // Arrow IPC (streaming format) file.
};

[[nodiscard]] std::optional<ResultFormat> parse_result_format(std::string_view name);

// Writes `result`'s batches to `output_path`, or stdout when std::nullopt.
// Throws kernellake::ExecutionError on any I/O or Arrow-side failure.
void write_query_result(const QueryResult& result, ResultFormat format,
                        const std::optional<std::string>& output_path);

}  // namespace kernellake::cli
