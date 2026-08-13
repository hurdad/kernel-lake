#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "kernellake/api/query_engine.hpp"

namespace kernellake::cli {

// This header only became clang-tidy-visible once it moved from src/cli/
// (outside .clang-tidy's HeaderFilterRegex) to include/kernellake/cli/
// (inside it) for testability -- see src/cli/CMakeLists.txt's own comment
// on the kernellake_cli split. std::uint8_t is plenty for four enumerators.
enum class ResultFormat : std::uint8_t {
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
