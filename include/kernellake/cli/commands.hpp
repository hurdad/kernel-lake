#pragma once

#include <string_view>
#include <vector>

#include "kernellake/common/config.hpp"

namespace kernellake::cli {

// Each subcommand parses its own `--flag value` style arguments (excluding
// the subcommand name itself) and returns a process exit code. Errors are
// reported to stderr with a "kernellake <command>: " prefix; nothing is
// printed to stdout on failure.
int run_inspect_parquet(const std::vector<std::string_view>& args, const EngineConfig& config);

int run_explain(const std::vector<std::string_view>& args, const EngineConfig& config);

int run_query(const std::vector<std::string_view>& args, const EngineConfig& config);

int run_generate_data(const std::vector<std::string_view>& args);

int run_benchmark_tpch(const std::vector<std::string_view>& args, const EngineConfig& config);

}  // namespace kernellake::cli
