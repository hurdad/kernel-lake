#include <spdlog/spdlog.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "commands.hpp"
#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/common/logging.hpp"
#include "kernellake/common/version.hpp"

namespace {

void print_usage() {
  std::printf(
      "kernellake - GPU-native analytics for the open lakehouse\n\n"
      "Usage:\n"
      "  kernellake --version\n"
      "  kernellake --help\n"
      "  kernellake [--config <path>] <command> ...\n\n"
      "Commands:\n"
      "  inspect-parquet --path <path> [--format text|json]\n"
      "  explain --sql <sql> [--format text|json] [--logical]\n"
      "  query (--sql <sql> | --file <path>) [--format table|csv|jsonl|arrow]\n"
      "         [--output <path>] [--stats]\n"
      "  generate-data --output <dir> --rows <n> [--files <n>] [--row-group-rows <n>]\n"
      "         [--region-cardinality <n>] [--category-cardinality <n>]\n"
      "         [--customer-cardinality <n>] [--null-rate <0..1>] [--skew <n>]\n"
      "         [--seed <n>] [--no-dictionary-encoding]\n"
      "  benchmark tpch --data <glob> --query <n> --mode cold|warm\n"
      "         [--scale-factor <n>] [--iterations <n>] [--warmup-iterations <n>]\n"
      "         [--query-file <path>] [--output <path.json>]\n"
      "\n"
      "TPC-H validation (`kernellake validate tpch` in the spec) is a Python tool, not a\n"
      "CLI subcommand -- see tools/validate_tpch.py.\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string_view> args(argv + 1, argv + argc);

  if (args.empty() || args[0] == "--help" || args[0] == "-h") {
    print_usage();
    return args.empty() ? 1 : 0;
  }

  if (args[0] == "--version") {
    std::printf("%s %s\n", kernellake::kProjectName, kernellake::kVersionString);
    return 0;
  }

  std::string config_path = "config/kernellake.yaml";
  bool explicit_config = false;
  std::size_t command_index = 0;
  if (args[0] == "--config") {
    if (args.size() < 2) {
      std::fprintf(stderr, "kernellake: --config requires a path argument\n");
      return 1;
    }
    config_path = args[1];
    explicit_config = true;
    command_index = 2;
  }

  if (command_index >= args.size()) {
    print_usage();
    return 1;
  }

  kernellake::EngineConfig config;
  try {
    if (explicit_config || std::filesystem::exists(config_path)) {
      config = kernellake::load_config_file(config_path);
    } else {
      config = kernellake::default_config();
    }
    kernellake::validate_config(config);
    kernellake::init_logging(config.logging);
  } catch (const kernellake::ConfigurationError& e) {
    std::fprintf(stderr, "kernellake: configuration error: %s\n", e.what());
    return 1;
  }

  const std::string_view command = args[command_index];
  const std::vector<std::string_view> command_args(args.begin() + static_cast<long>(command_index) + 1,
                                                   args.end());

  spdlog::info("kernellake starting: command='{}' config='{}'", command, config_path);

  if (command == "inspect-parquet") {
    return kernellake::cli::run_inspect_parquet(command_args);
  }
  if (command == "explain") {
    return kernellake::cli::run_explain(command_args, config);
  }
  if (command == "query") {
    return kernellake::cli::run_query(command_args, config);
  }
  if (command == "generate-data") {
    return kernellake::cli::run_generate_data(command_args);
  }
  if (command == "benchmark" && !command_args.empty() && command_args[0] == "tpch") {
    return kernellake::cli::run_benchmark_tpch(
        std::vector<std::string_view>(command_args.begin() + 1, command_args.end()), config);
  }

  std::fprintf(stderr, "kernellake: command '%.*s' is not yet implemented\n",
               static_cast<int>(command.size()), command.data());
  print_usage();
  return 1;
}
