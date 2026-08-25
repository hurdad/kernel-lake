#include "kernellake/cli/commands.hpp"

#include <fmt/format.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/observability/query_tracing.hpp"
#include "kernellake/cli/result_formatter.hpp"

namespace kernellake::cli {

namespace {

[[nodiscard]] std::string read_file_or_throw(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw ExecutionError(fmt::format("failed to open SQL file '{}'", path));
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

void print_optional(const char* label, const std::optional<std::int64_t>& value) {
  if (value) {
    std::fprintf(stderr, "  %s: %lld\n", label, static_cast<long long>(*value));
  }
}

void print_optional(const char* label, const std::optional<double>& value) {
  if (value) {
    std::fprintf(stderr, "  %s: %.6f\n", label, *value);
  }
}

void print_stats(const QueryResult& result, const std::optional<NvmeCacheMetricsSnapshot>& cache) {
  std::fprintf(stderr, "query stats:\n");
  print_optional("rows_returned", result.rows_returned);
  print_optional("files_considered", result.files_considered);
  print_optional("files_scanned", result.files_scanned);
  print_optional("row_groups_considered", result.row_groups_considered);
  print_optional("row_groups_scanned", result.row_groups_scanned);
  print_optional("peak_gpu_memory_bytes", result.peak_gpu_memory_bytes);
  print_optional("metadata_inspection_seconds", result.metadata_inspection_seconds);
  print_optional("parquet_decoding_seconds", result.parquet_decoding_seconds);
  print_optional("gpu_execution_seconds", result.gpu_execution_seconds);
  print_optional("cpu_execution_seconds", result.cpu_execution_seconds);
  print_optional("host_to_device_seconds", result.host_to_device_seconds);
  print_optional("device_to_host_seconds", result.device_to_host_seconds);
  print_optional("elapsed_wall_seconds", result.elapsed_wall_seconds);
  // Cumulative since this process/QueryEngine was constructed, not scoped
  // to just this one query -- see NvmeObjectCache::snapshot()'s own
  // comment. Since `kernellake query` is a one-query-per-process CLI
  // invocation, that's the same thing here in practice.
  if (cache) {
    std::fprintf(stderr, "  cache_hits: %llu\n", static_cast<unsigned long long>(cache->hits));
    std::fprintf(stderr, "  cache_misses: %llu\n", static_cast<unsigned long long>(cache->misses));
    std::fprintf(stderr, "  cache_evictions: %llu\n", static_cast<unsigned long long>(cache->evictions));
    std::fprintf(stderr, "  cache_current_bytes: %llu\n",
                 static_cast<unsigned long long>(cache->current_bytes));
    std::fprintf(stderr, "  cache_current_entries: %llu\n",
                 static_cast<unsigned long long>(cache->current_entries));
  }
}

}  // namespace

int run_query(const std::vector<std::string_view>& args, const CliConfig& config) {
  std::string sql;
  std::string file;
  std::string format_name = "table";
  std::optional<std::string> output_path;
  std::optional<std::string> backend_override;
  bool show_stats = false;

  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--sql" && i + 1 < args.size()) {
      sql = args[++i];
    } else if (args[i] == "--file" && i + 1 < args.size()) {
      file = args[++i];
    } else if (args[i] == "--format" && i + 1 < args.size()) {
      format_name = args[++i];
    } else if (args[i] == "--output" && i + 1 < args.size()) {
      output_path = std::string(args[++i]);
    } else if (args[i] == "--backend" && i + 1 < args.size()) {
      backend_override = std::string(args[++i]);
    } else if (args[i] == "--stats") {
      show_stats = true;
    }
  }

  if (backend_override && *backend_override != "cpu" && *backend_override != "gpu") {
    std::fprintf(stderr, "kernellake query: --backend must be one of cpu|gpu, got '%s'\n",
                 backend_override->c_str());
    return 1;
  }

  if (sql.empty() && file.empty()) {
    std::fprintf(stderr, "kernellake query: one of --sql or --file is required\n");
    return 1;
  }
  if (!sql.empty() && !file.empty()) {
    std::fprintf(stderr, "kernellake query: --sql and --file are mutually exclusive\n");
    return 1;
  }

  const std::optional<ResultFormat> format = parse_result_format(format_name);
  if (!format) {
    std::fprintf(stderr, "kernellake query: --format must be one of table|csv|jsonl|arrow, got '%s'\n",
                 format_name.c_str());
    return 1;
  }

  try {
    const std::string query_sql = file.empty() ? sql : read_file_or_throw(file);
    EngineConfig effective_engine_config = config.engine_config;
    if (backend_override) {
      effective_engine_config.engine.backend = *backend_override;
    }
    QueryEngine engine(effective_engine_config, config.device_id);

    observability::QuerySpan span = observability::start_query_span("kernellake.query");
    try {
      const QueryResult result = engine.execute(query_sql);
      span.finish(result, query_sql, effective_engine_config.engine.backend);
      write_query_result(result, *format, output_path);
      if (show_stats) {
        print_stats(result, engine.cache_metrics());
      }
    } catch (const KernelLakeError& e) {
      span.finish_error(e, query_sql, effective_engine_config.engine.backend);
      throw;
    }
  } catch (const KernelLakeError& e) {
    std::fprintf(stderr, "kernellake query: %s\n", e.what());
    return 1;
  }
  return 0;
}

}  // namespace kernellake::cli
