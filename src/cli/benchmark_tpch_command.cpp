#include "commands.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/storage/file_discovery.hpp"
#include "kernellake/storage/object_store_registry.hpp"

namespace kernellake::cli {

namespace {

// Best-effort cache eviction for a specific file, usable without root:
// POSIX_FADV_DONTNEED asks the kernel to drop that file's cached pages. It
// is a hint, not a guarantee, and does nothing for caches outside this
// process's control (e.g. a network filesystem's own cache) -- callers must
// still treat "cold" mode as approximate, per the spec's own allowance to
// "clearly state when operating-system or object-store caches could not be
// cleared."
void evict_from_page_cache(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return;
  }
  ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  ::close(fd);
}

std::string strip_comments_and_substitute(const std::string& text, const std::string& data_glob) {
  static const std::regex comment_line(R"(--[^\n]*\n)");
  std::string stripped = std::regex_replace(text, comment_line, "\n");
  const std::string placeholder = "{data}";
  std::size_t pos = 0;
  while ((pos = stripped.find(placeholder, pos)) != std::string::npos) {
    stripped.replace(pos, placeholder.size(), data_glob);
    pos += data_glob.size();
  }
  return stripped;
}

std::string read_file_or_throw(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw StorageError(fmt::format("failed to open TPC-H query file '{}'", path));
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

struct IterationMetrics {
  double wall_seconds = 0.0;
  std::int64_t rows_returned = 0;
  std::int64_t peak_gpu_memory_bytes = 0;
};

double mean_of(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double median_of(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  if (n % 2 == 1) {
    return values[n / 2];
  }
  return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

double stddev_of(const std::vector<double>& values, double mean) {
  if (values.size() < 2) {
    return 0.0;
  }
  double sum_sq = 0.0;
  for (const double v : values) {
    sum_sq += (v - mean) * (v - mean);
  }
  return std::sqrt(sum_sq / static_cast<double>(values.size() - 1));
}

}  // namespace

int run_benchmark_tpch(const std::vector<std::string_view>& args, const EngineConfig& config) {
  std::string data;
  std::string query_file_override;
  std::optional<double> scale_factor;
  int query_number = -1;
  std::string mode;
  int iterations = config.benchmark.default_iterations;
  int warmup_iterations = config.benchmark.warmup_iterations;
  std::optional<std::string> output_path;

  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--data" && i + 1 < args.size()) {
      data = args[++i];
    } else if (args[i] == "--query-file" && i + 1 < args.size()) {
      query_file_override = args[++i];
    } else if (args[i] == "--scale-factor" && i + 1 < args.size()) {
      scale_factor = std::stod(std::string(args[++i]));
    } else if (args[i] == "--query" && i + 1 < args.size()) {
      query_number = std::stoi(std::string(args[++i]));
    } else if (args[i] == "--mode" && i + 1 < args.size()) {
      mode = args[++i];
    } else if (args[i] == "--iterations" && i + 1 < args.size()) {
      iterations = std::stoi(std::string(args[++i]));
    } else if (args[i] == "--warmup-iterations" && i + 1 < args.size()) {
      warmup_iterations = std::stoi(std::string(args[++i]));
    } else if (args[i] == "--output" && i + 1 < args.size()) {
      output_path = std::string(args[++i]);
    }
  }

  if (data.empty()) {
    std::fprintf(stderr, "kernellake benchmark tpch: --data is required\n");
    return 1;
  }
  if (query_number < 0) {
    std::fprintf(stderr, "kernellake benchmark tpch: --query is required\n");
    return 1;
  }
  if (mode != "cold" && mode != "warm" && mode != "execution-only") {
    std::fprintf(stderr, "kernellake benchmark tpch: --mode must be cold|warm|execution-only\n");
    return 1;
  }
  if (mode == "execution-only") {
    std::fprintf(stderr,
                 "kernellake benchmark tpch: --mode execution-only is not implemented yet (it needs an "
                 "operator-tree entry point that skips ParquetScanOperator entirely, which does not exist -- "
                 "see docs/ROADMAP.md); use cold or warm mode instead\n");
    return 1;
  }
  if (iterations <= 0) {
    std::fprintf(stderr, "kernellake benchmark tpch: --iterations must be positive\n");
    return 1;
  }

  char default_query_file[64];
  std::snprintf(default_query_file, sizeof(default_query_file), "benchmarks/tpch/queries/q%02d.sql",
                query_number);
  const std::string query_file = query_file_override.empty() ? default_query_file : query_file_override;

  try {
    const std::string sql = strip_comments_and_substitute(read_file_or_throw(query_file), data);

    ObjectStoreRegistry store(config.storage);
    const std::vector<ObjectInfo> files = discover_parquet_files(store, {data});
    if (files.empty()) {
      std::fprintf(stderr, "kernellake benchmark tpch: no Parquet files matched '%s'\n", data.c_str());
      return 1;
    }

    QueryEngine engine(config);

    const auto run_once = [&]() -> IterationMetrics {
      if (mode == "cold") {
        for (const ObjectInfo& file : files) {
          evict_from_page_cache(file.uri.value());
        }
      }
      const QueryResult result = engine.execute(sql);
      return IterationMetrics{result.elapsed_wall_seconds.value_or(0.0), result.rows_returned.value_or(0),
                              result.peak_gpu_memory_bytes.value_or(0)};
    };

    for (int i = 0; i < warmup_iterations; ++i) {
      run_once();
    }

    std::vector<IterationMetrics> measurements;
    measurements.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
      measurements.push_back(run_once());
    }

    std::vector<double> wall_seconds;
    wall_seconds.reserve(measurements.size());
    for (const IterationMetrics& m : measurements) {
      wall_seconds.push_back(m.wall_seconds);
    }
    const double mean = mean_of(wall_seconds);

    nlohmann::json report;
    report["benchmark"] = "tpch";
    report["unofficial"] = true;
    report["disclaimer"] = "Unofficial TPC-H-derived benchmark. Not a certified TPC result.";
    report["query"] = query_number;
    report["query_file"] = query_file;
    report["mode"] = mode;
    report["data"] = data;
    if (scale_factor) {
      report["scale_factor"] = *scale_factor;
    }
    report["warmup_iterations"] = warmup_iterations;
    report["iterations"] = iterations;
    report["cache_clearing"] =
        mode == "cold"
            ? "best-effort per-file posix_fadvise(POSIX_FADV_DONTNEED) before each iteration; this is a "
              "hint, not a guarantee, and the OS-wide page cache cannot be dropped without root"
            : "none (warm mode measures with whatever the warmup iterations left cached)";
    report["result_validation_performed"] = false;
    report["result_validation_note"] =
        "run tools/validate_tpch.py separately to cross-check this query's results against DuckDB";

    nlohmann::json iteration_array = nlohmann::json::array();
    for (std::size_t i = 0; i < measurements.size(); ++i) {
      iteration_array.push_back({{"iteration", i},
                                 {"wall_seconds", measurements[i].wall_seconds},
                                 {"rows_returned", measurements[i].rows_returned},
                                 {"peak_gpu_memory_bytes", measurements[i].peak_gpu_memory_bytes}});
    }
    report["iteration_results"] = iteration_array;
    report["first_iteration_wall_seconds"] = wall_seconds.front();
    report["median_wall_seconds"] = median_of(wall_seconds);
    report["mean_wall_seconds"] = mean;
    report["min_wall_seconds"] = *std::min_element(wall_seconds.begin(), wall_seconds.end());
    report["max_wall_seconds"] = *std::max_element(wall_seconds.begin(), wall_seconds.end());
    report["stddev_wall_seconds"] = stddev_of(wall_seconds, mean);

    const std::string rendered = report.dump(2);
    if (output_path) {
      std::ofstream out(*output_path);
      if (!out) {
        throw StorageError(fmt::format("failed to open output file '{}'", *output_path));
      }
      out << rendered << "\n";
    } else {
      std::printf("%s\n", rendered.c_str());
    }
  } catch (const KernelLakeError& e) {
    std::fprintf(stderr, "kernellake benchmark tpch: %s\n", e.what());
    return 1;
  }
  return 0;
}

}  // namespace kernellake::cli
