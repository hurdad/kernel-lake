#pragma once

#include <cstdint>
#include <string>

namespace kernellake {

struct EngineSection {
  int device_id = 0;
  std::uint64_t batch_rows = 1'000'000;
  std::uint64_t result_batch_rows = 65'536;
  std::uint64_t query_memory_limit_bytes = 8ULL * 1024 * 1024 * 1024;
  // "gpu" (default) or "cpu" -- see docs/ARCHITECTURE.md's CPU backend
  // section. "gpu" on a CPU-only (dev preset) build throws the existing
  // clear ExecutionError explaining why, exactly as it always has;
  // requesting "cpu" works in *either* build, since the Acero-based CPU
  // backend needs no CUDA at all. `kernellake query --backend cpu|gpu`
  // overrides this per invocation without editing the config file.
  std::string backend = "gpu";
};

struct MemorySection {
  std::uint64_t pool_initial_bytes = 1ULL * 1024 * 1024 * 1024;
  std::uint64_t pool_max_bytes = 8ULL * 1024 * 1024 * 1024;
  bool use_async_allocator = true;
};

struct StorageSection {
  std::string local_root = "/";
  bool enable_s3 = false;
};

struct LoggingSection {
  std::string level = "info";
  bool json = false;
};

struct ProfilingSection {
  bool nvtx = true;
  bool operator_metrics = true;
};

struct BenchmarkSection {
  int default_iterations = 5;
  int warmup_iterations = 1;
  std::string output_format = "json";
  bool verify_results = true;
  std::string baseline = "duckdb";
};

struct EngineConfig {
  EngineSection engine;
  MemorySection memory;
  StorageSection storage;
  LoggingSection logging;
  ProfilingSection profiling;
  BenchmarkSection benchmark;
};

// Returns the built-in defaults, matching config/kernellake.yaml.
[[nodiscard]] EngineConfig default_config();

// Parses YAML text into an EngineConfig. Missing keys fall back to defaults.
// Throws ConfigurationError on malformed YAML or wrong value types.
[[nodiscard]] EngineConfig parse_config(const std::string& yaml_text);

// Loads and parses a YAML config file. Throws ConfigurationError if the file
// is missing, unreadable, or fails to parse.
[[nodiscard]] EngineConfig load_config_file(const std::string& path);

// Validates value ranges and known enumerations (log level, benchmark output
// format, benchmark baseline) and cross-field constraints (e.g. pool_max
// must be >= pool_initial). Does not validate GPU device availability here;
// CUDA device-count checks happen where a CUDA context is actually created,
// since this validation must also run in CPU-only builds.
// Throws ConfigurationError with an actionable message on the first
// violation found.
void validate_config(const EngineConfig& config);

}  // namespace kernellake
