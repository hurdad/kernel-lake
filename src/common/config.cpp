#include "kernellake/common/config.hpp"

#include <yaml-cpp/yaml.h>

#include <array>
#include <fstream>
#include <sstream>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

template <typename T>
T read_or(const YAML::Node& node, const char* key, T fallback) {
  if (!node || !node[key]) return fallback;
  try {
    return node[key].as<T>();
  } catch (const YAML::Exception& e) {
    throw ConfigurationError(std::string("invalid value for '") + key + "': " + e.what());
  }
}

}  // namespace

EngineConfig default_config() {
  return EngineConfig{};
}

EngineConfig parse_config(const std::string& yaml_text) {
  YAML::Node root;
  try {
    root = YAML::Load(yaml_text);
  } catch (const YAML::Exception& e) {
    throw ConfigurationError(std::string("failed to parse YAML configuration: ") + e.what());
  }

  EngineConfig config;

  const YAML::Node engine = root["engine"];
  config.engine.device_id = read_or(engine, "device_id", config.engine.device_id);
  config.engine.batch_rows = read_or(engine, "batch_rows", config.engine.batch_rows);
  config.engine.result_batch_rows = read_or(engine, "result_batch_rows", config.engine.result_batch_rows);
  config.engine.query_memory_limit_bytes =
      read_or(engine, "query_memory_limit_bytes", config.engine.query_memory_limit_bytes);
  config.engine.backend = read_or(engine, "backend", config.engine.backend);

  const YAML::Node memory = root["memory"];
  config.memory.pool_initial_bytes = read_or(memory, "pool_initial_bytes", config.memory.pool_initial_bytes);
  config.memory.pool_max_bytes = read_or(memory, "pool_max_bytes", config.memory.pool_max_bytes);
  config.memory.use_async_allocator =
      read_or(memory, "use_async_allocator", config.memory.use_async_allocator);

  const YAML::Node storage = root["storage"];
  config.storage.local_root = read_or(storage, "local_root", config.storage.local_root);
  config.storage.enable_s3 = read_or(storage, "enable_s3", config.storage.enable_s3);

  const YAML::Node logging = root["logging"];
  config.logging.level = read_or(logging, "level", config.logging.level);
  config.logging.json = read_or(logging, "json", config.logging.json);

  const YAML::Node profiling = root["profiling"];
  config.profiling.nvtx = read_or(profiling, "nvtx", config.profiling.nvtx);
  config.profiling.operator_metrics =
      read_or(profiling, "operator_metrics", config.profiling.operator_metrics);

  const YAML::Node benchmark = root["benchmark"];
  config.benchmark.default_iterations =
      read_or(benchmark, "default_iterations", config.benchmark.default_iterations);
  config.benchmark.warmup_iterations =
      read_or(benchmark, "warmup_iterations", config.benchmark.warmup_iterations);
  config.benchmark.output_format = read_or(benchmark, "output_format", config.benchmark.output_format);
  config.benchmark.verify_results = read_or(benchmark, "verify_results", config.benchmark.verify_results);
  config.benchmark.baseline = read_or(benchmark, "baseline", config.benchmark.baseline);

  const YAML::Node server = root["server"];
  config.server.host = read_or(server, "host", config.server.host);
  config.server.port = read_or(server, "port", config.server.port);

  return config;
}

EngineConfig load_config_file(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw ConfigurationError("cannot open configuration file '" + path +
                             "': check that the path exists and is readable");
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return parse_config(buffer.str());
}

void validate_config(const EngineConfig& config) {
  if (config.engine.device_id < 0) {
    throw ConfigurationError("engine.device_id must be >= 0, got " + std::to_string(config.engine.device_id));
  }
  if (config.engine.batch_rows == 0) {
    throw ConfigurationError("engine.batch_rows must be > 0");
  }
  if (config.engine.result_batch_rows == 0) {
    throw ConfigurationError("engine.result_batch_rows must be > 0");
  }
  if (config.engine.query_memory_limit_bytes == 0) {
    throw ConfigurationError("engine.query_memory_limit_bytes must be > 0");
  }
  if (config.engine.backend != "gpu" && config.engine.backend != "cpu") {
    throw ConfigurationError("engine.backend '" + config.engine.backend +
                             "' is unsupported (expected 'gpu' or 'cpu')");
  }

  if (config.memory.pool_initial_bytes == 0) {
    throw ConfigurationError("memory.pool_initial_bytes must be > 0");
  }
  if (config.memory.pool_max_bytes < config.memory.pool_initial_bytes) {
    throw ConfigurationError("memory.pool_max_bytes (" + std::to_string(config.memory.pool_max_bytes) +
                             ") must be >= memory.pool_initial_bytes (" +
                             std::to_string(config.memory.pool_initial_bytes) + ")");
  }

  static constexpr std::array<const char*, 7> kLogLevels = {"trace",   "debug", "info",    "warn",
                                                            "warning", "error", "critical"};
  bool level_ok = false;
  for (const char* level : kLogLevels) {
    if (config.logging.level == level) {
      level_ok = true;
      break;
    }
  }
  if (!level_ok) {
    throw ConfigurationError("logging.level '" + config.logging.level +
                             "' is not a recognized level "
                             "(expected one of trace/debug/info/warn/error/critical)");
  }

  if (config.benchmark.output_format != "json" && config.benchmark.output_format != "csv") {
    throw ConfigurationError("benchmark.output_format '" + config.benchmark.output_format +
                             "' is unsupported (expected 'json' or 'csv')");
  }
  if (config.benchmark.baseline != "duckdb" && config.benchmark.baseline != "none") {
    throw ConfigurationError("benchmark.baseline '" + config.benchmark.baseline +
                             "' is unsupported (expected 'duckdb' or 'none')");
  }
  if (config.benchmark.default_iterations <= 0) {
    throw ConfigurationError("benchmark.default_iterations must be > 0");
  }
  if (config.benchmark.warmup_iterations < 0) {
    throw ConfigurationError("benchmark.warmup_iterations must be >= 0");
  }

  if (config.server.port == 0) {
    throw ConfigurationError("server.port must be > 0");
  }
}

}  // namespace kernellake
