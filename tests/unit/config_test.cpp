#include <gtest/gtest.h>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {
namespace {

TEST(Config, DefaultsMatchSpec) {
  const EngineConfig config = default_config();
  EXPECT_EQ(config.engine.device_id, 0);
  EXPECT_EQ(config.engine.batch_rows, 1'000'000u);
  EXPECT_EQ(config.memory.pool_max_bytes, 8ULL * 1024 * 1024 * 1024);
  EXPECT_EQ(config.benchmark.baseline, "duckdb");
  EXPECT_NO_THROW(validate_config(config));
}

TEST(Config, ParsesOverrides) {
  const std::string yaml = R"(
engine:
  device_id: 2
  batch_rows: 500000
logging:
  level: debug
  json: true
benchmark:
  output_format: csv
)";
  const EngineConfig config = parse_config(yaml);
  EXPECT_EQ(config.engine.device_id, 2);
  EXPECT_EQ(config.engine.batch_rows, 500000u);
  EXPECT_EQ(config.logging.level, "debug");
  EXPECT_TRUE(config.logging.json);
  EXPECT_EQ(config.benchmark.output_format, "csv");
  // Fields not present in the override fall back to defaults.
  EXPECT_EQ(config.engine.result_batch_rows, default_config().engine.result_batch_rows);
}

TEST(Config, RejectsZeroBatchRows) {
  EngineConfig config = default_config();
  config.engine.batch_rows = 0;
  EXPECT_THROW(validate_config(config), ConfigurationError);
}

TEST(Config, RejectsContradictoryMemoryPool) {
  EngineConfig config = default_config();
  config.memory.pool_initial_bytes = 100;
  config.memory.pool_max_bytes = 10;
  EXPECT_THROW(validate_config(config), ConfigurationError);
}

TEST(Config, RejectsUnknownLogLevel) {
  EngineConfig config = default_config();
  config.logging.level = "verbose";
  EXPECT_THROW(validate_config(config), ConfigurationError);
}

TEST(Config, RejectsUnsupportedBenchmarkMode) {
  EngineConfig config = default_config();
  config.benchmark.output_format = "xml";
  EXPECT_THROW(validate_config(config), ConfigurationError);

  config = default_config();
  config.benchmark.baseline = "clickhouse-but-not-really";
  EXPECT_THROW(validate_config(config), ConfigurationError);
}

TEST(Config, RejectsMalformedYaml) {
  EXPECT_THROW(parse_config("engine: [this is not a map"), ConfigurationError);
}

TEST(Config, LoadConfigFileRejectsMissingPath) {
  EXPECT_THROW(load_config_file("/nonexistent/kernellake.yaml"), ConfigurationError);
}

}  // namespace
}  // namespace kernellake
