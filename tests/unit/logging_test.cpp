#include <gtest/gtest.h>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/common/logging.hpp"

namespace kernellake {
namespace {

TEST(Logging, AcceptsEveryLevelSpdlogItselfRecognizes) {
  // spdlog::level::from_str() (see <spdlog/common.h>'s SPDLOG_LEVEL_NAMES)
  // recognizes exactly these seven strings -- "warning", not "warn", and
  // "off" is a real, deliberately-supported level here (see
  // init_logging()'s own explicit `config.level != "off"` carve-out).
  for (const char* level : {"trace", "debug", "info", "warning", "error", "critical", "off"}) {
    LoggingSection config;
    config.level = level;
    EXPECT_NO_THROW((void)(init_logging(config))) << "level: " << level;
  }
}

TEST(Logging, RejectsLevelSpdlogDoesNotRecognize) {
  LoggingSection config;
  config.level = "verbose";
  EXPECT_THROW((void)(init_logging(config)), ConfigurationError);
}

// Diagnostic, not (by itself) a fix: validate_config()'s own log-level
// allowlist (config.cpp's kLogLevels: trace/debug/info/warn/warning/error/
// critical -- see config_test.cpp's RejectsUnknownLogLevel) and
// init_logging()'s here (spdlog::level::from_str()'s own recognized set,
// which -- confirmed by reading spdlog/common-inl.h's from_str() -- also
// special-cases "warn"/"err" as aliases for "warning"/"error", so "warn"
// itself is *not* actually a divergence between the two lists) disagree on
// exactly one entry: "off". init_logging() explicitly special-cases and
// accepts it (see this file's AcceptsEveryLevelSpdlogItselfRecognizes
// above), but kLogLevels has no "off" entry at all -- so
// validate_config(config) with logging.level == "off" throws
// ConfigurationError at startup, before init_logging() (which would have
// accepted it) is ever reached via the normal
// load_config_file()/validate_config()/init_logging() sequence in
// main.cpp. Not fixed here (out of this test's scope), just pinned down:
// this test demonstrates the asymmetry directly rather than asserting
// (wrongly) that the two lists actually diverge on "off"'s acceptance --
// they diverge on whether it's ever *reachable*.
TEST(Logging, OffIsRejectedByConfigValidationEvenThoughInitLoggingAcceptsIt) {
  EngineConfig config = default_config();
  config.logging.level = "off";
  EXPECT_THROW((void)(validate_config(config)), ConfigurationError);
  EXPECT_NO_THROW((void)(init_logging(config.logging)));
}

}  // namespace
}  // namespace kernellake
