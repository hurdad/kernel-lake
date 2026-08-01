#include "kernellake/common/logging.hpp"

#include <spdlog/spdlog.h>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {

void init_logging(const LoggingSection& config) {
  const spdlog::level::level_enum level = spdlog::level::from_str(config.level);
  if (level == spdlog::level::off && config.level != "off") {
    throw ConfigurationError("logging.level '" + config.level + "' is not a recognized level");
  }
  spdlog::set_level(level);

  if (config.json) {
    spdlog::set_pattern(R"({"time":"%Y-%m-%dT%H:%M:%S.%fZ","level":"%l","logger":"%n","message":"%v"})");
  } else {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
  }
}

}  // namespace kernellake
