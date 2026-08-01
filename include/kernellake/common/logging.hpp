#pragma once

#include <string>

namespace kernellake {

struct LoggingSection;

// Initializes the global spdlog default logger's level and output pattern.
// `json` selects a structured, JSON-Lines-shaped pattern (still produced by
// spdlog's pattern formatter, not a general-purpose JSON serializer) so log
// output stays greppable/parseable without pulling in a JSON logging
// dependency. Throws ConfigurationError if `level` is not a recognized
// spdlog level name.
void init_logging(const LoggingSection& config);

}  // namespace kernellake
