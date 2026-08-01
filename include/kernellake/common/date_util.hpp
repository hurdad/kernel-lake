#pragma once

#include <cstdint>
#include <string_view>

namespace kernellake {

// Parses a "YYYY-MM-DD" date literal into days since 1970-01-01, matching
// Arrow's date32 domain. Throws SqlError (with the offending text) if the
// format is wrong or the calendar values are out of range.
[[nodiscard]] std::int32_t parse_iso_date(std::string_view text);

}  // namespace kernellake
