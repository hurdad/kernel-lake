#include "kernellake/common/date_util.hpp"

#include <fmt/format.h>

#include <array>
#include <cctype>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

bool is_leap_year(int year) {
  return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
}

int days_in_month(int year, int month) {
  static constexpr std::array<int, 12> kDays = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (month == 2 && is_leap_year(year)) {
    return 29;
  }
  return kDays[static_cast<std::size_t>(month - 1)];
}

[[noreturn]] void fail(std::string_view text) {
  throw SqlError(
      fmt::format("invalid date literal '{}': expected 'YYYY-MM-DD' with valid calendar values", text));
}

}  // namespace

std::int32_t parse_iso_date(std::string_view text) {
  if (text.size() != 10 || text[4] != '-' || text[7] != '-') {
    fail(text);
  }
  for (std::size_t i : {0u, 1u, 2u, 3u, 5u, 6u, 8u, 9u}) {
    if (!std::isdigit(static_cast<unsigned char>(text[i]))) {
      fail(text);
    }
  }

  const int year = (text[0] - '0') * 1000 + (text[1] - '0') * 100 + (text[2] - '0') * 10 + (text[3] - '0');
  const int month = (text[5] - '0') * 10 + (text[6] - '0');
  const int day = (text[8] - '0') * 10 + (text[9] - '0');

  if (month < 1 || month > 12) {
    fail(text);
  }
  if (day < 1 || day > days_in_month(year, month)) {
    fail(text);
  }

  std::int64_t days = 0;
  if (year >= 1970) {
    for (int y = 1970; y < year; ++y) {
      days += is_leap_year(y) ? 366 : 365;
    }
  } else {
    for (int y = year; y < 1970; ++y) {
      days -= is_leap_year(y) ? 366 : 365;
    }
  }
  for (int m = 1; m < month; ++m) {
    days += days_in_month(year, m);
  }
  days += day - 1;

  return static_cast<std::int32_t>(days);
}

}  // namespace kernellake
