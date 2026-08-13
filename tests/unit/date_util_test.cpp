#include <gtest/gtest.h>

#include "kernellake/common/date_util.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/sql/ast.hpp"
#include "kernellake/sql/parser.hpp"

namespace kernellake {
namespace {

TEST(DateUtil, EpochIsDayZero) {
  EXPECT_EQ(parse_iso_date("1970-01-01"), 0);
}

TEST(DateUtil, StandardLeapYearIncludesFeb29) {
  // 2024 % 4 == 0 and 2024 % 100 != 0 -- an ordinary (non-century) leap
  // year, the common case is_leap_year()'s first disjunct covers.
  EXPECT_NO_THROW((void)(parse_iso_date("2024-02-29")));
  EXPECT_EQ(parse_iso_date("2024-03-01") - parse_iso_date("2024-02-28"), 2);  // Feb has 29 days in 2024.
}

TEST(DateUtil, NonLeapYearRejectsFeb29) {
  EXPECT_THROW((void)(parse_iso_date("2023-02-29")), SqlError);
  EXPECT_EQ(parse_iso_date("2023-03-01") - parse_iso_date("2023-02-28"), 1);  // Feb has 28 days in 2023.
}

TEST(DateUtil, CenturyYearDivisibleBy100ButNot400IsNotALeapYear) {
  // 1900 % 100 == 0 and 1900 % 400 != 0 -- the century exception
  // is_leap_year() must apply (year % 100 != 0 alone would wrongly call
  // this a leap year via the % 4 == 0 branch).
  EXPECT_THROW((void)(parse_iso_date("1900-02-29")), SqlError);
  EXPECT_EQ(parse_iso_date("1900-03-01") - parse_iso_date("1900-02-28"), 1);
}

TEST(DateUtil, CenturyYearDivisibleBy400IsALeapYear) {
  // 2000 % 100 == 0 and 2000 % 400 == 0 -- the exception-to-the-exception
  // that makes 2000 a leap year despite being a century year.
  EXPECT_NO_THROW((void)(parse_iso_date("2000-02-29")));
  EXPECT_EQ(parse_iso_date("2000-03-01") - parse_iso_date("2000-02-28"), 2);
}

TEST(DateUtil, PreEpochDateSeveralYearsBeforeIsExactlyRoundTripConsistent) {
  // Only a single day-before-epoch case (1969-12-31) existed anywhere
  // before this test -- not enough to exercise the backward-counting loop
  // in parse_iso_date() (the `for (int y = year; y < 1970; ++y)` branch),
  // which needs a year several iterations before 1970 to actually run more
  // than once. Expected value cross-checked against Python's
  // datetime.date(1965, 6, 15) - datetime.date(1970, 1, 1).
  EXPECT_EQ(parse_iso_date("1965-06-15"), -1661);
  // Sanity check the loop direction/magnitude independently: exactly one
  // full non-leap 365-day year earlier lands on the same month/day (the
  // span between the two dates doesn't cross a Feb 29, including 1964's
  // own leap day, which falls before June 15).
  EXPECT_EQ(parse_iso_date("1964-06-15") - parse_iso_date("1965-06-15"), -365);
}

TEST(DateUtil, RejectsMonthOutOfRange) {
  EXPECT_THROW((void)(parse_iso_date("2026-13-01")), SqlError);
  EXPECT_THROW((void)(parse_iso_date("2026-00-01")), SqlError);
}

TEST(DateUtil, RejectsDayOutOfRangeForMonth) {
  EXPECT_THROW((void)(parse_iso_date("2026-04-31")), SqlError);  // April has 30 days.
  EXPECT_THROW((void)(parse_iso_date("2026-01-32")), SqlError);
  EXPECT_THROW((void)(parse_iso_date("2026-01-00")), SqlError);
}

TEST(DateUtil, RejectsMalformedText) {
  EXPECT_THROW((void)(parse_iso_date("2026/01/01")), SqlError);
  EXPECT_THROW((void)(parse_iso_date("26-01-01")), SqlError);
  EXPECT_THROW((void)(parse_iso_date("not-a-date")), SqlError);
}

// SQL-level round trip: DATE '...' literals go through the exact same
// parse_iso_date() this file otherwise tests directly, via
// sql::parser.cpp's DATE-literal handling -- see sql_parser_test.cpp's own
// RejectsInvalidDateLiteral for the pre-existing (out-of-range month/day)
// case. These two specifically isolate the century leap-year exception at
// the SQL-parsing layer, not just the pure date-math layer above.
TEST(DateUtil, SqlLevelAcceptsCenturyLeapYearDateLiteral) {
  const sql::AstSelectStatement stmt =
      sql::parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE d >= DATE '2000-02-29'");
  ASSERT_NE(stmt.where, nullptr);
}

TEST(DateUtil, SqlLevelRejectsCenturyNonLeapYearDateLiteral) {
  EXPECT_THROW(
      (void)(sql::parse_sql("SELECT a FROM read_parquet('/x.parquet') WHERE d >= DATE '1900-02-29'")),
      SqlError);
}

}  // namespace
}  // namespace kernellake
