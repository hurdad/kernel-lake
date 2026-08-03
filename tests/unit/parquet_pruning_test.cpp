// Direct unit tests for evaluate_pruning()/predicate_proves_empty()
// (kernellake/io/parquet_pruning.hpp), constructing FileMetadata/
// RowGroupMetadata/ColumnStatistics by hand rather than through a real
// Parquet file -- parquet_metadata_test.cpp already covers the
// Equal/Less/no-predicates/missing-stats cases end-to-end against a real
// file; this file fills in the remaining comparison operators, value
// kinds, and multi-predicate/multi-row-group interactions that file
// doesn't exercise, since a wrong pruning decision here would silently
// corrupt query results rather than fail loudly.
#include <gtest/gtest.h>

#include "kernellake/io/parquet_pruning.hpp"

namespace kernellake {
namespace {

ColumnStatistics stats_with_range(LiteralStorage min_value, LiteralStorage max_value) {
  ColumnStatistics stats;
  stats.has_min_max = true;
  stats.min_value = std::move(min_value);
  stats.max_value = std::move(max_value);
  return stats;
}

ExpressionPtr int_literal(std::int64_t value) {
  return std::make_shared<LiteralExpression>(LiteralExpression::make_int64(value));
}

FileMetadata one_row_group_file(const std::string& column, ColumnStatistics stats) {
  RowGroupMetadata row_group;
  row_group.index = 0;
  row_group.column_statistics.emplace(column, std::move(stats));

  FileMetadata file;
  file.path = Uri("/x.parquet");
  file.row_groups.push_back(std::move(row_group));
  return file;
}

TEST(ParquetPruning, NotEqualSkipsOnlyWhenEveryValueEqualsLiteral) {
  // min == max == 5: every row equals 5, so `id != 5` is provably empty.
  const FileMetadata all_equal = one_row_group_file("id", stats_with_range(std::int64_t{5}, std::int64_t{5}));
  const ScanDecision skipped =
      evaluate_pruning(all_equal, {PushablePredicate{"id", BinaryOperator::NotEqual, int_literal(5)}});
  EXPECT_EQ(skipped.skipped_row_groups, std::vector<int>{0});

  // min=5, max=6: some rows could be 6 (!= 5 is true for them), must scan.
  const FileMetadata mixed = one_row_group_file("id", stats_with_range(std::int64_t{5}, std::int64_t{6}));
  const ScanDecision kept =
      evaluate_pruning(mixed, {PushablePredicate{"id", BinaryOperator::NotEqual, int_literal(5)}});
  EXPECT_TRUE(kept.skipped_row_groups.empty());
}

TEST(ParquetPruning, LessEqualSkipsWhenMinExceedsLiteral) {
  const FileMetadata file = one_row_group_file("id", stats_with_range(std::int64_t{10}, std::int64_t{20}));
  // id <= 9: min is 10, every row exceeds 9 -- provably empty.
  const ScanDecision skipped =
      evaluate_pruning(file, {PushablePredicate{"id", BinaryOperator::LessEqual, int_literal(9)}});
  EXPECT_EQ(skipped.skipped_row_groups, std::vector<int>{0});
  // id <= 10: min itself qualifies -- must scan.
  const ScanDecision kept =
      evaluate_pruning(file, {PushablePredicate{"id", BinaryOperator::LessEqual, int_literal(10)}});
  EXPECT_TRUE(kept.skipped_row_groups.empty());
}

TEST(ParquetPruning, GreaterSkipsWhenMaxDoesNotExceedLiteral) {
  const FileMetadata file = one_row_group_file("id", stats_with_range(std::int64_t{10}, std::int64_t{20}));
  // id > 20: max is 20, no row exceeds it -- provably empty.
  const ScanDecision skipped =
      evaluate_pruning(file, {PushablePredicate{"id", BinaryOperator::Greater, int_literal(20)}});
  EXPECT_EQ(skipped.skipped_row_groups, std::vector<int>{0});
  const ScanDecision kept =
      evaluate_pruning(file, {PushablePredicate{"id", BinaryOperator::Greater, int_literal(19)}});
  EXPECT_TRUE(kept.skipped_row_groups.empty());
}

TEST(ParquetPruning, GreaterEqualSkipsWhenMaxBelowLiteral) {
  const FileMetadata file = one_row_group_file("id", stats_with_range(std::int64_t{10}, std::int64_t{20}));
  // id >= 21: max is 20, every row is below it -- provably empty.
  const ScanDecision skipped =
      evaluate_pruning(file, {PushablePredicate{"id", BinaryOperator::GreaterEqual, int_literal(21)}});
  EXPECT_EQ(skipped.skipped_row_groups, std::vector<int>{0});
  const ScanDecision kept =
      evaluate_pruning(file, {PushablePredicate{"id", BinaryOperator::GreaterEqual, int_literal(20)}});
  EXPECT_TRUE(kept.skipped_row_groups.empty());
}

TEST(ParquetPruning, StringComparisonPrunesLexicographically) {
  const FileMetadata file =
      one_row_group_file("region", stats_with_range(std::string("A"), std::string("M")));
  auto literal = std::make_shared<LiteralExpression>(LiteralExpression::make_string("Z"));
  // region = 'Z': range is ['A', 'M'], 'Z' is outside it -- provably empty.
  const ScanDecision decision =
      evaluate_pruning(file, {PushablePredicate{"region", BinaryOperator::Equal, literal}});
  EXPECT_EQ(decision.skipped_row_groups, std::vector<int>{0});
}

TEST(ParquetPruning, BoolComparisonPrunesWhenAllValuesMatchOpposite) {
  const FileMetadata file = one_row_group_file("active", stats_with_range(false, false));
  auto literal = std::make_shared<LiteralExpression>(LiteralExpression::make_bool(true));
  // active = TRUE, but every value in the group is FALSE -- provably empty.
  const ScanDecision decision =
      evaluate_pruning(file, {PushablePredicate{"active", BinaryOperator::Equal, literal}});
  EXPECT_EQ(decision.skipped_row_groups, std::vector<int>{0});
}

TEST(ParquetPruning, IncomparableTypesNeverPrune) {
  // Statistics are numeric (int64) but the predicate literal is a string --
  // compare_literals() has no cross-type comparison, so this must fall back
  // to "must scan" rather than guess an ordering.
  const FileMetadata file = one_row_group_file("id", stats_with_range(std::int64_t{10}, std::int64_t{20}));
  auto literal = std::make_shared<LiteralExpression>(LiteralExpression::make_string("50"));
  const ScanDecision decision =
      evaluate_pruning(file, {PushablePredicate{"id", BinaryOperator::Equal, literal}});
  EXPECT_TRUE(decision.skipped_row_groups.empty());
}

TEST(ParquetPruning, MultiplePredicatesOnDifferentColumnsEachCanSkip) {
  RowGroupMetadata row_group;
  row_group.index = 0;
  row_group.column_statistics.emplace("id", stats_with_range(std::int64_t{0}, std::int64_t{9}));
  row_group.column_statistics.emplace("region", stats_with_range(std::string("A"), std::string("A")));
  FileMetadata file;
  file.path = Uri("/x.parquet");
  file.row_groups.push_back(std::move(row_group));

  // `id > 100` alone already proves this row group empty, even though
  // `region = 'B'` (the other predicate) isn't evaluated against it first.
  auto region_b = std::make_shared<LiteralExpression>(LiteralExpression::make_string("B"));
  const std::vector<PushablePredicate> predicates = {
      PushablePredicate{"id", BinaryOperator::Greater, int_literal(100)},
      PushablePredicate{"region", BinaryOperator::Equal, region_b},
  };
  const ScanDecision decision = evaluate_pruning(file, predicates);
  EXPECT_EQ(decision.skipped_row_groups, std::vector<int>{0});
}

TEST(ParquetPruning, ReasonExplainsNoRowGroupsCouldBeSkipped) {
  // A predicate that's present but can never prove any row group empty
  // (min/max span both sides of the literal): the "no row groups could be
  // proven safe" summary reason must be surfaced rather than an empty
  // `reasons` list silently implying nothing was even attempted.
  const FileMetadata file = one_row_group_file("id", stats_with_range(std::int64_t{0}, std::int64_t{100}));
  const ScanDecision decision =
      evaluate_pruning(file, {PushablePredicate{"id", BinaryOperator::Equal, int_literal(50)}});
  ASSERT_TRUE(decision.skipped_row_groups.empty());
  ASSERT_EQ(decision.reasons.size(), 1u);
  EXPECT_EQ(decision.reasons[0], "no row groups could be proven safe to skip from available statistics");
}

}  // namespace
}  // namespace kernellake
