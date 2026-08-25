// Direct unit tests for partition_values_prove_empty()
// (kernellake/iceberg/partition_pruning.hpp), constructing
// IcebergTableMetadata/IcebergPartitionSpec/PartitionFieldValue by hand
// rather than through a real REST catalog + manifest -- mirrors
// parquet_pruning_test.cpp's own "direct unit test" rationale, since a
// wrong pruning decision here would silently skip a file that still
// contains matching rows, a real correctness bug.
#include <gtest/gtest.h>

#include "kernellake/common/date_util.hpp"
#include "kernellake/iceberg/partition_pruning.hpp"

namespace kernellake::iceberg {
namespace {

ExpressionPtr string_literal(std::string value) {
  return std::make_shared<LiteralExpression>(LiteralExpression::make_string(std::move(value)));
}
ExpressionPtr int_literal(std::int64_t value) {
  return std::make_shared<LiteralExpression>(LiteralExpression::make_int64(value));
}
ExpressionPtr date_literal(const std::string& iso_date) {
  return std::make_shared<LiteralExpression>(LiteralExpression::make_date32(parse_iso_date(iso_date)));
}
ExpressionPtr timestamp_literal_micros(std::int64_t micros_since_epoch) {
  return std::make_shared<LiteralExpression>(LiteralExpression::make_timestamp(micros_since_epoch));
}
ExpressionPtr null_literal() {
  return std::make_shared<LiteralExpression>(LiteralExpression::make_null(int64_type()));
}

IcebergTableMetadata metadata_with_schema(std::vector<IcebergSchemaField> fields) {
  IcebergTableMetadata metadata;
  metadata.schema_fields = std::move(fields);
  return metadata;
}

IcebergPartitionSpec spec_with_field(std::int32_t source_id, std::string transform) {
  IcebergPartitionSpec spec;
  spec.spec_id = 0;
  spec.fields.push_back(
      IcebergPartitionField{source_id, /*field_id=*/1000, "partition_col", std::move(transform)});
  return spec;
}

// ---------------------------------------------------------------------------
// identity: no coarsening, so every comparison operator (including !=)
// prunes correctly.
// ---------------------------------------------------------------------------

TEST(PartitionPruning, IdentityEqualityPrunesMismatchedStringPartition) {
  const IcebergTableMetadata metadata =
      metadata_with_schema({IcebergSchemaField{1, "region", true, "string"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "identity");
  const std::vector<PartitionFieldValue> values = {std::string("US")};

  EXPECT_TRUE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"region", BinaryOperator::Equal, string_literal("EU")}}));
  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"region", BinaryOperator::Equal, string_literal("US")}}));
}

TEST(PartitionPruning, IdentityNotEqualPrunesOnlyWhenValueMatches) {
  const IcebergTableMetadata metadata =
      metadata_with_schema({IcebergSchemaField{1, "region", true, "string"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "identity");
  const std::vector<PartitionFieldValue> values = {std::string("US")};

  EXPECT_TRUE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"region", BinaryOperator::NotEqual, string_literal("US")}}));
  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"region", BinaryOperator::NotEqual, string_literal("EU")}}));
}

TEST(PartitionPruning, IdentityRangePrunesNumericPartitionColumn) {
  const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "id", true, "long"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "identity");
  const std::vector<PartitionFieldValue> values = {std::int64_t{5}};

  EXPECT_TRUE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"id", BinaryOperator::Greater, int_literal(10)}}));
  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"id", BinaryOperator::Greater, int_literal(3)}}));
}

// ---------------------------------------------------------------------------
// day/month/year/hour: coarsening transforms of a date/timestamp source --
// range operators prune correctly (monotonic), != never does.
// ---------------------------------------------------------------------------

TEST(PartitionPruning, DayTransformOnDateColumnPrunesByRange) {
  const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "d", true, "date"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "day");
  // day(Date32) is a direct pass-through of the day count itself.
  const std::vector<PartitionFieldValue> values = {static_cast<std::int64_t>(parse_iso_date("2024-01-01"))};

  EXPECT_TRUE(partition_values_prove_empty(
      metadata, spec, values,
      {PushablePredicate{"d", BinaryOperator::GreaterEqual, date_literal("2024-06-01")}}));
  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values,
      {PushablePredicate{"d", BinaryOperator::GreaterEqual, date_literal("2023-06-01")}}));
}

TEST(PartitionPruning, DayTransformNotEqualNeverPrunes) {
  const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "d", true, "date"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "day");
  const std::vector<PartitionFieldValue> values = {static_cast<std::int64_t>(parse_iso_date("2024-01-01"))};

  // Even though the transform values match exactly, other timestamps
  // within the same day could still satisfy `!= literal` -- must not
  // prune (see partition_pruning.hpp's own doc comment).
  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values,
      {PushablePredicate{"d", BinaryOperator::NotEqual, date_literal("2024-01-01")}}));
}

TEST(PartitionPruning, DayTransformOnTimestampColumnUsesFloorDivision) {
  const IcebergTableMetadata metadata =
      metadata_with_schema({IcebergSchemaField{1, "ts", true, "timestamp"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "day");
  // Partition value for 2024-01-01 (a real writer's day(ts) result).
  const std::vector<PartitionFieldValue> values = {static_cast<std::int64_t>(parse_iso_date("2024-01-01"))};

  // A timestamp one microsecond before epoch belongs to day -1 (1969-12-31),
  // not day 0 -- exercises floor (not truncating) division for a negative
  // numerator not evenly divisible by a day.
  EXPECT_TRUE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"ts", BinaryOperator::Less, timestamp_literal_micros(-1)}}));
}

// Regression coverage: proves_empty()'s strict Less/Greater cases used to
// apply the identity-only boundary rule (cmp >= 0 / cmp <= 0)
// unconditionally, even under a coarsening transform -- wrongly proving a
// file empty whenever the file's own day matched the literal's day
// exactly, even though rows earlier/later that same day could still
// satisfy a strict `<`/`>` predicate. Mirrors DayTransformNotEqualNeverPrunes
// above, for Less/Greater instead of NotEqual.
TEST(PartitionPruning, DayTransformStrictLessDoesNotPruneWithinSameDay) {
  const IcebergTableMetadata metadata =
      metadata_with_schema({IcebergSchemaField{1, "ts", true, "timestamp"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "day");
  // Every row in this file is on 2024-01-02.
  const std::vector<PartitionFieldValue> values = {static_cast<std::int64_t>(parse_iso_date("2024-01-02"))};
  const std::int64_t noon_jan_2 =
      static_cast<std::int64_t>(parse_iso_date("2024-01-02")) * 86'400'000'000LL + 12 * 3'600'000'000LL;
  const std::int64_t noon_jan_1 =
      static_cast<std::int64_t>(parse_iso_date("2024-01-01")) * 86'400'000'000LL + 12 * 3'600'000'000LL;

  // Literal at noon on the SAME day as the file's own day (cmp == 0): a
  // row earlier that day (e.g. 03:00) still satisfies `ts < noon` -- must
  // not be proven empty.
  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values,
      {PushablePredicate{"ts", BinaryOperator::Less, timestamp_literal_micros(noon_jan_2)}}));

  // Literal on the PRECEDING day: every row in this file (all on
  // 2024-01-02) is necessarily later than any 2024-01-01 timestamp, so
  // `ts < noon_jan_1` is impossible for every row -- correctly proven
  // empty.
  EXPECT_TRUE(partition_values_prove_empty(
      metadata, spec, values,
      {PushablePredicate{"ts", BinaryOperator::Less, timestamp_literal_micros(noon_jan_1)}}));
}

TEST(PartitionPruning, DayTransformStrictGreaterDoesNotPruneWithinSameDay) {
  const IcebergTableMetadata metadata =
      metadata_with_schema({IcebergSchemaField{1, "ts", true, "timestamp"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "day");
  // Every row in this file is on 2024-01-01.
  const std::vector<PartitionFieldValue> values = {static_cast<std::int64_t>(parse_iso_date("2024-01-01"))};
  const std::int64_t noon_jan_1 =
      static_cast<std::int64_t>(parse_iso_date("2024-01-01")) * 86'400'000'000LL + 12 * 3'600'000'000LL;
  const std::int64_t noon_jan_2 =
      static_cast<std::int64_t>(parse_iso_date("2024-01-02")) * 86'400'000'000LL + 12 * 3'600'000'000LL;

  // Literal at noon on the SAME day as the file's own day (cmp == 0): a
  // row later that day (e.g. 18:00) still satisfies `ts > noon` -- must
  // not be proven empty.
  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values,
      {PushablePredicate{"ts", BinaryOperator::Greater, timestamp_literal_micros(noon_jan_1)}}));

  // Literal on the FOLLOWING day: every row in this file (all on
  // 2024-01-01) is necessarily earlier than any 2024-01-02 timestamp, so
  // `ts > noon_jan_2` is impossible for every row -- correctly proven
  // empty.
  EXPECT_TRUE(partition_values_prove_empty(
      metadata, spec, values,
      {PushablePredicate{"ts", BinaryOperator::Greater, timestamp_literal_micros(noon_jan_2)}}));
}

TEST(PartitionPruning, HourTransformRequiresTimestampNotDate) {
  const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "d", true, "date"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "hour");
  const std::vector<PartitionFieldValue> values = {std::int64_t{0}};

  // Iceberg's `hour` transform is only defined for timestamp sources; a
  // Date32 literal has no time-of-day component, so this must never prune.
  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"d", BinaryOperator::Equal, date_literal("2024-01-01")}}));
}

TEST(PartitionPruning, HourTransformOnTimestampPrunesByFloorDivision) {
  const IcebergTableMetadata metadata =
      metadata_with_schema({IcebergSchemaField{1, "ts", true, "timestamp"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "hour");
  const std::vector<PartitionFieldValue> values = {std::int64_t{-1}};  // the hour ending at epoch

  EXPECT_TRUE(partition_values_prove_empty(
      metadata, spec, values,
      {PushablePredicate{"ts", BinaryOperator::GreaterEqual, timestamp_literal_micros(0)}}));
  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values,
      {PushablePredicate{"ts", BinaryOperator::GreaterEqual, timestamp_literal_micros(-1)}}));
}

// civil_from_days() is the exact inverse of date_util.cpp's parse_iso_date()
// forward (calendar -> days) computation -- these two dates specifically
// exercise its post-epoch and pre-epoch (negative remaining-days) branches.
TEST(PartitionPruning, YearAndMonthTransformsRecoverCorrectCalendarComponents) {
  {
    // 2000-03-01: year 2000 - 1970 = 30; month (2000-1970)*12 + (3-1) = 362.
    const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "d", true, "date"}});
    const std::vector<PartitionFieldValue> year_partition = {std::int64_t{30}};
    const IcebergPartitionSpec year_spec = spec_with_field(1, "year");
    EXPECT_FALSE(partition_values_prove_empty(
        metadata, year_spec, year_partition,
        {PushablePredicate{"d", BinaryOperator::Equal, date_literal("2000-03-01")}}));
    // A literal within the SAME year as the partition (cmp == 0): a file
    // partitioned by year(d)=2000 can still contain dates later in 2000
    // than 2000-03-01 (e.g. 2000-06-01), so `d > '2000-03-01'` must not
    // be proven empty -- see proves_empty()'s own comment on why strict
    // Less/Greater need the is_identity gate under coarsening.
    EXPECT_FALSE(partition_values_prove_empty(
        metadata, year_spec, year_partition,
        {PushablePredicate{"d", BinaryOperator::Greater, date_literal("2000-03-01")}}));
    // A literal in a STRICTLY LATER year: every row in this file is in
    // 2000, entirely before any 2001 date, so this is correctly proven
    // empty.
    EXPECT_TRUE(partition_values_prove_empty(
        metadata, year_spec, year_partition,
        {PushablePredicate{"d", BinaryOperator::Greater, date_literal("2001-01-01")}}));

    const std::vector<PartitionFieldValue> month_partition = {std::int64_t{362}};
    const IcebergPartitionSpec month_spec = spec_with_field(1, "month");
    EXPECT_FALSE(partition_values_prove_empty(
        metadata, month_spec, month_partition,
        {PushablePredicate{"d", BinaryOperator::Equal, date_literal("2000-03-01")}}));
    // Same reasoning at month granularity: a file partitioned by
    // month(d)=2000-03 can still contain dates later in March 2000.
    EXPECT_FALSE(partition_values_prove_empty(
        metadata, month_spec, month_partition,
        {PushablePredicate{"d", BinaryOperator::Greater, date_literal("2000-03-01")}}));
    EXPECT_TRUE(partition_values_prove_empty(
        metadata, month_spec, month_partition,
        {PushablePredicate{"d", BinaryOperator::Greater, date_literal("2000-04-01")}}));
  }
  {
    // 1969-12-31 (one day before epoch): year -1; month -1 -- exercises
    // civil_from_days()'s negative-remaining-days branch.
    const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "d", true, "date"}});
    const std::vector<PartitionFieldValue> year_partition = {std::int64_t{-1}};
    const IcebergPartitionSpec year_spec = spec_with_field(1, "year");
    EXPECT_FALSE(partition_values_prove_empty(
        metadata, year_spec, year_partition,
        {PushablePredicate{"d", BinaryOperator::Equal, date_literal("1969-12-31")}}));
    EXPECT_TRUE(partition_values_prove_empty(
        metadata, year_spec, year_partition,
        {PushablePredicate{"d", BinaryOperator::GreaterEqual, date_literal("1970-01-01")}}));
  }
}

// ---------------------------------------------------------------------------
// bucket/truncate: not evaluated at all -- always falls back to "must scan".
// ---------------------------------------------------------------------------

TEST(PartitionPruning, BucketTransformNeverPrunesEvenOnEquality) {
  const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "id", true, "long"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "bucket[4]");
  const std::vector<PartitionFieldValue> values = {std::int64_t{2}};

  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"id", BinaryOperator::Equal, int_literal(999)}}));
}

TEST(PartitionPruning, TruncateTransformNeverPrunes) {
  const IcebergTableMetadata metadata =
      metadata_with_schema({IcebergSchemaField{1, "region", true, "string"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "truncate[2]");
  const std::vector<PartitionFieldValue> values = {std::string("US")};

  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"region", BinaryOperator::Equal, string_literal("EUxx")}}));
}

// ---------------------------------------------------------------------------
// Defensive / fall-through cases -- all must degrade to "must scan", never
// crash or guess.
// ---------------------------------------------------------------------------

TEST(PartitionPruning, UnmatchedColumnNameNeverPrunes) {
  const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "id", true, "long"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "identity");
  const std::vector<PartitionFieldValue> values = {std::int64_t{5}};

  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"other_column", BinaryOperator::Equal, int_literal(999)}}));
}

TEST(PartitionPruning, SourceIdMissingFromSchemaNeverPrunes) {
  const IcebergTableMetadata metadata = metadata_with_schema({});  // source_id 1 not in schema_fields
  const IcebergPartitionSpec spec = spec_with_field(1, "identity");
  const std::vector<PartitionFieldValue> values = {std::int64_t{5}};

  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"id", BinaryOperator::Equal, int_literal(999)}}));
}

TEST(PartitionPruning, NullPartitionValueNeverPrunes) {
  const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "id", true, "long"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "identity");
  const std::vector<PartitionFieldValue> values = {std::monostate{}};

  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"id", BinaryOperator::Equal, int_literal(999)}}));
}

TEST(PartitionPruning, NullLiteralNeverPrunes) {
  const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "id", true, "long"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "identity");
  const std::vector<PartitionFieldValue> values = {std::int64_t{5}};

  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, values, {PushablePredicate{"id", BinaryOperator::Equal, null_literal()}}));
}

TEST(PartitionPruning, MismatchedFieldAndValueCountsAreHandledSafely) {
  const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "id", true, "long"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "identity");
  const std::vector<PartitionFieldValue> empty_values;  // shorter than spec.fields

  EXPECT_FALSE(partition_values_prove_empty(
      metadata, spec, empty_values, {PushablePredicate{"id", BinaryOperator::Equal, int_literal(999)}}));
}

TEST(PartitionPruning, AnyPredicateProvingEmptySkipsTheFile) {
  const IcebergTableMetadata metadata = metadata_with_schema({IcebergSchemaField{1, "id", true, "long"}});
  const IcebergPartitionSpec spec = spec_with_field(1, "identity");
  const std::vector<PartitionFieldValue> values = {std::int64_t{5}};

  const std::vector<PushablePredicate> predicates = {
      PushablePredicate{"unrelated_column", BinaryOperator::Equal, int_literal(1)},
      PushablePredicate{"id", BinaryOperator::Equal, int_literal(999)},
  };
  EXPECT_TRUE(partition_values_prove_empty(metadata, spec, values, predicates));
}

}  // namespace
}  // namespace kernellake::iceberg
