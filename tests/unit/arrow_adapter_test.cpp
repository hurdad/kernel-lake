#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/types/arrow_adapter.hpp"

namespace kernellake {
namespace {

TEST(ArrowAdapter, RoundTripsPrimitiveTypes) {
  for (const DataType& type : {int32_type(), int64_type(), float64_type(), string_type(), date32_type(),
                               timestamp_type(), boolean_type()}) {
    const auto arrow_type = to_arrow_type(type);
    const DataType round_tripped = from_arrow_type(arrow_type, type.nullable);
    EXPECT_EQ(round_tripped.id, type.id);
  }
}

TEST(ArrowAdapter, RoundTripsDecimal) {
  const DataType decimal = decimal_type(12, 3);
  const auto arrow_type = to_arrow_type(decimal);
  const DataType round_tripped = from_arrow_type(arrow_type, true);
  EXPECT_EQ(round_tripped.id, TypeId::Decimal);
  EXPECT_EQ(round_tripped.precision, 12);
  EXPECT_EQ(round_tripped.scale, 3);
}

TEST(ArrowAdapter, SchemaRoundTripPreservesFieldOrder) {
  Schema schema({Field{"region", string_type()}, Field{"amount", float64_type(false)}});
  const auto arrow_schema = to_arrow_schema(schema);
  ASSERT_EQ(arrow_schema->num_fields(), 2);
  EXPECT_EQ(arrow_schema->field(0)->name(), "region");
  EXPECT_EQ(arrow_schema->field(1)->name(), "amount");
  EXPECT_FALSE(arrow_schema->field(1)->nullable());

  const Schema round_tripped = from_arrow_schema(arrow_schema);
  EXPECT_TRUE(round_tripped.equals(schema));
}

TEST(ArrowAdapter, UnsupportedArrowTypeThrows) {
  auto list_type = arrow::list(arrow::int32());
  EXPECT_THROW((void)({ auto ignored = from_arrow_type(list_type, true); }), PlanningError);
}

}  // namespace
}  // namespace kernellake
