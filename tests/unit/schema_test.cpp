#include <gtest/gtest.h>

#include "kernellake/types/schema.hpp"

namespace kernellake {
namespace {

TEST(Schema, FindFieldReturnsIndex) {
  Schema schema({Field{"region", string_type()}, Field{"amount", float64_type()}});
  EXPECT_EQ(schema.find_field("amount"), 1u);
  EXPECT_EQ(schema.find_field("missing"), std::nullopt);
}

TEST(Schema, EqualsComparesNameAndType) {
  Schema a({Field{"x", int32_type()}});
  Schema b({Field{"x", int32_type()}});
  Schema c({Field{"x", int64_type()}});
  EXPECT_TRUE(a.equals(b));
  EXPECT_FALSE(a.equals(c));
}

TEST(DataType, ToStringIncludesNullability) {
  EXPECT_EQ(int32_type(true).to_string(), "INT32");
  EXPECT_EQ(int32_type(false).to_string(), "INT32 NOT NULL");
  EXPECT_EQ(decimal_type(10, 2).to_string(), "DECIMAL(10,2)");
}

}  // namespace
}  // namespace kernellake
