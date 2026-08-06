#include "kernellake/iceberg/schema_translation.hpp"

#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"

namespace kernellake::iceberg {
namespace {

TEST(SchemaTranslation, TranslatesEveryPrimitiveType) {
  const std::vector<IcebergSchemaField> fields = {
      {1, "a_bool", true, "boolean"},
      {2, "an_int", true, "int"},
      {3, "a_long", true, "long"},
      {4, "a_float", true, "float"},
      {5, "a_double", true, "double"},
      {6, "a_date", true, "date"},
      {7, "a_timestamp", true, "timestamp"},
      {8, "a_timestamptz", true, "timestamptz"},
      {9, "a_string", true, "string"},
  };
  const Schema schema = iceberg_schema_to_kernellake_schema(fields);
  ASSERT_EQ(schema.field_count(), 9u);
  EXPECT_EQ(schema.field(0).type.id, TypeId::Boolean);
  EXPECT_EQ(schema.field(1).type.id, TypeId::Int32);
  EXPECT_EQ(schema.field(2).type.id, TypeId::Int64);
  EXPECT_EQ(schema.field(3).type.id, TypeId::Float32);
  EXPECT_EQ(schema.field(4).type.id, TypeId::Float64);
  EXPECT_EQ(schema.field(5).type.id, TypeId::Date32);
  EXPECT_EQ(schema.field(6).type.id, TypeId::Timestamp);
  EXPECT_EQ(schema.field(7).type.id, TypeId::Timestamp);
  EXPECT_EQ(schema.field(8).type.id, TypeId::String);
}

TEST(SchemaTranslation, PreservesFieldNamesAndOrder) {
  const std::vector<IcebergSchemaField> fields = {
      {1, "id", true, "long"},
      {2, "name", false, "string"},
  };
  const Schema schema = iceberg_schema_to_kernellake_schema(fields);
  ASSERT_EQ(schema.field_count(), 2u);
  EXPECT_EQ(schema.field(0).name, "id");
  EXPECT_EQ(schema.field(1).name, "name");
}

TEST(SchemaTranslation, RequiredMapsToNonNullableAndOptionalToNullable) {
  const std::vector<IcebergSchemaField> fields = {
      {1, "required_col", true, "long"},
      {2, "optional_col", false, "long"},
  };
  const Schema schema = iceberg_schema_to_kernellake_schema(fields);
  EXPECT_FALSE(schema.field(0).type.nullable);
  EXPECT_TRUE(schema.field(1).type.nullable);
}

TEST(SchemaTranslation, ParsesDecimalPrecisionAndScale) {
  const std::vector<IcebergSchemaField> fields = {{1, "amount", true, "decimal(10,2)"}};
  const Schema schema = iceberg_schema_to_kernellake_schema(fields);
  ASSERT_EQ(schema.field_count(), 1u);
  EXPECT_EQ(schema.field(0).type.id, TypeId::Decimal);
  ASSERT_TRUE(schema.field(0).type.precision.has_value());
  EXPECT_EQ(*schema.field(0).type.precision, 10);
  ASSERT_TRUE(schema.field(0).type.scale.has_value());
  EXPECT_EQ(*schema.field(0).type.scale, 2);
}

TEST(SchemaTranslation, ThrowsOnUnsupportedTimeType) {
  const std::vector<IcebergSchemaField> fields = {{1, "t", true, "time"}};
  EXPECT_THROW((void)(iceberg_schema_to_kernellake_schema(fields)), StorageError);
}

TEST(SchemaTranslation, ThrowsOnUnsupportedUuidType) {
  const std::vector<IcebergSchemaField> fields = {{1, "u", true, "uuid"}};
  EXPECT_THROW((void)(iceberg_schema_to_kernellake_schema(fields)), StorageError);
}

TEST(SchemaTranslation, ThrowsOnNestedStructType) {
  const std::vector<IcebergSchemaField> fields = {
      {1, "nested", true, R"({"type":"struct","fields":[]})"}};
  EXPECT_THROW((void)(iceberg_schema_to_kernellake_schema(fields)), StorageError);
}

TEST(SchemaTranslation, ThrowsOnMalformedDecimal) {
  const std::vector<IcebergSchemaField> fields = {{1, "amount", true, "decimal(10)"}};
  EXPECT_THROW((void)(iceberg_schema_to_kernellake_schema(fields)), StorageError);
}

TEST(SchemaTranslation, EmptyFieldListProducesEmptySchema) {
  const Schema schema = iceberg_schema_to_kernellake_schema({});
  EXPECT_EQ(schema.field_count(), 0u);
}

}  // namespace
}  // namespace kernellake::iceberg
