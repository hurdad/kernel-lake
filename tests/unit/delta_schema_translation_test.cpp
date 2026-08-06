#include "kernellake/delta/schema_translation.hpp"

#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"

namespace kernellake::delta {
namespace {

TEST(DeltaSchemaTranslation, TranslatesEveryPrimitiveType) {
  const std::string schema_json = R"json({"type":"struct","fields":[
    {"name":"a_bool","type":"boolean","nullable":true,"metadata":{}},
    {"name":"a_byte","type":"byte","nullable":true,"metadata":{}},
    {"name":"a_short","type":"short","nullable":true,"metadata":{}},
    {"name":"an_integer","type":"integer","nullable":true,"metadata":{}},
    {"name":"a_long","type":"long","nullable":true,"metadata":{}},
    {"name":"a_float","type":"float","nullable":true,"metadata":{}},
    {"name":"a_double","type":"double","nullable":true,"metadata":{}},
    {"name":"a_date","type":"date","nullable":true,"metadata":{}},
    {"name":"a_timestamp","type":"timestamp","nullable":true,"metadata":{}},
    {"name":"a_timestamp_ntz","type":"timestamp_ntz","nullable":true,"metadata":{}},
    {"name":"a_string","type":"string","nullable":true,"metadata":{}}
  ]})json";
  const Schema schema = delta_schema_to_kernellake_schema(schema_json);
  ASSERT_EQ(schema.field_count(), 11u);
  EXPECT_EQ(schema.field(0).type.id, TypeId::Boolean);
  EXPECT_EQ(schema.field(1).type.id, TypeId::Int32);
  EXPECT_EQ(schema.field(2).type.id, TypeId::Int32);
  EXPECT_EQ(schema.field(3).type.id, TypeId::Int32);
  EXPECT_EQ(schema.field(4).type.id, TypeId::Int64);
  EXPECT_EQ(schema.field(5).type.id, TypeId::Float32);
  EXPECT_EQ(schema.field(6).type.id, TypeId::Float64);
  EXPECT_EQ(schema.field(7).type.id, TypeId::Date32);
  EXPECT_EQ(schema.field(8).type.id, TypeId::Timestamp);
  EXPECT_EQ(schema.field(9).type.id, TypeId::Timestamp);
  EXPECT_EQ(schema.field(10).type.id, TypeId::String);
}

TEST(DeltaSchemaTranslation, PreservesFieldNamesAndOrder) {
  const std::string schema_json = R"json({"type":"struct","fields":[
    {"name":"id","type":"long","nullable":false,"metadata":{}},
    {"name":"name","type":"string","nullable":true,"metadata":{}}
  ]})json";
  const Schema schema = delta_schema_to_kernellake_schema(schema_json);
  ASSERT_EQ(schema.field_count(), 2u);
  EXPECT_EQ(schema.field(0).name, "id");
  EXPECT_EQ(schema.field(1).name, "name");
}

TEST(DeltaSchemaTranslation, NullableMapsDirectlyToDataTypeNullable) {
  const std::string schema_json = R"json({"type":"struct","fields":[
    {"name":"not_nullable","type":"long","nullable":false,"metadata":{}},
    {"name":"nullable","type":"long","nullable":true,"metadata":{}}
  ]})json";
  const Schema schema = delta_schema_to_kernellake_schema(schema_json);
  EXPECT_FALSE(schema.field(0).type.nullable);
  EXPECT_TRUE(schema.field(1).type.nullable);
}

TEST(DeltaSchemaTranslation, DefaultsToNullableWhenKeyIsMissing) {
  const std::string schema_json =
      R"json({"type":"struct","fields":[{"name":"id","type":"long","metadata":{}}]})json";
  const Schema schema = delta_schema_to_kernellake_schema(schema_json);
  EXPECT_TRUE(schema.field(0).type.nullable);
}

TEST(DeltaSchemaTranslation, ParsesDecimalPrecisionAndScale) {
  const std::string schema_json =
      R"json({"type":"struct","fields":[{"name":"amount","type":"decimal(10,2)","nullable":true,"metadata":{}}]})json";
  const Schema schema = delta_schema_to_kernellake_schema(schema_json);
  ASSERT_EQ(schema.field_count(), 1u);
  EXPECT_EQ(schema.field(0).type.id, TypeId::Decimal);
  ASSERT_TRUE(schema.field(0).type.precision.has_value());
  EXPECT_EQ(*schema.field(0).type.precision, 10);
  ASSERT_TRUE(schema.field(0).type.scale.has_value());
  EXPECT_EQ(*schema.field(0).type.scale, 2);
}

TEST(DeltaSchemaTranslation, ThrowsOnUnsupportedBinaryType) {
  const std::string schema_json =
      R"json({"type":"struct","fields":[{"name":"b","type":"binary","nullable":true,"metadata":{}}]})json";
  EXPECT_THROW((void)(delta_schema_to_kernellake_schema(schema_json)), StorageError);
}

TEST(DeltaSchemaTranslation, ThrowsOnNestedArrayType) {
  const std::string schema_json =
      R"json({"type":"struct","fields":[{"name":"a","type":{"type":"array","elementType":"long","containsNull":true},"nullable":true,"metadata":{}}]})json";
  EXPECT_THROW((void)(delta_schema_to_kernellake_schema(schema_json)), StorageError);
}

TEST(DeltaSchemaTranslation, ThrowsOnMalformedDecimal) {
  const std::string schema_json =
      R"json({"type":"struct","fields":[{"name":"amount","type":"decimal(10)","nullable":true,"metadata":{}}]})json";
  EXPECT_THROW((void)(delta_schema_to_kernellake_schema(schema_json)), StorageError);
}

TEST(DeltaSchemaTranslation, EmptyFieldListProducesEmptySchema) {
  const Schema schema = delta_schema_to_kernellake_schema(R"json({"type":"struct","fields":[]})json");
  EXPECT_EQ(schema.field_count(), 0u);
}

TEST(DeltaSchemaTranslation, ThrowsOnMalformedJson) {
  EXPECT_THROW((void)(delta_schema_to_kernellake_schema("not json")), StorageError);
}

TEST(DeltaSchemaTranslation, ThrowsWhenFieldsKeyIsMissing) {
  EXPECT_THROW((void)(delta_schema_to_kernellake_schema(R"json({"type":"struct"})json")), StorageError);
}

}  // namespace
}  // namespace kernellake::delta
