#include <gtest/gtest.h>

#include "kernellake/common/error_context.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake {
namespace {

TEST(ErrorHierarchy, DerivesFromKernelLakeError) {
  static_assert(std::is_base_of_v<KernelLakeError, StorageError>);
  static_assert(std::is_base_of_v<KernelLakeError, SqlError>);
  static_assert(std::is_base_of_v<KernelLakeError, BindingError>);

  EXPECT_THROW(
      []() { throw StorageError("missing file"); }(),
      KernelLakeError);
}

TEST(ErrorHierarchy, PreservesMessage) {
  try {
    throw ConfigurationError("bad batch_rows value");
  } catch (const KernelLakeError& e) {
    EXPECT_STREQ(e.what(), "bad batch_rows value");
  }
}

TEST(ErrorContextBuilder, IncludesRequestedFields) {
  ErrorContextBuilder builder("parquet_scan");
  const std::string message =
      builder.uri("/data/sales/part-0.parquet").query_id("q-1").build("failed to open file");

  EXPECT_NE(message.find("parquet_scan"), std::string::npos);
  EXPECT_NE(message.find("/data/sales/part-0.parquet"), std::string::npos);
  EXPECT_NE(message.find("q-1"), std::string::npos);
}

}  // namespace
}  // namespace kernellake
