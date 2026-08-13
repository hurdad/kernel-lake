// Regression coverage for device_read()/device_read_async() -- confirmed,
// despite the class-level "GDS device-direct read" framing, these do not
// actually depend on cuFile/GDS hardware at all: device_read_async() reads
// through the same host_read() every host-side caller uses, then does a
// plain cudaMemcpyAsync to device memory (see object_store_datasource.cpp
// itself). That means this path is fully testable against ordinary local
// files, and was simply never tested -- real S3 scans currently only ever
// call host_read(), leaving this dormant in production but not untestable.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>

#include <cstring>
#include <filesystem>
#include <vector>

#include "kernellake/execution_gpu/object_store_datasource.hpp"
#include "kernellake/memory/rmm_environment.hpp"
#include "kernellake/storage/local_object_store.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

class ObjectStoreDatasourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_object_store_datasource_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "data.bin").string();
    contents_ = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    auto out = arrow::io::FileOutputStream::Open(path_).ValueOrDie();
    ASSERT_TRUE(out->Write(contents_.data(), static_cast<std::int64_t>(contents_.size())).ok());
    ASSERT_TRUE(out->Close().ok());
  }

  void TearDown() override { fs::remove_all(dir_); }

  std::unique_ptr<ObjectStoreDatasource> open_datasource() {
    return std::make_unique<ObjectStoreDatasource>(store_.open(Uri(path_)));
  }

  fs::path dir_;
  std::string path_;
  std::string contents_;
  LocalObjectStore store_;
};

TEST_F(ObjectStoreDatasourceTest, DeviceReadIntoDestinationBufferMatchesHostRead) {
  RmmEnvironment env(default_config());
  std::unique_ptr<ObjectStoreDatasource> datasource = open_datasource();

  constexpr std::size_t kOffset = 5;
  constexpr std::size_t kSize = 10;  // "56789ABCDE"
  rmm::device_buffer device_dst(kSize, cudf::get_default_stream());
  const std::size_t bytes_read = datasource->device_read(
      kOffset, kSize, static_cast<std::uint8_t*>(device_dst.data()), cudf::get_default_stream());
  ASSERT_EQ(bytes_read, kSize);
  cudf::get_default_stream().synchronize();

  std::vector<char> host_copy(kSize);
  cudaMemcpy(host_copy.data(), device_dst.data(), kSize, cudaMemcpyDeviceToHost);
  EXPECT_EQ(std::string(host_copy.data(), kSize), contents_.substr(kOffset, kSize));
}

TEST_F(ObjectStoreDatasourceTest, DeviceReadAsyncBufferOverloadMatchesHostRead) {
  RmmEnvironment env(default_config());
  std::unique_ptr<ObjectStoreDatasource> datasource = open_datasource();

  constexpr std::size_t kOffset = 0;
  constexpr std::size_t kSize = 5;  // "01234"
  std::unique_ptr<cudf::io::datasource::buffer> buffer =
      datasource->device_read(kOffset, kSize, cudf::get_default_stream());
  ASSERT_NE(buffer, nullptr);
  ASSERT_EQ(buffer->size(), kSize);
  cudf::get_default_stream().synchronize();

  std::vector<char> host_copy(kSize);
  cudaMemcpy(host_copy.data(), buffer->data(), kSize, cudaMemcpyDeviceToHost);
  EXPECT_EQ(std::string(host_copy.data(), kSize), contents_.substr(kOffset, kSize));
}

TEST_F(ObjectStoreDatasourceTest, DeviceReadAsyncFutureCompletesWithCorrectByteCount) {
  RmmEnvironment env(default_config());
  std::unique_ptr<ObjectStoreDatasource> datasource = open_datasource();

  constexpr std::size_t kOffset = 20;
  constexpr std::size_t kSize = 8;
  rmm::device_buffer device_dst(kSize, cudf::get_default_stream());
  std::future<std::size_t> pending = datasource->device_read_async(
      kOffset, kSize, static_cast<std::uint8_t*>(device_dst.data()), cudf::get_default_stream());
  const std::size_t bytes_read = pending.get();
  EXPECT_EQ(bytes_read, kSize);
  cudf::get_default_stream().synchronize();

  std::vector<char> host_copy(kSize);
  cudaMemcpy(host_copy.data(), device_dst.data(), kSize, cudaMemcpyDeviceToHost);
  EXPECT_EQ(std::string(host_copy.data(), kSize), contents_.substr(kOffset, kSize));
}

}  // namespace
}  // namespace kernellake
