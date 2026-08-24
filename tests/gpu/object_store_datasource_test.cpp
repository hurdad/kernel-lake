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

#include <rmm/mr/per_device_resource.hpp>

#include <cstring>
#include <filesystem>
#include <future>
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

  std::unique_ptr<ObjectStoreDatasource> open_datasource(const rmm::device_async_resource_ref& mr) {
    return std::make_unique<ObjectStoreDatasource>(store_.open(Uri(path_)), mr);
  }

  fs::path dir_;
  std::string path_;
  std::string contents_;
  LocalObjectStore store_;
};

TEST_F(ObjectStoreDatasourceTest, DeviceReadIntoDestinationBufferMatchesHostRead) {
  RmmEnvironment env(default_config());
  std::unique_ptr<ObjectStoreDatasource> datasource =
      open_datasource(rmm::mr::get_current_device_resource_ref());

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
  std::unique_ptr<ObjectStoreDatasource> datasource =
      open_datasource(rmm::mr::get_current_device_resource_ref());

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

TEST_F(ObjectStoreDatasourceTest, SizeMatchesUnderlyingObjectSize) {
  const std::unique_ptr<ObjectStoreDatasource> datasource =
      open_datasource(rmm::mr::get_current_device_resource_ref());
  EXPECT_EQ(datasource->size(), contents_.size());
}

// The dst-pointer host_read() overload (as opposed to the buffer-returning
// one every device_read()/device_read_async() test above goes through) had
// no coverage of its own -- cudf's own chunked_parquet_reader never calls
// it for a real scan (see this class's own comment), but it is still part
// of cudf::io::datasource's public override contract and must work
// correctly for any caller that does use it directly.
TEST_F(ObjectStoreDatasourceTest, HostReadIntoDestinationBufferMatchesHostReadBufferOverload) {
  std::unique_ptr<ObjectStoreDatasource> datasource =
      open_datasource(rmm::mr::get_current_device_resource_ref());

  constexpr std::size_t kOffset = 3;
  constexpr std::size_t kSize = 6;  // "3456789"[:6] == "345678"
  std::vector<std::uint8_t> dst(kSize);
  const std::size_t bytes_read = datasource->host_read(kOffset, kSize, dst.data());
  ASSERT_EQ(bytes_read, kSize);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(dst.data()), kSize), contents_.substr(kOffset, kSize));
}

// Same rationale as HostReadIntoDestinationBufferMatchesHostReadBufferOverload
// above -- neither host_read_async() overload is ever called by cudf's own
// real scan path either, but both are still real, callable overrides.
TEST_F(ObjectStoreDatasourceTest, HostReadAsyncBufferOverloadMatchesHostRead) {
  std::unique_ptr<ObjectStoreDatasource> datasource =
      open_datasource(rmm::mr::get_current_device_resource_ref());

  constexpr std::size_t kOffset = 10;
  constexpr std::size_t kSize = 8;
  std::future<std::unique_ptr<cudf::io::datasource::buffer>> pending =
      datasource->host_read_async(kOffset, kSize);
  const std::unique_ptr<cudf::io::datasource::buffer> buffer = pending.get();
  ASSERT_NE(buffer, nullptr);
  ASSERT_EQ(buffer->size(), kSize);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(buffer->data()), kSize),
            contents_.substr(kOffset, kSize));
}

TEST_F(ObjectStoreDatasourceTest, HostReadAsyncDestinationBufferOverloadMatchesHostRead) {
  std::unique_ptr<ObjectStoreDatasource> datasource =
      open_datasource(rmm::mr::get_current_device_resource_ref());

  constexpr std::size_t kOffset = 2;
  constexpr std::size_t kSize = 5;
  std::vector<std::uint8_t> dst(kSize);
  std::future<std::size_t> pending = datasource->host_read_async(kOffset, kSize, dst.data());
  const std::size_t bytes_read = pending.get();
  ASSERT_EQ(bytes_read, kSize);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(dst.data()), kSize), contents_.substr(kOffset, kSize));
}

// device_read()'s short-read handling (buffer.resize() down to the real
// bytes_read) had no coverage -- every existing device_read test above
// requests a range fully inside the file. Requesting past EOF is a
// legitimate, real caller pattern (a generous end-of-range guess), not an
// error case.
TEST_F(ObjectStoreDatasourceTest, DeviceReadPastEndOfFileShrinksBufferToActualBytesRead) {
  RmmEnvironment env(default_config());
  std::unique_ptr<ObjectStoreDatasource> datasource =
      open_datasource(rmm::mr::get_current_device_resource_ref());

  const std::size_t offset = contents_.size() - 5;
  const std::size_t requested_size = 50;  // far past EOF
  const std::size_t expected_bytes = contents_.size() - offset;
  std::unique_ptr<cudf::io::datasource::buffer> buffer =
      datasource->device_read(offset, requested_size, cudf::get_default_stream());
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->size(), expected_bytes);
  cudf::get_default_stream().synchronize();

  std::vector<char> host_copy(expected_bytes);
  cudaMemcpy(host_copy.data(), buffer->data(), expected_bytes, cudaMemcpyDeviceToHost);
  EXPECT_EQ(std::string(host_copy.data(), expected_bytes), contents_.substr(offset));
}

TEST_F(ObjectStoreDatasourceTest, DeviceReadAsyncFutureCompletesWithCorrectByteCount) {
  RmmEnvironment env(default_config());
  std::unique_ptr<ObjectStoreDatasource> datasource =
      open_datasource(rmm::mr::get_current_device_resource_ref());

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
