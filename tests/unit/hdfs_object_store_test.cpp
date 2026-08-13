// Direct unit tests for detail::strip_hdfs_authority()
// (src/storage/hdfs_object_store.cpp/generic_fs_object_store.hpp) -- pure
// string parsing, no real HDFS namenode/libhdfs needed. Unlike S3/GCS/Azure
// ("scheme://bucket/key", where the first path component is itself the
// addressed resource), an "hdfs://namenode:port/path" URI's authority names
// a namenode connection HadoopFileSystem already has from its own
// connection_config -- this function strips that authority before
// generic_fs_list()/generic_fs_open() (shared with S3/GCS/Azure) see the
// URI, leaving just the real path. See that function's own comment for the
// full rationale.
#include <gtest/gtest.h>

#include "generic_fs_object_store.hpp"

namespace kernellake {
namespace {

TEST(StripHdfsAuthority, StripsNamenodeAndPortLeavingJustThePath) {
  const Uri result = detail::strip_hdfs_authority(Uri("hdfs://namenode:8020/user/data/part.parquet"));
  EXPECT_EQ(result.value(), "hdfs:///user/data/part.parquet");
}

TEST(StripHdfsAuthority, NoSlashFoundLeavesAuthorityUnchanged) {
  // Documented edge case (see strip_hdfs_authority()'s own comment on the
  // find('/') == npos branch): a URI with an authority but no path at all
  // has nothing to strip, so the whole thing passes through unchanged.
  const Uri result = detail::strip_hdfs_authority(Uri("hdfs://namenode:8020"));
  EXPECT_EQ(result.value(), "hdfs://namenode:8020");
}

TEST(StripHdfsAuthority, TrailingSlashOnlyLeavesRootPath) {
  const Uri result = detail::strip_hdfs_authority(Uri("hdfs://namenode:8020/"));
  EXPECT_EQ(result.value(), "hdfs:///");
}

TEST(StripHdfsAuthority, EmptyUriHasNothingToStrip) {
  const Uri result = detail::strip_hdfs_authority(Uri(""));
  EXPECT_EQ(result.value(), "hdfs://");
}

}  // namespace
}  // namespace kernellake
