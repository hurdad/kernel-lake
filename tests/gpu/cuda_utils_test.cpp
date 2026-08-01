#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution/cuda_utils.hpp"

namespace kernellake {
namespace {

TEST(CudaUtils, CheckCudaPassesOnSuccess) {
  EXPECT_NO_THROW(check_cuda(cudaSuccess, "noop"));
}

TEST(CudaUtils, CheckCudaThrowsOnFailure) {
  EXPECT_THROW(check_cuda(cudaErrorInvalidValue, "bad_call"), CudaError);
}

TEST(CudaUtils, DeviceGuardRestoresPreviousDevice) {
  int before = -1;
  check_cuda(cudaGetDevice(&before), "cudaGetDevice");
  {
    CudaDeviceGuard guard(0);
    int during = -1;
    check_cuda(cudaGetDevice(&during), "cudaGetDevice");
    EXPECT_EQ(during, 0);
  }
  int after = -1;
  check_cuda(cudaGetDevice(&after), "cudaGetDevice");
  EXPECT_EQ(after, before);
}

TEST(CudaUtils, StreamIsUsableAndSynchronizable) {
  CudaStream stream;
  ASSERT_NE(stream.get(), nullptr);
  EXPECT_NO_THROW(check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize"));
}

TEST(CudaUtils, StreamMoveTransfersOwnership) {
  CudaStream a;
  cudaStream_t raw = a.get();
  CudaStream b = std::move(a);
  EXPECT_EQ(b.get(), raw);
}

}  // namespace
}  // namespace kernellake
