#include <gtest/gtest.h>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cuda_utils.hpp"

namespace kernellake {
namespace {

TEST(CudaUtils, CheckCudaPassesOnSuccess) {
  EXPECT_NO_THROW((void)(check_cuda(cudaSuccess, "noop")));
}

TEST(CudaUtils, CheckCudaThrowsOnFailure) {
  EXPECT_THROW((void)(check_cuda(cudaErrorInvalidValue, "bad_call")), CudaError);
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
  EXPECT_NO_THROW((void)(check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize")));
}

TEST(CudaUtils, StreamMoveTransfersOwnership) {
  CudaStream a;
  cudaStream_t raw = a.get();
  CudaStream b = std::move(a);
  EXPECT_EQ(b.get(), raw);
}

// Companion to StreamMoveTransfersOwnership above, which only exercises the
// move *constructor* -- operator=(CudaStream&&) had no coverage of its own
// (destroying the target's existing stream before taking ownership of the
// source's, unlike construction which starts from a null stream_).
TEST(CudaUtils, StreamMoveAssignmentDestroysPreviousStreamAndTransfersOwnership) {
  CudaStream a;
  CudaStream b;
  cudaStream_t raw_a = a.get();
  ASSERT_NE(raw_a, b.get());
  b = std::move(a);
  EXPECT_EQ(b.get(), raw_a);
  // b's original stream was destroyed, not leaked -- the moved-from a no
  // longer owns a stream either, so only b.get() should be a live handle
  // now; confirm it's still genuinely usable post-assignment.
  EXPECT_NO_THROW((void)(check_cuda(cudaStreamSynchronize(b.get()), "cudaStreamSynchronize")));
}

// Guards the `if (this != &other)` self-assignment check -- without it,
// `a = std::move(a)` would destroy a's own stream_ before the
// std::exchange on the right-hand side reads it, a real double-free/use-
// after-free. Routed through a pointer so the self-assignment isn't
// visible to the compiler as a literal `a = std::move(a)` expression.
TEST(CudaUtils, StreamSelfMoveAssignmentIsSafeNoOp) {
  CudaStream a;
  cudaStream_t raw = a.get();
  CudaStream* self_ptr = &a;
  *self_ptr = std::move(*self_ptr);
  EXPECT_EQ(a.get(), raw);
  EXPECT_NO_THROW((void)(check_cuda(cudaStreamSynchronize(a.get()), "cudaStreamSynchronize")));
}

}  // namespace
}  // namespace kernellake
