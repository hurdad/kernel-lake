#include "kernellake/execution_gpu/cuda_utils.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <string>
#include <utility>

#include "kernellake/common/errors.hpp"

namespace kernellake {

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw CudaError(fmt::format("{} failed: {}", operation, cudaGetErrorString(status)));
  }
}

CudaDeviceGuard::CudaDeviceGuard(int device_id) {
  check_cuda(cudaGetDevice(&previous_device_id_), "cudaGetDevice");
  check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
}

CudaDeviceGuard::~CudaDeviceGuard() {
  // Unlike every other CUDA call in this file, this one can't route through
  // check_cuda() (a destructor is implicitly noexcept, so a throw here
  // would call std::terminate()) -- but silently discarding the error, as
  // an earlier version of this destructor did, let a failed device restore
  // pass with no signal at all: a later CUDA call made without its own
  // explicit device selection could then silently run against the wrong
  // physical GPU on a multi-GPU host. Logged instead.
  const cudaError_t status = cudaSetDevice(previous_device_id_);
  if (status != cudaSuccess) {
    spdlog::warn("CudaDeviceGuard: cudaSetDevice({}) failed while restoring the previous device: {}",
                 previous_device_id_, cudaGetErrorString(status));
  }
}

// Deliberately plain cudaStreamCreate() (a "blocking" stream, in CUDA's
// terminology), not cudaStreamCreateWithFlags(..., cudaStreamNonBlocking).
// Every cudf call in every GPU operator takes this stream explicitly (see
// ExecutionContext::stream), but arrow_bridge.cpp's to_arrow_host()/
// from_arrow() calls do not -- they run on CUDA's legacy default/null
// stream instead. A blocking-flagged stream implicitly synchronizes with
// the null stream (null-stream work waits for everything previously
// queued on any blocking stream to finish first), which is the only
// reason those default-stream calls read correct, fully-computed device
// data. Switching this to cudaStreamNonBlocking would silently break that
// ordering guarantee and race the D2H copy in to_arrow_record_batch()
// against still-in-flight kernels on this stream.
CudaStream::CudaStream() {
  check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

CudaStream::~CudaStream() {
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  if (this != &other) {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    stream_ = std::exchange(other.stream_, nullptr);
  }
  return *this;
}

}  // namespace kernellake
