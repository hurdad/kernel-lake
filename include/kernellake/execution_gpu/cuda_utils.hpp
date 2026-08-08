#pragma once

#include <cuda_runtime.h>

#include <string_view>

namespace kernellake {

// Throws CudaError (naming `operation` and the CUDA error string) if
// `status` is not cudaSuccess. Every CUDA API call in KernelLake must be
// wrapped in this -- unchecked CUDA calls are not allowed.
void check_cuda(cudaError_t status, std::string_view operation);

// RAII CUDA device selection: sets the device on construction, restores
// whatever device was current beforehand on destruction.
class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(int device_id);
  ~CudaDeviceGuard();

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

 private:
  int previous_device_id_;
};

// RAII CUDA stream, deliberately *not* created with cudaStreamNonBlocking
// (see CudaStream::CudaStream() in cuda_utils.cpp for why that matters --
// it's load-bearing, not just an unset flag).
class CudaStream {
 public:
  CudaStream();
  ~CudaStream();

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

}  // namespace kernellake
