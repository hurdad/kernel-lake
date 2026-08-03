#include "kernellake/execution_gpu/cuda_utils.hpp"

#include <string>
#include <utility>

#include "kernellake/common/errors.hpp"

namespace kernellake {

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw CudaError(std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

CudaDeviceGuard::CudaDeviceGuard(int device_id) {
  check_cuda(cudaGetDevice(&previous_device_id_), "cudaGetDevice");
  check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
}

CudaDeviceGuard::~CudaDeviceGuard() {
  cudaSetDevice(previous_device_id_);
}

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
