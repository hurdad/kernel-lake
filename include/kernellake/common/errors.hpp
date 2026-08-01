#pragma once

#include <stdexcept>

namespace kernellake {

// Base class for every KernelLake-specific error. Callers should catch this
// (or a specific subclass) rather than std::exception when they want to
// distinguish KernelLake failures from third-party library exceptions.
class KernelLakeError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class ConfigurationError : public KernelLakeError {
public:
  using KernelLakeError::KernelLakeError;
};

class CudaError : public KernelLakeError {
public:
  using KernelLakeError::KernelLakeError;
};

class SqlError : public KernelLakeError {
public:
  using KernelLakeError::KernelLakeError;
};

class BindingError : public KernelLakeError {
public:
  using KernelLakeError::KernelLakeError;
};

class PlanningError : public KernelLakeError {
public:
  using KernelLakeError::KernelLakeError;
};

class OptimizationError : public KernelLakeError {
public:
  using KernelLakeError::KernelLakeError;
};

class ExecutionError : public KernelLakeError {
public:
  using KernelLakeError::KernelLakeError;
};

class StorageError : public KernelLakeError {
public:
  using KernelLakeError::KernelLakeError;
};

class BenchmarkError : public KernelLakeError {
public:
  using KernelLakeError::KernelLakeError;
};

class OutOfMemoryError : public KernelLakeError {
public:
  using KernelLakeError::KernelLakeError;
};

}  // namespace kernellake
