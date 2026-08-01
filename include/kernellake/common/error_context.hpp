#pragma once

#include <optional>
#include <sstream>
#include <string>

namespace kernellake {

// Builds actionable error messages that consistently include the operation
// name plus whichever contextual fields are relevant (file/URI, CUDA device,
// query ID, benchmark query/scale factor, underlying error, remediation).
// Every KernelLake error thrown from a non-trivial call site should be built
// through this rather than an ad hoc string, so messages stay uniform.
class ErrorContextBuilder {
public:
  explicit ErrorContextBuilder(std::string operation) : operation_(std::move(operation)) {}

  ErrorContextBuilder& uri(std::string value) {
    uri_ = std::move(value);
    return *this;
  }

  ErrorContextBuilder& cuda_device(int device_id) {
    cuda_device_ = device_id;
    return *this;
  }

  ErrorContextBuilder& query_id(std::string value) {
    query_id_ = std::move(value);
    return *this;
  }

  ErrorContextBuilder& benchmark(std::string query, int scale_factor) {
    benchmark_query_ = std::move(query);
    benchmark_scale_factor_ = scale_factor;
    return *this;
  }

  ErrorContextBuilder& underlying_error(std::string value) {
    underlying_error_ = std::move(value);
    return *this;
  }

  ErrorContextBuilder& remediation(std::string value) {
    remediation_ = std::move(value);
    return *this;
  }

  [[nodiscard]] std::string build(std::string_view message) const {
    std::ostringstream out;
    out << operation_ << ": " << message;
    if (uri_) out << " [uri=" << *uri_ << "]";
    if (cuda_device_) out << " [cuda_device=" << *cuda_device_ << "]";
    if (query_id_) out << " [query_id=" << *query_id_ << "]";
    if (benchmark_query_) {
      out << " [benchmark_query=" << *benchmark_query_
          << " scale_factor=" << benchmark_scale_factor_.value_or(0) << "]";
    }
    if (underlying_error_) out << " [underlying_error=" << *underlying_error_ << "]";
    if (remediation_) out << " [remediation=" << *remediation_ << "]";
    return out.str();
  }

private:
  std::string operation_;
  std::optional<std::string> uri_;
  std::optional<int> cuda_device_;
  std::optional<std::string> query_id_;
  std::optional<std::string> benchmark_query_;
  std::optional<int> benchmark_scale_factor_;
  std::optional<std::string> underlying_error_;
  std::optional<std::string> remediation_;
};

}  // namespace kernellake
