// Provides QueryEngine::execute() for CPU-only (KERNELLAKE_WITH_CUDA=OFF)
// builds. Mutually exclusive with query_engine_execute_gpu.cpp -- exactly
// one of the two is compiled into kernellake_api, selected by
// src/api/CMakeLists.txt based on KERNELLAKE_WITH_CUDA, so there is never an
// ODR conflict between them.
#include "kernellake/api/query_engine.hpp"

#include "kernellake/common/errors.hpp"

namespace kernellake {

QueryResult QueryEngine::execute(std::string_view /*sql*/) const {
  throw ExecutionError(
      "query execution requires GPU operators (libcudf/RMM), which are not part of this build; "
      "use `kernellake explain --sql ...` to see the plan KernelLake would run, or build with "
      "-DKERNELLAKE_WITH_CUDA=ON once libcudf/RMM are installed");
}

}  // namespace kernellake
