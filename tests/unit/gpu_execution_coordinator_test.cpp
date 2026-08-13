// Coverage for GpuExecutionCoordinator's KERNELLAKE_WITH_CUDA=OFF variant
// (gpu_execution_coordinator_stub.cpp) -- never constructed by any test
// before this file, in either build mode. This is a Flight SQL server type
// (server.engine.backend == "gpu" wiring, see the class's own doc comment
// in gpu_execution_coordinator.hpp), so like flight_sql_server_test.cpp/
// auth_middleware_test.cpp it only builds/links under KERNELLAKE_BUILD_SERVER
// (see tests/unit/CMakeLists.txt) -- e.g. the `server-dev` preset, not
// `cpu-dev` alone.
//
// The KERNELLAKE_WITH_CUDA=ON variant (gpu_execution_coordinator_gpu.cpp)
// needs a real GPU/CUDA toolkit and is out of scope here -- see
// docs/ARCHITECTURE.md and this repo's `gpu-dev-wsl2` preset for that build.
// src/server/CMakeLists.txt selects between the two variants via
// KERNELLAKE_WITH_CUDA, so this test is guarded by the same flag: it
// asserts on the stub's documented fail-fast behavior specifically, which
// would be a wrong assertion to compile (and, worse, silently never build)
// against the real GPU variant.
#include <gtest/gtest.h>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/server/gpu_execution_coordinator.hpp"

#ifndef KERNELLAKE_WITH_CUDA

namespace kernellake {
namespace {

TEST(GpuExecutionCoordinatorStub, ConstructorThrowsConfigurationErrorFailingFast) {
  EngineConfig config = default_config();
  config.engine.backend = "gpu";
  EXPECT_THROW((void)(GpuExecutionCoordinator(config)), ConfigurationError);
}

TEST(GpuExecutionCoordinatorStub, ConstructorErrorMessageExplainsHowToGetRealGpuSupport) {
  EngineConfig config = default_config();
  config.engine.backend = "gpu";
  try {
    (void)(GpuExecutionCoordinator(config));
    FAIL() << "expected GpuExecutionCoordinator construction to throw in a KERNELLAKE_WITH_CUDA=OFF build";
  } catch (const ConfigurationError& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("KERNELLAKE_WITH_CUDA"), std::string::npos) << message;
    EXPECT_NE(message.find("cpu"), std::string::npos) << message;
  }
}

}  // namespace
}  // namespace kernellake

#endif  // !KERNELLAKE_WITH_CUDA
