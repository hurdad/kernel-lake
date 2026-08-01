#pragma once

#include <cstdint>
#include <string>

namespace kernellake {

using QueryId = std::string;
using PipelineId = std::uint64_t;
using OperatorId = std::uint64_t;
using PartitionId = std::uint32_t;
using FragmentId = std::uint32_t;
using WorkerId = std::string;

}  // namespace kernellake
