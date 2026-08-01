#include "commands.hpp"

#include <cstdio>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/planner/logical_plan.hpp"
#include "kernellake/planner/physical_plan.hpp"

namespace kernellake::cli {

int run_explain(const std::vector<std::string_view>& args, const EngineConfig& config) {
  std::string sql;
  std::string format = "text";
  bool logical = false;

  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--sql" && i + 1 < args.size()) {
      sql = args[++i];
    } else if (args[i] == "--format" && i + 1 < args.size()) {
      format = args[++i];
    } else if (args[i] == "--logical") {
      logical = true;
    }
  }

  if (sql.empty()) {
    std::fprintf(stderr, "kernellake explain: --sql is required\n");
    return 1;
  }
  if (format != "text" && format != "json") {
    std::fprintf(stderr, "kernellake explain: --format must be 'text' or 'json', got '%s'\n",
                 format.c_str());
    return 1;
  }

  try {
    QueryEngine engine(config);
    if (logical) {
      const LogicalPlanPtr plan = engine.explain_logical(sql);
      std::printf("%s\n", (format == "json" ? explain_json(*plan) : explain_text(*plan)).c_str());
    } else {
      const PhysicalPlanPtr plan = engine.explain(sql);
      std::printf("%s\n", (format == "json" ? explain_json(*plan) : explain_text(*plan)).c_str());
    }
  } catch (const KernelLakeError& e) {
    std::fprintf(stderr, "kernellake explain: %s\n", e.what());
    return 1;
  }
  return 0;
}

}  // namespace kernellake::cli
