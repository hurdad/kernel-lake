#include "kernellake/planner/physical_plan.hpp"

#include <nlohmann/json.hpp>

#include <sstream>

namespace kernellake {

namespace {

std::string join_names(const std::vector<std::string>& names) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i > 0) out << ", ";
    out << names[i];
  }
  out << "]";
  return out.str();
}

std::string join_named_expressions(const std::vector<NamedExpression>& items) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i > 0) out << ", ";
    out << items[i].expr->to_string() << " AS " << items[i].name;
  }
  out << "]";
  return out.str();
}

}  // namespace

std::vector<std::pair<std::string, std::string>> ParquetScanNode::explain_attributes() const {
  int total_row_groups = 0;
  int selected_row_groups = 0;
  for (const PhysicalFileFragment& fragment : fragments_) {
    total_row_groups += fragment.total_row_groups;
    selected_row_groups += static_cast<int>(fragment.selected_row_groups.size());
  }
  // Files entirely pruned away are considered but not counted in
  // total_row_groups above (they contributed no fragment), which is correct
  // for a "row groups scanned / row groups considered among scanned files"
  // metric; a fuller "considered across all files" count would need the
  // per-file total_row_groups even for skipped files, which the physical
  // planner does not currently retain for fully-pruned files.
  return {
      {"files", std::to_string(fragments_.size()) + "/" + std::to_string(files_considered_)},
      {"row_groups", std::to_string(selected_row_groups) + "/" + std::to_string(total_row_groups)},
      {"columns", join_names(columns_)},
  };
}

std::vector<std::pair<std::string, std::string>> ProjectionNode::explain_attributes() const {
  return {{"items", join_named_expressions(items_)}};
}

std::vector<std::pair<std::string, std::string>> HashAggregateNode::explain_attributes() const {
  std::vector<std::string> group_by_names;
  for (const NamedExpression& item : group_by_) group_by_names.push_back(item.name);
  return {
      {"group_by", join_names(group_by_names)},
      {"aggregates", join_named_expressions(aggregates_)},
  };
}

std::vector<std::pair<std::string, std::string>> ScalarAggregateNode::explain_attributes() const {
  return {{"aggregates", join_named_expressions(aggregates_)}};
}

namespace {

void explain_text_recursive(const PhysicalPlanNode& node, const std::string& prefix,
                            std::ostringstream& out) {
  out << node.node_name() << "\n";
  const std::string attr_prefix = prefix + "    ";
  for (const auto& [key, value] : node.explain_attributes()) {
    out << attr_prefix << key << ": " << value << "\n";
  }
  for (const PhysicalPlanPtr& child : node.children()) {
    out << attr_prefix << "└── ";
    explain_text_recursive(*child, attr_prefix, out);
  }
}

nlohmann::json explain_json_recursive(const PhysicalPlanNode& node) {
  nlohmann::json j;
  j["node"] = std::string(node.node_name());
  for (const auto& [key, value] : node.explain_attributes()) {
    j["attributes"][key] = value;
  }
  nlohmann::json children = nlohmann::json::array();
  for (const PhysicalPlanPtr& child : node.children()) {
    children.push_back(explain_json_recursive(*child));
  }
  j["children"] = std::move(children);
  return j;
}

}  // namespace

std::string explain_text(const PhysicalPlanNode& root) {
  std::ostringstream out;
  explain_text_recursive(root, "", out);
  return out.str();
}

std::string explain_json(const PhysicalPlanNode& root) {
  return explain_json_recursive(root).dump(2);
}

}  // namespace kernellake
