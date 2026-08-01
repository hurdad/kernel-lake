#include "kernellake/planner/logical_plan.hpp"

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

std::vector<std::pair<std::string, std::string>> LogicalScan::explain_attributes() const {
  std::vector<std::string> column_names;
  for (const Field& field : schema_.fields()) column_names.push_back(field.name);
  std::vector<std::pair<std::string, std::string>> attrs = {
      {"source", join_names(source_paths_)},
      {"columns", join_names(column_names)},
  };
  if (required_columns_.size() != column_names.size()) {
    attrs.push_back({"required_columns", join_names(required_columns_)});
  }
  if (!pushable_predicates_.empty()) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < pushable_predicates_.size(); ++i) {
      if (i > 0) out << ", ";
      const PushablePredicate& p = pushable_predicates_[i];
      out << p.column_name << " " << to_string(p.op) << " " << p.literal->to_string();
    }
    out << "]";
    attrs.push_back({"pushable_predicates", out.str()});
  }
  return attrs;
}

std::vector<std::pair<std::string, std::string>> LogicalProjection::explain_attributes() const {
  return {{"items", join_named_expressions(items_)}};
}

std::vector<std::pair<std::string, std::string>> LogicalAggregate::explain_attributes() const {
  std::vector<std::string> group_by_names;
  for (const NamedExpression& item : group_by_) group_by_names.push_back(item.name);
  return {
      {"group_by", join_names(group_by_names)},
      {"aggregates", join_named_expressions(aggregates_)},
  };
}

std::vector<std::pair<std::string, std::string>> LogicalSort::explain_attributes() const {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < keys_.size(); ++i) {
    if (i > 0) out << ", ";
    out << keys_[i].expr->to_string() << (keys_[i].ascending ? " ASC" : " DESC");
  }
  out << "]";
  return {{"keys", out.str()}};
}

namespace {

void explain_text_recursive(const LogicalPlanNode& node, const std::string& prefix,
                             bool is_root, std::ostringstream& out) {
  out << node.node_name() << "\n";
  const std::string attr_prefix = prefix + (is_root ? "    " : "    ");
  for (const auto& [key, value] : node.explain_attributes()) {
    out << attr_prefix << key << ": " << value << "\n";
  }
  if (node.estimated_rows.has_value()) {
    out << attr_prefix << "estimated_rows: " << *node.estimated_rows << "\n";
  }
  const std::vector<LogicalPlanPtr> children = node.children();
  for (std::size_t i = 0; i < children.size(); ++i) {
    out << attr_prefix << "└── ";
    explain_text_recursive(*children[i], attr_prefix, false, out);
  }
}

nlohmann::json explain_json_recursive(const LogicalPlanNode& node) {
  nlohmann::json j;
  j["node"] = std::string(node.node_name());
  for (const auto& [key, value] : node.explain_attributes()) {
    j["attributes"][key] = value;
  }
  if (node.estimated_rows.has_value()) {
    j["estimated_rows"] = *node.estimated_rows;
  }
  nlohmann::json children = nlohmann::json::array();
  for (const LogicalPlanPtr& child : node.children()) {
    children.push_back(explain_json_recursive(*child));
  }
  j["children"] = std::move(children);
  return j;
}

}  // namespace

std::string explain_text(const LogicalPlanNode& root) {
  std::ostringstream out;
  explain_text_recursive(root, "", true, out);
  return out.str();
}

std::string explain_json(const LogicalPlanNode& root) {
  return explain_json_recursive(root).dump(2);
}

}  // namespace kernellake
