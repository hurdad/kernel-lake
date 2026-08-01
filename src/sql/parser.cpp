#include "kernellake/sql/parser.hpp"

#include <SQLParser.h>

#include <regex>
#include <string>
#include <vector>

#include "kernellake/common/date_util.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake::sql {

namespace {

constexpr std::string_view kPlaceholderTable = "kernellake_parquet_source";

struct Preprocessed {
  std::string sql;
  std::vector<std::string> paths;
};

// hyrise/sql-parser's FROM-clause grammar only accepts table names, joins,
// and subqueries -- it has no notion of a table-valued function call like
// `read_parquet('...')`. We recognize that one specific shape ourselves,
// extract its string-literal path arguments, and substitute a plain
// placeholder identifier before handing the query to the parser. This is a
// deliberate, narrow, documented preprocessing step -- not general
// SQL-string rewriting -- and any FROM clause that does not match it fails
// clearly rather than being silently reinterpreted.
Preprocessed preprocess_from_read_parquet(const std::string& sql) {
  static const std::regex kFromPattern(
      R"(FROM\s+read_parquet\s*\(\s*((?:'(?:[^'\\]|\\.)*'\s*,\s*)*'(?:[^'\\]|\\.)*')\s*\))",
      std::regex::icase);

  std::smatch match;
  if (!std::regex_search(sql, match, kFromPattern)) {
    throw SqlError(
        "KernelLake requires a single data source of the form "
        "FROM read_parquet('path' [, 'path2', ...]); no such clause was found in the query");
  }

  const std::string args = match[1].str();
  static const std::regex kStringLiteralPattern(R"('((?:[^'\\]|\\.)*)')");
  std::vector<std::string> paths;
  for (auto it = std::sregex_iterator(args.begin(), args.end(), kStringLiteralPattern);
       it != std::sregex_iterator(); ++it) {
    paths.push_back((*it)[1].str());
  }

  std::string rewritten = sql;
  rewritten.replace(static_cast<std::size_t>(match.position(0)),
                     static_cast<std::size_t>(match.length(0)),
                     "FROM " + std::string(kPlaceholderTable));
  return Preprocessed{std::move(rewritten), std::move(paths)};
}

[[noreturn]] void unsupported(std::string_view what) {
  throw SqlError("unsupported SQL construct: " + std::string(what));
}

AstExprPtr wrap(std::variant<AstColumnRef, AstStar, AstLiteral, AstBinary, AstUnary, AstBetween,
                              AstAggregate>
                    node,
                std::optional<std::string> alias = std::nullopt) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = std::move(node);
  expr->alias = std::move(alias);
  return expr;
}

std::optional<std::string> alias_of(const hsql::Expr& e) {
  if (e.alias == nullptr) return std::nullopt;
  return std::string(e.alias);
}

AstExprPtr convert_expr(const hsql::Expr* e);

AstBinaryOp to_binary_op(hsql::OperatorType op) {
  switch (op) {
    case hsql::kOpPlus:
      return AstBinaryOp::Add;
    case hsql::kOpMinus:
      return AstBinaryOp::Subtract;
    case hsql::kOpAsterisk:
      return AstBinaryOp::Multiply;
    case hsql::kOpSlash:
      return AstBinaryOp::Divide;
    case hsql::kOpEquals:
      return AstBinaryOp::Eq;
    case hsql::kOpNotEquals:
      return AstBinaryOp::NotEq;
    case hsql::kOpLess:
      return AstBinaryOp::Lt;
    case hsql::kOpLessEq:
      return AstBinaryOp::LtEq;
    case hsql::kOpGreater:
      return AstBinaryOp::Gt;
    case hsql::kOpGreaterEq:
      return AstBinaryOp::GtEq;
    case hsql::kOpAnd:
      return AstBinaryOp::And;
    case hsql::kOpOr:
      return AstBinaryOp::Or;
    default:
      unsupported("binary operator");
  }
}

AstExprPtr convert_operator(const hsql::Expr* e) {
  switch (e->opType) {
    case hsql::kOpBetween: {
      if (e->exprList == nullptr || e->exprList->size() != 2 || e->expr == nullptr) {
        unsupported("malformed BETWEEN expression");
      }
      AstBetween between{convert_expr(e->expr), convert_expr((*e->exprList)[0]),
                          convert_expr((*e->exprList)[1])};
      return wrap(std::move(between), alias_of(*e));
    }
    case hsql::kOpUnaryMinus:
      return wrap(AstUnary{AstUnaryOp::Negate, convert_expr(e->expr)}, alias_of(*e));
    case hsql::kOpIsNull:
      return wrap(AstUnary{AstUnaryOp::IsNull, convert_expr(e->expr)}, alias_of(*e));
    case hsql::kOpNot: {
      // hsql represents "x IS NOT NULL" as NOT(IS NULL(x)); recover the
      // cleaner IsNotNull node instead of double-negating.
      if (e->expr != nullptr && e->expr->type == hsql::kExprOperator &&
          e->expr->opType == hsql::kOpIsNull) {
        return wrap(AstUnary{AstUnaryOp::IsNotNull, convert_expr(e->expr->expr)}, alias_of(*e));
      }
      return wrap(AstUnary{AstUnaryOp::Not, convert_expr(e->expr)}, alias_of(*e));
    }
    default:
      break;
  }
  if (e->expr == nullptr || e->expr2 == nullptr) unsupported("binary operator with missing operand");
  return wrap(AstBinary{to_binary_op(e->opType), convert_expr(e->expr), convert_expr(e->expr2)},
              alias_of(*e));
}

AstExprPtr convert_function(const hsql::Expr* e) {
  if (e->distinct) unsupported("DISTINCT inside an aggregate function");
  if (e->windowDescription != nullptr) unsupported("window functions");
  if (e->name == nullptr) unsupported("function call with no name");

  std::string name(e->name);
  for (char& c : name) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

  if (name == "COUNT") {
    if (e->exprList != nullptr && e->exprList->size() == 1 &&
        (*e->exprList)[0]->type == hsql::kExprStar) {
      return wrap(AstAggregate{AstAggregateFunc::CountStar, nullptr}, alias_of(*e));
    }
    if (e->exprList == nullptr || e->exprList->size() != 1) unsupported("COUNT with != 1 argument");
    return wrap(AstAggregate{AstAggregateFunc::Count, convert_expr((*e->exprList)[0])},
                alias_of(*e));
  }

  static const std::vector<std::pair<std::string, AstAggregateFunc>> kAggregates = {
      {"SUM", AstAggregateFunc::Sum},
      {"MIN", AstAggregateFunc::Min},
      {"MAX", AstAggregateFunc::Max},
      {"AVG", AstAggregateFunc::Avg},
  };
  for (const auto& [fn_name, fn] : kAggregates) {
    if (name != fn_name) continue;
    if (e->exprList == nullptr || e->exprList->size() != 1) {
      unsupported(fn_name + " with != 1 argument");
    }
    return wrap(AstAggregate{fn, convert_expr((*e->exprList)[0])}, alias_of(*e));
  }

  unsupported("function '" + name + "' (only SUM/COUNT/MIN/MAX/AVG are supported)");
}

AstExprPtr convert_expr(const hsql::Expr* e) {
  if (e == nullptr) unsupported("null expression");
  switch (e->type) {
    case hsql::kExprColumnRef:
      if (e->name == nullptr) unsupported("column reference with no name");
      return wrap(AstColumnRef{std::string(e->name)}, alias_of(*e));
    case hsql::kExprStar:
      return wrap(AstStar{}, alias_of(*e));
    case hsql::kExprLiteralInt:
      if (e->isBoolLiteral) {
        return wrap(AstLiteral{AstLiteralKind::Boolean, 0, 0.0, {}, e->ival != 0}, alias_of(*e));
      }
      return wrap(AstLiteral{AstLiteralKind::Integer, e->ival, 0.0, {}, false}, alias_of(*e));
    case hsql::kExprLiteralFloat:
      return wrap(AstLiteral{AstLiteralKind::Float, 0, e->fval, {}, false}, alias_of(*e));
    case hsql::kExprLiteralString:
      return wrap(
          AstLiteral{AstLiteralKind::String, 0, 0.0, e->name ? std::string(e->name) : "", false},
          alias_of(*e));
    case hsql::kExprLiteralNull:
      return wrap(AstLiteral{AstLiteralKind::Null, 0, 0.0, {}, false}, alias_of(*e));
    case hsql::kExprLiteralDate: {
      if (e->name == nullptr) unsupported("date literal with no text");
      const std::int32_t days = parse_iso_date(e->name);
      return wrap(AstLiteral{AstLiteralKind::Date, days, 0.0, {}, false}, alias_of(*e));
    }
    case hsql::kExprOperator:
      return convert_operator(e);
    case hsql::kExprFunctionRef:
      return convert_function(e);
    default:
      unsupported("expression type not in the supported grammar (subqueries, CASE, LIKE, IN, "
                  "CAST, EXTRACT, and arrays are not yet supported)");
  }
}

}  // namespace

AstSelectStatement parse_sql(std::string_view sql_view) {
  const std::string sql(sql_view);
  Preprocessed preprocessed = preprocess_from_read_parquet(sql);

  hsql::SQLParserResult result;
  hsql::SQLParser::parse(preprocessed.sql, &result);
  if (!result.isValid()) {
    throw SqlError("SQL parse error: " + std::string(result.errorMsg()) + " (line " +
                   std::to_string(result.errorLine()) + ", column " +
                   std::to_string(result.errorColumn()) + ")");
  }
  if (result.size() != 1) {
    throw SqlError("KernelLake supports exactly one SQL statement per query, got " +
                    std::to_string(result.size()));
  }

  const hsql::SQLStatement* raw_stmt = result.getStatement(0);
  if (raw_stmt->type() != hsql::kStmtSelect) {
    unsupported("only SELECT statements are supported");
  }
  const auto* stmt = static_cast<const hsql::SelectStatement*>(raw_stmt);

  if (stmt->selectDistinct) unsupported("SELECT DISTINCT");
  if (stmt->setOperations != nullptr && !stmt->setOperations->empty()) {
    unsupported("UNION/INTERSECT/EXCEPT");
  }
  if (stmt->withDescriptions != nullptr && !stmt->withDescriptions->empty()) {
    unsupported("WITH (common table expressions)");
  }
  if (stmt->groupBy != nullptr && stmt->groupBy->having != nullptr) {
    unsupported("HAVING");
  }
  if (stmt->fromTable == nullptr || stmt->fromTable->type != hsql::kTableName ||
      stmt->fromTable->name == nullptr || stmt->fromTable->name != kPlaceholderTable) {
    unsupported("joins and subqueries (only a single FROM read_parquet(...) source is supported)");
  }

  AstSelectStatement out;
  out.from.paths = preprocessed.paths;

  if (stmt->selectList == nullptr || stmt->selectList->empty()) {
    throw SqlError("SELECT list must not be empty");
  }
  for (const hsql::Expr* item : *stmt->selectList) {
    out.select_list.push_back(convert_expr(item));
  }

  if (stmt->whereClause != nullptr) {
    out.where = convert_expr(stmt->whereClause);
  }

  if (stmt->groupBy != nullptr && stmt->groupBy->columns != nullptr) {
    for (const hsql::Expr* column : *stmt->groupBy->columns) {
      out.group_by.push_back(convert_expr(column));
    }
  }

  if (stmt->order != nullptr) {
    for (const hsql::OrderDescription* order : *stmt->order) {
      out.order_by.push_back(AstOrderByItem{convert_expr(order->expr),
                                             order->type == hsql::kOrderAsc});
    }
  }

  if (stmt->limit != nullptr) {
    if (stmt->limit->offset != nullptr) unsupported("OFFSET");
    if (stmt->limit->limit == nullptr || stmt->limit->limit->type != hsql::kExprLiteralInt) {
      unsupported("non-literal LIMIT");
    }
    out.limit = stmt->limit->limit->ival;
  }

  return out;
}

}  // namespace kernellake::sql
