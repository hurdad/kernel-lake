#include "kernellake/sql/parser.hpp"

#include <SQLParser.h>
#include <fmt/format.h>

#include <algorithm>
#include <regex>
#include <string>
#include <vector>

#include "kernellake/common/date_util.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake::sql {

namespace {

// hyrise/sql-parser (bison/flex, recursive-descent) recurses once per
// nesting level of a parenthesized expression, with no depth limit of its
// own -- a query with many thousands of nested parens
// ("SELECT (((((...1...))))) FROM ...") drives it into a C-stack overflow,
// which crashes the process outright rather than raising a catchable
// exception. This is a real concern for kernellake-server specifically:
// its Flight SQL surface accepts arbitrary-length query text over the wire
// with no prior validation. A cheap pre-scan here (well before any parsing
// work, real or hsql's) rejects both pathologically long SQL text and
// pathologically deep nesting with a clean SqlError instead. The limits
// are generous relative to any real query (TPC-H's own queries, the
// deepest in this project, nest nowhere close to either) specifically to
// avoid false positives on legitimate SQL.
constexpr std::size_t kMaxSqlBytes = 1 << 20;  // 1 MiB
constexpr int kMaxParenDepth = 500;

void check_sql_within_limits(std::string_view sql) {
  if (sql.size() > kMaxSqlBytes) {
    throw SqlError(fmt::format("SQL text is too long ({} bytes, max {})", sql.size(), kMaxSqlBytes));
  }
  int depth = 0;
  int max_depth = 0;
  for (const char c : sql) {
    if (c == '(') {
      max_depth = std::max(max_depth, ++depth);
    } else if (c == ')' && depth > 0) {
      --depth;
    }
  }
  if (max_depth > kMaxParenDepth) {
    throw SqlError(
        fmt::format("SQL expression nesting is too deep ({} levels, max {})", max_depth, kMaxParenDepth));
  }
}

struct PlaceholderSource {
  std::string placeholder;
  std::vector<std::string> paths;
};

struct Preprocessed {
  std::string sql;
  std::vector<PlaceholderSource> sources;  // in order of appearance
};

// hyrise/sql-parser's FROM-clause grammar only accepts table names, joins,
// and subqueries -- it has no notion of a table-valued function call like
// `read_parquet('...')`. We find every occurrence of that one specific
// shape ourselves, extract its string-literal path arguments, and
// substitute a distinct placeholder identifier for each occurrence before
// handing the query to the parser -- this leaves the surrounding syntax
// (JOIN/ON/aliases/commas) completely untouched, so hsql's own grammar
// still does the real work of parsing table references and join structure.
// This is a deliberate, narrow, documented preprocessing step -- not
// general SQL-string rewriting -- and any FROM clause whose shape isn't
// recognized after this substitution fails clearly in parse_sql() below
// rather than being silently reinterpreted.
Preprocessed preprocess_from_read_parquet(const std::string& sql) {
  static const std::regex kReadParquetPattern(
      R"(read_parquet\s*\(\s*((?:'(?:[^'\\]|\\.)*'\s*,\s*)*'(?:[^'\\]|\\.)*')\s*\))", std::regex::icase);
  static const std::regex kStringLiteralPattern(R"('((?:[^'\\]|\\.)*)')");

  std::string rewritten;
  std::vector<PlaceholderSource> sources;
  std::size_t last_pos = 0;
  for (auto it = std::sregex_iterator(sql.begin(), sql.end(), kReadParquetPattern);
       it != std::sregex_iterator(); ++it) {
    const std::smatch& match = *it;
    const auto match_pos = static_cast<std::size_t>(match.position(0));
    rewritten.append(sql, last_pos, match_pos - last_pos);

    std::string placeholder = "kernellake_parquet_source_" + std::to_string(sources.size());
    const std::string args = match[1].str();
    std::vector<std::string> paths;
    for (auto path_it = std::sregex_iterator(args.begin(), args.end(), kStringLiteralPattern);
         path_it != std::sregex_iterator(); ++path_it) {
      paths.push_back((*path_it)[1].str());
    }
    rewritten += placeholder;
    sources.push_back(PlaceholderSource{std::move(placeholder), std::move(paths)});

    last_pos = match_pos + static_cast<std::size_t>(match.length(0));
  }
  rewritten.append(sql, last_pos, sql.size() - last_pos);

  if (sources.empty()) {
    throw SqlError(
        "KernelLake requires at least one data source of the form "
        "read_parquet('path' [, 'path2', ...]); no such clause was found in the query");
  }
  // Generous relative to any real query (TPC-H's own deepest join, Q8,
  // needs 7) -- purely a guard against a pathological number of joined
  // sources driving unbounded chain-building work, same rationale as
  // kMaxParenDepth above.
  constexpr std::size_t kMaxJoinSources = 12;
  if (sources.size() > kMaxJoinSources) {
    throw SqlError(
        fmt::format("KernelLake supports at most {} read_parquet(...) sources in a JOIN chain, got {}",
                    kMaxJoinSources, sources.size()));
  }
  return Preprocessed{std::move(rewritten), std::move(sources)};
}

[[noreturn]] void unsupported(std::string_view what) {
  throw SqlError(fmt::format("unsupported SQL construct: {}", what));
}

AstExprPtr wrap(std::variant<AstColumnRef, AstStar, AstLiteral, AstBinary, AstUnary, AstBetween, AstAggregate,
                             AstLike, AstIn, AstCase, AstCast>
                    node,
                std::optional<std::string> alias = std::nullopt) {
  auto expr = std::make_shared<AstExpr>();
  expr->node = std::move(node);
  expr->alias = std::move(alias);
  return expr;
}

std::optional<std::string> alias_of(const hsql::Expr& e) {
  if (e.alias == nullptr) {
    return std::nullopt;
  }
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

AstIn convert_in(const hsql::Expr* e) {
  if (e->select != nullptr) {
    unsupported("IN with a subquery (only a literal list is supported)");
  }
  if (e->expr == nullptr || e->exprList == nullptr) {
    unsupported("malformed IN expression");
  }
  AstIn in;
  in.value = convert_expr(e->expr);
  in.list.reserve(e->exprList->size());
  for (const hsql::Expr* item : *e->exprList) {
    in.list.push_back(convert_expr(item));
  }
  return in;
}

AstCase convert_case(const hsql::Expr* e) {
  if (e->exprList == nullptr || e->exprList->empty()) {
    unsupported("CASE with no WHEN branches");
  }
  AstCase result;
  result.when_then.reserve(e->exprList->size());
  for (const hsql::Expr* element : *e->exprList) {
    if (element->opType != hsql::kOpCaseListElement || element->expr == nullptr ||
        element->expr2 == nullptr) {
      unsupported("malformed CASE WHEN branch");
    }
    AstExprPtr when = convert_expr(element->expr);
    // Simple CASE ("CASE x WHEN v THEN ... END") compares each WHEN value
    // against the CASE operand; searched CASE ("CASE WHEN cond THEN ...
    // END", e->expr == nullptr) uses each WHEN as a boolean condition
    // directly. Desugar the simple form into the searched form here so
    // the binder/executor only ever need to handle one shape.
    AstExprPtr condition = e->expr != nullptr
                               ? wrap(AstBinary{AstBinaryOp::Eq, convert_expr(e->expr), std::move(when)})
                               : std::move(when);
    result.when_then.emplace_back(std::move(condition), convert_expr(element->expr2));
  }
  if (e->expr2 != nullptr) {
    result.else_branch = convert_expr(e->expr2);
  }
  return result;
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
      // hsql represents "x IS NOT NULL" as NOT(IS NULL(x)) and "x NOT IN
      // (...)" as NOT(IN(x, ...)); recover the cleaner negated node
      // instead of double-negating.
      if (e->expr != nullptr && e->expr->type == hsql::kExprOperator && e->expr->opType == hsql::kOpIsNull) {
        return wrap(AstUnary{AstUnaryOp::IsNotNull, convert_expr(e->expr->expr)}, alias_of(*e));
      }
      if (e->expr != nullptr && e->expr->type == hsql::kExprOperator && e->expr->opType == hsql::kOpIn) {
        AstIn in = convert_in(e->expr);
        in.negated = true;
        return wrap(std::move(in), alias_of(*e));
      }
      return wrap(AstUnary{AstUnaryOp::Not, convert_expr(e->expr)}, alias_of(*e));
    }
    case hsql::kOpLike:
    case hsql::kOpNotLike: {
      if (e->expr == nullptr || e->expr2 == nullptr) {
        unsupported("malformed LIKE expression");
      }
      return wrap(AstLike{convert_expr(e->expr), convert_expr(e->expr2), e->opType == hsql::kOpNotLike},
                  alias_of(*e));
    }
    case hsql::kOpILike:
      unsupported("ILIKE (case-insensitive LIKE) -- use LIKE");
    case hsql::kOpIn:
      return wrap(convert_in(e), alias_of(*e));
    case hsql::kOpCase:
      return wrap(convert_case(e), alias_of(*e));
    default:
      break;
  }
  if (e->expr == nullptr || e->expr2 == nullptr) {
    unsupported("binary operator with missing operand");
  }
  return wrap(AstBinary{to_binary_op(e->opType), convert_expr(e->expr), convert_expr(e->expr2)},
              alias_of(*e));
}

AstExprPtr convert_function(const hsql::Expr* e) {
  if (e->distinct) {
    unsupported("DISTINCT inside an aggregate function");
  }
  if (e->windowDescription != nullptr) {
    unsupported("window functions");
  }
  if (e->name == nullptr) {
    unsupported("function call with no name");
  }

  std::string name(e->name);
  for (char& c : name) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }

  if (name == "COUNT") {
    if (e->exprList != nullptr && e->exprList->size() == 1 && (*e->exprList)[0]->type == hsql::kExprStar) {
      return wrap(AstAggregate{AstAggregateFunc::CountStar, nullptr}, alias_of(*e));
    }
    if (e->exprList == nullptr || e->exprList->size() != 1) {
      unsupported("COUNT with != 1 argument");
    }
    return wrap(AstAggregate{AstAggregateFunc::Count, convert_expr((*e->exprList)[0])}, alias_of(*e));
  }

  static const std::vector<std::pair<std::string, AstAggregateFunc>> kAggregates = {
      {"SUM", AstAggregateFunc::Sum},
      {"MIN", AstAggregateFunc::Min},
      {"MAX", AstAggregateFunc::Max},
      {"AVG", AstAggregateFunc::Avg},
  };
  for (const auto& [fn_name, fn] : kAggregates) {
    if (name != fn_name) {
      continue;
    }
    if (e->exprList == nullptr || e->exprList->size() != 1) {
      unsupported(fn_name + " with != 1 argument");
    }
    return wrap(AstAggregate{fn, convert_expr((*e->exprList)[0])}, alias_of(*e));
  }

  unsupported("function '" + name + "' (only SUM/COUNT/MIN/MAX/AVG are supported)");
}

// Names deliberately match SQL keywords, not KernelLake's own TypeId names
// -- this keeps the AST (and this mapping) independent of both hsql and
// KernelLake's type system; binder.cpp owns the actual mapping to
// DataType/TypeId.
std::string to_cast_type_name(hsql::DataType type) {
  switch (type) {
    case hsql::DataType::BIGINT:
    case hsql::DataType::LONG:
      return "BIGINT";
    case hsql::DataType::INT:
    case hsql::DataType::SMALLINT:
      return "INT";
    case hsql::DataType::BOOLEAN:
      return "BOOLEAN";
    case hsql::DataType::DOUBLE:
    case hsql::DataType::REAL:
      return "DOUBLE";
    case hsql::DataType::FLOAT:
      return "FLOAT";
    case hsql::DataType::DATE:
      return "DATE";
    case hsql::DataType::DATETIME:
      return "DATETIME";
    case hsql::DataType::VARCHAR:
    case hsql::DataType::TEXT:
    case hsql::DataType::CHAR:
      return "VARCHAR";
    case hsql::DataType::DECIMAL:
      return "DECIMAL";
    case hsql::DataType::UNKNOWN:
      unsupported("CAST to an unrecognized type");
  }
  unsupported("CAST to an unrecognized type");
}

AstExprPtr convert_expr(const hsql::Expr* e) {
  if (e == nullptr) {
    unsupported("null expression");
  }
  switch (e->type) {
    case hsql::kExprColumnRef:
      if (e->name == nullptr) {
        unsupported("column reference with no name");
      }
      return wrap(AstColumnRef{std::string(e->name),
                               e->table != nullptr ? std::optional<std::string>(e->table) : std::nullopt},
                  alias_of(*e));
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
      return wrap(AstLiteral{AstLiteralKind::String, 0, 0.0, e->name ? std::string(e->name) : "", false},
                  alias_of(*e));
    case hsql::kExprLiteralNull:
      return wrap(AstLiteral{AstLiteralKind::Null, 0, 0.0, {}, false}, alias_of(*e));
    case hsql::kExprLiteralDate: {
      if (e->name == nullptr) {
        unsupported("date literal with no text");
      }
      const std::int32_t days = parse_iso_date(e->name);
      return wrap(AstLiteral{AstLiteralKind::Date, days, 0.0, {}, false}, alias_of(*e));
    }
    case hsql::kExprOperator:
      return convert_operator(e);
    case hsql::kExprFunctionRef:
      return convert_function(e);
    case hsql::kExprCast: {
      if (e->expr == nullptr) {
        unsupported("malformed CAST expression");
      }
      return wrap(AstCast{convert_expr(e->expr), to_cast_type_name(e->columnType.data_type),
                          e->columnType.precision, e->columnType.scale},
                  alias_of(*e));
    }
    default:
      unsupported(
          "expression type not in the supported grammar (subqueries, EXTRACT, and arrays are not "
          "yet supported)");
  }
}

// Resolves one JOIN side's TableRef (a leaf, never itself a nested JOIN --
// see flatten_join_chain()'s own comment) to its read_parquet(...) paths
// and required alias.
AstParquetSource convert_join_source(const hsql::TableRef* ref, const Preprocessed& preprocessed) {
  if (ref == nullptr || ref->type != hsql::kTableName || ref->name == nullptr) {
    unsupported("JOIN sides must each be a single read_parquet(...) source, not a subquery or nested join");
  }
  if (ref->alias == nullptr || ref->alias->name == nullptr) {
    unsupported("both sides of a JOIN must be aliased, e.g. read_parquet('a.parquet') AS a");
  }
  for (const PlaceholderSource& source : preprocessed.sources) {
    if (source.placeholder == ref->name) {
      return AstParquetSource{source.paths, std::string(ref->alias->name)};
    }
  }
  unsupported("JOIN source does not reference a read_parquet(...) call");
}

// hsql parses a multi-way JOIN left-associatively: `A JOIN B ON c1 JOIN C
// ON c2` becomes a TableRef tree `(A JOIN B ON c1) JOIN C ON c2`, i.e.
// `join->left` is itself a kTableJoin for a chain of 3+ sources, while
// `join->right` is always a plain leaf table -- so this recurses only on
// `join->left`, appending each level's `join->right`/condition as one more
// AstJoinStep, in left-to-right source order. Every join in the chain must
// be a plain INNER JOIN between a (possibly-nested) JOIN and a single
// aliased read_parquet(...) source -- anything else (a comma-style join, a
// subquery on either side, a non-INNER join type) fails clearly via
// convert_join_source()/the checks below, at whichever level of the chain
// it appears.
AstJoinClause flatten_join_chain(const hsql::TableRef* table_ref, const Preprocessed& preprocessed) {
  if (table_ref == nullptr || table_ref->type != hsql::kTableJoin || table_ref->join == nullptr) {
    unsupported("malformed JOIN");
  }
  const hsql::JoinDefinition* join = table_ref->join;
  if (join->left == nullptr || join->right == nullptr) {
    unsupported("malformed JOIN");
  }
  if (join->type != hsql::kJoinInner) {
    unsupported("JOIN types other than INNER JOIN");
  }
  if (join->condition == nullptr) {
    unsupported("JOIN with no ON condition");
  }

  AstJoinClause clause;
  if (join->left->type == hsql::kTableJoin) {
    clause = flatten_join_chain(join->left, preprocessed);
  } else {
    clause.first = convert_join_source(join->left, preprocessed);
  }
  clause.steps.push_back(
      AstJoinStep{convert_join_source(join->right, preprocessed), convert_expr(join->condition)});
  return clause;
}

}  // namespace

AstSelectStatement parse_sql(std::string_view sql_view) {
  check_sql_within_limits(sql_view);
  const std::string sql(sql_view);
  Preprocessed preprocessed = preprocess_from_read_parquet(sql);

  hsql::SQLParserResult result;
  hsql::SQLParser::parse(preprocessed.sql, &result);
  if (!result.isValid()) {
    throw SqlError(fmt::format("SQL parse error: {} (line {}, column {})", result.errorMsg(),
                               result.errorLine(), result.errorColumn()));
  }
  if (result.size() != 1) {
    throw SqlError(
        fmt::format("KernelLake supports exactly one SQL statement per query, got {}", result.size()));
  }

  const hsql::SQLStatement* raw_stmt = result.getStatement(0);
  if (raw_stmt->type() != hsql::kStmtSelect) {
    unsupported("only SELECT statements are supported");
  }
  // Tag-checked downcast: the `!= hsql::kStmtSelect` check just above already
  // guarantees raw_stmt is really a SelectStatement.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
  const auto* stmt = static_cast<const hsql::SelectStatement*>(raw_stmt);

  if (stmt->selectDistinct) {
    unsupported("SELECT DISTINCT");
  }
  if (stmt->setOperations != nullptr && !stmt->setOperations->empty()) {
    unsupported("UNION/INTERSECT/EXCEPT");
  }
  if (stmt->withDescriptions != nullptr && !stmt->withDescriptions->empty()) {
    unsupported("WITH (common table expressions)");
  }
  if (stmt->groupBy != nullptr && stmt->groupBy->having != nullptr) {
    unsupported("HAVING");
  }
  AstSelectStatement out;

  if (stmt->fromTable != nullptr && stmt->fromTable->type == hsql::kTableName &&
      stmt->fromTable->name != nullptr) {
    if (preprocessed.sources.size() != 1 || stmt->fromTable->name != preprocessed.sources[0].placeholder) {
      unsupported(
          "joins and subqueries (only a single FROM read_parquet(...) source, or a two-table JOIN, "
          "is supported)");
    }
    out.from.paths = preprocessed.sources[0].paths;
    if (stmt->fromTable->alias != nullptr && stmt->fromTable->alias->name != nullptr) {
      out.from.alias = std::string(stmt->fromTable->alias->name);
    }
  } else if (stmt->fromTable != nullptr && stmt->fromTable->type == hsql::kTableJoin) {
    AstJoinClause clause = flatten_join_chain(stmt->fromTable, preprocessed);
    // clause.steps.size() + 1 (the "first" source) must equal every
    // read_parquet(...) call actually found in the query text -- catches a
    // read_parquet(...) source that appears in the FROM clause but outside
    // the JOIN chain the parser understood (this project has no other FROM
    // shape a source could legitimately appear in).
    if (preprocessed.sources.size() != clause.steps.size() + 1) {
      unsupported("a JOIN requires exactly one read_parquet(...) source per table in the chain");
    }
    out.join = std::move(clause);
  } else {
    unsupported(
        "joins and subqueries (only a single FROM read_parquet(...) source, or a two-table JOIN, is "
        "supported)");
  }

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
      out.order_by.push_back(AstOrderByItem{convert_expr(order->expr), order->type == hsql::kOrderAsc});
    }
  }

  if (stmt->limit != nullptr) {
    if (stmt->limit->offset != nullptr) {
      unsupported("OFFSET");
    }
    if (stmt->limit->limit == nullptr || stmt->limit->limit->type != hsql::kExprLiteralInt) {
      unsupported("non-literal LIMIT");
    }
    out.limit = stmt->limit->limit->ival;
  }

  return out;
}

}  // namespace kernellake::sql
