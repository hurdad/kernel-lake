#include "kernellake/sql/parser.hpp"

#include <SQLParser.h>
#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernellake/common/date_util.hpp"
#include "kernellake/common/errors.hpp"

namespace kernellake::sql {

namespace {

bool starts_with_ci(std::string_view sql, std::size_t pos, std::string_view lower_literal) {
  if (pos + lower_literal.size() > sql.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lower_literal.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(sql[pos + i])) != lower_literal[i]) {
      return false;
    }
  }
  return true;
}

// hyrise/sql-parser (bison/flex, recursive-descent) recurses once per
// nesting level of both a parenthesized expression AND a chained JOIN
// (`A JOIN B JOIN C JOIN ...`, built as a left-deep TableRef tree -- see
// flatten_join_chain() below), with no depth limit of its own in either
// case -- confirmed for real: a query with many thousands of nested parens
// *or* many thousands of chained JOINs (~40,000, well under this
// function's own 1 MiB byte cap -- KernelLake's own semantic
// kMaxJoinSources limit below can't help here, since that only runs
// *after* hsql::SQLParser::parse() has already built its full tree)
// each independently drove it into a C-stack overflow, which crashes the
// process outright rather than raising a catchable exception. This is a
// real concern for kernellake-server specifically: its Flight SQL surface
// accepts arbitrary-length query text over the wire with no prior
// validation. A cheap pre-scan here (well before any parsing work, real
// or hsql's) rejects pathologically long SQL text, pathologically deep
// paren nesting, and a pathologically long JOIN chain with a clean
// SqlError instead. The limits are generous relative to any real query
// (TPC-H's own queries, the deepest in this project, nest nowhere close
// to any of these) specifically to avoid false positives on legitimate
// SQL.
constexpr std::size_t kMaxSqlBytes = 1 << 20;  // 1 MiB
constexpr int kMaxParenDepth = 500;
// Generous relative to KernelLake's own semantic ceiling (kMaxJoinSources
// = 12 read_parquet(...) sources per JOIN chain, checked later, after
// hsql has already parsed): counting the substring "join" case-
// insensitively (not a real tokenizer -- same coarse-but-safe philosophy
// as counting bare '(' / ')' above, and just as unaffected by a stray
// "join" appearing inside a string literal, since the threshold is so far
// above any legitimate count) is enough to reject the pathological case
// long before hsql ever sees the text.
constexpr int kMaxJoinKeywords = 64;

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
  int join_count = 0;
  for (std::size_t pos = 0; pos < sql.size(); ++pos) {
    if (starts_with_ci(sql, pos, "join")) {
      ++join_count;
    }
  }
  if (join_count > kMaxJoinKeywords) {
    throw SqlError(
        fmt::format("SQL JOIN chain is too long ({} JOIN keywords, max {})", join_count, kMaxJoinKeywords));
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

// Threaded through every conversion function (instead of a plain
// `const Preprocessed&`) once a query can contain more than one SELECT
// statement (a HAVING subquery, see AstSubquery) -- `preprocessed.sources`
// is one flat, occurrence-ordered list spanning the *entire* raw SQL text
// (outer query and any nested subquery alike; preprocess_from_read_parquet()
// scans linearly with no awareness of parens/nesting, so each occurrence
// anywhere in the text -- including inside a `(SELECT ...)` -- gets its own
// globally unique placeholder). `consumed` tracks, by index into that same
// list, which source has been claimed by some statement's own FROM/JOIN
// clause so far across the whole parse -- see find_and_consume_source().
// Replaces the older, simpler "preprocessed.sources.size() must exactly
// equal this one statement's own source count" checks, which assumed
// there was only ever one statement to begin with.
struct ConversionState {
  const Preprocessed& preprocessed;
  std::vector<bool>& consumed;
};

std::size_t skip_whitespace(std::string_view sql, std::size_t pos) {
  while (pos < sql.size() && std::isspace(static_cast<unsigned char>(sql[pos]))) {
    ++pos;
  }
  return pos;
}

// Parses a single-quoted string literal starting at `sql[pos]` (which must
// be '\''). The returned content is the raw, unprocessed text between the
// quotes -- a backslash-escaped character is skipped over (so an escaped
// quote doesn't end the string early) but copied through verbatim rather
// than being unescaped; this matches this function's own former regex
// equivalent (`'((?:[^'\\]|\\.)*)'` captures the escape sequence as-is, it
// never transforms it). Returns nullopt if `pos` isn't a quote, or the
// string never closes.
std::optional<std::pair<std::string, std::size_t>> try_parse_quoted_string(std::string_view sql,
                                                                           std::size_t pos) {
  if (pos >= sql.size() || sql[pos] != '\'') {
    return std::nullopt;
  }
  std::string content;
  std::size_t i = pos + 1;
  while (i < sql.size()) {
    const char c = sql[i];
    if (c == '\'') {
      return std::make_pair(std::move(content), i + 1);
    }
    if (c == '\\' && i + 1 < sql.size()) {
      content += c;
      content += sql[i + 1];
      i += 2;
      continue;
    }
    content += c;
    ++i;
  }
  return std::nullopt;
}

// Hand-rolled replacement for a former std::regex pattern
// (`read_parquet\s*\(\s*(('...'\s*,\s*)*'...')\s*\)`): libstdc++'s
// std::regex recurses once per repetition of a `(...)*` group, so that
// pattern's repeated comma-separated-string group crashed the whole
// process via a C-stack overflow given attacker-controlled input well
// within this file's own 1 MiB SQL-length cap below -- confirmed for
// real, a single `read_parquet('<~35,000-character path>')` (let alone
// many comma-separated paths) segfaulted the process instead of raising
// a catchable error, defeating the exact threat model
// check_sql_within_limits() above already exists to guard against
// (kernellake-server's Flight SQL surface accepts arbitrary-length query
// text over the wire with no other validation). This scanner is a single
// linear pass with no recursion and no backtracking, so its stack usage
// is O(1) regardless of input size or shape. `start` must point just past
// "read_parquet"; returns the parsed path list and the position just past
// the closing ')', or nullopt if the shape there isn't a parenthesized
// comma-separated list of single-quoted strings -- exactly the old
// regex's "no match" case, left for hsql itself to reject with its own
// parse error.
std::optional<std::pair<std::vector<std::string>, std::size_t>> try_parse_read_parquet_args(
    std::string_view sql, std::size_t start) {
  std::size_t pos = skip_whitespace(sql, start);
  if (pos >= sql.size() || sql[pos] != '(') {
    return std::nullopt;
  }
  pos = skip_whitespace(sql, pos + 1);

  std::vector<std::string> paths;
  while (true) {
    std::optional<std::pair<std::string, std::size_t>> literal = try_parse_quoted_string(sql, pos);
    if (!literal) {
      return std::nullopt;
    }
    paths.push_back(std::move(literal->first));
    pos = skip_whitespace(sql, literal->second);
    if (pos < sql.size() && sql[pos] == ',') {
      pos = skip_whitespace(sql, pos + 1);
      continue;
    }
    break;
  }
  if (pos >= sql.size() || sql[pos] != ')') {
    return std::nullopt;
  }
  return std::make_pair(std::move(paths), pos + 1);
}

}  // namespace

// hyrise/sql-parser's FROM-clause grammar only accepts table names, joins,
// and subqueries -- it has no notion of a table-valued function call like
// `read_parquet('...')` or `read_iceberg('...')`. We find every occurrence
// of either specific shape ourselves, extract its string-literal
// argument(s), and substitute a distinct placeholder identifier for each
// occurrence before handing the query to the parser -- this leaves the
// surrounding syntax (JOIN/ON/aliases/commas) completely untouched, so
// hsql's own grammar still does the real work of parsing table references
// and join structure. This is a deliberate, narrow, documented
// preprocessing step -- not general SQL-string rewriting -- and any FROM
// clause whose shape isn't recognized after this substitution fails
// clearly in parse_sql() below rather than being silently reinterpreted.
//
// `read_iceberg('catalog.namespace.table')` takes exactly one string
// argument (unlike read_parquet's comma-separated path list), which this
// preprocessor re-encodes as a single path "iceberg://catalog.namespace.table"
// -- reusing kernellake::Uri's existing scheme-dispatch idiom (the same one
// ObjectStoreRegistry already uses for S3/GCS/Azure/HDFS) so that
// AstParquetSource, the binder, and LogicalScan need no "source kind"
// concept of their own; only IcebergSourceResolver
// (kernellake/iceberg/iceberg_source_resolver.hpp) ever looks at the
// prefix. Splitting "catalog.namespace.table" into its parts, and
// validating the catalog name against configured catalogs, happens there
// too -- not here, matching how a read_parquet(...) path's own syntax
// isn't validated at this layer either.
//
// `read_delta('<table_uri>')` follows the exact same one-string-argument
// shape, re-encoded as "delta://<table_uri>" -- only DeltaSourceResolver
// (kernellake/delta/delta_source_resolver.hpp) ever looks at that prefix.
// Unlike Iceberg's catalog-qualified name, `<table_uri>` is itself already
// a full URI (e.g. "s3://bucket/warehouse/orders"), so the encoded source
// legitimately contains a second "://" of its own; Uri::scheme() only
// looks at the first occurrence, so this composes correctly without any
// extra escaping here.
namespace {
Preprocessed preprocess_from_read_parquet(const std::string& sql) {
  static constexpr std::string_view kParquetFunctionName = "read_parquet";
  static constexpr std::string_view kIcebergFunctionName = "read_iceberg";
  static constexpr std::string_view kDeltaFunctionName = "read_delta";
  static constexpr std::string_view kUnityCatalogFunctionName = "read_unity_catalog";

  std::string rewritten;
  std::vector<PlaceholderSource> sources;
  std::size_t pos = 0;
  while (pos < sql.size()) {
    const bool is_parquet = starts_with_ci(sql, pos, kParquetFunctionName);
    const bool is_iceberg = !is_parquet && starts_with_ci(sql, pos, kIcebergFunctionName);
    const bool is_delta = !is_parquet && !is_iceberg && starts_with_ci(sql, pos, kDeltaFunctionName);
    // Checked after kDeltaFunctionName despite "read_delta" not being a
    // prefix of "read_unity_catalog" (unlike, say, "read_i..." shapes):
    // kept in the same "not yet matched anything" chain as every other
    // function name here for one consistent shape, not because ordering
    // matters between these two specifically.
    const bool is_unity_catalog =
        !is_parquet && !is_iceberg && !is_delta && starts_with_ci(sql, pos, kUnityCatalogFunctionName);
    if (!is_parquet && !is_iceberg && !is_delta && !is_unity_catalog) {
      rewritten += sql[pos];
      ++pos;
      continue;
    }
    const std::string_view function_name = is_parquet   ? kParquetFunctionName
                                           : is_iceberg ? kIcebergFunctionName
                                           : is_delta   ? kDeltaFunctionName
                                                        : kUnityCatalogFunctionName;
    std::optional<std::pair<std::vector<std::string>, std::size_t>> parsed =
        try_parse_read_parquet_args(sql, pos + function_name.size());
    if (!parsed || ((is_iceberg || is_delta || is_unity_catalog) && parsed->first.size() != 1)) {
      // Doesn't have the shape this preprocessor understands (e.g. a
      // non-string argument, or more than one argument to
      // read_iceberg(...)/read_delta(...)/read_unity_catalog(...)) -- leave
      // the text untouched; hsql's own grammar will reject it with a
      // normal parse error.
      rewritten += sql[pos];
      ++pos;
      continue;
    }
    std::string placeholder = "kernellake_parquet_source_" + std::to_string(sources.size());
    rewritten += placeholder;
    std::vector<std::string> paths;
    if (is_iceberg) {
      paths = {"iceberg://" + parsed->first.front()};
    } else if (is_delta) {
      paths = {"delta://" + parsed->first.front()};
    } else if (is_unity_catalog) {
      paths = {"unitycatalog://" + parsed->first.front()};
    } else {
      paths = std::move(parsed->first);
    }
    sources.push_back(PlaceholderSource{placeholder, std::move(paths)});
    pos = parsed->second;
  }

  if (sources.empty()) {
    throw SqlError(
        "KernelLake requires at least one data source of the form "
        "read_parquet('path' [, 'path2', ...]), read_iceberg('catalog.namespace.table'), "
        "read_delta('table_uri'), or read_unity_catalog('instance.catalog.schema.table'); no such clause "
        "was found in the query");
  }
  // Generous relative to any real query (TPC-H's own deepest join, Q8,
  // needs 7) -- purely a guard against a pathological number of joined
  // sources driving unbounded chain-building work, same rationale as
  // kMaxParenDepth above.
  constexpr std::size_t kMaxJoinSources = 12;
  if (sources.size() > kMaxJoinSources) {
    throw SqlError(
        fmt::format("KernelLake supports at most {} read_parquet(...)/read_iceberg(...)/read_delta(...)/"
                    "read_unity_catalog(...) sources in a JOIN chain, got {}",
                    kMaxJoinSources, sources.size()));
  }
  return Preprocessed{std::move(rewritten), std::move(sources)};
}

[[noreturn]] void unsupported(std::string_view what) {
  throw SqlError(fmt::format("unsupported SQL construct: {}", what));
}

AstExprPtr wrap(std::variant<AstColumnRef, AstStar, AstLiteral, AstBinary, AstUnary, AstBetween, AstAggregate,
                             AstLike, AstIn, AstCase, AstCast, AstExtract, AstSubquery>
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

AstExprPtr convert_expr(const hsql::Expr* e, ConversionState& state);
AstSelectStatement convert_select_statement(const hsql::SelectStatement* stmt, ConversionState& state);

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

AstIn convert_in(const hsql::Expr* e, ConversionState& state) {
  if (e->expr == nullptr) {
    unsupported("malformed IN expression");
  }
  AstIn in;
  in.value = convert_expr(e->expr, state);
  if (e->select != nullptr) {
    in.subquery = std::make_shared<AstSelectStatement>(convert_select_statement(e->select, state));
    return in;
  }
  if (e->exprList == nullptr) {
    unsupported("malformed IN expression");
  }
  in.list.reserve(e->exprList->size());
  for (const hsql::Expr* item : *e->exprList) {
    in.list.push_back(convert_expr(item, state));
  }
  return in;
}

AstCase convert_case(const hsql::Expr* e, ConversionState& state) {
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
    AstExprPtr when = convert_expr(element->expr, state);
    // Simple CASE ("CASE x WHEN v THEN ... END") compares each WHEN value
    // against the CASE operand; searched CASE ("CASE WHEN cond THEN ...
    // END", e->expr == nullptr) uses each WHEN as a boolean condition
    // directly. Desugar the simple form into the searched form here so
    // the binder/executor only ever need to handle one shape.
    AstExprPtr condition =
        e->expr != nullptr ? wrap(AstBinary{AstBinaryOp::Eq, convert_expr(e->expr, state), std::move(when)})
                           : std::move(when);
    result.when_then.emplace_back(std::move(condition), convert_expr(element->expr2, state));
  }
  if (e->expr2 != nullptr) {
    result.else_branch = convert_expr(e->expr2, state);
  }
  return result;
}

AstExprPtr convert_operator(const hsql::Expr* e, ConversionState& state) {
  switch (e->opType) {
    case hsql::kOpBetween: {
      if (e->exprList == nullptr || e->exprList->size() != 2 || e->expr == nullptr) {
        unsupported("malformed BETWEEN expression");
      }
      AstBetween between{convert_expr(e->expr, state), convert_expr((*e->exprList)[0], state),
                         convert_expr((*e->exprList)[1], state)};
      return wrap(std::move(between), alias_of(*e));
    }
    case hsql::kOpUnaryMinus:
      return wrap(AstUnary{AstUnaryOp::Negate, convert_expr(e->expr, state)}, alias_of(*e));
    case hsql::kOpIsNull:
      return wrap(AstUnary{AstUnaryOp::IsNull, convert_expr(e->expr, state)}, alias_of(*e));
    case hsql::kOpNot: {
      // hsql represents "x IS NOT NULL" as NOT(IS NULL(x)) and "x NOT IN
      // (...)" as NOT(IN(x, ...)); recover the cleaner negated node
      // instead of double-negating.
      if (e->expr != nullptr && e->expr->type == hsql::kExprOperator && e->expr->opType == hsql::kOpIsNull) {
        return wrap(AstUnary{AstUnaryOp::IsNotNull, convert_expr(e->expr->expr, state)}, alias_of(*e));
      }
      if (e->expr != nullptr && e->expr->type == hsql::kExprOperator && e->expr->opType == hsql::kOpIn) {
        AstIn in = convert_in(e->expr, state);
        in.negated = true;
        return wrap(std::move(in), alias_of(*e));
      }
      return wrap(AstUnary{AstUnaryOp::Not, convert_expr(e->expr, state)}, alias_of(*e));
    }
    case hsql::kOpLike:
    case hsql::kOpNotLike: {
      if (e->expr == nullptr || e->expr2 == nullptr) {
        unsupported("malformed LIKE expression");
      }
      return wrap(
          AstLike{convert_expr(e->expr, state), convert_expr(e->expr2, state), e->opType == hsql::kOpNotLike},
          alias_of(*e));
    }
    case hsql::kOpILike:
      unsupported("ILIKE (case-insensitive LIKE) -- use LIKE");
    case hsql::kOpIn:
      return wrap(convert_in(e, state), alias_of(*e));
    case hsql::kOpCase:
      return wrap(convert_case(e, state), alias_of(*e));
    default:
      break;
  }
  if (e->expr == nullptr || e->expr2 == nullptr) {
    unsupported("binary operator with missing operand");
  }
  return wrap(AstBinary{to_binary_op(e->opType), convert_expr(e->expr, state), convert_expr(e->expr2, state)},
              alias_of(*e));
}

AstExprPtr convert_function(const hsql::Expr* e, ConversionState& state) {
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
    return wrap(AstAggregate{AstAggregateFunc::Count, convert_expr((*e->exprList)[0], state)}, alias_of(*e));
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
    return wrap(AstAggregate{fn, convert_expr((*e->exprList)[0], state)}, alias_of(*e));
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

AstExprPtr convert_expr(const hsql::Expr* e, ConversionState& state) {
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
      return convert_operator(e, state);
    case hsql::kExprFunctionRef:
      return convert_function(e, state);
    case hsql::kExprCast: {
      if (e->expr == nullptr) {
        unsupported("malformed CAST expression");
      }
      return wrap(AstCast{convert_expr(e->expr, state), to_cast_type_name(e->columnType.data_type),
                          e->columnType.precision, e->columnType.scale},
                  alias_of(*e));
    }
    case hsql::kExprExtract: {
      if (e->expr == nullptr) {
        unsupported("malformed EXTRACT expression");
      }
      AstExtractField field;
      switch (e->datetimeField) {
        case hsql::kDatetimeYear:
          field = AstExtractField::Year;
          break;
        case hsql::kDatetimeMonth:
          field = AstExtractField::Month;
          break;
        case hsql::kDatetimeDay:
          field = AstExtractField::Day;
          break;
        default:
          // HOUR/MINUTE/SECOND: no generated table has a time-of-day
          // component (every date column is DATE, not DATETIME/TIMESTAMP),
          // so these fields are structurally meaningless here, not just
          // unimplemented.
          unsupported(
              "EXTRACT field (only YEAR/MONTH/DAY are supported -- no generated table has a "
              "time-of-day component)");
      }
      return wrap(AstExtract{field, convert_expr(e->expr, state)}, alias_of(*e));
    }
    case hsql::kExprSelect: {
      // `(SELECT ...)` used as a value expression -- only meaningful as an
      // operand inside HAVING (see AstSubquery's own comment); legal
      // anywhere hsql's grammar allows a value expression, purely so this
      // parses instead of erroring, but resolve_subqueries() only ever
      // looks for/resolves one inside AstSelectStatement::having --
      // anywhere else, it survives unresolved all the way to the binder,
      // which rejects it with a clear, specific error (see
      // Binder::bind_node(const AstSubquery&, bool)) rather than this
      // layer guessing where it's "supposed" to be disallowed.
      if (e->select == nullptr) {
        unsupported("malformed subquery expression");
      }
      auto statement = std::make_shared<AstSelectStatement>(convert_select_statement(e->select, state));
      return wrap(AstSubquery{std::move(statement)}, alias_of(*e));
    }
    default:
      unsupported("expression type not in the supported grammar (arrays are not yet supported)");
  }
}

// Finds the source (by placeholder identifier) a FROM/JOIN clause
// referenced, and marks it consumed -- see ConversionState's own comment
// for why this replaces the older "preprocessed.sources.size() must
// exactly match this one statement" checks. Throws if `placeholder` isn't
// found at all, or (defensively; shouldn't happen given every occurrence
// gets its own globally unique placeholder -- see
// preprocess_from_read_parquet()) was already consumed by an earlier
// FROM/JOIN clause somewhere in this same parse.
const PlaceholderSource& find_and_consume_source(const std::string& placeholder, ConversionState& state) {
  for (std::size_t i = 0; i < state.preprocessed.sources.size(); ++i) {
    if (state.preprocessed.sources[i].placeholder != placeholder) {
      continue;
    }
    if (state.consumed[i]) {
      unsupported(
          "internal error: a read_parquet(...)/read_iceberg(...)/read_delta(...) source was referenced by "
          "more than one FROM/JOIN clause");
    }
    state.consumed[i] = true;
    return state.preprocessed.sources[i];
  }
  unsupported(
      "FROM/JOIN source does not reference a read_parquet(...)/read_iceberg(...)/read_delta(...) call");
}

// Resolves one JOIN side's TableRef (a leaf, never itself a nested JOIN --
// see flatten_join_chain()'s own comment) to its read_parquet(...) paths
// and required alias.
AstParquetSource convert_join_source(const hsql::TableRef* ref, ConversionState& state) {
  if (ref == nullptr || ref->type != hsql::kTableName || ref->name == nullptr) {
    unsupported(
        "JOIN sides must each be a single read_parquet(...)/read_iceberg(...)/read_delta(...) source, not a "
        "subquery or "
        "nested join");
  }
  if (ref->alias == nullptr || ref->alias->name == nullptr) {
    unsupported("both sides of a JOIN must be aliased, e.g. read_parquet('a.parquet') AS a");
  }
  const PlaceholderSource& source = find_and_consume_source(ref->name, state);
  return AstParquetSource{source.paths, std::string(ref->alias->name)};
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
AstJoinClause flatten_join_chain(const hsql::TableRef* table_ref, ConversionState& state) {
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
    clause = flatten_join_chain(join->left, state);
  } else {
    clause.first = convert_join_source(join->left, state);
  }
  clause.steps.push_back(
      AstJoinStep{convert_join_source(join->right, state), convert_expr(join->condition, state)});
  return clause;
}

// The body of what used to be all of parse_sql() -- factored out so it can
// be called recursively for a HAVING subquery's own nested
// hsql::SelectStatement (see convert_expr()'s kExprSelect case), not just
// the top-level statement. `state` is shared across every recursive call
// within one parse_sql() invocation (one Preprocessed/consumed pair for
// the whole raw SQL text, outer query and every nested subquery alike).
AstSelectStatement convert_select_statement(const hsql::SelectStatement* stmt, ConversionState& state) {
  if (stmt->selectDistinct) {
    unsupported("SELECT DISTINCT");
  }
  if (stmt->setOperations != nullptr && !stmt->setOperations->empty()) {
    unsupported("UNION/INTERSECT/EXCEPT");
  }
  if (stmt->withDescriptions != nullptr && !stmt->withDescriptions->empty()) {
    unsupported("WITH (common table expressions)");
  }

  AstSelectStatement out;

  if (stmt->fromTable != nullptr && stmt->fromTable->type == hsql::kTableName &&
      stmt->fromTable->name != nullptr) {
    const PlaceholderSource& source = find_and_consume_source(stmt->fromTable->name, state);
    out.from.paths = source.paths;
    if (stmt->fromTable->alias != nullptr && stmt->fromTable->alias->name != nullptr) {
      out.from.alias = std::string(stmt->fromTable->alias->name);
    }
  } else if (stmt->fromTable != nullptr && stmt->fromTable->type == hsql::kTableJoin) {
    out.join = flatten_join_chain(stmt->fromTable, state);
  } else {
    unsupported(
        "joins and subqueries (only a single FROM read_parquet(...)/read_iceberg(...)/read_delta(...) "
        "source, or a "
        "two-table JOIN, is supported)");
  }

  if (stmt->selectList == nullptr || stmt->selectList->empty()) {
    throw SqlError("SELECT list must not be empty");
  }
  for (const hsql::Expr* item : *stmt->selectList) {
    out.select_list.push_back(convert_expr(item, state));
  }

  if (stmt->whereClause != nullptr) {
    out.where = convert_expr(stmt->whereClause, state);
  }

  if (stmt->groupBy != nullptr && stmt->groupBy->columns != nullptr) {
    for (const hsql::Expr* column : *stmt->groupBy->columns) {
      out.group_by.push_back(convert_expr(column, state));
    }
  }
  if (stmt->groupBy != nullptr && stmt->groupBy->having != nullptr) {
    out.having = convert_expr(stmt->groupBy->having, state);
  }

  if (stmt->order != nullptr) {
    for (const hsql::OrderDescription* order : *stmt->order) {
      out.order_by.push_back(
          AstOrderByItem{convert_expr(order->expr, state), order->type == hsql::kOrderAsc});
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

  std::vector<bool> consumed(preprocessed.sources.size(), false);
  ConversionState state{preprocessed, consumed};
  AstSelectStatement out = convert_select_statement(stmt, state);

  // Every read_parquet(...)/read_iceberg(...)/read_delta(...) occurrence
  // anywhere in the raw SQL text (outer query or nested HAVING subquery
  // alike) must have been claimed by exactly one FROM/JOIN clause
  // somewhere in the parse -- catches a source that appears in the query
  // text but outside any FROM/JOIN shape this parser understands (this
  // project has no other place a source could legitimately appear).
  for (std::size_t i = 0; i < preprocessed.sources.size(); ++i) {
    if (!consumed[i]) {
      unsupported(
          "a read_parquet(...)/read_iceberg(...)/read_delta(...) source appears in the query but is not "
          "referenced by any recognized FROM/JOIN clause");
    }
  }

  return out;
}

}  // namespace kernellake::sql
