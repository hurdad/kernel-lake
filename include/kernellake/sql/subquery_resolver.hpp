#pragma once

#include <functional>

#include "kernellake/sql/ast.hpp"

namespace kernellake::sql {

// Walks `expr`'s tree, replacing every `AstSubquery` node found (wherever
// it's nested -- e.g. as one side of a comparison inside a larger boolean
// expression) with the literal `evaluate(*node.statement)` returns, and
// returns the resulting (possibly-unchanged) tree.
//
// Pure AST-to-AST transformation with zero I/O/storage dependency of its
// own -- `evaluate` carries all the capability needed to actually run a
// nested query (parse is already done; evaluate binds, plans, and
// executes it), kept out of this function so kernellake::sql stays
// independent of kernellake_api the same way parser.cpp's own header
// comment already establishes for the rest of this module. See
// QueryEngine::evaluate_scalar_subquery() for the real implementation
// callers pass.
//
// Only ever called on `AstSelectStatement::having` today (see
// QueryEngine::plan_logical()) -- a subquery appearing anywhere else
// (WHERE, SELECT list, GROUP BY, JOIN ON) is never visited by this
// function, so it survives unresolved all the way to the binder, which
// rejects it with a clear, specific error (see
// Binder::bind_node(const AstSubquery&, bool)). This function itself has
// no opinion on *where* a subquery is legal -- that's the caller's
// choice of which expression(s) to run it over.
[[nodiscard]] AstExprPtr resolve_subqueries(
    const AstExprPtr& expr, const std::function<AstLiteral(const AstSelectStatement&)>& evaluate);

// Walks `expr`'s tree, replacing every `AstIn` node whose `subquery` field
// is set (i.e. `value IN (SELECT ...)`, TPC-H Q18's shape) with an
// equivalent `AstIn` whose `list` is populated from
// `evaluate(*node.subquery)` and whose `subquery` is cleared -- from
// `bind_node(const AstIn&, bool)`'s own perspective (binder.cpp), the
// result is indistinguishable from an IN whose source was always a
// literal list, so the binder needs no changes for this feature at all.
//
// Deliberately a separate function from `resolve_subqueries()` above
// (which stays HAVING-scalar-subquery-only, unchanged) rather than a
// generalization of it: an IN-subquery is non-correlated but can
// legitimately return many rows (unlike HAVING's exactly-one-row/one-
// column contract), so `evaluate` here returns a `std::vector<AstLiteral>`
// instead of a single `AstLiteral`. See
// QueryEngine::evaluate_list_subquery() for the real implementation
// callers pass, and this function's own scale caveat: the returned list
// becomes an OR-chain of equality comparisons at bind time (the same
// desugar a literal IN list already gets), so this is a narrow mechanism
// for a subquery expected to return a modest number of rows, not a
// general-purpose semi-join.
//
// Only ever called on `AstSelectStatement::where` today (see
// QueryEngine::plan_logical()) -- an IN-subquery appearing anywhere else
// (HAVING, SELECT list, GROUP BY, JOIN ON) is never visited by this
// function, so its `AstIn::subquery` field survives unresolved all the
// way to the binder, which rejects it with a clear, specific error.
[[nodiscard]] AstExprPtr resolve_in_subqueries(
    const AstExprPtr& expr,
    const std::function<std::vector<AstLiteral>(const AstSelectStatement&)>& evaluate);

// Extracts every top-level `WHERE`-clause `AND`-conjunct that's an
// `AstExists` node (see that type's own comment in ast.hpp for the exact
// scope) and appends it as a new step onto `stmt`'s own join chain
// (`JoinType::LeftSemi` for `EXISTS`, `LeftAnti` for `NOT EXISTS`) --
// creating one, from `stmt.from`, if `stmt` had none yet. The extracted
// step's `condition` is `subquery`'s own `WHERE` clause, completely
// unmodified -- it gets resolved by the exact same
// `extract_join_step_keys()` machinery (binder.cpp) a real `JOIN ... ON`
// step's own condition already goes through, which is what actually
// enforces "one correlation equality plus only-the-subquery's-own-columns
// auxiliary predicates" -- this function does no such resolution itself
// (it has no schema access, running well before binding), only the
// purely structural checks below.
//
// Unlike resolve_subqueries()/resolve_in_subqueries() above, this is a
// pure AST-to-AST transform with no I/O/execution dependency at all: an
// `EXISTS`'s own correlated subquery becomes part of the *outer* query's
// own join chain, executed by the same physical plan as any other JOIN
// step -- never run separately beforehand the way a HAVING/IN subquery
// is. Operates on (and returns) a whole `AstSelectStatement`, not a
// single expression, since it moves content from `.where` into `.join`.
//
// A conjunct that isn't rewritable is left exactly where it is in
// `.where`, to be rejected later at bind time with a clear, specific
// error (`Binder::bind_node(const AstExists&, bool)`) -- not rewritable
// means any of: the outer query is itself a derived table or (with no
// existing join chain) has no alias on its own `FROM` source; `subquery`
// has its own `JOIN`/derived-table `FROM`, `GROUP BY`, `HAVING`,
// `ORDER BY`, or `LIMIT`; `subquery`'s single `FROM` source has no alias;
// or `subquery` has no `WHERE` clause at all (nothing to become a join
// condition, and a non-correlated `EXISTS (SELECT ...)` with no
// correlation key isn't representable as a keyed join step). This
// function has no opinion on error *messages* -- only on what it can
// mechanically rewrite; the binder still owns explaining *why* the rest
// is unsupported.
[[nodiscard]] AstSelectStatement rewrite_exists_subqueries(AstSelectStatement stmt);

// Extracts every top-level `WHERE`-clause `AND`-conjunct of shape
// `<expr> <comparison> (SELECT <agg-expr> FROM ... WHERE <correlation>)`
// or `(SELECT ...) <comparison> <expr>` (TPC-H Q17's `l_quantity < (SELECT
// 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)`) and
// decorrelates it into a JOIN against a synthesized, `GROUP BY`-
// aggregated derived table, appended as a new step onto `stmt`'s own join
// chain (creating one, from `stmt.from`, if `stmt` had none yet, mirroring
// `rewrite_exists_subqueries()`'s identical promotion) -- the matched
// conjunct itself is rewritten in place to compare against the derived
// table's own output column instead of the subquery.
//
// The subquery's own `WHERE` clause is scanned for top-level `AND`-
// conjuncts that are a plain equality between a column qualified by one
// of `stmt`'s own (outer) aliases and a column qualified by one of the
// subquery's own (inner) aliases -- each such conjunct is a correlation
// key, removed from the subquery's own `WHERE` and instead becomes: the
// first one, the new join step's `ON`-clause equality key; any further
// ones (TPC-H Q20's two-column `l_partkey = ps_partkey AND l_suppkey =
// ps_suppkey` correlation), additional top-level conjuncts appended to
// the *outer* query's own `WHERE` after the join (the exact same
// "one key in `ON`, the rest as a post-join `WHERE` filter" idiom this
// project's Q9 already uses for `partsupp`'s own two-column join
// condition, and correct for the same reason: the derived table's
// composite `GROUP BY` still groups by every correlation column
// together, so a many-to-one join on the first key alone, filtered by
// the rest afterward, produces the same result set as a true multi-key
// join would). The subquery's own remaining (non-correlation) `WHERE`
// conjuncts, if any (TPC-H Q2's own `r_name = '[REGION]'`), stay on the
// synthesized derived table's `WHERE` clause unchanged; its own `FROM`
// (a single source or a full multi-way `JOIN` chain, TPC-H Q2's own
// 4-way `partsupp`/`supplier`/`nation`/`region` shape) is moved onto the
// derived table as-is, uninspected -- this function has no opinion on
// what's inside it, only on the subquery's own top-level shape (exactly
// one `SELECT`-list item, no `GROUP BY`/`HAVING`/`ORDER BY`/`LIMIT`/
// derived-table `FROM` of its own, at least one correlation conjunct
// found).
//
// Like `rewrite_exists_subqueries()`, a purely structural AST-to-AST
// transform with no I/O/execution dependency: the decorrelated subquery
// becomes part of the *outer* query's own join chain and `GROUP BY`
// binding, reusing the exact same, already-tested machinery any real
// multi-way `JOIN`-plus-aggregate query goes through -- never run
// separately beforehand the way a HAVING/IN subquery is. Runs generically
// on whatever `AstSelectStatement` it's given, so it applies equally
// however deeply this statement is nested (a top-level query, an `IN
// (SELECT ...)`'s own body, a derived table's own inner query, ...) as
// long as the caller runs it wherever `QueryEngine::plan_logical_unoptimized()`
// itself recurses -- see that function's own call site.
//
// A conjunct that isn't rewritable (any restriction above unmet, or no
// correlation conjunct found in the subquery's own `WHERE` at all -- a
// non-correlated scalar subquery, `resolve_subqueries()`'s own job, run
// separately) is left exactly where it is, to be resolved by that other
// pass or, failing that, rejected at bind time with a clear, specific
// error the same way any other unsupported subquery shape already is.
[[nodiscard]] AstSelectStatement rewrite_correlated_scalar_subqueries(AstSelectStatement stmt);

}  // namespace kernellake::sql
