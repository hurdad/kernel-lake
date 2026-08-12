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

}  // namespace kernellake::sql
