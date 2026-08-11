#include "kernellake/server/auth_middleware.hpp"

#include <arrow/flight/server.h>
#include <arrow/flight/types.h>

namespace kernellake {

namespace {

// Ordinary std::string_view::operator== short-circuits on a length
// mismatch and then calls memcmp, both of which let an attacker who can
// measure response latency narrow down the token byte-by-byte. The token
// compared here is a real bearer credential crossing a network boundary
// (unlike, say, delta.api_key, which this process only ever sends, never
// checks) -- worth the few extra lines. Not a full defense (branching on
// header presence/length above still leaks a little), just cheap
// hardening of the one comparison that matters.
bool ConstantTimeEquals(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) {
    return false;
  }
  unsigned char diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff |= static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i]);
  }
  return diff == 0;
}

}  // namespace

BearerTokenMiddlewareFactory::BearerTokenMiddlewareFactory(const std::string& token)
    : expected_header_("Bearer " + token) {}

arrow::Status BearerTokenMiddlewareFactory::StartCall(
    const arrow::flight::CallInfo& /*info*/, const arrow::flight::ServerCallContext& context,
    std::shared_ptr<arrow::flight::ServerMiddleware>* middleware) {
  *middleware = nullptr;
  const arrow::flight::CallHeaders& headers = context.incoming_headers();
  const auto it = headers.find("authorization");
  if (it == headers.end() || !ConstantTimeEquals(it->second, expected_header_)) {
    return arrow::flight::MakeFlightError(arrow::flight::FlightStatusCode::Unauthenticated,
                                          "missing or invalid bearer token");
  }
  return arrow::Status::OK();
}

}  // namespace kernellake
