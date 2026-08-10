#pragma once

#include <arrow/flight/server_middleware.h>

#include <string>

namespace kernellake {

// Rejects every RPC (including Handshake) whose "authorization" header is
// not exactly "Bearer <token>", where <token> is server.auth_token
// (include/kernellake/common/config.hpp). Registered in
// FlightServerOptions::middleware by main.cpp when server.auth_enabled is
// set. StartCall returning a non-OK Status is Arrow Flight's own mechanism
// for rejecting a call before it reaches the RPC handler (see
// arrow/flight/server_middleware.h) -- KernelLakeFlightSqlServer itself
// needs no changes for this to take effect.
class BearerTokenMiddlewareFactory : public arrow::flight::ServerMiddlewareFactory {
 public:
  explicit BearerTokenMiddlewareFactory(std::string token);

  arrow::Status StartCall(const arrow::flight::CallInfo& info,
                          const arrow::flight::ServerCallContext& context,
                          std::shared_ptr<arrow::flight::ServerMiddleware>* middleware) override;

 private:
  // Precomputed once from the configured token rather than re-concatenating
  // "Bearer " + token_ on every call.
  std::string expected_header_;
};

}  // namespace kernellake
