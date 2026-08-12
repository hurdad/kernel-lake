#pragma once

// Shared test-only HTTP stub, used by any REST-client test file that needs
// to exercise a real curl request/response over a real loopback socket
// without a live external server (see rest_catalog_client_test.cpp,
// unity_catalog_client_test.cpp).
#include <arpa/inet.h>
#include <fmt/format.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>
#include <thread>
#include <vector>

namespace kernellake {

// A minimal single-connection-at-a-time loopback HTTP stub: not a general
// HTTP server. REST client requests exercised against this are small
// enough (headers + a short JSON/form body) to always land in one TCP
// segment on loopback, so a single recv() into a generous buffer is enough
// to capture the whole request -- a real HTTP server would need to loop on
// recv() and parse Content-Length, this doesn't.
class LoopbackHttpServer {
 public:
  LoopbackHttpServer() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    ::listen(listen_fd_, /*backlog=*/4);

    timeval timeout{5, 0};
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  }

  ~LoopbackHttpServer() {
    if (thread_.joinable()) {
      thread_.join();
    }
    ::close(listen_fd_);
  }

  [[nodiscard]] std::string base_url() const { return fmt::format("http://127.0.0.1:{}", port_); }

  // Accepts `response_count` connections in order, one per element of
  // `responses`, capturing each raw request into the matching slot of
  // `captured_requests` (if non-null). Runs on a background thread since
  // the calling test's main thread is blocked inside the client call that
  // connects to this server.
  void respond(std::vector<std::string> responses, std::vector<std::string>* captured_requests = nullptr) {
    thread_ = std::thread([this, responses = std::move(responses), captured_requests] {
      for (size_t i = 0; i < responses.size(); ++i) {
        const int conn_fd = ::accept(listen_fd_, nullptr, nullptr);
        if (conn_fd < 0) {
          return;
        }
        timeval timeout{5, 0};
        ::setsockopt(conn_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

        std::string request(65536, '\0');
        const ssize_t received = ::recv(conn_fd, request.data(), request.size(), 0);
        request.resize(received > 0 ? static_cast<size_t>(received) : 0);
        if (captured_requests != nullptr) {
          (*captured_requests)[i] = request;
        }

        const std::string& response = responses[i];
        ::send(conn_fd, response.data(), response.size(), 0);
        ::close(conn_fd);
      }
    });
  }

  void join() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  int listen_fd_ = -1;
  int port_ = 0;
  std::thread thread_;
};

inline std::string http_ok_json(const std::string& json_body) {
  return fmt::format(
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: "
      "close\r\n\r\n{}",
      json_body.size(), json_body);
}

inline std::string http_status(int status_code, const std::string& reason, const std::string& body = "") {
  return fmt::format("HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", status_code,
                     reason, body.size(), body);
}

}  // namespace kernellake
