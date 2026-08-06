#include "kernellake/delta/delta_txn_client.hpp"

#include <fmt/format.h>
#include <grpcpp/grpcpp.h>

#include <fstream>
#include <sstream>

#include "delta_txn.grpc.pb.h"
#include "kernellake/common/errors.hpp"
#include "kernellake/observability/query_tracing.hpp"

namespace kernellake::delta {

namespace {

std::string read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw ConfigurationError(
        fmt::format("delta txn client: couldn't open delta.tls_ca_cert_path '{}'", path));
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

std::shared_ptr<grpc::ChannelCredentials> build_credentials(const DeltaSection& config) {
  if (!config.use_tls) {
    return grpc::InsecureChannelCredentials();
  }
  grpc::SslCredentialsOptions options;
  if (!config.tls_ca_cert_path.empty()) {
    options.pem_root_certs = read_file(config.tls_ca_cert_path);
  }
  return grpc::SslCredentials(options);
}

void add_auth_metadata(grpc::ClientContext& context, const DeltaSection& config) {
  if (!config.api_key.empty()) {
    context.AddMetadata("x-api-key", config.api_key);
  }
}

// Injects `span`'s W3C trace-context headers as gRPC metadata -- a no-op
// when tracing is disabled/not built (ClientSpan::inject() itself never
// calls the setter in that case), matching delta-txn-service's own
// TraceContextLayer (telemetry/trace_context.rs) on the receiving end,
// which extracts these same headers to parent its "grpc.request" span
// under this call's ClientSpan, completing the distributed trace across
// the process boundary instead of leaving it as two disconnected trees.
void inject_trace_context(const observability::ClientSpan& span, grpc::ClientContext& context) {
  span.inject([&context](std::string_view key, std::string_view value) {
    context.AddMetadata(std::string(key), std::string(value));
  });
}

DeltaTableInfo translate_table_metadata(const ::delta::txn::v1::TableMetadata& metadata, int64_t version) {
  DeltaTableInfo info;
  info.version = version;
  info.schema_string = metadata.schema_string();
  info.partition_columns.assign(metadata.partition_columns().begin(), metadata.partition_columns().end());
  return info;
}

DeltaActiveFile translate_add_file(const ::delta::txn::v1::AddFile& file) {
  DeltaActiveFile active_file;
  active_file.path = file.path();
  active_file.size = file.size();
  active_file.modification_time = file.modification_time();
  active_file.partition_values.insert(file.partition_values().begin(), file.partition_values().end());
  active_file.record_count = file.has_stats() ? file.stats().num_records() : 0;
  return active_file;
}

}  // namespace

struct DeltaTxnClient::Impl {
  DeltaSection config;
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<::delta::txn::v1::DeltaTxnService::Stub> stub;
};

DeltaTxnClient::DeltaTxnClient(DeltaSection config) : impl_(std::make_unique<Impl>()) {
  if (config.grpc_endpoint.empty()) {
    throw ConfigurationError("delta txn client: delta.grpc_endpoint is not configured");
  }
  impl_->channel = grpc::CreateChannel(config.grpc_endpoint, build_credentials(config));
  impl_->stub = ::delta::txn::v1::DeltaTxnService::NewStub(impl_->channel);
  impl_->config = std::move(config);
}

DeltaTxnClient::~DeltaTxnClient() = default;

DeltaTableInfo DeltaTxnClient::get_table(const std::string& table_uri) {
  observability::ClientSpan span = observability::start_client_span("delta_txn.GetTable");
  try {
    ::delta::txn::v1::GetTableRequest request;
    request.set_table_uri(table_uri);

    grpc::ClientContext context;
    add_auth_metadata(context, impl_->config);
    inject_trace_context(span, context);

    ::delta::txn::v1::GetTableResponse response;
    const grpc::Status status = impl_->stub->GetTable(&context, request, &response);
    if (!status.ok()) {
      throw StorageError(
          fmt::format("delta txn client: GetTable('{}') failed: {}", table_uri, status.error_message()));
    }
    if (!response.has_metadata()) {
      throw StorageError(
          fmt::format("delta txn client: GetTable('{}') response is missing metadata", table_uri));
    }
    DeltaTableInfo info = translate_table_metadata(response.metadata(), response.version());
    span.finish_ok();
    return info;
  } catch (const std::exception& e) {
    span.finish_error(e.what());
    throw;
  }
}

DeltaActiveFileListing DeltaTxnClient::list_active_files(const std::string& table_uri) {
  observability::ClientSpan span = observability::start_client_span("delta_txn.ListActiveFiles");
  try {
    ::delta::txn::v1::ListActiveFilesRequest request;
    request.set_table_uri(table_uri);

    grpc::ClientContext context;
    add_auth_metadata(context, impl_->config);
    inject_trace_context(span, context);

    const std::unique_ptr<grpc::ClientReader<::delta::txn::v1::ListActiveFilesResponse>> reader =
        impl_->stub->ListActiveFiles(&context, request);

    DeltaActiveFileListing listing;
    bool header_seen = false;
    ::delta::txn::v1::ListActiveFilesResponse response;
    while (reader->Read(&response)) {
      if (response.has_header()) {
        if (!response.header().has_metadata()) {
          throw StorageError(
              fmt::format("delta txn client: ListActiveFiles('{}') header is missing metadata", table_uri));
        }
        listing.table = translate_table_metadata(response.header().metadata(), response.header().version());
        header_seen = true;
      } else if (response.has_batch()) {
        for (const ::delta::txn::v1::AddFile& file : response.batch().files()) {
          listing.files.push_back(translate_add_file(file));
        }
      }
    }
    const grpc::Status status = reader->Finish();
    if (!status.ok()) {
      throw StorageError(fmt::format("delta txn client: ListActiveFiles('{}') failed: {}", table_uri,
                                     status.error_message()));
    }
    if (!header_seen) {
      throw StorageError(fmt::format(
          "delta txn client: ListActiveFiles('{}') stream ended without a header message", table_uri));
    }
    span.finish_ok();
    return listing;
  } catch (const std::exception& e) {
    span.finish_error(e.what());
    throw;
  }
}

}  // namespace kernellake::delta
