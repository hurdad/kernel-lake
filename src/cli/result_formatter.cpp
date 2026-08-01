#include "result_formatter.hpp"

#include <arrow/api.h>
#include <arrow/csv/writer.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>

#include <algorithm>
#include <cstdio>

#include "kernellake/common/errors.hpp"

namespace kernellake::cli {

namespace {

[[nodiscard]] std::string arrow_status_message(const arrow::Status& status, std::string_view context) {
  return std::string(context) + ": " + status.ToString();
}

// A generic scalar-to-text conversion covering every Arrow type KernelLake
// can produce (see kernellake/types/arrow_adapter.hpp): arrow::Scalar's own
// ToString() already renders each concrete type correctly, so there is no
// need to hand-roll per-type formatting here.
[[nodiscard]] std::string scalar_text(const arrow::Array& column, std::int64_t row) {
  if (column.IsNull(row)) return "NULL";
  arrow::Result<std::shared_ptr<arrow::Scalar>> scalar = column.GetScalar(row);
  if (!scalar.ok()) throw ExecutionError(arrow_status_message(scalar.status(), "failed to read result value"));
  return (*scalar)->ToString();
}

[[nodiscard]] bool is_json_numeric_or_bool(const arrow::DataType& type) {
  switch (type.id()) {
    case arrow::Type::BOOL:
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] std::string json_escape(const std::string& text) {
  std::string escaped;
  escaped.reserve(text.size() + 2);
  for (const char c : text) {
    switch (c) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
    }
  }
  return escaped;
}

void write_table_format(const QueryResult& result, std::FILE* out) {
  const std::vector<std::string> headers = [&] {
    std::vector<std::string> names;
    if (result.schema) {
      for (const std::shared_ptr<arrow::Field>& field : result.schema->fields()) names.push_back(field->name());
    }
    return names;
  }();

  std::vector<std::size_t> widths;
  widths.reserve(headers.size());
  for (const std::string& header : headers) widths.push_back(header.size());

  // Column widths must span every row across every batch, so render every
  // cell to text once and keep it rather than computing widths in one pass
  // and re-stringifying values in a second.
  std::vector<std::vector<std::string>> rows;
  for (const std::shared_ptr<arrow::RecordBatch>& batch : result.batches) {
    for (std::int64_t row = 0; row < batch->num_rows(); ++row) {
      std::vector<std::string> cells;
      cells.reserve(headers.size());
      for (int col = 0; col < batch->num_columns(); ++col) {
        std::string text = scalar_text(*batch->column(col), row);
        widths[static_cast<std::size_t>(col)] = std::max(widths[static_cast<std::size_t>(col)], text.size());
        cells.push_back(std::move(text));
      }
      rows.push_back(std::move(cells));
    }
  }

  auto print_row = [&](const std::vector<std::string>& cells) {
    for (std::size_t col = 0; col < cells.size(); ++col) {
      std::fprintf(out, "%-*s", static_cast<int>(widths[col]) + 2, cells[col].c_str());
    }
    std::fputc('\n', out);
  };
  print_row(headers);
  for (std::size_t col = 0; col < headers.size(); ++col) {
    std::fprintf(out, "%s  ", std::string(widths[col], '-').c_str());
  }
  std::fputc('\n', out);
  for (const std::vector<std::string>& row : rows) print_row(row);
}

void write_jsonl_format(const QueryResult& result, std::FILE* out) {
  for (const std::shared_ptr<arrow::RecordBatch>& batch : result.batches) {
    for (std::int64_t row = 0; row < batch->num_rows(); ++row) {
      std::string line = "{";
      for (int col = 0; col < batch->num_columns(); ++col) {
        if (col > 0) line += ",";
        line += "\"" + json_escape(batch->schema()->field(col)->name()) + "\":";
        const arrow::Array& column = *batch->column(col);
        if (column.IsNull(row)) {
          line += "null";
        } else if (is_json_numeric_or_bool(*column.type())) {
          line += scalar_text(column, row);
        } else {
          line += "\"" + json_escape(scalar_text(column, row)) + "\"";
        }
      }
      line += "}";
      std::fprintf(out, "%s\n", line.c_str());
    }
  }
}

std::shared_ptr<arrow::io::OutputStream> open_binary_sink(const std::optional<std::string>& output_path) {
  arrow::Result<std::shared_ptr<arrow::io::FileOutputStream>> sink =
      output_path ? arrow::io::FileOutputStream::Open(*output_path) : arrow::io::FileOutputStream::Open(1);
  if (!sink.ok()) {
    throw ExecutionError(arrow_status_message(sink.status(), "failed to open query result output"));
  }
  return *sink;
}

void write_csv_format(const QueryResult& result, const std::optional<std::string>& output_path) {
  const std::shared_ptr<arrow::io::OutputStream> sink = open_binary_sink(output_path);
  arrow::Result<std::shared_ptr<arrow::Table>> table =
      arrow::Table::FromRecordBatches(result.schema, result.batches);
  if (!table.ok()) throw ExecutionError(arrow_status_message(table.status(), "failed to assemble CSV output"));
  const arrow::Status status = arrow::csv::WriteCSV(**table, arrow::csv::WriteOptions::Defaults(), sink.get());
  if (!status.ok()) throw ExecutionError(arrow_status_message(status, "failed to write CSV output"));
  const arrow::Status close_status = sink->Close();
  if (!close_status.ok()) throw ExecutionError(arrow_status_message(close_status, "failed to close CSV output"));
}

void write_arrow_ipc_format(const QueryResult& result, const std::optional<std::string>& output_path) {
  const std::shared_ptr<arrow::io::OutputStream> sink = open_binary_sink(output_path);
  arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchWriter>> writer =
      arrow::ipc::MakeFileWriter(sink, result.schema);
  if (!writer.ok()) {
    throw ExecutionError(arrow_status_message(writer.status(), "failed to open Arrow IPC output"));
  }
  for (const std::shared_ptr<arrow::RecordBatch>& batch : result.batches) {
    const arrow::Status status = (*writer)->WriteRecordBatch(*batch);
    if (!status.ok()) throw ExecutionError(arrow_status_message(status, "failed to write Arrow IPC batch"));
  }
  const arrow::Status close_status = (*writer)->Close();
  if (!close_status.ok()) {
    throw ExecutionError(arrow_status_message(close_status, "failed to close Arrow IPC output"));
  }
  const arrow::Status sink_close_status = sink->Close();
  if (!sink_close_status.ok()) {
    throw ExecutionError(arrow_status_message(sink_close_status, "failed to close Arrow IPC output"));
  }
}

}  // namespace

std::optional<ResultFormat> parse_result_format(std::string_view name) {
  if (name == "table") return ResultFormat::Table;
  if (name == "csv") return ResultFormat::Csv;
  if (name == "jsonl") return ResultFormat::JsonLines;
  if (name == "arrow") return ResultFormat::ArrowIpc;
  return std::nullopt;
}

void write_query_result(const QueryResult& result, ResultFormat format,
                         const std::optional<std::string>& output_path) {
  switch (format) {
    case ResultFormat::Table: {
      std::FILE* out = output_path ? std::fopen(output_path->c_str(), "w") : stdout;
      if (out == nullptr) throw ExecutionError("failed to open query result output '" + *output_path + "'");
      write_table_format(result, out);
      if (output_path) std::fclose(out);
      return;
    }
    case ResultFormat::JsonLines: {
      std::FILE* out = output_path ? std::fopen(output_path->c_str(), "w") : stdout;
      if (out == nullptr) throw ExecutionError("failed to open query result output '" + *output_path + "'");
      write_jsonl_format(result, out);
      if (output_path) std::fclose(out);
      return;
    }
    case ResultFormat::Csv:
      write_csv_format(result, output_path);
      return;
    case ResultFormat::ArrowIpc:
      write_arrow_ipc_format(result, output_path);
      return;
  }
}

}  // namespace kernellake::cli
