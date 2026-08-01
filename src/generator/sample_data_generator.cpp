#include "kernellake/generator/sample_data_generator.hpp"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <cmath>
#include <filesystem>
#include <random>

#include "kernellake/common/errors.hpp"

namespace kernellake {

namespace {

namespace fs = std::filesystem;

// Fixed, wall-clock-independent date range so the same seed always produces
// byte-identical output regardless of when generate-data actually runs.
constexpr std::int32_t kMinEventDate = 19723;  // 2024-01-01
constexpr std::int32_t kMaxEventDate = 20818;  // 2026-12-31
constexpr std::int64_t kMicrosPerDay = 86'400'000'000;

void validate(const SampleDataGeneratorOptions& options) {
  if (options.output_dir.empty()) throw ConfigurationError("generate-data: --output is required");
  if (options.rows <= 0) throw ConfigurationError("generate-data: --rows must be positive");
  if (options.files <= 0) throw ConfigurationError("generate-data: --files must be positive");
  if (options.row_group_rows <= 0) {
    throw ConfigurationError("generate-data: --row-group-rows must be positive");
  }
  if (options.region_cardinality <= 0 || options.category_cardinality <= 0 ||
      options.customer_cardinality <= 0) {
    throw ConfigurationError("generate-data: cardinalities must be positive");
  }
  if (options.null_rate < 0.0 || options.null_rate > 1.0) {
    throw ConfigurationError("generate-data: --null-rate must be within [0, 1]");
  }
  if (options.skew < 0.0) throw ConfigurationError("generate-data: --skew must be non-negative");
}

// skew == 0 samples every index in [0, cardinality) uniformly; increasing
// skew concentrates draws toward the low-index values (index 0 is always
// the mode), giving generate-data's "skew" knob real effect without needing
// a full Zipfian table.
int skewed_index(std::mt19937_64& rng, int cardinality, double skew) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const double u = unit(rng);
  const double biased = std::pow(u, 1.0 + skew);
  int index = static_cast<int>(biased * static_cast<double>(cardinality));
  if (index >= cardinality) index = cardinality - 1;
  return index;
}

std::shared_ptr<arrow::Schema> sample_schema() {
  return arrow::schema({
      arrow::field("order_id", arrow::int64(), false),
      arrow::field("customer_id", arrow::int64(), false),
      arrow::field("region", arrow::utf8(), false),
      arrow::field("amount", arrow::float64(), false),
      arrow::field("event_date", arrow::date32(), false),
      arrow::field("event_time", arrow::timestamp(arrow::TimeUnit::MICRO), false),
      arrow::field("category", arrow::utf8(), false),
      arrow::field("discount", arrow::float64(), true),
  });
}

std::shared_ptr<arrow::Table> generate_batch_table(const SampleDataGeneratorOptions& options,
                                                    std::int64_t first_order_id, std::int64_t row_count,
                                                    std::mt19937_64& rng) {
  arrow::Int64Builder order_id_builder;
  arrow::Int64Builder customer_id_builder;
  arrow::StringBuilder region_builder;
  arrow::DoubleBuilder amount_builder;
  arrow::Date32Builder event_date_builder;
  arrow::TimestampBuilder event_time_builder(arrow::timestamp(arrow::TimeUnit::MICRO),
                                              arrow::default_memory_pool());
  arrow::StringBuilder category_builder;
  arrow::DoubleBuilder discount_builder;

  std::uniform_int_distribution<std::int64_t> customer_dist(0, options.customer_cardinality - 1);
  std::uniform_real_distribution<double> amount_dist(1.0, 1000.0);
  std::uniform_int_distribution<std::int32_t> date_dist(kMinEventDate, kMaxEventDate);
  std::uniform_int_distribution<std::int64_t> time_of_day_dist(0, kMicrosPerDay - 1);
  std::uniform_real_distribution<double> discount_dist(0.0, 0.3);
  std::bernoulli_distribution null_dist(options.null_rate);

  arrow::Status status;
  for (std::int64_t i = 0; i < row_count; ++i) {
    status = order_id_builder.Append(first_order_id + i);
    if (!status.ok()) throw StorageError("generate-data: " + status.ToString());
    status = customer_id_builder.Append(customer_dist(rng));
    if (!status.ok()) throw StorageError("generate-data: " + status.ToString());
    status = region_builder.Append("region-" +
                                    std::to_string(skewed_index(rng, options.region_cardinality, options.skew)));
    if (!status.ok()) throw StorageError("generate-data: " + status.ToString());
    status = amount_builder.Append(amount_dist(rng));
    if (!status.ok()) throw StorageError("generate-data: " + status.ToString());
    const std::int32_t event_date = date_dist(rng);
    status = event_date_builder.Append(event_date);
    if (!status.ok()) throw StorageError("generate-data: " + status.ToString());
    status = event_time_builder.Append(static_cast<std::int64_t>(event_date) * kMicrosPerDay +
                                        time_of_day_dist(rng));
    if (!status.ok()) throw StorageError("generate-data: " + status.ToString());
    status = category_builder.Append(
        "category-" + std::to_string(skewed_index(rng, options.category_cardinality, options.skew)));
    if (!status.ok()) throw StorageError("generate-data: " + status.ToString());
    if (null_dist(rng)) {
      status = discount_builder.AppendNull();
    } else {
      status = discount_builder.Append(discount_dist(rng));
    }
    if (!status.ok()) throw StorageError("generate-data: " + status.ToString());
  }

  std::shared_ptr<arrow::Array> order_id, customer_id, region, amount, event_date, event_time, category,
      discount;
  if (!order_id_builder.Finish(&order_id).ok() || !customer_id_builder.Finish(&customer_id).ok() ||
      !region_builder.Finish(&region).ok() || !amount_builder.Finish(&amount).ok() ||
      !event_date_builder.Finish(&event_date).ok() || !event_time_builder.Finish(&event_time).ok() ||
      !category_builder.Finish(&category).ok() || !discount_builder.Finish(&discount).ok()) {
    throw StorageError("generate-data: failed to finalize generated columns");
  }

  return arrow::Table::Make(sample_schema(), {order_id, customer_id, region, amount, event_date, event_time,
                                               category, discount});
}

}  // namespace

SampleDataGenerationResult generate_sample_data(const SampleDataGeneratorOptions& options) {
  validate(options);

  std::error_code ec;
  fs::create_directories(options.output_dir, ec);
  if (ec) {
    throw StorageError("generate-data: failed to create output directory '" + options.output_dir +
                        "': " + ec.message());
  }

  const std::shared_ptr<parquet::WriterProperties> writer_properties = [&] {
    parquet::WriterProperties::Builder builder;
    builder.compression(parquet::Compression::SNAPPY);
    if (options.dictionary_encoding) {
      builder.enable_dictionary();
    } else {
      builder.disable_dictionary();
    }
    return builder.build();
  }();

  std::mt19937_64 rng(options.seed);

  const std::int64_t base_rows_per_file = options.rows / options.files;
  const std::int64_t extra_rows = options.rows % options.files;

  SampleDataGenerationResult result;
  result.file_paths.reserve(static_cast<std::size_t>(options.files));

  std::int64_t next_order_id = 0;
  for (int file_index = 0; file_index < options.files; ++file_index) {
    const std::int64_t rows_this_file = base_rows_per_file + (file_index < extra_rows ? 1 : 0);
    if (rows_this_file == 0) continue;

    const std::shared_ptr<arrow::Table> table =
        generate_batch_table(options, next_order_id, rows_this_file, rng);
    next_order_id += rows_this_file;

    char name_buffer[32];
    std::snprintf(name_buffer, sizeof(name_buffer), "part-%05d.parquet", file_index);
    const fs::path file_path = fs::path(options.output_dir) / name_buffer;

    arrow::Result<std::shared_ptr<arrow::io::FileOutputStream>> sink =
        arrow::io::FileOutputStream::Open(file_path.string());
    if (!sink.ok()) {
      throw StorageError("generate-data: failed to open '" + file_path.string() + "': " +
                          sink.status().ToString());
    }
    const arrow::Status status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink,
                                                              options.row_group_rows, writer_properties);
    if (!status.ok()) {
      throw StorageError("generate-data: failed to write '" + file_path.string() + "': " + status.ToString());
    }

    result.file_paths.push_back(file_path.string());
    result.rows_written += rows_this_file;
  }

  return result;
}

}  // namespace kernellake
