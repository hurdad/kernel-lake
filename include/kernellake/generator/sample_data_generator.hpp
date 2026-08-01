#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kernellake {

// Configures kernellake generate-data's deterministic synthetic "sales"
// dataset (see docs/architecture.md for the schema this produces:
// order_id/customer_id/region/amount/event_date/event_time/category/
// discount). The same options always produce byte-identical Parquet output,
// so benchmark and correctness runs stay reproducible across machines.
struct SampleDataGeneratorOptions {
  std::string output_dir;
  std::int64_t rows = 10'000'000;
  int files = 1;
  std::int64_t row_group_rows = 250'000;

  // Number of distinct values a low-cardinality dimension takes; "skew"
  // biases selection within that range toward the low-index values instead
  // of sampling them uniformly (0 = uniform, higher = more concentrated).
  int region_cardinality = 10;
  int category_cardinality = 8;
  std::int64_t customer_cardinality = 100'000;
  double skew = 0.0;

  // Fraction of `discount` values that are NULL (the dataset's one nullable
  // column, per the spec's example schema).
  double null_rate = 0.1;

  std::uint64_t seed = 42;
  bool dictionary_encoding = true;
};

struct SampleDataGenerationResult {
  std::vector<std::string> file_paths;
  std::int64_t rows_written = 0;
};

// Throws kernellake::ConfigurationError for invalid options (e.g. rows < 0,
// files <= 0) and kernellake::StorageError if the output directory can't be
// created or a file can't be written.
[[nodiscard]] SampleDataGenerationResult generate_sample_data(const SampleDataGeneratorOptions& options);

}  // namespace kernellake
