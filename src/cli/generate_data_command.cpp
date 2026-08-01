#include "commands.hpp"

#include <charconv>
#include <cstdio>

#include "kernellake/common/errors.hpp"
#include "kernellake/generator/sample_data_generator.hpp"

namespace kernellake::cli {

namespace {

template <typename T>
[[nodiscard]] bool parse_number(std::string_view text, T& out) {
  const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
  return ec == std::errc() && ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_double(std::string_view text, double& out) {
  try {
    std::size_t consumed = 0;
    out = std::stod(std::string(text), &consumed);
    return consumed == text.size();
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace

int run_generate_data(const std::vector<std::string_view>& args) {
  SampleDataGeneratorOptions options;
  bool bad_arg = false;

  for (std::size_t i = 0; i < args.size() && !bad_arg; ++i) {
    if (args[i] == "--output" && i + 1 < args.size()) {
      options.output_dir = args[++i];
    } else if (args[i] == "--rows" && i + 1 < args.size()) {
      bad_arg = !parse_number(args[++i], options.rows);
    } else if (args[i] == "--files" && i + 1 < args.size()) {
      bad_arg = !parse_number(args[++i], options.files);
    } else if (args[i] == "--row-group-rows" && i + 1 < args.size()) {
      bad_arg = !parse_number(args[++i], options.row_group_rows);
    } else if (args[i] == "--region-cardinality" && i + 1 < args.size()) {
      bad_arg = !parse_number(args[++i], options.region_cardinality);
    } else if (args[i] == "--category-cardinality" && i + 1 < args.size()) {
      bad_arg = !parse_number(args[++i], options.category_cardinality);
    } else if (args[i] == "--customer-cardinality" && i + 1 < args.size()) {
      bad_arg = !parse_number(args[++i], options.customer_cardinality);
    } else if (args[i] == "--null-rate" && i + 1 < args.size()) {
      bad_arg = !parse_double(args[++i], options.null_rate);
    } else if (args[i] == "--skew" && i + 1 < args.size()) {
      bad_arg = !parse_double(args[++i], options.skew);
    } else if (args[i] == "--seed" && i + 1 < args.size()) {
      bad_arg = !parse_number(args[++i], options.seed);
    } else if (args[i] == "--no-dictionary-encoding") {
      options.dictionary_encoding = false;
    } else {
      std::fprintf(stderr, "kernellake generate-data: unrecognized argument '%.*s'\n",
                   static_cast<int>(args[i].size()), args[i].data());
      return 1;
    }
  }

  if (bad_arg) {
    std::fprintf(stderr, "kernellake generate-data: invalid numeric argument value\n");
    return 1;
  }

  try {
    const SampleDataGenerationResult result = generate_sample_data(options);
    std::printf("wrote %lld rows across %zu file(s) to %s\n", static_cast<long long>(result.rows_written),
                result.file_paths.size(), options.output_dir.c_str());
  } catch (const KernelLakeError& e) {
    std::fprintf(stderr, "kernellake generate-data: %s\n", e.what());
    return 1;
  }
  return 0;
}

}  // namespace kernellake::cli
