#include "config.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>

namespace btc_gold {

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  --mode <0-6>          Search mode (default: 0)\n"
              << "  --input <file>        Database file\n"
              << "  --input-type <type>   address|hash160|pubkey\n"
              << "  --threads <n>         Number of threads\n"
              << "  --start <n>           Start value\n"
              << "  --end <n>             End value\n"
              << "  --verbose             Enable verbose output\n"
              << "  --help                Show this help\n";
}

bool parse_args(int argc, char** argv, Config& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            return false;
        } else if (arg == "--mode" && i + 1 < argc) {
            config.mode = static_cast<Config::Mode>(std::stoi(argv[++i]));
        } else if (arg == "--input" && i + 1 < argc) {
            config.database_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--log-file" && i + 1 < argc) {
            config.log_file = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--start" && i + 1 < argc) {
            config.start_value = std::stoull(argv[++i]);
        } else if (arg == "--end" && i + 1 < argc) {
            config.end_value = std::stoull(argv[++i]);
        } else if (arg == "--input-type" && i + 1 < argc) {
            std::string type = argv[++i];
            if (type == "address") config.input_type = Config::InputType::ADDRESS;
            else if (type == "hash160") config.input_type = Config::InputType::HASH160;
            else if (type == "pubkey") config.input_type = Config::InputType::PUBKEY;
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--stop-on-find") {
            config.stop_on_find = true;
        } else if (arg == "--range-min" && i + 1 < argc) {
            config.range_min_bit = std::stoi(argv[++i]);
        } else if (arg == "--range-max" && i + 1 < argc) {
            config.range_max_bit = std::stoi(argv[++i]);
        } else if (arg == "--multiplier" && i + 1 < argc) {
            config.multiplier = std::stoull(argv[++i]);
        }
    }
    return true;
}

}  // namespace btc_gold
