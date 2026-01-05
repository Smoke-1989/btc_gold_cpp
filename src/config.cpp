#include "config.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <algorithm>

namespace btc_gold {

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  --mode <mode>         Search mode (linear|random|doubling|hamming|modular-stride)\n"
              << "  --input <file>        Database file\n"
              << "  --input-type <type>   address|hash160|pubkey\n"
              << "  --threads <n>         Number of threads\n"
              << "  --start <n>           Start value (hex/dec)\n"
              << "  --end <n>             End value (hex/dec)\n"
              << "  --min-bit <n>         Min bit range (Doubling/Hamming)\n"
              << "  --max-bit <n>         Max bit range (Doubling/Hamming)\n"
              << "  --verbose             Enable verbose output\n"
              << "  --help                Show this help\n";
}

bool parse_args(int argc, char** argv, Config& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            return false;
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string val = argv[++i];
            // Normalize to lower case
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            
            if (val == "linear" || val == "0") config.mode = Config::Mode::LINEAR;
            else if (val == "random" || val == "1") config.mode = Config::Mode::RANDOM;
            else if (val == "geometric" || val == "2") config.mode = Config::Mode::GEOMETRIC;
            else if (val == "terminator" || val == "3") config.mode = Config::Mode::TERMINATOR;
            else if (val == "doubling" || val == "4") config.mode = Config::Mode::DOUBLING;
            else if (val == "hamming" || val == "5") config.mode = Config::Mode::HAMMING;
            else if (val == "modular-stride" || val == "modular" || val == "6") config.mode = Config::Mode::MODULAR_STRIDE;
            else {
                // Try integer fallback or default
                try {
                    config.mode = static_cast<Config::Mode>(std::stoi(val));
                } catch (...) {
                    std::cerr << "Unknown mode: " << val << ". Using default (LINEAR).\n";
                    config.mode = Config::Mode::LINEAR;
                }
            }
        } else if (arg == "--input" && i + 1 < argc) {
            config.database_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--log-file" && i + 1 < argc) {
            config.log_file = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--start" && i + 1 < argc) {
            // Use base 0 to auto-detect hex (0x) vs decimal
            config.start_value = std::stoull(argv[++i], nullptr, 0);
        } else if (arg == "--end" && i + 1 < argc) {
            config.end_value = std::stoull(argv[++i], nullptr, 0);
        } else if (arg == "--input-type" && i + 1 < argc) {
            std::string type = argv[++i];
            if (type == "address") config.input_type = Config::InputType::ADDRESS;
            else if (type == "hash160") config.input_type = Config::InputType::HASH160;
            else if (type == "pubkey") config.input_type = Config::InputType::PUBKEY;
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--stop-on-find") {
            config.stop_on_find = true;
        } else if ((arg == "--range-min" || arg == "--min-bit") && i + 1 < argc) {
            config.range_min_bit = std::stoi(argv[++i]);
        } else if ((arg == "--range-max" || arg == "--max-bit") && i + 1 < argc) {
            config.range_max_bit = std::stoi(argv[++i]);
        } else if (arg == "--multiplier" && i + 1 < argc) {
            config.multiplier = std::stoull(argv[++i], nullptr, 0);
        }
    }
    return true;
}

}  // namespace btc_gold
