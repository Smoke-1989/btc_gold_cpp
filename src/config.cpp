#include "config.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace btc_gold {

// ============================================================================
// ROBUST HEX/DECIMAL PARSER - Supports: 0x3fff, 3fff, 123456, etc.
// ============================================================================

uint64_t parse_number(const std::string& str) {
    if (str.empty()) {
        throw std::invalid_argument("Empty number string");
    }
    
    std::string clean = str;
    
    // Remove any whitespace
    clean.erase(std::remove_if(clean.begin(), clean.end(), ::isspace), clean.end());
    
    // Check if it looks like hex (starts with 0x, 0X, or contains a-f/A-F)
    bool is_hex = false;
    
    if (clean.size() >= 2 && (clean[0] == '0' && (clean[1] == 'x' || clean[1] == 'X'))) {
        // Has 0x prefix
        is_hex = true;
        clean = clean.substr(2); // Remove 0x
    } else {
        // Check if contains hex digits (a-f, A-F)
        for (char c : clean) {
            if ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
                is_hex = true;
                break;
            }
        }
    }
    
    // Parse as hex or decimal
    try {
        if (is_hex) {
            // Parse as hexadecimal
            return std::stoull(clean, nullptr, 16);
        } else {
            // Parse as decimal
            return std::stoull(clean, nullptr, 10);
        }
    } catch (const std::exception& e) {
        throw std::invalid_argument("Invalid number format: " + str + " (" + e.what() + ")");
    }
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  --mode <mode>         Search mode (linear|random|doubling|hamming|modular-stride)\n"
              << "  --input <file>        Database file\n"
              << "  --input-type <type>   address|hash160|pubkey\n"
              << "  --threads <n>         Number of threads\n"
              << "  --start <n>           Start value (hex: 0x3fff or 3fff, decimal: 123456)\n"
              << "  --end <n>             End value (hex: 0x3fff or 3fff, decimal: 123456)\n"
              << "  --min-bit <n>         Min bit range (Doubling/Hamming)\n"
              << "  --max-bit <n>         Max bit range (Doubling/Hamming)\n"
              << "  --multiplier <n>      Multiplier for modular stride mode\n"
              << "  --verbose             Enable verbose output\n"
              << "  --stop-on-find        Stop after first match\n"
              << "  --help                Show this help\n"
              << "\nExamples:\n"
              << "  " << prog_name << " --mode linear --start 0x1000 --end 0xFFFF\n"
              << "  " << prog_name << " --mode linear --start 1000 --end FFFF\n"
              << "  " << prog_name << " --mode doubling --min-bit 40 --max-bit 66\n";
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
            // v4.0: Robust hex/decimal parser
            try {
                config.start_value = parse_number(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing --start: " << e.what() << "\n";
                return false;
            }
        } else if (arg == "--end" && i + 1 < argc) {
            // v4.0: Robust hex/decimal parser
            try {
                config.end_value = parse_number(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing --end: " << e.what() << "\n";
                return false;
            }
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
            // v4.0: Robust hex/decimal parser
            try {
                config.multiplier = parse_number(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing --multiplier: " << e.what() << "\n";
                return false;
            }
        }
    }
    return true;
}

}  // namespace btc_gold
