#include "config.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace btc_gold {

// ============================================================================
// 256-BIT HEX PARSER - Converts hex string to PrivateKey (32 bytes)
// ============================================================================

bool parse_hex_to_privkey(const std::string& hex_str, PrivateKey& out) {
    std::string clean = hex_str;
    
    // Remove 0x prefix if present
    if (clean.size() >= 2 && clean[0] == '0' && (clean[1] == 'x' || clean[1] == 'X')) {
        clean = clean.substr(2);
    }
    
    // Remove spaces
    clean.erase(std::remove_if(clean.begin(), clean.end(), ::isspace), clean.end());
    
    // Check length
    if (clean.empty() || clean.size() > 64) {
        return false;
    }
    
    // Pad with leading zeros to 64 chars (32 bytes)
    while (clean.size() < 64) {
        clean = "0" + clean;
    }
    
    // Convert hex to bytes (big-endian)
    for (size_t i = 0; i < 32; i++) {
        std::string byte_str = clean.substr(i * 2, 2);
        try {
            out[i] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
        } catch (...) {
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// FLEXIBLE NUMBER PARSER - Supports hex and decimal (up to 64-bit)
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
    
    // If hex and more than 16 chars (64 bits), it's too big for uint64_t
    if (is_hex && clean.size() > 16) {
        throw std::overflow_error("Hex value exceeds 64-bit limit");
    }
    
    // Parse as hex or decimal (both fit in uint64_t)
    try {
        if (is_hex) {
            return std::stoull(clean, nullptr, 16);
        } else {
            return std::stoull(clean, nullptr, 10);
        }
    } catch (const std::exception& e) {
        throw std::invalid_argument("Invalid number format: " + str);
    }
}

void print_usage(const char* prog_name) {
    std::cout << "\n"
              << "═══════════════════════════════════════════════════════════════════════════════\n"
              << "  🔥 BTC GOLD C++ v4.0 EXTERMINATOR - USAGE 🔥\n"
              << "═══════════════════════════════════════════════════════════════════════════════\n"
              << "\n"
              << "USAGE: " << prog_name << " [options]\n"
              << "\n"
              << "MODES:\n"
              << "  0 | linear          Sequential TURBO mode (Point Addition optimization)\n"
              << "  1 | random          Full 256-bit random search\n"
              << "  2 | geometric       3-Phase geometric search (Border/Ceiling/Hamming)\n"
              << "  3 | terminator      Multiplicative descent EXTERMINATOR mode\n"
              << "  4 | doubling        Powers of 2 search (2^n)\n"
              << "  5 | hamming         Low Hamming weight keys (sparse bits)\n"
              << "  6 | modular-stride  Arithmetic progression with custom stride\n"
              << "\n"
              << "OPTIONS:\n"
              << "  --mode <mode>           Search mode (see MODES above)\n"
              << "  --input <file>          Database file with targets\n"
              << "  --input-type <type>     address | hash160 | pubkey\n"
              << "  --threads <n>           Number of threads (0 = auto-detect)\n"
              << "\n"
              << "  RANGE (for Linear/Modular modes):\n"
              << "  --start <value>         Start value (hex or decimal, up to 64-bit)\n"
              << "                          Examples: 0x3fff, 3fff, 16383\n"
              << "  --end <value>           End value (hex or decimal, up to 64-bit)\n"
              << "\n"
              << "  256-BIT RANGE (for values > 64-bit):\n"
              << "  --start-hex <hex>       Start value as 256-bit hex (with or without 0x)\n"
              << "                          Example: 3fffffffffffffffff or 0x3fff\n"
              << "  --end-hex <hex>         End value as 256-bit hex (with or without 0x)\n"
              << "\n"
              << "  BIT RANGE (for Doubling/Hamming modes):\n"
              << "  --min-bit <n>           Minimum bit position (1-256)\n"
              << "  --max-bit <n>           Maximum bit position (1-256)\n"
              << "\n"
              << "  MODULAR STRIDE:\n"
              << "  --multiplier <n>        Stride multiplier (default: 1000000)\n"
              << "\n"
              << "  OUTPUT:\n"
              << "  --output <file>         Output file (default: found.txt)\n"
              << "  --log-file <file>       Log file path\n"
              << "\n"
              << "  FLAGS:\n"
              << "  --verbose               Enable verbose progress output\n"
              << "  --stop-on-find          Stop immediately after first match\n"
              << "  --help                  Show this help\n"
              << "\n"
              << "EXAMPLES:\n"
              << "  # Linear mode with 64-bit range\n"
              << "  " << prog_name << " --mode linear --start 0x1000 --end 0xFFFF\n"
              << "\n"
              << "  # Linear mode with 256-bit range (FULL SUPPORT)\n"
              << "  " << prog_name << " --mode linear --start-hex 3fffffffffffffffff \\\n"
              << "                --end-hex 7fffffffffffffffff\n"
              << "\n"
              << "  # Doubling mode (powers of 2)\n"
              << "  " << prog_name << " --mode doubling --min-bit 40 --max-bit 66\n"
              << "\n"
              << "  # TERMINATOR mode (multiplicative descent)\n"
              << "  " << prog_name << " --mode terminator --start 0x1000000000\n"
              << "\n"
              << "═══════════════════════════════════════════════════════════════════════════════\n"
              << "\n";
}

bool parse_args(int argc, char** argv, Config& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            return false;
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string val = argv[++i];
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            
            if (val == "linear" || val == "0") config.mode = Config::Mode::LINEAR;
            else if (val == "random" || val == "1") config.mode = Config::Mode::RANDOM;
            else if (val == "geometric" || val == "2") config.mode = Config::Mode::GEOMETRIC;
            else if (val == "terminator" || val == "exterminator" || val == "3") config.mode = Config::Mode::TERMINATOR;
            else if (val == "doubling" || val == "4") config.mode = Config::Mode::DOUBLING;
            else if (val == "hamming" || val == "5") config.mode = Config::Mode::HAMMING;
            else if (val == "modular-stride" || val == "modular" || val == "6") config.mode = Config::Mode::MODULAR_STRIDE;
            else {
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
            // 64-bit parser
            try {
                config.start_value = parse_number(argv[++i]);
                config.use_256bit_range = false; // Use 64-bit mode
            } catch (const std::exception& e) {
                std::cerr << "Error parsing --start: " << e.what() << "\n";
                std::cerr << "For values > 64-bit, use --start-hex with hex format\n";
                return false;
            }
        } else if (arg == "--end" && i + 1 < argc) {
            // 64-bit parser
            try {
                config.end_value = parse_number(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing --end: " << e.what() << "\n";
                std::cerr << "For values > 64-bit, use --end-hex with hex format\n";
                return false;
            }
        } else if (arg == "--start-hex" && i + 1 < argc) {
            // v4.0: FULL 256-BIT HEX PARSER
            std::string hex_val = argv[++i];
            if (!parse_hex_to_privkey(hex_val, config.start_key_256)) {
                std::cerr << "Error: Invalid hex format for --start-hex: " << hex_val << "\n";
                return false;
            }
            config.use_256bit_range = true;
            
            // Log the parsed value
            std::cout << "[INFO] Start key (256-bit): ";
            for (int j = 0; j < 32; j++) {
                std::cout << std::hex << std::setw(2) << std::setfill('0') 
                         << static_cast<int>(config.start_key_256[j]);
            }
            std::cout << std::dec << "\n";
            
        } else if (arg == "--end-hex" && i + 1 < argc) {
            // v4.0: FULL 256-BIT HEX PARSER
            std::string hex_val = argv[++i];
            if (!parse_hex_to_privkey(hex_val, config.end_key_256)) {
                std::cerr << "Error: Invalid hex format for --end-hex: " << hex_val << "\n";
                return false;
            }
            config.use_256bit_range = true;
            
            // Log the parsed value
            std::cout << "[INFO] End key (256-bit): ";
            for (int j = 0; j < 32; j++) {
                std::cout << std::hex << std::setw(2) << std::setfill('0') 
                         << static_cast<int>(config.end_key_256[j]);
            }
            std::cout << std::dec << "\n";
            
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
