#pragma once

#include "types.h"
#include <string>
#include <vector>

namespace btc_gold {

// ============================================================================
// CONFIGURATION
// ============================================================================

struct Config {
    // Mode Selection
    enum class Mode : int {
        LINEAR = 0,         // Sequential search with TURBO optimization
        RANDOM = 1,         // Random 256-bit search
        GEOMETRIC = 2,      // 3-Phase geometric search
        TERMINATOR = 3,     // Multiplicative descent
        DOUBLING = 4,       // Powers of 2 search
        HAMMING = 5,        // Low-weight keys (Hamming distance)
        MODULAR_STRIDE = 6  // Arithmetic progression
    };

    // Input Type Selection
    enum class InputType : int {
        ADDRESS = 0,    // Standard Bitcoin addresses (P2PKH)
        HASH160 = 1,    // Raw 20-byte Hash160 hex
        PUBKEY = 2      // Raw 33/65-byte Public Key hex
    };

    // Core Settings
    Mode mode = Mode::LINEAR;
    InputType input_type = InputType::ADDRESS;
    int num_threads = 0;        // 0 = auto-detect
    bool verbose = false;
    
    // File Paths
    std::string database_file;  // File containing targets
    std::string output_file = "found.txt";
    std::string log_file;       // Log file path
    
    // Search Range (for Linear/Geometric/Doubling modes)
    uint64_t start_value = 1;
    uint64_t end_value = 0xFFFFFFFFFFFFFFFF;
    
    // Bit Range (for Doubling/Hamming modes)
    int range_min_bit = 1;
    int range_max_bit = 255;
    
    // Stride (for Modular Stride mode)
    uint64_t multiplier = 1000000;
    
    // Flags
    bool stop_on_find = false;
};

// ============================================================================
// ARGUMENT PARSING
// ============================================================================

// Parse command line arguments into Config struct
bool parse_args(int argc, char** argv, Config& config);

// Print usage information
void print_usage(const char* prog_name);

}  // namespace btc_gold
