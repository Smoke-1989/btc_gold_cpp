#pragma once

#include <cstdint>
#include <array>
#include <string>

namespace btc_gold {

// ============================================================================
// FUNDAMENTAL TYPES
// ============================================================================

using Hash160 = std::array<uint8_t, 20>;  // 160-bit hash
using PrivateKey = std::array<uint8_t, 32>; // 256-bit private key
using PublicKey = std::array<uint8_t, 33>;  // 33-byte compressed public key

// ============================================================================
// KEY RESULT
// ============================================================================

struct KeyResult {
    PrivateKey privkey;
    Hash160 hash160;
    std::string address;
    std::string wif_compressed;
    std::string wif_uncompressed;
    bool found = false;
    int scan_mode = 0;  // 1=compressed, 2=uncompressed, 3=both
};

// ============================================================================
// CONFIGURATION
// ============================================================================

struct Config {
    // Operation mode
    enum Mode { 
        LINEAR = 1, 
        RANDOM = 2, 
        GEOMETRIC = 3,
        TERMINATOR = 4  // <--- O EXTERMINADOR DO FUTURO
    };
    Mode mode = LINEAR;
    
    // Database input type
    enum InputType { ADDRESS = 1, HASH160 = 2, PUBKEY = 3 };
    InputType input_type = ADDRESS;
    
    // Scan mode
    enum ScanMode { COMPRESSED = 1, UNCOMPRESSED = 2, BOTH = 3 };
    ScanMode scan_mode = COMPRESSED;
    
    // Parameters
    uint64_t start_value = 1;
    uint64_t end_value = 0xFFFFFFFFFFFFFFFF;
    
    // Terminator Mode Specifics
    uint64_t multiplier = 2; // Initial Salto
    int range_min_bit = 1;   // Start Bit (e.g. 66)
    int range_max_bit = 256; // End Bit (e.g. 67)
    
    uint64_t stride = 1;
    
    // Threading
    int num_threads = 0;  // 0 = auto-detect
    
    // Behavior
    bool stop_on_find = false;
    bool verbose = true;
    
    // Files
    std::string database_file = "alvos.txt";
    std::string output_file = "found_gold.txt";
};

}  // namespace btc_gold
