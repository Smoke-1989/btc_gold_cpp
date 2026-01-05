#pragma once

#include <cstdint>
#include <array>
#include <string>

namespace btc_gold {

// ============================================================================
// FUNDAMENTAL TYPES
// ============================================================================

using Hash160 = std::array<uint8_t, 20>;      // 160-bit hash
using PrivateKey = std::array<uint8_t, 32>;   // 256-bit private key
using PublicKey = std::array<uint8_t, 33>;    // 33-byte compressed public key

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

}  // namespace btc_gold
