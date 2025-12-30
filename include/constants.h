#pragma once

#include <cstdint>
#include <limits>

namespace btc_gold {

// ============================================================================
// SECP256K1 CONSTANTS
// ============================================================================

// Bitcoin curve order (n)
constexpr uint64_t CURVE_ORDER_UPPER = 0xFFFFFFFEBAAEDCE6ULL;
constexpr uint64_t CURVE_ORDER_LOWER = 0xC6AF48A03BBFD25EULL;

// Maximum valid private key (n-1)
constexpr uint64_t MAX_PRIVATE_KEY = 0xFFFFFFFEBAAEDCE6ULL;  // Simplified

// ============================================================================
// PERFORMANCE CONSTANTS
// ============================================================================

// Batch size for processing
constexpr int BATCH_SIZE = 10000;

// Update interval for progress
constexpr int UPDATE_INTERVAL = 100000;

// Memory pool size
constexpr int MEMORY_POOL_SIZE = 1024;

// ============================================================================
// STRING CONSTANTS
// ============================================================================

constexpr const char* LOGO = R"(
    ███████╗ ██████╗ ██╗   ██╗███╗   ██╗ ██████╗ 
    ██╔════╝██╔═══██╗██║   ██║████╗  ██║██╔════╝ 
    ███████╗██║   ██║██║   ██║██╔██╗ ██║██║  ███╗
    ╚════██║██║   ██║██║   ██║██║╚██╗██║██║   ██║
    ███████║╚██████╔╝╚██████╔╝██║ ╚████║╚██████╔╝
    ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ 
    BTC GOLD C++ - Professional Edition v1.0
)";

}  // namespace btc_gold
