#pragma once

#include "types.h"
#include "logger.h"
#include <unordered_set>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace btc_gold {

// ============================================================================
// DATABASE - Production-grade Target Management
// ============================================================================

class Database {
public:
    // Constructor: Default (empty database)
    Database();
    
    // Constructor: Load targets from file on initialization
    explicit Database(const std::string& filename, Config::InputType type);
    
    // Destructor
    ~Database() = default;
    
    // Delete copy constructor and assignment (RAII pattern)
    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;
    
    // Move constructor and assignment (allow moving)
    Database(Database&&) noexcept = default;
    Database& operator=(Database&&) noexcept = default;
    
    // PUBLIC: Load targets from file
    // Returns: true if at least one target loaded, false on total failure
    // Throws: std::runtime_error on critical errors
    bool load(const std::string& filename, Config::InputType type);
    
    // PUBLIC: Check if hash160 exists in database
    bool contains(const Hash160& hash) const;
    
    // PUBLIC: Get number of loaded targets
    size_t size() const { return targets_.size(); }
    
    // PUBLIC: Check if database is empty
    bool empty() const { return targets_.empty(); }
    
    // PUBLIC: Clear all targets
    void clear() { targets_.clear(); }
    
    // PUBLIC: Get const reference to targets (for advanced usage)
    const std::unordered_set<std::string>& get_targets() const {
        return targets_;
    }
    
    // PUBLIC: Set logger for error reporting
    void set_logger(Logger* logger) { logger_ = logger; }
    
private:
    // Database storage: hash160 as hex string for fast lookup
    std::unordered_set<std::string> targets_;
    
    // Logger pointer (optional, for error reporting)
    Logger* logger_;
    
    // Parse Bitcoin Address (P2PKH or P2SH) -> hash160
    Hash160 parse_address(const std::string& address);
    
    // Parse HASH160 hex string (40 characters)
    Hash160 parse_hash160(const std::string& hex);
    
    // Parse Public Key (compressed or uncompressed) -> hash160
    Hash160 parse_pubkey(const std::string& hex);
    
    // Helper: Decode Base58 string
    static std::vector<uint8_t> decode_base58(const std::string& encoded);
    
    // Helper: Convert hex string to bytes
    static std::vector<uint8_t> hex_to_bytes(const std::string& hex);
};

}  // namespace btc_gold
