#pragma once

#include "types.h"
#include <unordered_set>
#include <string>
#include <vector>

namespace btc_gold {

class Database {
public:
    Database();
    
    /**
     * Load targets from file
     * Supports: Bitcoin addresses, HASH160, Public Keys
     */
    bool load(const std::string& filename, Config::InputType type);
    
    /**
     * Check if hash160 exists in database
     */
    bool contains(const Hash160& hash) const;
    
    /**
     * Get number of loaded targets
     */
    size_t size() const { return targets_.size(); }
    
    /**
     * Get reference to targets
     */
    const std::unordered_set<std::string>& get_targets() const {
        return targets_;
    }
    
private:
    std::unordered_set<std::string> targets_;
    
    Hash160 parse_address(const std::string& address);
    Hash160 parse_hash160(const std::string& hex);
    Hash160 parse_pubkey(const std::string& hex);
};

}  // namespace btc_gold
