#include "database.h"
#include "logger.h"
#include <fstream>
#include <sstream>

namespace btc_gold {

Database::Database() {}

bool Database::load(const std::string& filename, Config::InputType type) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        Logger::instance().error("Cannot open file: %s", filename.c_str());
        return false;
    }
    
    std::string line;
    int loaded = 0;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        try {
            Hash160 hash;
            if (type == Config::InputType::ADDRESS) {
                hash = parse_address(line);
            } else if (type == Config::InputType::HASH160) {
                hash = parse_hash160(line);
            } else {
                hash = parse_pubkey(line);
            }
            
            std::string hex_str;
            for (auto byte : hash) {
                char buf[3];
                snprintf(buf, sizeof(buf), "%02x", byte);
                hex_str += buf;
            }
            targets_.insert(hex_str);
            loaded++;
        }
        catch (...) {
            continue;
        }
    }
    
    file.close();
    Logger::instance().info("Loaded %d targets", loaded);
    return loaded > 0;
}

bool Database::contains(const Hash160& hash) const {
    std::string hex_str;
    for (auto byte : hash) {
        char buf[3];
        snprintf(buf, sizeof(buf), "%02x", byte);
        hex_str += buf;
    }
    return targets_.find(hex_str) != targets_.end();
}

Hash160 Database::parse_address(const std::string& address) {
    Hash160 result;
    // Simplified: would need base58 decoding
    return result;
}

Hash160 Database::parse_hash160(const std::string& hex) {
    Hash160 result;
    if (hex.length() != 40) throw std::runtime_error("Invalid hash160");
    
    for (int i = 0; i < 20; ++i) {
        result[i] = std::stoi(hex.substr(i*2, 2), nullptr, 16);
    }
    return result;
}

Hash160 Database::parse_pubkey(const std::string& hex) {
    Hash160 result;
    // Simplified: would need SHA256 + RIPEMD160
    return result;
}

}  // namespace btc_gold
