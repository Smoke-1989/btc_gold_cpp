#include "database.h"
#include "logger.h"
#include <fstream>
#include <sstream>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <iomanip>

namespace btc_gold {

// ============================================================================
// Base58 Alphabet for Bitcoin
// ============================================================================
static const std::string BASE58_ALPHABET = 
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// ============================================================================
// Helper: Decode Base58 string
// ============================================================================
static std::vector<uint8_t> decode_base58(const std::string& encoded) {
    // Convert base58 to big integer
    std::vector<uint8_t> decoded(25, 0);
    
    for (char c : encoded) {
        size_t digit = BASE58_ALPHABET.find(c);
        if (digit == std::string::npos) {
            throw std::runtime_error("Invalid base58 character");
        }
        
        // Multiply current value by 58
        int carry = digit;
        for (int i = decoded.size() - 1; i >= 0; --i) {
            carry += decoded[i] * 58;
            decoded[i] = carry & 0xFF;
            carry >>= 8;
        }
        
        if (carry > 0) {
            throw std::runtime_error("Base58 overflow");
        }
    }
    
    // Handle leading zeros
    size_t leading_zeros = 0;
    for (char c : encoded) {
        if (c == '1') leading_zeros++;
        else break;
    }
    
    std::vector<uint8_t> result(leading_zeros);
    result.insert(result.end(), decoded.begin() + (25 - 20), decoded.end());
    
    return result;
}

// ============================================================================
// Helper: Hex string to bytes
// ============================================================================
static std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    if (hex.length() % 2 != 0) {
        throw std::runtime_error("Invalid hex string length");
    }
    
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

// ============================================================================
// Constructor
// ============================================================================
Database::Database() {}

// ============================================================================
// Load targets from file
// ============================================================================
bool Database::load(const std::string& filename, Config::InputType type) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        Logger::instance().error("Cannot open file: %s", filename.c_str());
        return false;
    }
    
    std::string line;
    int loaded = 0;
    int errors = 0;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty()) continue;
        
        try {
            Hash160 hash;
            
            if (type == Config::InputType::ADDRESS) {
                hash = parse_address(line);
            } else if (type == Config::InputType::HASH160) {
                hash = parse_hash160(line);
            } else {
                hash = parse_pubkey(line);
            }
            
            // Convert hash to hex string for storage
            std::string hex_str;
            for (auto byte : hash) {
                char buf[3];
                snprintf(buf, sizeof(buf), "%02x", byte);
                hex_str += buf;
            }
            
            targets_.insert(hex_str);
            loaded++;
        }
        catch (const std::exception& e) {
            errors++;
            std::string error_msg = std::string("Error parsing line: ") + line + 
                                    " | Reason: " + e.what();
            Logger::instance().warn(error_msg);
            continue;
        }
    }
    
    file.close();
    Logger::instance().info("Loaded %d targets", loaded);
    return loaded > 0;
}

// ============================================================================
// Check if hash160 exists in database
// ============================================================================
bool Database::contains(const Hash160& hash) const {
    std::string hex_str;
    for (auto byte : hash) {
        char buf[3];
        snprintf(buf, sizeof(buf), "%02x", byte);
        hex_str += buf;
    }
    return targets_.find(hex_str) != targets_.end();
}

// ============================================================================
// Parse Bitcoin Address (P2PKH or P2SH)
// ============================================================================
Hash160 Database::parse_address(const std::string& address) {
    if (address.empty()) {
        throw std::runtime_error("Empty address");
    }
    
    try {
        // Decode from base58
        std::vector<uint8_t> decoded = decode_base58(address);
        
        if (decoded.size() != 25) {
            throw std::runtime_error("Invalid address length (expected 25 bytes)");
        }
        
        // Verify checksum (last 4 bytes)
        std::vector<uint8_t> payload(decoded.begin(), decoded.end() - 4);
        std::vector<uint8_t> checksum(decoded.end() - 4, decoded.end());
        
        // Calculate checksum: SHA256(SHA256(payload))
        unsigned char sha256_1[SHA256_DIGEST_LENGTH];
        unsigned char sha256_2[SHA256_DIGEST_LENGTH];
        
        SHA256(payload.data(), payload.size(), sha256_1);
        SHA256(sha256_1, SHA256_DIGEST_LENGTH, sha256_2);
        
        // Verify checksum
        if (std::vector<uint8_t>(sha256_2, sha256_2 + 4) != checksum) {
            throw std::runtime_error("Invalid address checksum");
        }
        
        // Extract hash160 (bytes 1-21)
        Hash160 result;
        std::copy(payload.begin() + 1, payload.end(), result.begin());
        
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Address parse error: ") + e.what());
    }
}

// ============================================================================
// Parse Hash160 (40 hex characters)
// ============================================================================
Hash160 Database::parse_hash160(const std::string& hex) {
    if (hex.length() != 40) {
        throw std::runtime_error("Invalid hash160 length (expected 40 hex chars)");
    }
    
    // Verify it's valid hex
    for (char c : hex) {
        if (!std::isxdigit(c)) {
            throw std::runtime_error("Invalid hex characters in hash160");
        }
    }
    
    Hash160 result;
    for (int i = 0; i < 20; ++i) {
        result[i] = std::stoi(hex.substr(i * 2, 2), nullptr, 16);
    }
    return result;
}

// ============================================================================
// Parse Public Key (compressed or uncompressed) and hash it
// ============================================================================
Hash160 Database::parse_pubkey(const std::string& hex) {
    if (hex.length() != 66 && hex.length() != 130) {
        throw std::runtime_error(
            "Invalid pubkey length (expected 66 for compressed or 130 for uncompressed)"
        );
    }
    
    // Verify it's valid hex
    for (char c : hex) {
        if (!std::isxdigit(c)) {
            throw std::runtime_error("Invalid hex characters in pubkey");
        }
    }
    
    try {
        // Convert hex to bytes
        std::vector<uint8_t> pubkey = hex_to_bytes(hex);
        
        // Validate pubkey format
        if (pubkey.size() == 33) {
            // Compressed: 02 or 03 prefix
            if (pubkey[0] != 0x02 && pubkey[0] != 0x03) {
                throw std::runtime_error("Invalid compressed pubkey prefix");
            }
        } else if (pubkey.size() == 65) {
            // Uncompressed: 04 prefix
            if (pubkey[0] != 0x04) {
                throw std::runtime_error("Invalid uncompressed pubkey prefix");
            }
        } else {
            throw std::runtime_error("Invalid pubkey size");
        }
        
        // Hash160 = RIPEMD160(SHA256(pubkey))
        unsigned char sha256_hash[SHA256_DIGEST_LENGTH];
        SHA256(pubkey.data(), pubkey.size(), sha256_hash);
        
        unsigned char ripemd160_hash[RIPEMD160_DIGEST_LENGTH];
        RIPEMD160(sha256_hash, SHA256_DIGEST_LENGTH, ripemd160_hash);
        
        Hash160 result;
        std::copy(ripemd160_hash, ripemd160_hash + 20, result.begin());
        
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Pubkey parse error: ") + e.what());
    }
}

}  // namespace btc_gold
