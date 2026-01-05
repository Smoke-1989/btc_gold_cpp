#pragma once

#include "types.h"
#include <vector>
#include <secp256k1.h>

namespace btc_gold {

// ============================================================================
// SECP256K1 WRAPPER - Cryptographic Operations
// ============================================================================

class Secp256k1Wrapper {
public:
    // Constructor/Destructor
    Secp256k1Wrapper();
    ~Secp256k1Wrapper();
    
    // ========================================================================
    // PUBKEY GENERATION
    // ========================================================================
    
    // Generate compressed public key (33 bytes)
    PublicKey pubkey_compressed(const PrivateKey& privkey) const;
    
    // Generate uncompressed public key (65 bytes)
    std::vector<uint8_t> pubkey_uncompressed(const PrivateKey& privkey) const;
    
    // Verify private key is valid
    bool verify_privkey(const PrivateKey& privkey) const;
    
    // ========================================================================
    // OPTIMIZATION: Point Addition for Linear Mode TURBO
    // ========================================================================
    
    // Add scalar to public key point (EC point addition)
    // Used for TURBO mode: incremental pubkey calculation
    // Much faster than recalculating pubkey from privkey each time
    bool pubkey_tweak_add(PublicKey& pubkey_bytes, int increment) const;
    
    // ========================================================================
    // HASHING: SHA256 + RIPEMD160
    // ========================================================================
    
    // Hash160: SHA256(RIPEMD160(data))
    // Standard Bitcoin hash for addresses
    Hash160 hash160(const PublicKey& pubkey) const;
    
    // Hash160 of hash160
    Hash160 hash160_of_hash160(const Hash160& hash) const;
    
    // ========================================================================
    // PRIVATE KEY CONVERSION
    // ========================================================================
    
    // Convert uint64 integer to private key (32-byte array)
    // Used for range-based searching
    void int_to_privkey(uint64_t value, PrivateKey& privkey) const;
    
    // ========================================================================
    // BASE58 ENCODING
    // ========================================================================
    
    // Base58 encode (Bitcoin standard)
    std::string encode_base58(const std::vector<uint8_t>& data) const;
    
    // Base58Check encode (Base58 + SHA256 checksum)
    std::string encode_base58check(const std::vector<uint8_t>& data) const;
    
    // ========================================================================
    // WIF (Wallet Import Format)
    // ========================================================================
    
    // Private key to WIF (Wallet Import Format)
    // compressed: true for compressed pubkey, false for uncompressed
    std::string privkey_to_wif(const PrivateKey& privkey, bool compressed) const;
    
    // ========================================================================
    // ADDRESS CONVERSION
    // ========================================================================
    
    // Hash160 to Bitcoin address
    std::string hash160_to_address(const Hash160& hash160) const;
    
    // Public key to Bitcoin address
    std::string pubkey_to_address(const PublicKey& pubkey) const;
    
    // ========================================================================
    // CONTEXT ACCESS
    // ========================================================================
    
    secp256k1_context* get_context() const { return context_; }

private:
    secp256k1_context* context_;
};

}  // namespace btc_gold
