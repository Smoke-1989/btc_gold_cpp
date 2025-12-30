#pragma once

#include "types.h"
#include <vector>
#include <secp256k1.h>

namespace btc_gold {

class Secp256k1 {
public:
    static Secp256k1& instance();
    
    // Core crypto
    PublicKey pubkey_compressed(const PrivateKey& privkey) const;
    std::vector<uint8_t> pubkey_uncompressed(const PrivateKey& privkey) const;
    bool verify_privkey(const PrivateKey& privkey) const;
    
    // Optimization: Add scalar to public key (Point Addition)
    // Returns true on success, updates pubkey_bytes in place
    bool pubkey_tweak_add(std::vector<uint8_t>& pubkey_bytes, const uint8_t* tweak) const;
    
    // Utils
    std::string to_wif(const PrivateKey& privkey, bool compressed) const;
    std::string to_address(const std::vector<uint8_t>& pubkey_bytes) const;
    std::string encode_base58(const std::vector<uint8_t>& data) const;
    std::string encode_base58check(const std::vector<uint8_t>& data) const;

    secp256k1_context* get_context() const { return context_; }

private:
    Secp256k1();
    ~Secp256k1();
    
    secp256k1_context* context_;
};

}  // namespace btc_gold
