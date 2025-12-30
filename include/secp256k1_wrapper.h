#pragma once

#include "types.h"
#include <vector>
#include <memory>

namespace btc_gold {

class Secp256k1 {
public:
    static Secp256k1& instance();
    
    /**
     * Generate compressed public key from private key
     */
    PublicKey pubkey_compressed(const PrivateKey& privkey) const;
    
    /**
     * Get uncompressed public key (65 bytes)
     */
    std::vector<uint8_t> pubkey_uncompressed(const PrivateKey& privkey) const;
    
    /**
     * Verify private key is valid
     */
    bool verify_privkey(const PrivateKey& privkey) const;
    
    /**
     * Fast batch public key generation
     */
    void batch_pubkeys_compressed(
        const std::vector<PrivateKey>& privkeys,
        std::vector<PublicKey>& pubkeys
    ) const;
    
private:
    Secp256k1();
    ~Secp256k1();
    Secp256k1(const Secp256k1&) = delete;
    Secp256k1& operator=(const Secp256k1&) = delete;
};

}  // namespace btc_gold
