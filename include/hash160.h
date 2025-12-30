#pragma once

#include "types.h"
#include <vector>

namespace btc_gold {

class Hash160Engine {
public:
    Hash160Engine();
    ~Hash160Engine();
    
    /**
     * Compute SHA256(RIPEMD160(data))
     * @param data Input data (typically 33 bytes for compressed pubkey)
     * @return 20-byte hash160
     */
    Hash160 compute(const std::vector<uint8_t>& data) const;
    
    /**
     * Compute hash160 from public key bytes
     */
    Hash160 compute_from_pubkey(const std::vector<uint8_t>& pubkey) const;
    
    /**
     * Fast batch computation of hash160 values
     */
    void batch_compute(
        const std::vector<std::vector<uint8_t>>& inputs,
        std::vector<Hash160>& outputs
    ) const;
    
private:
    // Implementation details hidden
};

}  // namespace btc_gold
