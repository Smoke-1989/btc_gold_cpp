#pragma once

#include "types.h"
#include <vector>
#include <openssl/evp.h>

namespace btc_gold {

class Hash160Engine {
public:
    Hash160Engine();
    ~Hash160Engine();
    
    Hash160 compute(const std::vector<unsigned char>& data);
    Hash160 compute(const unsigned char* data, size_t len);

private:
    EVP_MD_CTX *sha256_ctx_;
    EVP_MD_CTX *ripemd160_ctx_;
};

}  // namespace btc_gold
