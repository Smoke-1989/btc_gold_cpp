#include "hash160.h"
#include <openssl/evp.h>
#include <cstring>

namespace btc_gold {

Hash160Engine::Hash160Engine() {
    sha256_ctx_ = EVP_MD_CTX_new();
    ripemd160_ctx_ = EVP_MD_CTX_new();
}

Hash160Engine::~Hash160Engine() {
    if (sha256_ctx_) EVP_MD_CTX_free(sha256_ctx_);
    if (ripemd160_ctx_) EVP_MD_CTX_free(ripemd160_ctx_);
}

Hash160 Hash160Engine::compute(const std::vector<unsigned char>& data) {
    return compute(data.data(), data.size());
}

Hash160 Hash160Engine::compute(const unsigned char* data, size_t len) {
    unsigned char sha256_result[32];
    unsigned char ripemd160_result[20];
    unsigned int out_len = 0;

    // 1. SHA-256 (High Performance EVP)
    EVP_DigestInit_ex(sha256_ctx_, EVP_sha256(), NULL);
    EVP_DigestUpdate(sha256_ctx_, data, len);
    EVP_DigestFinal_ex(sha256_ctx_, sha256_result, &out_len);

    // 2. RIPEMD-160
    const EVP_MD* md = EVP_ripemd160();
    
    EVP_DigestInit_ex(ripemd160_ctx_, md, NULL);
    EVP_DigestUpdate(ripemd160_ctx_, sha256_result, 32);
    EVP_DigestFinal_ex(ripemd160_ctx_, ripemd160_result, &out_len);

    Hash160 result;
    std::memcpy(result.data(), ripemd160_result, 20);
    return result;
}

} // namespace btc_gold
