#include "hash160.h"
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <cstring>

namespace btc_gold {

Hash160Engine::Hash160Engine() {
}

Hash160Engine::~Hash160Engine() {
}

Hash160 Hash160Engine::compute(const std::vector<uint8_t>& data) const {
    Hash160 result;
    
    // SHA256
    unsigned char sha_hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha_ctx;
    SHA256_Init(&sha_ctx);
    SHA256_Update(&sha_ctx, data.data(), data.size());
    SHA256_Final(sha_hash, &sha_ctx);
    
    // RIPEMD160
    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    const EVP_MD* md = EVP_ripemd160();
    unsigned int len = 0;
    
    EVP_DigestInit_ex(mdctx, md, nullptr);
    EVP_DigestUpdate(mdctx, sha_hash, SHA256_DIGEST_LENGTH);
    EVP_DigestFinal_ex(mdctx, result.data(), &len);
    EVP_MD_CTX_free(mdctx);
    
    return result;
}

Hash160 Hash160Engine::compute_from_pubkey(const std::vector<uint8_t>& pubkey) const {
    return compute(pubkey);
}

void Hash160Engine::batch_compute(
    const std::vector<std::vector<uint8_t>>& inputs,
    std::vector<Hash160>& outputs
) const {
    outputs.clear();
    for (const auto& input : inputs) {
        outputs.push_back(compute(input));
    }
}

}  // namespace btc_gold
