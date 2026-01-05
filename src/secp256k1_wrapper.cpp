#include "secp256k1_wrapper.h"
#include <secp256k1.h>
#include <secp256k1_recovery.h>
#include <cstring>
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <openssl/ripemd.h>
#include <algorithm>
#include <cmath>

namespace btc_gold {

static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

Secp256k1Wrapper::Secp256k1Wrapper() {
    context_ = secp256k1_context_create(
        SECP256K1_CONTEXT_VERIFY | SECP256K1_CONTEXT_SIGN
    );
    if (!context_) {
        throw std::runtime_error("Failed to create secp256k1 context");
    }
}

Secp256k1Wrapper::~Secp256k1Wrapper() {
    if (context_) {
        secp256k1_context_destroy(context_);
    }
}

// ============================================================================
// PUBKEY GENERATION
// ============================================================================

PublicKey Secp256k1Wrapper::pubkey_compressed(const PrivateKey& privkey) const {
    PublicKey result;
    secp256k1_pubkey pubkey;
    
    if (!secp256k1_ec_pubkey_create(context_, &pubkey, privkey.data())) {
        throw std::runtime_error("Failed to create public key");
    }
    
    size_t output_len = 33;
    if (!secp256k1_ec_pubkey_serialize(
        context_,
        result.data(),
        &output_len,
        &pubkey,
        SECP256K1_EC_COMPRESSED)) {
        throw std::runtime_error("Failed to serialize public key");
    }
    
    return result;
}

std::vector<uint8_t> Secp256k1Wrapper::pubkey_uncompressed(const PrivateKey& privkey) const {
    std::vector<uint8_t> result(65);
    secp256k1_pubkey pubkey;
    
    if (!secp256k1_ec_pubkey_create(context_, &pubkey, privkey.data())) {
        throw std::runtime_error("Failed to create public key");
    }
    
    size_t output_len = 65;
    if (!secp256k1_ec_pubkey_serialize(
        context_,
        result.data(),
        &output_len,
        &pubkey,
        SECP256K1_EC_UNCOMPRESSED)) {
        throw std::runtime_error("Failed to serialize public key");
    }
    
    return result;
}

bool Secp256k1Wrapper::verify_privkey(const PrivateKey& privkey) const {
    return secp256k1_ec_seckey_verify(context_, privkey.data());
}

// ============================================================================
// OPTIMIZATION: Point Addition for Linear Mode TURBO
// ============================================================================

bool Secp256k1Wrapper::pubkey_tweak_add(PublicKey& pubkey_bytes, int increment) const {
    try {
        secp256k1_pubkey pubkey;
        
        // Parse current pubkey
        if (!secp256k1_ec_pubkey_parse(context_, &pubkey, pubkey_bytes.data(), pubkey_bytes.size())) {
            return false;
        }
        
        // Create tweak from increment
        uint8_t tweak[32] = {0};
        if (increment > 0) {
            // Simple byte assignment
            for (int i = 0; i < 4 && i < 32; i++) {
                tweak[31 - i] = (increment >> (8 * i)) & 0xFF;
            }
        }
        
        // Add tweak (scalar) to point
        if (!secp256k1_ec_pubkey_tweak_add(context_, &pubkey, tweak)) {
            return false;
        }
        
        // Serialize back (compressed)
        size_t len = 33;
        if (!secp256k1_ec_pubkey_serialize(
            context_, 
            pubkey_bytes.data(), 
            &len, 
            &pubkey, 
            SECP256K1_EC_COMPRESSED)) {
            return false;
        }
        
        return true;
        
    } catch (...) {
        return false;
    }
}

// ============================================================================
// HASHING - WITH FIX FOR RIPEMD160 DEPRECATION
// ============================================================================

Hash160 Secp256k1Wrapper::hash160(const PublicKey& pubkey) const {
    Hash160 result;
    
    // SHA256
    unsigned char sha256_hash[SHA256_DIGEST_LENGTH];
    SHA256(pubkey.data(), pubkey.size(), sha256_hash);
    
    // RIPEMD160 (with deprecation handling)
    unsigned char ripemd160_hash[RIPEMD160_DIGEST_LENGTH];
    
    // For OpenSSL 3.0+, use EVP interface
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx, EVP_ripemd160(), NULL);
    EVP_DigestUpdate(mdctx, sha256_hash, SHA256_DIGEST_LENGTH);
    unsigned int md_len;
    EVP_DigestFinal_ex(mdctx, ripemd160_hash, &md_len);
    EVP_MD_CTX_free(mdctx);
#else
    RIPEMD160(sha256_hash, SHA256_DIGEST_LENGTH, ripemd160_hash);
#endif
    
    std::copy(ripemd160_hash, ripemd160_hash + RIPEMD160_DIGEST_LENGTH, result.begin());
    return result;
}

Hash160 Secp256k1Wrapper::hash160_of_hash160(const Hash160& hash) const {
    Hash160 result;
    
    // SHA256
    unsigned char sha256_hash[SHA256_DIGEST_LENGTH];
    SHA256(hash.data(), hash.size(), sha256_hash);
    
    // RIPEMD160
    unsigned char ripemd160_hash[RIPEMD160_DIGEST_LENGTH];
    
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx, EVP_ripemd160(), NULL);
    EVP_DigestUpdate(mdctx, sha256_hash, SHA256_DIGEST_LENGTH);
    unsigned int md_len;
    EVP_DigestFinal_ex(mdctx, ripemd160_hash, &md_len);
    EVP_MD_CTX_free(mdctx);
#else
    RIPEMD160(sha256_hash, SHA256_DIGEST_LENGTH, ripemd160_hash);
#endif
    
    std::copy(ripemd160_hash, ripemd160_hash + RIPEMD160_DIGEST_LENGTH, result.begin());
    return result;
}

// ============================================================================
// PRIVATE KEY CONVERSION
// ============================================================================

void Secp256k1Wrapper::int_to_privkey(uint64_t value, PrivateKey& privkey) const {
    privkey.fill(0);
    
    // Convert little-endian 64-bit to big-endian bytes
    // Place in the last 8 bytes of the 32-byte key
    for (int i = 0; i < 8; i++) {
        privkey[31 - i] = (value >> (8 * i)) & 0xFF;
    }
}

// ============================================================================
// BASE58 ENCODING
// ============================================================================

std::string Secp256k1Wrapper::encode_base58(const std::vector<uint8_t>& data) const {
    // Count leading zeros
    int zeros = 0;
    while (zeros < (int)data.size() && data[zeros] == 0) {
        zeros++;
    }
    
    // Convert to base58
    std::vector<unsigned char> b58((data.size() * 138 / 100) + 1, 0);
    std::vector<unsigned char> input = data;
    
    size_t size = 0;
    for (auto byte : input) {
        int carry = byte;
        for (size_t i = 0; i < size; ++i) {
            carry += 256 * b58[i];
            b58[i] = carry % 58;
            carry /= 58;
        }
        while (carry > 0) {
            b58[size++] = carry % 58;
            carry /= 58;
        }
    }
    
    // Build string
    std::string str(zeros, '1');
    for (size_t i = 0; i < size; ++i) {
        str += BASE58_ALPHABET[b58[size - 1 - i]];
    }
    
    return str;
}

std::string Secp256k1Wrapper::encode_base58check(const std::vector<uint8_t>& data) const {
    std::vector<uint8_t> payload = data;
    
    // Double SHA256 checksum
    unsigned char hash1[SHA256_DIGEST_LENGTH];
    unsigned char hash2[SHA256_DIGEST_LENGTH];
    
    SHA256(payload.data(), payload.size(), hash1);
    SHA256(hash1, SHA256_DIGEST_LENGTH, hash2);
    
    // Append first 4 bytes of checksum
    payload.insert(payload.end(), hash2, hash2 + 4);
    
    return encode_base58(payload);
}

// ============================================================================
// WIF ENCODING
// ============================================================================

std::string Secp256k1Wrapper::privkey_to_wif(const PrivateKey& privkey, bool compressed) const {
    std::vector<uint8_t> data;
    data.push_back(0x80);  // Mainnet prefix
    data.insert(data.end(), privkey.begin(), privkey.end());
    if (compressed) {
        data.push_back(0x01);
    }
    return encode_base58check(data);
}

// ============================================================================
// ADDRESS CONVERSION
// ============================================================================

std::string Secp256k1Wrapper::hash160_to_address(const Hash160& hash160) const {
    std::vector<uint8_t> payload;
    payload.push_back(0x00);  // Version byte (Bitcoin mainnet P2PKH)
    payload.insert(payload.end(), hash160.begin(), hash160.end());
    return encode_base58check(payload);
}

std::string Secp256k1Wrapper::pubkey_to_address(const PublicKey& pubkey) const {
    Hash160 h = hash160(pubkey);
    return hash160_to_address(h);
}

}  // namespace btc_gold
