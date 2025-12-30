#include "secp256k1_wrapper.h"
#include <secp256k1.h>
#include <secp256k1_recovery.h>
#include <cstring>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <algorithm>

namespace btc_gold {

static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

Secp256k1& Secp256k1::instance() {
    static Secp256k1 instance;
    return instance;
}

Secp256k1::Secp256k1() {
    context_ = secp256k1_context_create(
        SECP256K1_CONTEXT_VERIFY | SECP256K1_CONTEXT_SIGN
    );
}

Secp256k1::~Secp256k1() {
    if (context_) {
        secp256k1_context_destroy(context_);
    }
}

PublicKey Secp256k1::pubkey_compressed(const PrivateKey& privkey) const {
    PublicKey result;
    secp256k1_pubkey pubkey;
    
    secp256k1_ec_pubkey_create(context_, &pubkey, privkey.data());
    
    size_t output_len = 33;
    secp256k1_ec_pubkey_serialize(
        context_,
        result.data(),
        &output_len,
        &pubkey,
        SECP256K1_EC_COMPRESSED
    );
    
    return result;
}

std::vector<uint8_t> Secp256k1::pubkey_uncompressed(const PrivateKey& privkey) const {
    std::vector<uint8_t> result(65);
    secp256k1_pubkey pubkey;
    
    secp256k1_ec_pubkey_create(context_, &pubkey, privkey.data());
    
    size_t output_len = 65;
    secp256k1_ec_pubkey_serialize(
        context_,
        result.data(),
        &output_len,
        &pubkey,
        SECP256K1_EC_UNCOMPRESSED
    );
    
    return result;
}

bool Secp256k1::verify_privkey(const PrivateKey& privkey) const {
    return secp256k1_ec_seckey_verify(context_, privkey.data());
}

// Optimization for Linear Mode
bool Secp256k1::pubkey_tweak_add(std::vector<uint8_t>& pubkey_bytes, const uint8_t* tweak) const {
    secp256k1_pubkey pubkey;
    
    // Parse current pubkey
    if (!secp256k1_ec_pubkey_parse(context_, &pubkey, pubkey_bytes.data(), pubkey_bytes.size())) {
        return false;
    }
    
    // Add tweak (scalar) to point
    if (!secp256k1_ec_pubkey_tweak_add(context_, &pubkey, tweak)) {
        return false;
    }
    
    // Serialize back
    size_t len = pubkey_bytes.size();
    secp256k1_ec_pubkey_serialize(
        context_, 
        pubkey_bytes.data(), 
        &len, 
        &pubkey, 
        len == 33 ? SECP256K1_EC_COMPRESSED : SECP256K1_EC_UNCOMPRESSED
    );
    
    return true;
}

// Base58 Utils
std::string Secp256k1::encode_base58(const std::vector<uint8_t>& data) const {
    // Count leading zeros
    int zeros = 0;
    while (zeros < data.size() && data[zeros] == 0) zeros++;
    
    // Convert to big integer
    std::vector<unsigned char> b58((data.size() * 138 / 100) + 1);
    std::vector<unsigned char> input = data;
    
    // Process the bytes
    size_t size = 0;
    for (auto& source : input) {
        int carry = source;
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
    
    // Encode string
    std::string str(zeros, '1');
    for (size_t i = 0; i < size; ++i) {
        str += BASE58_ALPHABET[b58[size - 1 - i]];
    }
    return str;
}

std::string Secp256k1::encode_base58check(const std::vector<uint8_t>& data) const {
    std::vector<uint8_t> payload = data;
    
    // Double SHA256 checksum
    unsigned char hash1[SHA256_DIGEST_LENGTH];
    unsigned char hash2[SHA256_DIGEST_LENGTH];
    
    SHA256(payload.data(), payload.size(), hash1);
    SHA256(hash1, SHA256_DIGEST_LENGTH, hash2);
    
    // Append first 4 bytes
    payload.insert(payload.end(), hash2, hash2 + 4);
    
    return encode_base58(payload);
}

std::string Secp256k1::to_wif(const PrivateKey& privkey, bool compressed) const {
    std::vector<uint8_t> data;
    data.push_back(0x80); // Mainnet prefix
    data.insert(data.end(), privkey.begin(), privkey.end());
    if (compressed) {
        data.push_back(0x01);
    }
    return encode_base58check(data);
}

std::string Secp256k1::to_address(const std::vector<uint8_t>& pubkey_bytes) const {
    // SHA256
    unsigned char sha256_hash[SHA256_DIGEST_LENGTH];
    SHA256(pubkey_bytes.data(), pubkey_bytes.size(), sha256_hash);
    
    // RIPEMD160
    unsigned char ripemd160_hash[RIPEMD160_DIGEST_LENGTH];
    RIPEMD160(sha256_hash, SHA256_DIGEST_LENGTH, ripemd160_hash);
    
    // Add version byte (0x00 for Mainnet)
    std::vector<uint8_t> payload;
    payload.push_back(0x00);
    payload.insert(payload.end(), ripemd160_hash, ripemd160_hash + RIPEMD160_DIGEST_LENGTH);
    
    return encode_base58check(payload);
}

}  // namespace btc_gold
