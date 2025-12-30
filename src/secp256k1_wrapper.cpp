#include "secp256k1_wrapper.h"
#include <secp256k1.h>
#include <secp256k1_recovery.h>
#include <cstring>

namespace btc_gold {

static secp256k1_context* global_context = nullptr;

Secp256k1& Secp256k1::instance() {
    static Secp256k1 instance;
    return instance;
}

Secp256k1::Secp256k1() {
    global_context = secp256k1_context_create(
        SECP256K1_CONTEXT_VERIFY | SECP256K1_CONTEXT_SIGN
    );
}

Secp256k1::~Secp256k1() {
    if (global_context) {
        secp256k1_context_destroy(global_context);
    }
}

PublicKey Secp256k1::pubkey_compressed(const PrivateKey& privkey) const {
    PublicKey result;
    secp256k1_pubkey pubkey;
    
    secp256k1_ec_pubkey_create(global_context, &pubkey, privkey.data());
    
    size_t output_len = 33;
    secp256k1_ec_pubkey_serialize(
        global_context,
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
    
    secp256k1_ec_pubkey_create(global_context, &pubkey, privkey.data());
    
    size_t output_len = 65;
    secp256k1_ec_pubkey_serialize(
        global_context,
        result.data(),
        &output_len,
        &pubkey,
        SECP256K1_EC_UNCOMPRESSED
    );
    
    return result;
}

bool Secp256k1::verify_privkey(const PrivateKey& privkey) const {
    // Check if non-zero and less than curve order
    return secp256k1_ec_seckey_verify(global_context, privkey.data());
}

void Secp256k1::batch_pubkeys_compressed(
    const std::vector<PrivateKey>& privkeys,
    std::vector<PublicKey>& pubkeys
) const {
    pubkeys.clear();
    for (const auto& privkey : privkeys) {
        pubkeys.push_back(pubkey_compressed(privkey));
    }
}

}  // namespace btc_gold
