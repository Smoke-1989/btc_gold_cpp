#include "worker.h"
#include "logger.h"
#include <random>
#include <fstream>

namespace btc_gold {

Worker::Worker(
    int worker_id,
    const Config& config,
    const Database& database,
    Stats& stats
) : worker_id_(worker_id), config_(config), database_(database), stats_(stats) {
}

void Worker::run() {
    switch (config_.mode) {
        case Config::Mode::LINEAR:
            run_linear_mode();
            break;
        case Config::Mode::RANDOM:
            run_random_mode();
            break;
        case Config::Mode::GEOMETRIC:
            run_geometric_mode();
            break;
    }
}

void Worker::run_linear_mode() {
    uint64_t current = config_.start_value + worker_id_;
    
    while (!stats_.should_stop) {
        PrivateKey privkey_bytes;
        for (int i = 0; i < 32; ++i) {
            privkey_bytes[i] = (current >> (i * 8)) & 0xFF;
        }
        
        if (secp256k1_.verify_privkey(privkey_bytes)) {
            auto pubkey = secp256k1_.pubkey_compressed(privkey_bytes);
            std::vector<uint8_t> pubkey_vec(pubkey.begin(), pubkey.end());
            auto hash = hash_engine_.compute(pubkey_vec);
            check_and_save(privkey_bytes, hash);
            
            if (config_.scan_mode == Config::ScanMode::BOTH || 
                config_.scan_mode == Config::ScanMode::UNCOMPRESSED) {
                auto pubkey_u = secp256k1_.pubkey_uncompressed(privkey_bytes);
                auto hash_u = hash_engine_.compute(pubkey_u);
                check_and_save(privkey_bytes, hash_u);
            }
        }
        
        current += config_.stride;
        stats_.total_keys++;
    }
}

void Worker::run_random_mode() {
    std::mt19937_64 rng(worker_id_ + std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist(config_.start_value, config_.end_value);
    
    while (!stats_.should_stop) {
        uint64_t rand_val = dist(rng);
        PrivateKey privkey_bytes;
        for (int i = 0; i < 32; ++i) {
            privkey_bytes[i] = (rand_val >> (i * 8)) & 0xFF;
        }
        
        if (secp256k1_.verify_privkey(privkey_bytes)) {
            auto pubkey = secp256k1_.pubkey_compressed(privkey_bytes);
            std::vector<uint8_t> pubkey_vec(pubkey.begin(), pubkey.end());
            auto hash = hash_engine_.compute(pubkey_vec);
            check_and_save(privkey_bytes, hash);
        }
        
        stats_.total_keys++;
    }
}

void Worker::run_geometric_mode() {
    uint64_t current = config_.start_value + worker_id_;
    
    while (!stats_.should_stop && current < config_.end_value) {
        PrivateKey privkey_bytes;
        for (int i = 0; i < 32; ++i) {
            privkey_bytes[i] = (current >> (i * 8)) & 0xFF;
        }
        
        if (secp256k1_.verify_privkey(privkey_bytes)) {
            auto pubkey = secp256k1_.pubkey_compressed(privkey_bytes);
            std::vector<uint8_t> pubkey_vec(pubkey.begin(), pubkey.end());
            auto hash = hash_engine_.compute(pubkey_vec);
            check_and_save(privkey_bytes, hash);
        }
        
        current *= config_.multiplier;
        stats_.total_keys++;
    }
}

void Worker::check_and_save(const PrivateKey& privkey, const Hash160& hash160) {
    std::string hex_str;
    for (auto byte : hash160) {
        char buf[3];
        snprintf(buf, sizeof(buf), "%02x", byte);
        hex_str += buf;
    }
    
    if (database_.contains(hash160)) {
        stats_.found_count++;
        
        std::ofstream out("found_gold.txt", std::ios::app);
        std::string privkey_hex;
        for (auto byte : privkey) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", byte);
            privkey_hex += buf;
        }
        
        out << "[FOUND] Hash160: " << hex_str << " PrivKey: " << privkey_hex << "\n";
        out.close();
        
        if (config_.stop_on_find) {
            stats_.should_stop = true;
        }
    }
}

}  // namespace btc_gold
