#include "worker.h"
#include "logger.h"
#include <random>
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>

namespace btc_gold {

static std::mutex file_mutex;

Worker::Worker(
    int worker_id,
    const Config& config,
    const Database& database,
    Stats& stats
) : worker_id_(worker_id), config_(config), database_(database), stats_(stats) {
    hash_engine_ = std::make_unique<Hash160Engine>();
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

inline void int_to_privkey(uint64_t val, PrivateKey& privkey) {
    std::fill(privkey.begin(), privkey.end(), 0);
    for (int i = 0; i < 8; ++i) {
        privkey[31 - i] = (val >> (i * 8)) & 0xFF;
    }
}

// Optimization: Use Point Addition (TweakAdd) instead of full multiplication
void Worker::run_linear_mode() {
    uint64_t current = config_.start_value + worker_id_;
    uint64_t stride_val = config_.num_threads * config_.stride;
    
    PrivateKey privkey_bytes;
    int_to_privkey(current, privkey_bytes);
    
    // Tweak value is the stride (converted to 32 bytes Big Endian)
    uint8_t tweak[32] = {0};
    for(int i=0; i<8; i++) tweak[31-i] = (stride_val >> (i*8)) & 0xFF;

    // Initial Public Keys
    std::vector<uint8_t> pubkey_c;  // Compressed buffer
    std::vector<uint8_t> pubkey_u;  // Uncompressed buffer

    // Initialize starting points
    if (config_.scan_mode == Config::ScanMode::COMPRESSED || config_.scan_mode == Config::ScanMode::BOTH) {
        auto pk = secp256k1_.pubkey_compressed(privkey_bytes);
        pubkey_c.assign(pk.begin(), pk.end());
    }
    if (config_.scan_mode == Config::ScanMode::UNCOMPRESSED || config_.scan_mode == Config::ScanMode::BOTH) {
        pubkey_u = secp256k1_.pubkey_uncompressed(privkey_bytes);
    }
    
    while (!stats_.should_stop) {
        if (config_.end_value > 0 && current > config_.end_value) break;

        // Check Compressed
        if (!pubkey_c.empty()) {
            auto hash = hash_engine_->compute(pubkey_c);
            if (database_.contains(hash)) {
                // Recover privkey for logging (since we only have the counter)
                int_to_privkey(current, privkey_bytes);
                check_and_save(privkey_bytes, hash, true);
            }
            // Fast Update: Add stride to point
            secp256k1_.pubkey_tweak_add(pubkey_c, tweak);
        }

        // Check Uncompressed
        if (!pubkey_u.empty()) {
            auto hash = hash_engine_->compute(pubkey_u);
            if (database_.contains(hash)) {
                int_to_privkey(current, privkey_bytes);
                check_and_save(privkey_bytes, hash, false);
            }
            secp256k1_.pubkey_tweak_add(pubkey_u, tweak);
        }

        current += stride_val;
        stats_.total_keys++;
    }
}

void Worker::run_random_mode() {
    std::mt19937_64 rng(worker_id_ + std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist(config_.start_value, config_.end_value);
    
    PrivateKey privkey_bytes;
    std::vector<uint8_t> pubkey_vec;
    pubkey_vec.reserve(65);
    
    while (!stats_.should_stop) {
        uint64_t rand_val = dist(rng);
        int_to_privkey(rand_val, privkey_bytes);
        
        if (config_.scan_mode != Config::ScanMode::UNCOMPRESSED) {
            auto pk = secp256k1_.pubkey_compressed(privkey_bytes);
            pubkey_vec.assign(pk.begin(), pk.end());
            auto hash = hash_engine_->compute(pubkey_vec);
            if (database_.contains(hash)) check_and_save(privkey_bytes, hash, true);
        }
        
        if (config_.scan_mode != Config::ScanMode::COMPRESSED) {
            auto pk = secp256k1_.pubkey_uncompressed(privkey_bytes);
            pubkey_vec = std::move(pk);
            auto hash = hash_engine_->compute(pubkey_vec);
            if (database_.contains(hash)) check_and_save(privkey_bytes, hash, false);
        }
        
        stats_.total_keys++;
    }
}

void Worker::run_geometric_mode() {
    // Fix: Each worker handles a distinct interleaved sequence of powers
    // Worker 0: Start * m^0, Start * m^(0+N), Start * m^(0+2N)...
    // Worker 1: Start * m^1, Start * m^(1+N), Start * m^(1+2N)...
    // Where N is num_threads
    
    uint64_t current = config_.start_value;
    uint64_t multiplier = config_.multiplier;
    
    // Fast forward to worker's first offset
    for (int i = 0; i < worker_id_; ++i) {
        current *= multiplier;
        if (current > config_.end_value) return; 
    }

    // Calculate super-multiplier for stride (multiplier ^ num_threads)
    uint64_t stride_multiplier = 1;
    for (int i = 0; i < config_.num_threads; ++i) {
        stride_multiplier *= multiplier;
    }
    
    PrivateKey privkey_bytes;
    std::vector<uint8_t> pubkey_vec;
    
    while (!stats_.should_stop && current <= config_.end_value && current > 0) {
        int_to_privkey(current, privkey_bytes);
        
        if (config_.scan_mode != Config::ScanMode::UNCOMPRESSED) {
            auto pk = secp256k1_.pubkey_compressed(privkey_bytes);
            pubkey_vec.assign(pk.begin(), pk.end());
            auto hash = hash_engine_->compute(pubkey_vec);
            if (database_.contains(hash)) check_and_save(privkey_bytes, hash, true);
        }
        
        if (config_.scan_mode != Config::ScanMode::COMPRESSED) {
            auto pk = secp256k1_.pubkey_uncompressed(privkey_bytes);
            pubkey_vec = std::move(pk);
            auto hash = hash_engine_->compute(pubkey_vec);
            if (database_.contains(hash)) check_and_save(privkey_bytes, hash, false);
        }
        
        uint64_t next = current * stride_multiplier;
        if (next < current) break; // Overflow check
        current = next;
        stats_.total_keys++;
    }
}

void Worker::check_and_save(const PrivateKey& privkey, const Hash160& hash160, bool compressed) {
    stats_.found_count++;
    
    // Re-generate pubkey bytes for address generation
    std::vector<uint8_t> pubkey_bytes;
    if (compressed) {
        auto pk = secp256k1_.pubkey_compressed(privkey);
        pubkey_bytes.assign(pk.begin(), pk.end());
    } else {
        pubkey_bytes = secp256k1_.pubkey_uncompressed(privkey);
    }

    std::string address = secp256k1_.to_address(pubkey_bytes);
    std::string wif_c = secp256k1_.to_wif(privkey, true);
    std::string wif_u = secp256k1_.to_wif(privkey, false);
    
    std::stringstream ss_hash, ss_priv, ss_pub;
    ss_hash << std::hex << std::setfill('0');
    for (auto byte : hash160) ss_hash << std::setw(2) << (int)byte;
    
    ss_priv << std::hex << std::setfill('0');
    for (auto byte : privkey) ss_priv << std::setw(2) << (int)byte;

    ss_pub << std::hex << std::setfill('0');
    for (auto byte : pubkey_bytes) ss_pub << std::setw(2) << (int)byte;
    
    std::string log_msg = "[FOUND] " + address;
    Logger::instance().info(log_msg);
    
    {
        std::lock_guard<std::mutex> lock(file_mutex);
        std::ofstream out("found_gold.txt", std::ios::app);
        out << "================================================================================\n";
        out << "FOUND GOLD!\n";
        out << "================================================================================\n";
        out << "Address:            " << address << " (" << (compressed ? "Compressed" : "Uncompressed") << ")\n";
        out << "Private Key (HEX):  " << ss_priv.str() << "\n";
        out << "Public Key (HEX):   " << ss_pub.str() << "\n";
        out << "Hash160:            " << ss_hash.str() << "\n";
        out << "WIF (Compressed):   " << wif_c << "\n";
        out << "WIF (Uncompressed): " << wif_u << "\n";
        out << "================================================================================\n";
    }
    
    if (config_.stop_on_find) {
        stats_.should_stop = true;
    }
}

}  // namespace btc_gold
