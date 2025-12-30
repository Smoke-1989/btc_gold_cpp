#include "worker.h"
#include "logger.h"
#include <random>
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <unordered_set>
#include <iostream>

namespace btc_gold {

static std::mutex file_mutex;
static std::mutex found_set_mutex;
static std::unordered_set<std::string> found_hashes;

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
        case Config::Mode::TERMINATOR:
            run_terminator_mode();
            break;
    }
}

inline void int_to_privkey(uint64_t val, PrivateKey& privkey) {
    std::fill(privkey.begin(), privkey.end(), 0);
    for (int i = 0; i < 8; ++i) {
        privkey[31 - i] = (val >> (i * 8)) & 0xFF;
    }
}

// Overload for 128-bit support
inline void int128_to_privkey(unsigned __int128 val, PrivateKey& privkey) {
    std::fill(privkey.begin(), privkey.end(), 0);
    for (int i = 0; i < 16 && i < 32; ++i) {
        privkey[31 - i] = (uint8_t)((val >> (i * 8)) & 0xFF);
    }
}

void Worker::run_linear_mode() {
    uint64_t current = config_.start_value + worker_id_;
    uint64_t stride_val = config_.num_threads * config_.stride;
    
    PrivateKey privkey_bytes;
    int_to_privkey(current, privkey_bytes);
    
    uint8_t tweak[32] = {0};
    for(int i=0; i<8; i++) tweak[31-i] = (stride_val >> (i*8)) & 0xFF;

    std::vector<uint8_t> pubkey_c;
    std::vector<uint8_t> pubkey_u;

    if (config_.scan_mode == Config::ScanMode::COMPRESSED || config_.scan_mode == Config::ScanMode::BOTH) {
        auto pk = secp256k1_.pubkey_compressed(privkey_bytes);
        pubkey_c.assign(pk.begin(), pk.end());
    }
    if (config_.scan_mode == Config::ScanMode::UNCOMPRESSED || config_.scan_mode == Config::ScanMode::BOTH) {
        pubkey_u = secp256k1_.pubkey_uncompressed(privkey_bytes);
    }
    
    while (!stats_.should_stop) {
        if (config_.end_value > 0 && current > config_.end_value) break;

        if (!pubkey_c.empty()) {
            auto hash = hash_engine_->compute(pubkey_c);
            if (database_.contains(hash)) {
                int_to_privkey(current, privkey_bytes);
                check_and_save(privkey_bytes, hash, true);
            }
            secp256k1_.pubkey_tweak_add(pubkey_c, tweak);
        }

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
    uint64_t current_base = config_.start_value;
    uint64_t multiplier = config_.multiplier;
    const uint64_t RANGE_PER_STEP = 1000000; 
    
    PrivateKey privkey_bytes;
    uint64_t stride_val = config_.num_threads;
    
    uint8_t tweak[32] = {0};
    for(int i=0; i<8; i++) tweak[31-i] = (stride_val >> (i*8)) & 0xFF;

    while (!stats_.should_stop && current_base <= config_.end_value && current_base > 0) {
        uint64_t current = current_base + worker_id_;
        uint64_t step_end = current_base + RANGE_PER_STEP;
        int_to_privkey(current, privkey_bytes);
        
        std::vector<uint8_t> pubkey_c, pubkey_u;
        if (config_.scan_mode != Config::ScanMode::UNCOMPRESSED) {
            auto pk = secp256k1_.pubkey_compressed(privkey_bytes);
            pubkey_c.assign(pk.begin(), pk.end());
        }
        if (config_.scan_mode != Config::ScanMode::COMPRESSED) {
            pubkey_u = secp256k1_.pubkey_uncompressed(privkey_bytes);
        }

        while (current < step_end && !stats_.should_stop) {
             if (config_.end_value > 0 && current > config_.end_value) break;

            if (!pubkey_c.empty()) {
                auto hash = hash_engine_->compute(pubkey_c);
                if (database_.contains(hash)) {
                    int_to_privkey(current, privkey_bytes);
                    check_and_save(privkey_bytes, hash, true);
                }
                secp256k1_.pubkey_tweak_add(pubkey_c, tweak);
            }
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
        uint64_t next_base = current_base * multiplier;
        if (next_base <= current_base) break; 
        current_base = next_base;
    }
}

void Worker::run_terminator_mode() {
    unsigned __int128 multiplier = config_.multiplier;
    multiplier -= worker_id_;
    
    unsigned __int128 min_val = (unsigned __int128)1 << (config_.range_min_bit - 1);
    unsigned __int128 max_val = ((unsigned __int128)1 << config_.range_max_bit) - 1; 
    
    if (config_.range_max_bit >= 128) max_val = ~((unsigned __int128)0); 

    PrivateKey privkey_bytes;
    std::vector<uint8_t> pubkey_vec;

    if (worker_id_ == 0) {
        std::stringstream ss;
        ss << "[TERMINATOR] Range: 2^" << (config_.range_min_bit - 1) 
           << " to 2^" << config_.range_max_bit << " - 1";
        ss << " | Starting Multiplier: " << (uint64_t)config_.multiplier;
        Logger::instance().info(ss.str());
    }

    uint64_t decrements = 0;

    while (!stats_.should_stop && multiplier > 1) {
        
        unsigned __int128 current = multiplier;
        
        // Fast forward to first value >= min_val
        bool overflow = false;
        while (current < min_val) {
            unsigned __int128 next = current * multiplier;
            if (next < current) { 
                overflow = true;
                break; 
            }
            current = next;
        }
        
        if (!overflow) {
            while (current <= max_val && !stats_.should_stop) {
                int128_to_privkey(current, privkey_bytes);
                
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
                
                unsigned __int128 next = current * multiplier;
                if (next < current) break; 
                current = next;
            }
        }

        // Decrement multiplier logic
        if (multiplier <= (unsigned __int128)config_.num_threads) break;
        multiplier -= config_.num_threads;
        
        // Log progress every 10000 decrements (worker 0 only)
        if (worker_id_ == 0) {
            decrements++;
            if (decrements % 10000 == 0) {
                 std::stringstream ss;
                 ss << "[TERMINATOR] Current Multiplier: " << (uint64_t)multiplier;
                 Logger::instance().info(ss.str());
            }
        }
    }
}

void Worker::check_and_save(const PrivateKey& privkey, const Hash160& hash160, bool compressed) {
    std::stringstream ss_hash;
    ss_hash << std::hex << std::setfill('0');
    for (auto byte : hash160) ss_hash << std::setw(2) << (int)byte;
    std::string hash_str = ss_hash.str();

    {
        std::lock_guard<std::mutex> lock(found_set_mutex);
        if (found_hashes.count(hash_str)) return;
        found_hashes.insert(hash_str);
    }

    stats_.found_count++;
    
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
    
    std::stringstream ss_priv, ss_pub;
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
        out << "Hash160:            " << hash_str << "\n";
        out << "WIF (Compressed):   " << wif_c << "\n";
        out << "WIF (Uncompressed): " << wif_u << "\n";
        out << "================================================================================\n";
    }
    
    if (config_.stop_on_find) {
        stats_.should_stop = true;
    }
}

}  // namespace btc_gold
