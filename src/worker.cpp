#include "worker.h"
#include "logger.h"
#include <random>
#include <fstream>
#include <iomanip>

namespace btc_gold {

Worker::Worker(
    int worker_id,
    const Config& config,
    const Database& database,
    Stats& stats
) : worker_id_(worker_id), config_(config), database_(database), stats_(stats) {
    // Initialize hash engine once per worker
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

// Helper to convert uint64_t to 32-byte private key buffer (big endian for libsecp256k1)
inline void int_to_privkey(uint64_t val, PrivateKey& privkey) {
    // Initialize all to zero
    std::fill(privkey.begin(), privkey.end(), 0);
    
    // Fill the last 8 bytes (since we are using uint64_t range)
    // Bitcoin uses Big Endian for private keys
    for (int i = 0; i < 8; ++i) {
        privkey[31 - i] = (val >> (i * 8)) & 0xFF;
    }
}

void Worker::run_linear_mode() {
    uint64_t current = config_.start_value + worker_id_;
    uint64_t stride = config_.num_threads * config_.stride;
    
    PrivateKey privkey_bytes;
    
    // Pre-allocate vector for pubkey to avoid reallocations
    std::vector<uint8_t> pubkey_vec;
    pubkey_vec.reserve(65);
    
    while (!stats_.should_stop) {
        if (config_.end_value > 0 && current > config_.end_value) break;

        int_to_privkey(current, privkey_bytes);
        
        // Fast check: verify only if necessary (usually always valid for low ranges)
        // Optimization: secp256k1_ec_seckey_verify is relatively cheap
        
        // --- Compressed Scan ---
        if (config_.scan_mode == Config::ScanMode::COMPRESSED || 
            config_.scan_mode == Config::ScanMode::BOTH) {
            
            auto pubkey = secp256k1_.pubkey_compressed(privkey_bytes);
            
            // Avoid vector construction overhead
            pubkey_vec.assign(pubkey.begin(), pubkey.end());
            
            auto hash = hash_engine_->compute(pubkey_vec);
            if (database_.contains(hash)) {
                check_and_save(privkey_bytes, hash);
            }
        }
        
        // --- Uncompressed Scan ---
        if (config_.scan_mode == Config::ScanMode::UNCOMPRESSED || 
            config_.scan_mode == Config::ScanMode::BOTH) {
            
            auto pubkey_u = secp256k1_.pubkey_uncompressed(privkey_bytes);
            
            // Reuse vector buffer
            pubkey_vec = std::move(pubkey_u);
            
            auto hash_u = hash_engine_->compute(pubkey_vec);
            if (database_.contains(hash_u)) {
                check_and_save(privkey_bytes, hash_u);
            }
        }
        
        current += stride;
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
        
        // --- Compressed Scan ---
        if (config_.scan_mode == Config::ScanMode::COMPRESSED || 
            config_.scan_mode == Config::ScanMode::BOTH) {
            
            auto pubkey = secp256k1_.pubkey_compressed(privkey_bytes);
            pubkey_vec.assign(pubkey.begin(), pubkey.end());
            auto hash = hash_engine_->compute(pubkey_vec);
            if (database_.contains(hash)) {
                check_and_save(privkey_bytes, hash);
            }
        }
        
        // --- Uncompressed Scan ---
        if (config_.scan_mode == Config::ScanMode::UNCOMPRESSED || 
            config_.scan_mode == Config::ScanMode::BOTH) {
            
            auto pubkey_u = secp256k1_.pubkey_uncompressed(privkey_bytes);
            pubkey_vec = std::move(pubkey_u);
            auto hash_u = hash_engine_->compute(pubkey_vec);
            if (database_.contains(hash_u)) {
                check_and_save(privkey_bytes, hash_u);
            }
        }
        
        stats_.total_keys++;
    }
}

void Worker::run_geometric_mode() {
    uint64_t current = config_.start_value;
    
    // Offset for worker threads if necessary, or just identical search
    // For geometric, usually we want distinct ranges, but simple multiplier implies shared space
    // Let's keep it simple: Workers search independently?
    // Better: Worker ID alters the starting seed slightly if desired.
    
    PrivateKey privkey_bytes;
    std::vector<uint8_t> pubkey_vec;
    pubkey_vec.reserve(65);
    
    while (!stats_.should_stop && current < config_.end_value) {
        int_to_privkey(current, privkey_bytes);
        
        // --- Compressed Scan ---
        if (config_.scan_mode == Config::ScanMode::COMPRESSED || 
            config_.scan_mode == Config::ScanMode::BOTH) {
            auto pubkey = secp256k1_.pubkey_compressed(privkey_bytes);
            pubkey_vec.assign(pubkey.begin(), pubkey.end());
            auto hash = hash_engine_->compute(pubkey_vec);
            if (database_.contains(hash)) {
                check_and_save(privkey_bytes, hash);
            }
        }
        
        current *= config_.multiplier;
        if (current == 0) break; // Overflow protection
        stats_.total_keys++;
    }
}

void Worker::check_and_save(const PrivateKey& privkey, const Hash160& hash160) {
    // Only verify FULL database if hash check passes (Optimized)
    // Note: database_.contains(hash) is already done before calling this function
    // So this function is only called when we FOUND a match.
    
    stats_.found_count++;
    
    // Convert to hex for logging
    std::stringstream ss_hash, ss_priv;
    ss_hash << std::hex << std::setfill('0');
    for (auto byte : hash160) ss_hash << std::setw(2) << (int)byte;
    
    ss_priv << std::hex << std::setfill('0');
    for (auto byte : privkey) ss_priv << std::setw(2) << (int)byte;
    
    std::string msg = "[FOUND] Hash160: " + ss_hash.str() + " PrivKey: " + ss_priv.str();
    Logger::instance().info(msg);
    
    // Append to file
    std::ofstream out("found_gold.txt", std::ios::app);
    out << msg << "\n";
    out.close();
    
    if (config_.stop_on_find) {
        stats_.should_stop = true;
    }
}

}  // namespace btc_gold
