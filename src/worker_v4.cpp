#include "worker.h"
#include "bigint256.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstring>

namespace btc_gold {

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

WorkerEngine::WorkerEngine(const Config& config, Logger& logger, Database& database)
    : config_(config),
      logger_(logger),
      database_(database),
      secp256k1_() {}

WorkerEngine::~WorkerEngine() {
    should_stop_ = true;
    
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    flush_hits();
}

// ============================================================================
// MAIN RUN METHOD - Dispatcher
// ============================================================================

void WorkerEngine::run() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    logger_.info("[INFO] Starting BTC GOLD v4.2 EXTERMINATOR - ENTERPRISE EDITION");
    logger_.info("[INFO] Mode: " + std::to_string(static_cast<int>(config_.mode)));
    logger_.info("[INFO] Threads: " + std::to_string(
        config_.num_threads > 0 ? config_.num_threads : std::thread::hardware_concurrency()));
    logger_.info("[INFO] Database size: " + std::to_string(database_.size()));
    
    if (config_.use_256bit_range) {
        logger_.info("[INFO] ✅ 256-BIT RANGE MODE ACTIVE - Full support enabled");
    }
    
    try {
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
            case Config::Mode::DOUBLING:
                run_doubling_mode();
                break;
            case Config::Mode::HAMMING:
                run_hamming_mode();
                break;
            case Config::Mode::MODULAR_STRIDE:
                run_modular_stride_mode();
                break;
            default:
                throw std::runtime_error("Unknown mode: " + std::to_string(static_cast<int>(config_.mode)));
        }
    } catch (const std::exception& e) {
        logger_.error("[ERROR] Fatal error during scanning: " + std::string(e.what()));
        throw;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    logger_.info("[DONE] Scanning completed in " + std::to_string(duration.count()) + "s");
    logger_.info("[RESULTS] Total keys checked: " + std::to_string(keys_checked_.load()));
    logger_.info("[RESULTS] Matches found: " + std::to_string(found_count_.load()));
}

// ============================================================================
// MODE 1: LINEAR - ENTERPRISE GRADE (ZERO DUPLICATES)
// ============================================================================

void WorkerEngine::run_linear_mode() {
    logger_.info("[LINEAR] Initializing TURBO mode (Point Addition)");
    
    if (config_.use_256bit_range) {
        logger_.info("[LINEAR] 🔥 256-BIT MODE ACTIVE");
        BigInt256 start_big(config_.start_key_256);
        BigInt256 end_big(config_.end_key_256);
        logger_.info("[LINEAR] Range: 0x" + start_big.to_hex() + " to 0x" + end_big.to_hex());
    } else {
        logger_.info("[LINEAR] Range: " + std::to_string(config_.start_value) + 
                    " to " + std::to_string(config_.end_value));
    }
    
    int num_threads = config_.num_threads > 0 ? config_.num_threads : std::thread::hardware_concurrency();
    
    for (int i = 0; i < num_threads; i++) {
        workers_.emplace_back(&WorkerEngine::linear_worker_turbo, this, i);
    }
    
    auto progress_thread = std::thread(&WorkerEngine::report_progress, this);
    
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    
    should_stop_ = true;
    if (progress_thread.joinable()) progress_thread.join();
}

void WorkerEngine::linear_worker_turbo(int thread_id) {
    try {
        if (config_.use_256bit_range) {
            linear_worker_256bit(thread_id);
        } else {
            linear_worker_64bit(thread_id);
        }
    } catch (const std::exception& e) {
        logger_.error("[ERROR] Linear worker " + std::to_string(thread_id) + ": " + std::string(e.what()));
    }
}

// ============================================================================
// LINEAR WORKER - 256-BIT (ENTERPRISE: STRICT PARTITIONING)
// ============================================================================

void WorkerEngine::linear_worker_256bit(int thread_id) {
    BigInt256 start_big(config_.start_key_256);
    BigInt256 end_big(config_.end_key_256);
    BigInt256 total_range = end_big - start_big;
    
    int num_workers = workers_.size();
    BigInt256 chunk_size = total_range / num_workers;
    
    // CRITICAL: Semi-open intervals [start, end) - NO OVERLAP
    BigInt256 thread_start = start_big + (chunk_size * thread_id);
    BigInt256 thread_end;
    
    if (thread_id == num_workers - 1) {
        thread_end = end_big; // Last thread covers remainder
    } else {
        thread_end = thread_start + chunk_size; // [start, start+chunk)
    }
    
    // Handle edge case: If chunk_size is zero (range < threads), idle this thread
    if (chunk_size.is_zero() && thread_id > 0) {
        logger_.info("[T" + std::to_string(thread_id) + "] Idle (range too small for this thread)");
        return;
    }
    
    BigInt256 current = thread_start;
    PrivateKey privkey;
    current.to_privkey(privkey);
    
    PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
    Hash160 hash160 = secp256k1_.hash160(pubkey);
    
    logger_.info("[T" + std::to_string(thread_id) + "] Range: 0x" + thread_start.to_hex() + 
                " to 0x" + thread_end.to_hex());
    
    while (current < thread_end && !should_stop_) {
        if (check_match(privkey, pubkey, hash160)) {
            HitBuffer::Hit hit;
            hit.privkey = privkey;
            hit.hash160 = hash160;
            hit.address = secp256k1_.hash160_to_address(hash160);
            hit.wif_compressed = secp256k1_.privkey_to_wif(privkey, true);
            hit.extra_info = current.to_hex();
            
            hit_buffer_.add(hit);
            found_count_++;
            logger_.warning("[💰 FOUND] Match at 0x" + current.to_hex());
            flush_hits();
            
            if (config_.stop_on_find) {
                should_stop_ = true;
                break;
            }
        }
        
        secp256k1_.pubkey_tweak_add(pubkey, 1);
        hash160 = secp256k1_.hash160(pubkey);
        ++current;
        
        // Periodic privkey resync every 1M keys
        if ((keys_checked_.fetch_add(1) & 0xFFFFF) == 0) {
            current.to_privkey(privkey);
        }
    }
    
    flush_hits();
    logger_.info("[T" + std::to_string(thread_id) + "] Completed");
}

// ============================================================================
// LINEAR WORKER - 64-BIT (ENTERPRISE: STRICT PARTITIONING)
// ============================================================================

void WorkerEngine::linear_worker_64bit(int thread_id) {
    uint64_t total_range = config_.end_value - config_.start_value;
    int num_workers = workers_.size();
    uint64_t chunk_size = total_range / num_workers;
    
    uint64_t thread_start = config_.start_value + (chunk_size * thread_id);
    uint64_t thread_end;
    
    if (thread_id == num_workers - 1) {
        thread_end = config_.end_value;
    } else {
        thread_end = thread_start + chunk_size;
    }
    
    // Edge case: idle if range too small
    if (chunk_size == 0 && thread_id > 0) {
        logger_.info("[T" + std::to_string(thread_id) + "] Idle (range too small)");
        return;
    }
    
    PrivateKey privkey;
    secp256k1_.int_to_privkey(thread_start, privkey);
    
    PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
    Hash160 hash160 = secp256k1_.hash160(pubkey);
    
    logger_.info("[T" + std::to_string(thread_id) + "] Range: " + std::to_string(thread_start) + 
                " to " + std::to_string(thread_end));
    
    for (uint64_t current = thread_start; current < thread_end && !should_stop_; current++) {
        if (check_match(privkey, pubkey, hash160)) {
            HitBuffer::Hit hit;
            secp256k1_.int_to_privkey(current, hit.privkey);
            hit.hash160 = hash160;
            hit.address = secp256k1_.hash160_to_address(hash160);
            hit.wif_compressed = secp256k1_.privkey_to_wif(hit.privkey, true);
            hit.extra_info = std::to_string(current);
            
            hit_buffer_.add(hit);
            found_count_++;
            logger_.warning("[💰 FOUND] Match at " + std::to_string(current));
            flush_hits();
            
            if (config_.stop_on_find) {
                should_stop_ = true;
            }
        }
        
        secp256k1_.pubkey_tweak_add(pubkey, 1);
        hash160 = secp256k1_.hash160(pubkey);
        keys_checked_++;
    }
    
    flush_hits();
}

// ============================================================================
// MODE 2: RANDOM
// ============================================================================

void WorkerEngine::run_random_mode() {
    logger_.info("[RANDOM] Full 256-bit random search");
    
    int num_threads = config_.num_threads > 0 ? config_.num_threads : std::thread::hardware_concurrency();
    
    for (int i = 0; i < num_threads; i++) {
        workers_.emplace_back(&WorkerEngine::random_worker, this, i);
    }
    
    auto progress_thread = std::thread(&WorkerEngine::report_progress, this);
    
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    
    should_stop_ = true;
    if (progress_thread.joinable()) progress_thread.join();
}

void WorkerEngine::random_worker(int thread_id) {
    try {
        std::random_device rd;
        std::mt19937_64 rng(rd() + thread_id);
        std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
        
        logger_.info("[T" + std::to_string(thread_id) + "] Random worker started");
        
        while (!should_stop_) {
            PrivateKey privkey;
            for (int k = 0; k < 32; k += 8) {
                uint64_t part = dist(rng);
                std::memcpy(&privkey[k], &part, 8);
            }
            
            PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
            Hash160 hash160 = secp256k1_.hash160(pubkey);
            
            if (check_match(privkey, pubkey, hash160)) {
                HitBuffer::Hit hit;
                hit.privkey = privkey;
                hit.hash160 = hash160;
                hit.address = secp256k1_.hash160_to_address(hash160);
                hit.wif_compressed = secp256k1_.privkey_to_wif(privkey, true);
                hit.extra_info = "random";
                
                hit_buffer_.add(hit);
                found_count_++;
                logger_.warning("[💰 FOUND] Random match");
                flush_hits();
                
                if (config_.stop_on_find) {
                    should_stop_ = true;
                }
            }
            
            keys_checked_++;
        }
    } catch (const std::exception& e) {
        logger_.error("[ERROR] Random worker: " + std::string(e.what()));
    }
}

// ============================================================================
// MODE 3: GEOMETRIC (3-Phase)
// ============================================================================

void WorkerEngine::run_geometric_mode() {
    logger_.info("[GEOMETRIC] 3-Phase: Border, Ceiling, Hamming");
    logger_.debug("[GEOMETRIC] Implementation pending - future release");
}

void WorkerEngine::geometric_worker(int thread_id) {
    logger_.info("[T" + std::to_string(thread_id) + "] Geometric worker");
}

// ============================================================================
// MODE 4: TERMINATOR - ENTERPRISE GRADE (GEOMETRIC PROGRESSION)
// ============================================================================

void WorkerEngine::run_terminator_mode() {
    logger_.info("[TERMINATOR] 🔥 Multiplicative Geometric Progression Mode");
    
    if (config_.use_256bit_range) {
        logger_.info("[TERMINATOR] Using 256-bit start/end range");
    } else {
        logger_.info("[TERMINATOR] Using 64-bit start range");
    }
    
    logger_.info("[TERMINATOR] Multiplier: " + std::to_string(config_.multiplier));
    
    int num_threads = config_.num_threads > 0 ? config_.num_threads : std::thread::hardware_concurrency();
    
    for (int i = 0; i < num_threads; i++) {
        workers_.emplace_back(&WorkerEngine::terminator_worker, this, i);
    }
    
    auto progress_thread = std::thread(&WorkerEngine::report_progress, this);
    
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    
    should_stop_ = true;
    if (progress_thread.joinable()) progress_thread.join();
}

void WorkerEngine::terminator_worker(int thread_id) {
    try {
        logger_.info("[T" + std::to_string(thread_id) + "] Terminator worker started");
        
        if (config_.use_256bit_range) {
            // 256-bit geometric progression
            BigInt256 current(config_.start_key_256);
            BigInt256 end_limit(config_.end_key_256);
            BigInt256 multiplier(config_.multiplier);
            
            // Thread offset: Each thread starts at start * multiplier^thread_id
            for (int offset = 0; offset < thread_id; offset++) {
                current = current * config_.multiplier;
                if (current > end_limit) {
                    logger_.info("[T" + std::to_string(thread_id) + "] Start position exceeds end limit - idle");
                    return;
                }
            }
            
            logger_.info("[T" + std::to_string(thread_id) + "] Starting at: 0x" + current.to_hex());
            
            int num_workers = workers_.size();
            BigInt256 step_multiplier(1);
            for (int i = 0; i < num_workers; i++) {
                step_multiplier = step_multiplier * config_.multiplier;
            }
            
            while (current <= end_limit && !should_stop_) {
                PrivateKey privkey;
                current.to_privkey(privkey);
                
                PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
                Hash160 hash160 = secp256k1_.hash160(pubkey);
                
                if (check_match(privkey, pubkey, hash160)) {
                    HitBuffer::Hit hit;
                    hit.privkey = privkey;
                    hit.hash160 = hash160;
                    hit.address = secp256k1_.hash160_to_address(hash160);
                    hit.wif_compressed = secp256k1_.privkey_to_wif(privkey, true);
                    hit.extra_info = current.to_hex();
                    
                    hit_buffer_.add(hit);
                    found_count_++;
                    logger_.warning("[💰 FOUND] Match at 0x" + current.to_hex());
                    flush_hits();
                    
                    if (config_.stop_on_find) {
                        should_stop_ = true;
                        break;
                    }
                }
                
                current = current * step_multiplier;
                keys_checked_++;
            }
            
        } else {
            // 64-bit geometric progression (legacy)
            uint64_t current = config_.start_value;
            uint64_t multiplier = config_.multiplier;
            
            // Thread offset
            for (int offset = 0; offset < thread_id; offset++) {
                if (current > UINT64_MAX / multiplier) {
                    logger_.info("[T" + std::to_string(thread_id) + "] Overflow - idle");
                    return;
                }
                current *= multiplier;
            }
            
            logger_.info("[T" + std::to_string(thread_id) + "] Starting at: " + std::to_string(current));
            
            int num_workers = workers_.size();
            uint64_t step_multiplier = 1;
            for (int i = 0; i < num_workers; i++) {
                step_multiplier *= multiplier;
            }
            
            while (current <= config_.end_value && !should_stop_) {
                PrivateKey privkey;
                secp256k1_.int_to_privkey(current, privkey);
                
                PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
                Hash160 hash160 = secp256k1_.hash160(pubkey);
                
                if (check_match(privkey, pubkey, hash160)) {
                    HitBuffer::Hit hit;
                    hit.privkey = privkey;
                    hit.hash160 = hash160;
                    hit.address = secp256k1_.hash160_to_address(hash160);
                    hit.wif_compressed = secp256k1_.privkey_to_wif(privkey, true);
                    hit.extra_info = std::to_string(current);
                    
                    hit_buffer_.add(hit);
                    found_count_++;
                    logger_.warning("[💰 FOUND] Match at " + std::to_string(current));
                    flush_hits();
                    
                    if (config_.stop_on_find) {
                        should_stop_ = true;
                        break;
                    }
                }
                
                if (current > UINT64_MAX / step_multiplier) {
                    logger_.info("[T" + std::to_string(thread_id) + "] Overflow - stopping");
                    break;
                }
                
                current *= step_multiplier;
                keys_checked_++;
            }
        }
        
        flush_hits();
        logger_.info("[T" + std::to_string(thread_id) + "] Completed");
        
    } catch (const std::exception& e) {
        logger_.error("[ERROR] Terminator worker: " + std::string(e.what()));
    }
}

// ============================================================================
// MODE 5: DOUBLING (Powers of 2)
// ============================================================================

void WorkerEngine::run_doubling_mode() {
    logger_.info("[DOUBLING] Powers of 2 mode");
    logger_.info("[DOUBLING] Range: 2^" + std::to_string(config_.range_min_bit) +
                " to 2^" + std::to_string(config_.range_max_bit));
    doubling_worker();
}

void WorkerEngine::doubling_worker() {
    try {
        logger_.info("[DOUBLING] Starting doubling worker");
        
        for (int bit = config_.range_min_bit - 1; bit <= config_.range_max_bit && !should_stop_; bit++) {
            PrivateKey privkey = {};
            
            int byte_idx = bit / 8;
            int bit_idx = bit % 8;
            if (byte_idx < 32) {
                privkey[byte_idx] = 1 << bit_idx;
            }
            
            PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
            Hash160 hash160 = secp256k1_.hash160(pubkey);
            
            if (check_match(privkey, pubkey, hash160)) {
                HitBuffer::Hit hit;
                hit.privkey = privkey;
                hit.hash160 = hash160;
                hit.address = secp256k1_.hash160_to_address(hash160);
                hit.wif_compressed = secp256k1_.privkey_to_wif(privkey, true);
                hit.extra_info = "2^" + std::to_string(bit);
                
                hit_buffer_.add(hit);
                found_count_++;
                logger_.warning("[💰 FOUND] 2^" + std::to_string(bit));
                flush_hits();
                
                if (config_.stop_on_find) {
                    should_stop_ = true;
                }
            }
            
            keys_checked_++;
        }
        
        flush_hits();
        logger_.info("[DOUBLING] Complete");
        
    } catch (const std::exception& e) {
        logger_.error("[ERROR] Doubling worker: " + std::string(e.what()));
    }
}

// ============================================================================
// MODE 6: HAMMING (Low-Weight Keys)
// ============================================================================

void WorkerEngine::run_hamming_mode() {
    logger_.info("[HAMMING] Low-weight key search (2-bit combinations)");
    logger_.info("[HAMMING] Range: bits " + std::to_string(config_.range_min_bit) +
                " to " + std::to_string(config_.range_max_bit));
    hamming_worker();
}

void WorkerEngine::hamming_worker() {
    try {
        logger_.info("[HAMMING] Starting hamming worker");
        
        int min_bit = config_.range_min_bit;
        int max_bit = config_.range_max_bit;
        
        for (int bit1 = min_bit; bit1 <= max_bit && !should_stop_; bit1++) {
            for (int bit2 = bit1 + 1; bit2 <= max_bit && !should_stop_; bit2++) {
                PrivateKey privkey = {};
                
                if (bit1 < 256) {
                    int byte_idx = bit1 / 8;
                    int bit_idx = bit1 % 8;
                    privkey[byte_idx] |= (1 << bit_idx);
                }
                
                if (bit2 < 256) {
                    int byte_idx = bit2 / 8;
                    int bit_idx = bit2 % 8;
                    privkey[byte_idx] |= (1 << bit_idx);
                }
                
                PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
                Hash160 hash160 = secp256k1_.hash160(pubkey);
                
                if (check_match(privkey, pubkey, hash160)) {
                    HitBuffer::Hit hit;
                    hit.privkey = privkey;
                    hit.hash160 = hash160;
                    hit.address = secp256k1_.hash160_to_address(hash160);
                    hit.wif_compressed = secp256k1_.privkey_to_wif(privkey, true);
                    hit.extra_info = "2^" + std::to_string(bit1) + " + 2^" + std::to_string(bit2);
                    
                    hit_buffer_.add(hit);
                    found_count_++;
                    logger_.warning("[💰 FOUND] 2^" + std::to_string(bit1) + " + 2^" + std::to_string(bit2));
                    flush_hits();
                    
                    if (config_.stop_on_find) {
                        should_stop_ = true;
                    }
                }
                
                keys_checked_++;
            }
        }
        
        flush_hits();
        logger_.info("[HAMMING] Complete");
        
    } catch (const std::exception& e) {
        logger_.error("[ERROR] Hamming worker: " + std::string(e.what()));
    }
}

// ============================================================================
// MODE 7: MODULAR STRIDE (Arithmetic Progression)
// ============================================================================

void WorkerEngine::run_modular_stride_mode() {
    logger_.info("[MODULAR_STRIDE] Arithmetic progression mode");
    logger_.info("[MODULAR_STRIDE] Start: " + std::to_string(config_.start_value) +
                ", Multiplier: " + std::to_string(config_.multiplier));
    
    int num_threads = config_.num_threads > 0 ? config_.num_threads : std::thread::hardware_concurrency();
    
    for (int i = 0; i < num_threads; i++) {
        workers_.emplace_back(&WorkerEngine::modular_stride_worker, this, i);
    }
    
    auto progress_thread = std::thread(&WorkerEngine::report_progress, this);
    
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    
    should_stop_ = true;
    if (progress_thread.joinable()) progress_thread.join();
}

void WorkerEngine::modular_stride_worker(int thread_id) {
    try {
        logger_.info("[T" + std::to_string(thread_id) + "] Modular stride worker started");
        
        uint64_t offset = thread_id;
        uint64_t current = config_.start_value + offset;
        
        while (current <= config_.end_value && !should_stop_) {
            PrivateKey privkey;
            secp256k1_.int_to_privkey(current, privkey);
            
            PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
            Hash160 hash160 = secp256k1_.hash160(pubkey);
            
            if (check_match(privkey, pubkey, hash160)) {
                HitBuffer::Hit hit;
                hit.privkey = privkey;
                hit.hash160 = hash160;
                hit.address = secp256k1_.hash160_to_address(hash160);
                hit.wif_compressed = secp256k1_.privkey_to_wif(privkey, true);
                hit.extra_info = std::to_string(current);
                
                hit_buffer_.add(hit);
                found_count_++;
                logger_.warning("[💰 FOUND] Match at " + std::to_string(current));
                flush_hits();
                
                if (config_.stop_on_find) {
                    should_stop_ = true;
                }
            }
            
            keys_checked_++;
            current += config_.multiplier;
        }
        
        flush_hits();
        
    } catch (const std::exception& e) {
        logger_.error("[ERROR] Modular stride worker: " + std::string(e.what()));
    }
}

// ============================================================================
// HELPER METHODS
// ============================================================================

bool WorkerEngine::check_match(const PrivateKey& privkey, const PublicKey& pubkey,
                               const Hash160& hash160) {
    (void)privkey; (void)pubkey; // Suppress unused warnings
    return database_.contains(hash160);
}

void WorkerEngine::flush_hits() {
    auto hits = hit_buffer_.flush();
    
    if (!hits.empty()) {
        std::lock_guard<std::mutex> guard(hit_buffer_.lock);
        
        std::ofstream outfile(config_.output_file, std::ios::app);
        if (!outfile.is_open()) {
            logger_.error("Cannot open output file: " + config_.output_file);
            return;
        }
        
        for (const auto& hit : hits) {
            std::stringstream privkey_hex_stream;
            privkey_hex_stream << std::hex << std::setfill('0');
            for (int i = 0; i < 32; i++) {
                privkey_hex_stream << std::setw(2) << static_cast<int>(hit.privkey[i]);
            }
            std::string privkey_hex = privkey_hex_stream.str();
            
            PublicKey pubkey = secp256k1_.pubkey_compressed(hit.privkey);
            std::stringstream pubkey_hex_stream;
            pubkey_hex_stream << std::hex << std::setfill('0');
            for (int i = 0; i < 33; i++) {
                pubkey_hex_stream << std::setw(2) << static_cast<int>(pubkey[i]);
            }
            std::string pubkey_hex = pubkey_hex_stream.str();
            
            std::stringstream hash160_hex_stream;
            hash160_hex_stream << std::hex << std::setfill('0');
            for (int i = 0; i < 20; i++) {
                hash160_hex_stream << std::setw(2) << static_cast<int>(hit.hash160[i]);
            }
            std::string hash160_hex = hash160_hex_stream.str();
            
            std::string wif_uncompressed = secp256k1_.privkey_to_wif(hit.privkey, false);
            
            outfile << "================================================================================\n";
            outfile << "FOUND GOLD!\n";
            outfile << "================================================================================\n";
            outfile << "Address:            " << hit.address << " (Compressed)\n";
            outfile << "Private Key (HEX):  " << privkey_hex << "\n";
            outfile << "Public Key (HEX):   " << pubkey_hex << "\n";
            outfile << "Hash160:            " << hash160_hex << "\n";
            outfile << "WIF (Compressed):   " << hit.wif_compressed << "\n";
            outfile << "WIF (Uncompressed): " << wif_uncompressed << "\n";
            outfile << "Extra Info:         " << hit.extra_info << "\n";
            outfile << "================================================================================\n";
        }
        
        outfile.close();
        logger_.debug("Flushed " + std::to_string(hits.size()) + " hits");
    }
}

void WorkerEngine::report_progress() {
    auto last_keys = 0UL;
    
    while (!should_stop_) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        auto current_keys = keys_checked_.load();
        auto rate = current_keys - last_keys;
        
        if (config_.verbose) {
            logger_.debug("[PROGRESS] Speed: " + std::to_string(rate / 1000000) + "M k/s | " +
                         "Total: " + std::to_string(current_keys / 1000000) + "M keys | " +
                         "Found: " + std::to_string(found_count_.load()));
        }
        
        last_keys = current_keys;
    }
}

}  // namespace btc_gold
