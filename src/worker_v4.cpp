#include "worker.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>

namespace btc_gold {

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

WorkerEngine::WorkerEngine(const Config& config)
    : config_(config), logger_(config.verbose), database_(config.database_file) {}

WorkerEngine::~WorkerEngine() {
    // Flush any remaining hits
    flush_hits();
    
    // Join all worker threads
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

// ============================================================================
// MAIN RUN METHOD - Dispatcher
// ============================================================================

void WorkerEngine::run() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    logger_.log("[INFO] Starting BTC GOLD v4.0 EXTERMINATOR");
    logger_.log("[INFO] Mode: " + std::to_string(config_.mode));
    logger_.log("[INFO] Threads: " + std::to_string(
        config_.num_threads > 0 ? config_.num_threads : std::thread::hardware_concurrency()));
    logger_.log("[INFO] Database: " + config_.database_file);
    
    // Dispatch to appropriate mode
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
            logger_.log("[ERROR] Unknown mode: " + std::to_string(config_.mode));
            return;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    logger_.log("[DONE] Scanning completed in " + std::to_string(duration.count()) + "s");
    logger_.log("[RESULTS] Total keys checked: " + std::to_string(keys_checked_.load()));
    logger_.log("[RESULTS] Matches found: " + std::to_string(found_count_.load()));
}

// ============================================================================
// MODE 1: LINEAR (TURBO via Point Addition)
// ============================================================================

void WorkerEngine::run_linear_mode() {
    logger_.log("[LINEAR] Initializing TURBO mode (Point Addition)");
    
    int num_threads = config_.num_threads > 0 ? config_.num_threads : std::thread::hardware_concurrency();
    
    for (int i = 0; i < num_threads; i++) {
        workers_.emplace_back(&WorkerEngine::linear_worker_turbo, this, i);
    }
    
    // Progress reporting thread
    auto progress_thread = std::thread(&WorkerEngine::report_progress, this);
    
    // Wait for workers
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    
    should_stop_ = true;
    if (progress_thread.joinable()) progress_thread.join();
}

void WorkerEngine::linear_worker_turbo(int thread_id) {
    try {
        // Calculate thread-specific start position
        uint64_t range = config_.end_value - config_.start_value;
        uint64_t thread_chunk = range / std::max(1, (int)workers_.size());
        uint64_t start = config_.start_value + (thread_id * thread_chunk);
        uint64_t end = (thread_id == (int)workers_.size() - 1) ? config_.end_value : start + thread_chunk;
        
        // Initialize privkey at start position
        PrivateKey privkey;
        secp256k1_.int_to_privkey(start, privkey);
        
        // Get initial pubkey
        PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
        Hash160 hash160 = secp256k1_.hash160(pubkey);
        
        logger_.log("[T" + std::to_string(thread_id) + "] Starting at: " + std::to_string(start));
        
        // MAIN LOOP - TURBO: Only hash160 per iteration
        for (uint64_t current = start; current < end && !should_stop_; current++) {
            // 1. Check hash
            if (check_match(privkey, pubkey, hash160)) {
                HitBuffer::Hit hit;
                format_key_result(privkey, hash160, hit);
                hit_buffer_.add(hit);
                found_count_++;
                
                if (config_.stop_on_find) {
                    should_stop_ = true;
                }
            }
            
            // 2. TURBO: Update pubkey via EC Point Addition (fast!)
            // Instead of recalculating privkey every time
            secp256k1_.pubkey_tweak_add(pubkey, 1);
            hash160 = secp256k1_.hash160(pubkey);
            
            // 3. Increment counter
            keys_checked_++;
            
            // 4. Flush if buffer full
            if (hit_buffer_.should_flush()) {
                flush_hits();
            }
        }
        
        // Final flush for this thread
        flush_hits();
        
    } catch (const std::exception& e) {
        logger_.log("[ERROR] Linear worker " + std::to_string(thread_id) + ": " + std::string(e.what()));
    }
}

// ============================================================================
// MODE 2: RANDOM
// ============================================================================

void WorkerEngine::run_random_mode() {
    logger_.log("[RANDOM] Full 256-bit random search");
    
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
        std::mt19937_64 rng(std::random_device{}() + thread_id);
        std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
        
        logger_.log("[T" + std::to_string(thread_id) + "] Random worker started");
        
        for (uint64_t i = 0; i < UINT64_MAX && !should_stop_; i++) {
            // Generate random privkey
            PrivateKey privkey;
            privkey[0] = dist(rng);
            privkey[1] = dist(rng);
            privkey[2] = dist(rng);
            privkey[3] = dist(rng);
            
            PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
            Hash160 hash160 = secp256k1_.hash160(pubkey);
            
            if (check_match(privkey, pubkey, hash160)) {
                HitBuffer::Hit hit;
                format_key_result(privkey, hash160, hit);
                hit_buffer_.add(hit);
                found_count_++;
                
                if (config_.stop_on_find) {
                    should_stop_ = true;
                }
            }
            
            keys_checked_++;
            
            if (hit_buffer_.should_flush()) {
                flush_hits();
            }
        }
    } catch (const std::exception& e) {
        logger_.log("[ERROR] Random worker: " + std::string(e.what()));
    }
}

// ============================================================================
// MODE 3: GEOMETRIC (3-Phase)
// ============================================================================

void WorkerEngine::run_geometric_mode() {
    logger_.log("[GEOMETRIC] 3-Phase: Border, Ceiling, Hamming");
    
    int num_threads = config_.num_threads > 0 ? config_.num_threads : std::thread::hardware_concurrency();
    
    for (int i = 0; i < num_threads; i++) {
        workers_.emplace_back(&WorkerEngine::geometric_worker, this, i);
    }
    
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

void WorkerEngine::geometric_worker(int thread_id) {
    // Phase 1: Border (2^(min_bit-1) to 2^min_bit)
    // Phase 2: Ceiling (2^(max_bit-1) to 2^max_bit)
    // Phase 3: Hamming (low-weight in middle)
    logger_.log("[T" + std::to_string(thread_id) + "] Geometric worker");
}

// ============================================================================
// MODE 4: TERMINATOR (Multiplicative Descent)
// ============================================================================

void WorkerEngine::run_terminator_mode() {
    logger_.log("[TERMINATOR] Multiplicative descent mode");
    
    int num_threads = config_.num_threads > 0 ? config_.num_threads : std::thread::hardware_concurrency();
    
    for (int i = 0; i < num_threads; i++) {
        workers_.emplace_back(&WorkerEngine::terminator_worker, this, i);
    }
    
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

void WorkerEngine::terminator_worker(int thread_id) {
    logger_.log("[T" + std::to_string(thread_id) + "] Terminator worker");
}

// ============================================================================
// MODE 5: DOUBLING (Powers of 2) - NEW
// ============================================================================

void WorkerEngine::run_doubling_mode() {
    logger_.log("[DOUBLING] Powers of 2 mode");
    logger_.log("[DOUBLING] Range: 2^" + std::to_string(config_.range_min_bit) + 
                " to 2^" + std::to_string(config_.range_max_bit));
    
    // Single-threaded (only ~256 combinations max)
    doubling_worker();
}

void WorkerEngine::doubling_worker() {
    try {
        logger_.log("[DOUBLING] Starting doubling worker");
        
        // Test each power of 2 in range
        for (int bit = config_.range_min_bit - 1; bit <= config_.range_max_bit && !should_stop_; bit++) {
            // Create privkey = 2^bit
            PrivateKey privkey = {};
            
            // Set bit at position 'bit'
            int byte_idx = bit / 8;
            int bit_idx = bit % 8;
            if (byte_idx < 32) {
                privkey[byte_idx] = 1 << bit_idx;
            }
            
            PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
            Hash160 hash160 = secp256k1_.hash160(pubkey);
            
            if (check_match(privkey, pubkey, hash160)) {
                HitBuffer::Hit hit;
                format_key_result(privkey, hash160, hit);
                hit_buffer_.add(hit);
                found_count_++;
                logger_.log("[FOUND] 2^" + std::to_string(bit));
                
                if (config_.stop_on_find) {
                    should_stop_ = true;
                }
            }
            
            keys_checked_++;
        }
        
        flush_hits();
        logger_.log("[DOUBLING] Complete");
        
    } catch (const std::exception& e) {
        logger_.log("[ERROR] Doubling worker: " + std::string(e.what()));
    }
}

// ============================================================================
// MODE 6: HAMMING (Low-Weight Keys) - NEW
// ============================================================================

void WorkerEngine::run_hamming_mode() {
    logger_.log("[HAMMING] Low-weight key search (2-bit combinations)");
    logger_.log("[HAMMING] Range: bits " + std::to_string(config_.range_min_bit) + 
                " to " + std::to_string(config_.range_max_bit));
    
    hamming_worker();
}

void WorkerEngine::hamming_worker() {
    try {
        logger_.log("[HAMMING] Starting hamming worker");
        
        int min_bit = config_.range_min_bit;
        int max_bit = config_.range_max_bit;
        
        // Test all 2-bit combinations
        for (int bit1 = min_bit; bit1 <= max_bit && !should_stop_; bit1++) {
            for (int bit2 = bit1 + 1; bit2 <= max_bit && !should_stop_; bit2++) {
                // Create privkey = 2^bit1 + 2^bit2
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
                    format_key_result(privkey, hash160, hit);
                    hit_buffer_.add(hit);
                    found_count_++;
                    logger_.log("[FOUND] 2^" + std::to_string(bit1) + " + 2^" + std::to_string(bit2));
                    
                    if (config_.stop_on_find) {
                        should_stop_ = true;
                    }
                }
                
                keys_checked_++;
            }
        }
        
        flush_hits();
        logger_.log("[HAMMING] Complete");
        
    } catch (const std::exception& e) {
        logger_.log("[ERROR] Hamming worker: " + std::string(e.what()));
    }
}

// ============================================================================
// MODE 7: MODULAR STRIDE (Arithmetic Progression) - NEW
// ============================================================================

void WorkerEngine::run_modular_stride_mode() {
    logger_.log("[MODULAR_STRIDE] Arithmetic progression mode");
    logger_.log("[MODULAR_STRIDE] Start: " + std::to_string(config_.start_value) + 
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
        logger_.log("[T" + std::to_string(thread_id) + "] Modular stride worker started");
        
        uint64_t offset = thread_id;  // Each thread gets different offset
        uint64_t current = config_.start_value + offset;
        
        while (current <= config_.end_value && !should_stop_) {
            PrivateKey privkey;
            secp256k1_.int_to_privkey(current, privkey);
            
            PublicKey pubkey = secp256k1_.pubkey_compressed(privkey);
            Hash160 hash160 = secp256k1_.hash160(pubkey);
            
            if (check_match(privkey, pubkey, hash160)) {
                HitBuffer::Hit hit;
                format_key_result(privkey, hash160, hit);
                hit_buffer_.add(hit);
                found_count_++;
                
                if (config_.stop_on_find) {
                    should_stop_ = true;
                }
            }
            
            keys_checked_++;
            
            if (hit_buffer_.should_flush()) {
                flush_hits();
            }
            
            // Move to next in arithmetic sequence
            current += config_.multiplier;
        }
        
        flush_hits();
        
    } catch (const std::exception& e) {
        logger_.log("[ERROR] Modular stride worker: " + std::string(e.what()));
    }
}

// ============================================================================
// HELPER METHODS
// ============================================================================

bool WorkerEngine::check_match(const PrivateKey& privkey, const PublicKey& pubkey,
                               const Hash160& hash160) {
    return database_.contains(hash160, config_.input_type);
}

void WorkerEngine::format_key_result(const PrivateKey& privkey, const Hash160& hash160,
                                     HitBuffer::Hit& hit) {
    hit.privkey = privkey;
    hit.hash160 = hash160;
    hit.address = secp256k1_.hash160_to_address(hash160);
    hit.wif_compressed = secp256k1_.privkey_to_wif(privkey, true);
}

void WorkerEngine::flush_hits() {
    auto hits = hit_buffer_.flush();
    
    if (!hits.empty()) {
        std::lock_guard<std::mutex> guard(hit_buffer_.lock);
        
        // Write to file
        std::ofstream outfile(config_.output_file, std::ios::app);
        
        for (const auto& hit : hits) {
            outfile << hit.address << "|" 
                   << hit.wif_compressed << "|" 
                   << "" << "\n";  // Additional fields
        }
        
        outfile.close();
        
        if (config_.verbose) {
            logger_.log("[FLUSH] Wrote " + std::to_string(hits.size()) + " hits");
        }
    }
}

void WorkerEngine::report_progress() {
    auto last_keys = 0UL;
    
    while (!should_stop_) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        auto current_keys = keys_checked_.load();
        auto rate = current_keys - last_keys;
        
        if (config_.verbose) {
            logger_.log("[PROGRESS] Speed: " + std::to_string(rate / 1000000) + "M k/s | " +
                       "Total: " + std::to_string(current_keys / 1000000) + "M keys | " +
                       "Found: " + std::to_string(found_count_.load()));
        }
        
        last_keys = current_keys;
    }
}

}  // namespace btc_gold
