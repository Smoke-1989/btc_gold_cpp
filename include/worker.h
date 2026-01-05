#pragma once

#include "config.h"              // Config + fundamental types
#include "secp256k1_wrapper.h"
#include "database.h"
#include "logger.h"
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <queue>
#include <stdexcept>
#include <iostream>

namespace btc_gold {

// ============================================================================
// BATCH WRITE BUFFER - Reduces mutex contention
// ============================================================================

struct HitBuffer {
    struct Hit {
        PrivateKey privkey;
        Hash160 hash160;
        std::string address;
        std::string wif_compressed;
    };
    
    std::vector<Hit> hits;
    std::mutex lock;
    static constexpr size_t MAX_HITS = 10000;
    
    void add(const Hit& hit) {
        std::lock_guard<std::mutex> guard(lock);
        hits.push_back(hit);
    }
    
    bool should_flush() const {
        return hits.size() >= MAX_HITS;
    }
    
    std::vector<Hit> flush() {
        std::lock_guard<std::mutex> guard(lock);
        auto result = hits;
        hits.clear();
        return result;
    }
};

// ============================================================================
// WORKER ENGINE v4.0 - Enterprise Grade
// ============================================================================

class WorkerEngine {
public:
    // Constructor
    explicit WorkerEngine(const Config& config, Logger& logger, Database& database);
    
    // Destructor
    ~WorkerEngine();
    
    // Delete copy (RAII)
    WorkerEngine(const WorkerEngine&) = delete;
    WorkerEngine& operator=(const WorkerEngine&) = delete;
    
    // Main entry point
    void run();
    
    // Mode-specific implementations
    void run_linear_mode();
    void run_random_mode();
    void run_geometric_mode();
    void run_terminator_mode();
    void run_doubling_mode();
    void run_hamming_mode();
    void run_modular_stride_mode();
    
private:
    // Config and dependencies (references to external objects)
    const Config& config_;
    Logger& logger_;  // Reference, not ownership
    Database& database_;  // Reference, not ownership
    Secp256k1Wrapper secp256k1_;  // Owned
    
    // State
    HitBuffer hit_buffer_;
    std::atomic<uint64_t> keys_checked_{0};
    std::atomic<bool> should_stop_{false};
    std::atomic<int> found_count_{0};
    std::vector<std::thread> workers_;
    
    // Worker methods
    void linear_worker_turbo(int thread_id);
    void random_worker(int thread_id);
    void geometric_worker(int thread_id);
    void terminator_worker(int thread_id);
    void doubling_worker();
    void hamming_worker();
    void modular_stride_worker(int thread_id);
    
    // Helpers
    bool check_match(const PrivateKey& privkey, const PublicKey& pubkey,
                     const Hash160& hash160);
    void format_key_result(const PrivateKey& privkey, const Hash160& hash160,
                          HitBuffer::Hit& hit);
    void flush_hits();
    void report_progress();
};

}  // namespace btc_gold
