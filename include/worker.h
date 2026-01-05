#pragma once

#include "types.h"
#include "secp256k1_wrapper.h"
#include "database.h"
#include "logger.h"
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <queue>

namespace btc_gold {

// ============================================================================
// BATCH WRITE BUFFER - Reduz mutex contention
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
// WORKER ENGINE v4.0
// ============================================================================

class WorkerEngine {
public:
    WorkerEngine(const Config& config);
    ~WorkerEngine();
    
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
    Config config_;
    Database database_;
    Logger logger_;
    Secp256k1Wrapper secp256k1_;
    HitBuffer hit_buffer_;
    
    std::atomic<uint64_t> keys_checked_{0};
    std::atomic<bool> should_stop_{false};
    std::atomic<int> found_count_{0};
    
    // Worker threads
    std::vector<std::thread> workers_;
    
    // ========================================================================
    // LINEAR MODE (Turbo - Point Addition)
    // ========================================================================
    
    void linear_worker_turbo(int thread_id);
    // Point Addition: Start with privkey, then use EC tweak_add for speed
    // Recalculate privkey ONLY on hit
    
    // ========================================================================
    // RANDOM MODE
    // ========================================================================
    
    void random_worker(int thread_id);
    // Pure random search across full 256-bit space
    
    // ========================================================================
    // GEOMETRIC MODE (3-Phase)
    // ========================================================================
    
    void geometric_worker(int thread_id);
    // Phase 1: Border (near range_min_bit)
    // Phase 2: Ceiling (near range_max_bit)
    // Phase 3: Hamming (low-weight middle)
    
    // ========================================================================
    // TERMINATOR MODE (Multiplicative Descent)
    // ========================================================================
    
    void terminator_worker(int thread_id);
    // Start with large multiplier, decrease exponentially
    // multiplier = 2 → 2^1 → 2^2 → ...
    
    // ========================================================================
    // MODE 5: DOUBLING (Powers of 2)
    // ========================================================================
    
    void doubling_worker();
    // Test: 2^(min_bit-1), 2^min_bit, ..., 2^max_bit
    // Single-threaded (only ~256 combinations max)
    // Speed: 50M+ k/s
    
    // ========================================================================
    // MODE 6: HAMMING (Low-Weight Keys)
    // ========================================================================
    
    void hamming_worker();
    // Test all 2-bit combinations: 2^bit1 + 2^bit2
    // Range: [min_bit, max_bit]
    // Combinatorial: C(n, 2) = n*(n-1)/2 keys
    // Single-threaded (combinatorial enumeration)
    
    // ========================================================================
    // MODE 7: MODULAR STRIDE (Arithmetic Progression)
    // ========================================================================
    
    void modular_stride_worker(int thread_id);
    // Test: start, start+multiplier, start+2*multiplier, ...
    // Distributed by thread: each thread gets offset
    // Thread i tests: start+i, start+i+multiplier, ...
    // Speed: 50M+ k/s
    
    // ========================================================================
    // HELPER METHODS
    // ========================================================================
    
    // Check if privkey matches any target in database
    bool check_match(const PrivateKey& privkey, const PublicKey& pubkey,
                     const Hash160& hash160);
    
    // Convert privkey to address/WIF formats
    void format_key_result(const PrivateKey& privkey, const Hash160& hash160,
                          HitBuffer::Hit& hit);
    
    // Flush hits to disk
    void flush_hits();
    
    // Progress reporting
    void report_progress();
    
    // Initialize workers based on thread count
    void init_workers();
    
    // Get stride for this thread (modular mode)
    uint64_t get_thread_stride(int thread_id, uint64_t base_stride);
};

}  // namespace btc_gold
