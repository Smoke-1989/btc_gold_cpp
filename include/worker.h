#pragma once

#include "types.h"
#include "database.h"
#include "hash160.h"
#include "secp256k1_wrapper.h"
#include <thread>
#include <atomic>
#include <vector>
#include <cstdint>
#include <memory>

namespace btc_gold {

class Worker {
public:
    struct Stats {
        std::atomic<uint64_t> total_keys{0};
        std::atomic<uint64_t> found_count{0};
        std::atomic<bool> should_stop{false};
        uint64_t start_time = 0;
    };
    
    Worker(
        int worker_id,
        const Config& config,
        const Database& database,
        Stats& stats
    );
    
    void run();
    
private:
    int worker_id_;
    const Config& config_;
    const Database& database_;
    Stats& stats_;
    
    std::unique_ptr<Hash160Engine> hash_engine_;
    const Secp256k1& secp256k1_ = Secp256k1::instance();
    
    void run_linear_mode();
    void run_random_mode();
    void run_geometric_mode();
    
    void check_and_save(const PrivateKey& privkey, const Hash160& hash160);
};

}  // namespace btc_gold
