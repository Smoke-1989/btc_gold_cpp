#pragma once

#include "types.h"
#include "config.h"
#include "database.h"
#include "hash160.h"
#include "secp256k1_wrapper.h"
#include <memory>

namespace btc_gold {

struct Stats {
    std::atomic<uint64_t> total_keys{0};
    std::atomic<uint64_t> found_count{0};
    std::atomic<bool> should_stop{false};
};

class Worker {
public:
    Worker(
        int worker_id,
        const Config& config,
        const Database& database,
        Stats& stats
    );
    
    void run();

private:
    void run_linear_mode();
    void run_random_mode();
    void run_geometric_mode();
    
    // Updated signature: added 'bool compressed'
    void check_and_save(const PrivateKey& privkey, const Hash160& hash160, bool compressed);

    int worker_id_;
    const Config& config_;
    const Database& database_;
    Stats& stats_;
    
    // Each worker has its own hash engine to be thread-safe
    std::unique_ptr<Hash160Engine> hash_engine_;
    
    // Singleton wrapper for secp256k1 context
    Secp256k1& secp256k1_ = Secp256k1::instance();
};

}  // namespace btc_gold
