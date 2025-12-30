#pragma once

#include "types.h"
#include "config.h"
#include "database.h"
#include "hash160.h"
#include "secp256k1_wrapper.h"
#include <memory>
#include <atomic>

namespace btc_gold {

// Alinhamento de 64 bytes para evitar False Sharing entre threads
struct alignas(64) Stats {
    std::atomic<uint64_t> total_keys{0};
    std::atomic<uint64_t> found_count{0};
    std::atomic<bool> should_stop{false};
    uint64_t start_time = 0;
    
    // Padding para preencher a linha de cache
    char padding[64 - (sizeof(std::atomic<uint64_t>)*2 + sizeof(std::atomic<bool>) + sizeof(uint64_t)) % 64];
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
    void run_terminator_mode();
    
    void check_and_save(const PrivateKey& privkey, const Hash160& hash160, bool compressed);

    int worker_id_;
    const Config& config_;
    const Database& database_;
    Stats& stats_;
    
    std::unique_ptr<Hash160Engine> hash_engine_;
    Secp256k1& secp256k1_ = Secp256k1::instance();
};

}  // namespace btc_gold
