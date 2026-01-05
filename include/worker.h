#pragma once

#include "types.h"
#include "config.h"
#include "database.h"
#include "hash160.h"
#include "secp256k1_wrapper.h"
#include <memory>
#include <atomic>
#include <vector>

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

// Buffer de hits para batch writing (evita mutex contention)
struct FoundKey {
    std::string address;
    std::string privkey_hex;
    std::string pubkey_hex;
    std::string hash160_hex;
    std::string wif_c;
    std::string wif_u;
    bool compressed;
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
    
    // Retorna hits encontrados por este worker para batch write
    const std::vector<FoundKey>& get_found_keys() const { return found_keys_buffer_; }

private:
    void run_linear_mode();
    void run_linear_mode_turbo();  // v4.0: Otimização agressiva
    void run_random_mode();
    void run_geometric_mode();
    void run_terminator_mode();
    void run_doubling_mode();      // v4.0: Novo modo (Modo 5)
    void run_hamming_mode();       // v4.0: Novo modo otimizado (Modo 6)
    void run_modular_stride_mode(); // v4.0: Novo modo (Modo 7)
    
    void check_and_save(const PrivateKey& privkey, const Hash160& hash160, bool compressed);
    void batch_write_found_keys();  // v4.0: Escreve buffer de hits em batch

    int worker_id_;
    const Config& config_;
    const Database& database_;
    Stats& stats_;
    
    std::unique_ptr<Hash160Engine> hash_engine_;
    Secp256k1& secp256k1_ = Secp256k1::instance();
    
    // v4.0: Buffer local para hits (sem mutex, apenas ao final)
    std::vector<FoundKey> found_keys_buffer_;
    static constexpr size_t BATCH_SIZE = 10000;  // Flush após 10k hits
};

}  // namespace btc_gold
