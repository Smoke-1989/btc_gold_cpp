#include "worker.h"
#include "logger.h"
#include <random>
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <unordered_set>
#include <iostream>
#include <cmath>

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

// Representa 2^power em formato de chave privada (32 bytes, big-endian)
inline void big_int_power_of_2_to_privkey(int power, PrivateKey& privkey) {
    std::fill(privkey.begin(), privkey.end(), 0);
    if (power < 0) return;
    int byte_index = power / 8;
    int bit_index  = power % 8;
    if (byte_index < 32) {
        privkey[31 - byte_index] = static_cast<uint8_t>(1 << bit_index);
    }
}

// Soma um inteiro de 64 bits (delta) em uma chave privada (little-endian nos 8 bytes finais)
inline void add_uint64_to_privkey(PrivateKey& privkey, uint64_t delta) {
    int i = 31;
    while (delta > 0 && i >= 0) {
        uint64_t sum = static_cast<uint64_t>(privkey[i]) + (delta & 0xFFULL);
        privkey[i]   = static_cast<uint8_t>(sum & 0xFFULL);
        delta        = (delta >> 8) + (sum >> 8);
        --i;
    }
}

// Seta um bit específico (2^power) dentro da chave privada
inline void set_bit_in_privkey(int power, PrivateKey& privkey) {
    if (power < 0) return;
    int byte_index = power / 8;
    int bit_index  = power % 8;
    if (byte_index < 32) {
        privkey[31 - byte_index] |= static_cast<uint8_t>(1 << bit_index);
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
    // Bits configurados pelo usuário (pensando em puzzles: 66, 67, 68...)
    int min_bit = config_.range_min_bit;
    int max_bit = config_.range_max_bit;

    if (min_bit < 1) min_bit = 1;
    if (max_bit <= 0) return;

    // Curva secp256k1 ~ 2^256, então bit 256 já estoura
    if (max_bit > 255) max_bit = 255;
    if (min_bit > max_bit) return;

    // Tamanho da "fronteira" em cada bit (Border Patrol)
    const uint64_t WINDOW = 1000000ULL; // 1 milhão de chaves por bit

    uint64_t stride_val = config_.num_threads * config_.stride;
    if (stride_val == 0) stride_val = config_.num_threads;
    if (stride_val == 0) stride_val = 1;

    if (worker_id_ == 0) {
        std::stringstream ss;
        ss << "[GEOMETRIC] Border Patrol: bits " << min_bit << " a " << max_bit
           << " | janela " << WINDOW << " chaves por bit";
        Logger::instance().info(ss.str());
    }

    // Tweak para incremento constante na chave (igual ao Linear Mode)
    uint8_t tweak[32] = {0};
    for (int i = 0; i < 8; ++i) tweak[31 - i] = (stride_val >> (i * 8)) & 0xFFULL;

    PrivateKey base_priv;
    PrivateKey current_priv;
    std::vector<uint8_t> pubkey_c, pubkey_u;

    // -----------------------------
    // FASE 1: BORDER PATROL POR BIT
    // -----------------------------
    for (int bit = min_bit + worker_id_; bit <= max_bit; bit += config_.num_threads) {
        if (stats_.should_stop) break;

        // Para um puzzle de N bits, o range começa em 2^(N-1)
        int base_power = bit - 1;
        if (base_power < 0) base_power = 0;

        // Base = 2^(base_power)
        big_int_power_of_2_to_privkey(base_power, base_priv);
        // Offset por thread (cada worker começa em base + worker_id_)
        current_priv = base_priv;
        add_uint64_to_privkey(current_priv, static_cast<uint64_t>(worker_id_));

        pubkey_c.clear();
        pubkey_u.clear();

        if (config_.scan_mode != Config::ScanMode::UNCOMPRESSED) {
            auto pk = secp256k1_.pubkey_compressed(current_priv);
            pubkey_c.assign(pk.begin(), pk.end());
        }
        if (config_.scan_mode != Config::ScanMode::COMPRESSED) {
            pubkey_u = secp256k1_.pubkey_uncompressed(current_priv);
        }

        if (pubkey_c.empty() && pubkey_u.empty()) continue;

        // Cada thread cobre índices: i = worker_id_, worker_id_ + stride_val, ... < WINDOW
        if (static_cast<uint64_t>(worker_id_) >= WINDOW) continue;
        uint64_t max_index = WINDOW - 1;
        uint64_t remaining = ((max_index - static_cast<uint64_t>(worker_id_)) / stride_val) + 1;

        while (!stats_.should_stop && remaining--) {
            if (!pubkey_c.empty()) {
                auto hash = hash_engine_->compute(pubkey_c);
                if (database_.contains(hash)) {
                    check_and_save(current_priv, hash, true);
                }
                secp256k1_.pubkey_tweak_add(pubkey_c, tweak);
            }

            if (!pubkey_u.empty()) {
                auto hash = hash_engine_->compute(pubkey_u);
                if (database_.contains(hash)) {
                    check_and_save(current_priv, hash, false);
                }
                secp256k1_.pubkey_tweak_add(pubkey_u, tweak);
            }

            // Avança a chave privada em "stride_val" para manter consistência com o pubkey
            add_uint64_to_privkey(current_priv, stride_val);
            stats_.total_keys++;
        }
    }

    // ----------------------------------------
    // FASE 2: LOW HAMMING WEIGHT (2 bits ligados)
    // ----------------------------------------
    // Para evitar explosão combinatória, apenas a thread 0 executa.
    if (worker_id_ == 0) {
        const int MAX_SECOND_BITS = 32; // limita os segundos bits mais baixos avaliados

        std::stringstream ss;
        ss << "[GEOMETRIC] Low Hamming Weight: ativado (ate " << MAX_SECOND_BITS
           << " bits baixos por faixa)";
        Logger::instance().info(ss.str());

        for (int bit = min_bit; bit <= max_bit && !stats_.should_stop; ++bit) {
            int base_power = bit - 1;
            if (base_power < 0) base_power = 0;

            int max_k = std::min(base_power, MAX_SECOND_BITS);
            for (int k = 0; k <= max_k && !stats_.should_stop; ++k) {
                PrivateKey hw_priv;
                std::fill(hw_priv.begin(), hw_priv.end(), 0);
                set_bit_in_privkey(base_power, hw_priv);
                set_bit_in_privkey(k, hw_priv);

                if (config_.scan_mode != Config::ScanMode::UNCOMPRESSED) {
                    auto pk = secp256k1_.pubkey_compressed(hw_priv);
                    pubkey_c.assign(pk.begin(), pk.end());
                    auto hash = hash_engine_->compute(pubkey_c);
                    if (database_.contains(hash)) check_and_save(hw_priv, hash, true);
                }

                if (config_.scan_mode != Config::ScanMode::COMPRESSED) {
                    auto pk = secp256k1_.pubkey_uncompressed(hw_priv);
                    pubkey_u = std::move(pk);
                    auto hash = hash_engine_->compute(pubkey_u);
                    if (database_.contains(hash)) check_and_save(hw_priv, hash, false);
                }

                stats_.total_keys++;
            }
        }
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
