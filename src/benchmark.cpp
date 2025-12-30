#include "hash160.h"
#include "secp256k1_wrapper.h"
#include "logger.h"
#include <iostream>
#include <chrono>
#include <random>

using namespace btc_gold;

int main() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "BTC GOLD C++ BENCHMARK v1.0\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    Hash160Engine hash_engine;
    Secp256k1& secp = Secp256k1::instance();
    
    // Test 1: Hash160 Performance
    std::cout << "[TEST 1] Hash160 Performance\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100000; ++i) {
        std::vector<uint8_t> data(33, 0x02);
        data[0] = i & 0xFF;
        auto hash = hash_engine.compute(data);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double kps = (100000.0 / duration.count()) * 1000;
    
    std::cout << "Hash160: " << kps << " k/s\n\n";
    
    // Test 2: Full Key Generation + Hash
    std::cout << "[TEST 2] Full Pipeline (KeyGen + Hash160)\n";
    start = std::chrono::high_resolution_clock::now();
    
    std::mt19937_64 rng(42);
    for (int i = 0; i < 50000; ++i) {
        PrivateKey privkey;
        for (int j = 0; j < 32; ++j) {
            privkey[j] = rng() & 0xFF;
        }
        
        if (secp.verify_privkey(privkey)) {
            auto pubkey = secp.pubkey_compressed(privkey);
            std::vector<uint8_t> pubkey_vec(pubkey.begin(), pubkey.end());
            auto hash = hash_engine.compute(pubkey_vec);
        }
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    kps = (50000.0 / duration.count()) * 1000;
    
    std::cout << "Full Pipeline: " << kps << " k/s\n\n";
    
    std::cout << std::string(70, '=') << "\n";
    std::cout << "[EXPECTED] C++ should achieve 50-100M keys/sec on modern CPU\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    return 0;
}
