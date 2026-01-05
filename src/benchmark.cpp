#include "secp256k1_wrapper.h"
#include "logger.h"
#include "database.h"
#include "types.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace btc_gold;

int main() {
    try {
        Logger logger("", Logger::Level::INFO);
        logger.info("=" + std::string(70, '='));
        logger.info("BTC GOLD C++ v4.0 - Performance Benchmark");
        logger.info("=" + std::string(70, '='));
        
        Secp256k1Wrapper secp256k1;
        
        // Test 1: Public key generation (compressed)
        {
            logger.info("\nTest 1: Public key generation (compressed)");
            
            PrivateKey privkey;
            secp256k1.int_to_privkey(1, privkey);
            
            constexpr int iterations = 10000;
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < iterations; i++) {
                PublicKey pubkey = secp256k1.pubkey_compressed(privkey);
                (void)pubkey;  // Suppress unused warning
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            double ops_per_sec = (iterations * 1000000.0) / duration.count();
            logger.info("Iterations: " + std::to_string(iterations));
            logger.info("Time: " + std::to_string(duration.count() / 1000.0) + " ms");
            logger.info("Speed: " + std::to_string(static_cast<int>(ops_per_sec)) + " ops/sec");
        }
        
        // Test 2: Hash160 calculation
        {
            logger.info("\nTest 2: Hash160 calculation");
            
            PrivateKey privkey;
            secp256k1.int_to_privkey(1, privkey);
            PublicKey pubkey = secp256k1.pubkey_compressed(privkey);
            
            constexpr int iterations = 10000;
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < iterations; i++) {
                Hash160 hash = secp256k1.hash160(pubkey);
                (void)hash;  // Suppress unused warning
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            double ops_per_sec = (iterations * 1000000.0) / duration.count();
            logger.info("Iterations: " + std::to_string(iterations));
            logger.info("Time: " + std::to_string(duration.count() / 1000.0) + " ms");
            logger.info("Speed: " + std::to_string(static_cast<int>(ops_per_sec)) + " ops/sec");
        }
        
        // Test 3: Point addition (TURBO mode)
        {
            logger.info("\nTest 3: Point addition (TURBO mode)");
            
            PrivateKey privkey;
            secp256k1.int_to_privkey(1, privkey);
            PublicKey pubkey = secp256k1.pubkey_compressed(privkey);
            
            constexpr int iterations = 10000;
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < iterations; i++) {
                secp256k1.pubkey_tweak_add(pubkey, 1);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            double ops_per_sec = (iterations * 1000000.0) / duration.count();
            logger.info("Iterations: " + std::to_string(iterations));
            logger.info("Time: " + std::to_string(duration.count() / 1000.0) + " ms");
            logger.info("Speed: " + std::to_string(static_cast<int>(ops_per_sec)) + " ops/sec");
        }
        
        // Test 4: Base58 encoding
        {
            logger.info("\nTest 4: Base58 encoding");
            
            Hash160 hash;
            hash.fill(0x01);
            
            constexpr int iterations = 1000;
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < iterations; i++) {
                std::string address = secp256k1.hash160_to_address(hash);
                (void)address;  // Suppress unused warning
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            double ops_per_sec = (iterations * 1000000.0) / duration.count();
            logger.info("Iterations: " + std::to_string(iterations));
            logger.info("Time: " + std::to_string(duration.count() / 1000.0) + " ms");
            logger.info("Speed: " + std::to_string(static_cast<int>(ops_per_sec)) + " ops/sec");
        }
        
        logger.info("\n" + std::string(70, '='));
        logger.info("Benchmark completed successfully");
        logger.info(std::string(70, '='));
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
}
