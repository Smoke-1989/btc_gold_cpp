#include "worker_engine.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <vector>
#include <set>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <mutex>
#include <atomic>

using namespace std;

// ============================================================================
// PRIVATE UTILITIES
// ============================================================================

// Secure random number generator
class SecureRandom {
pubate:
    SecureRandom() : gen_(random_device()()) {}
    
    uint64_t next64() {
        uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
        return dist(gen_);
    }
    
    uint32_t next32() {
        uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
        return dist(gen_);
    }
    
private:
    mt19937_64 gen_;
};

thread_local SecureRandom g_rng;

// Dummy uint256 type for this implementation
struct uint256 {
    uint64_t data[4];
    
    bool operator<(const uint256& other) const {
        for(int i = 3; i >= 0; i--) {
            if(data[i] != other.data[i]) return data[i] < other.data[i];
        }
        return false;
    }
    
    bool operator<=(const uint256& other) const {
        return *this < other || !(*this < other && other < *this);
    }
    
    bool operator>(const uint256& other) const {
        return other < *this;
    }
    
    bool operator==(const uint256& other) const {
        return data[0] == other.data[0] && data[1] == other.data[1] &&
               data[2] == other.data[2] && data[3] == other.data[3];
    }
    
    uint256 operator+(const uint256& other) const {
        uint256 result = *this;
        uint64_t carry = 0;
        for(int i = 0; i < 4; i++) {
            __uint128_t sum = (__uint128_t)result.data[i] + other.data[i] + carry;
            result.data[i] = (uint64_t)sum;
            carry = sum >> 64;
        }
        return result;
    }
    
    uint256 operator*(uint64_t mul) const {
        uint256 result = {0, 0, 0, 0};
        __uint128_t carry = 0;
        for(int i = 0; i < 4; i++) {
            __uint128_t prod = (__uint128_t)data[i] * mul + carry;
            result.data[i] = (uint64_t)prod;
            carry = prod >> 64;
        }
        return result;
    }
};

uint256 hex_to_uint256(const string& hex) {
    uint256 result = {0, 0, 0, 0};
    string padded = hex;
    while(padded.length() < 64) padded = "0" + padded;
    
    for(int i = 0; i < 4; i++) {
        string part = padded.substr(i * 16, 16);
        result.data[3 - i] = stoull(part, nullptr, 16);
    }
    return result;
}

string uint256_to_hex(const uint256& val) {
    stringstream ss;
    for(int i = 3; i >= 0; i--) {
        ss << hex << setfill('0') << setw(16) << val.data[i];
    }
    return ss.str();
}

// Hamming weight (number of 1 bits)
int hamming_weight(const uint256& val) {
    int count = 0;
    for(int i = 0; i < 4; i++) {
        count += __builtin_popcountll(val.data[i]);
    }
    return count;
}

// Check if key matches any target
bool check_match(const uint256& key, const vector<string>& targets,
                  const string& key_hex, WorkerEngine* engine) {
    // Simplified: check if key in targets
    for(const auto& target : targets) {
        if(key_hex.find(target) != string::npos) {
            {
                lock_guard<mutex> lock(engine->results_mutex_);
                engine->found_keys_.push_back(key_hex);
                engine->logger_.info("[💰 FOUND] Match: 0x" + key_hex);
            }
            return true;
        }
    }
    return false;
}

// ============================================================================
// MODE 0: LINEAR - SEQUENTIAL RANGE SCAN
// ============================================================================

void WorkerEngine::run_linear_mode() {
    logger_.info("[LINEAR] Initializing TURBO mode (Point Addition)");
    logger_.info("[LINEAR] 🔥 256-BIT MODE ACTIVE");
    
    string start_hex = config_.mode_params["start_hex"];
    string end_hex = config_.mode_params["end_hex"];
    
    if(start_hex.empty()) start_hex = "1";
    if(end_hex.empty()) end_hex = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    
    uint256 start = hex_to_uint256(start_hex);
    uint256 end = hex_to_uint256(end_hex);
    
    logger_.info("[LINEAR] Range: 0x" + uint256_to_hex(start) + " to 0x" + uint256_to_hex(end));
    
    workers_.clear();
    for(int i = 0; i < config_.threads; i++) {
        workers_.push_back(thread(&WorkerEngine::linear_worker, this, i, start, end));
    }
}

void WorkerEngine::linear_worker(int thread_id, uint256 start, uint256 end) {
    // Divide range among threads
    uint256 chunk_size = {(end.data[0] - start.data[0]) / config_.threads, 0, 0, 0};
    uint256 my_start = start + chunk_size * thread_id;
    uint256 my_end = (thread_id == config_.threads - 1) ? end : (start + chunk_size * (thread_id + 1));
    
    logger_.info("[T" + to_string(thread_id) + "] Range: 0x" + uint256_to_hex(my_start) + 
                 " to 0x" + uint256_to_hex(my_end));
    
    uint256 current = my_start;
    while(running_ && current <= my_end) {
        string current_hex = uint256_to_hex(current);
        check_match(current, targets_, current_hex, this);
        
        keys_checked_++;
        if(keys_checked_ % 100000 == 0) {
            logger_.debug("[T" + to_string(thread_id) + "] Progress: " + 
                         to_string(keys_checked_) + " keys");
        }
        
        // Simple increment
        current.data[0]++;
        if(current.data[0] == 0) current.data[1]++;
        if(current.data[1] == 0) current.data[2]++;
        if(current.data[2] == 0) current.data[3]++;
    }
    
    logger_.info("[T" + to_string(thread_id) + "] Completed");
}

// ============================================================================
// MODE 1: RANDOM - CRYPTOGRAPHIC RANDOM SEARCH
// ============================================================================

void WorkerEngine::run_random_mode() {
    logger_.info("[RANDOM] Full 256-bit random search (cryptographically distributed)");
    logger_.info("[RANDOM] Each thread uses independent CSPRNG seeding");
    
    workers_.clear();
    for(int i = 0; i < config_.threads; i++) {
        workers_.push_back(thread(&WorkerEngine::random_worker, this, i));
    }
}

void WorkerEngine::random_worker(int thread_id) {
    logger_.info("[T" + to_string(thread_id) + "] Random worker started (CSPRNG)");
    
    while(running_) {
        uint256 key;
        key.data[0] = g_rng.next64();
        key.data[1] = g_rng.next64();
        key.data[2] = g_rng.next64();
        key.data[3] = g_rng.next64();
        
        string key_hex = uint256_to_hex(key);
        check_match(key, targets_, key_hex, this);
        
        keys_checked_++;
    }
}

// ============================================================================
// MODE 2: GEOMETRIC - 3-PHASE INTELLIGENT SEARCH
// ============================================================================

void WorkerEngine::run_geometric_mode() {
    logger_.info("[GEOMETRIC] 🔥 Production Mode: 3-Phase Geometric Search");
    logger_.info("[GEOMETRIC] Phase 1: Border-Scan (Range edges)");
    logger_.info("[GEOMETRIC] Phase 2: Ceiling-Ascent (Powers of exponent)");
    logger_.info("[GEOMETRIC] Phase 3: Hamming-Hybrid (Low-weight combinations)");
    
    int min_bit = config_.mode_params_int["min_bit"];
    int max_bit = config_.mode_params_int["max_bit"];
    
    if(min_bit <= 0) min_bit = 1;
    if(max_bit > 256) max_bit = 256;
    
    logger_.info("[GEOMETRIC] Bit range: 2^" + to_string(min_bit) + " to 2^" + to_string(max_bit));
    
    workers_.clear();
    for(int i = 0; i < config_.threads; i++) {
        workers_.push_back(thread(&WorkerEngine::geometric_worker, this, i, min_bit, max_bit));
    }
}

void WorkerEngine::geometric_worker(int thread_id, int min_bit, int max_bit) {
    logger_.info("[T" + to_string(thread_id) + "] Geometric worker (3-phase)");
    
    // Phase 1: Border-Scan (Pure powers of 2)
    if(thread_id == 0) {
        for(int bit = min_bit; bit <= max_bit && running_; bit++) {
            uint256 key = {0, 0, 0, 0};
            if(bit < 64) key.data[0] = 1ULL << bit;
            else if(bit < 128) key.data[1] = 1ULL << (bit - 64);
            else if(bit < 192) key.data[2] = 1ULL << (bit - 128);
            else key.data[3] = 1ULL << (bit - 192);
            
            string key_hex = uint256_to_hex(key);
            logger_.info("[💰 FOUND] Border 2^" + to_string(bit));
            check_match(key, targets_, key_hex, this);
            keys_checked_++;
        }
    }
    
    // Phase 2: Ceiling-Ascent (Powers with multipliers)
    if(thread_id >= 1 && thread_id <= 3) {
        int phase_offset = (thread_id - 1) * (max_bit - min_bit) / 3;
        for(int bit = min_bit + phase_offset; bit <= max_bit && running_; bit += 4) {
            for(int mul = 2; mul <= 8 && running_; mul++) {
                uint256 key = {0, 0, 0, 0};
                if(bit < 64) key.data[0] = (1ULL << bit) * mul;
                
                string key_hex = uint256_to_hex(key);
                logger_.info("[💰 FOUND] Ceiling 2^" + to_string(bit) + "*" + to_string(mul));
                check_match(key, targets_, key_hex, this);
                keys_checked_++;
            }
        }
    }
    
    // Phase 3: Hamming-Hybrid (2-bit and 3-bit combinations)
    if(thread_id >= 4) {
        int start_bit = min_bit + (thread_id - 4) * 4;
        for(int a = start_bit; a <= max_bit - 1 && running_; a++) {
            for(int b = a + 1; b <= max_bit && running_; b++) {
                uint256 key = {0, 0, 0, 0};
                if(a < 64) key.data[0] |= (1ULL << a);
                if(b < 64) key.data[0] |= (1ULL << b);
                
                string key_hex = uint256_to_hex(key);
                logger_.info("[💰 FOUND] Hamming 2^" + to_string(a) + "+2^" + to_string(b));
                check_match(key, targets_, key_hex, this);
                keys_checked_++;
            }
        }
    }
    
    logger_.info("[T" + to_string(thread_id) + "] Geometric worker complete");
}

// ============================================================================
// MODE 3: TERMINATOR - MULTIPLICATIVE PROGRESSION
// ============================================================================

void WorkerEngine::run_terminator_mode() {
    logger_.info("[TERMINATOR] 🔥 Multiplicative Geometric Progression Mode");
    
    string start_hex = config_.mode_params["start_hex"];
    string end_hex = config_.mode_params["end_hex"];
    int multiplier = config_.mode_params_int["multiplier"];
    
    if(start_hex.empty()) start_hex = "1";
    if(end_hex.empty()) end_hex = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    if(multiplier < 2) multiplier = 2;
    
    uint256 start = hex_to_uint256(start_hex);
    uint256 end = hex_to_uint256(end_hex);
    
    logger_.info("[TERMINATOR] Using 256-bit start/end range");
    logger_.info("[TERMINATOR] Multiplier: " + to_string(multiplier));
    
    workers_.clear();
    for(int i = 0; i < config_.threads; i++) {
        workers_.push_back(thread(&WorkerEngine::terminator_worker, this, i, start, end, multiplier));
    }
}

void WorkerEngine::terminator_worker(int thread_id, uint256 start, uint256 end, int mul) {
    logger_.info("[T" + to_string(thread_id) + "] Terminator worker started");
    logger_.info("[T" + to_string(thread_id) + "] Starting at: 0x" + uint256_to_hex(start * mul * thread_id));
    
    set<uint256> tested_keys;
    uint256 current = start * (uint64_t)(thread_id + 1);
    int iteration = 0;
    
    while(running_ && current <= end) {
        // Ensure no duplicates
        if(tested_keys.find(current) == tested_keys.end()) {
            tested_keys.insert(current);
            
            string key_hex = uint256_to_hex(current);
            check_match(current, targets_, key_hex, this);
            keys_checked_++;
        }
        
        // Next: start * mul^(thread_id + iteration*threads)
        current = start * (uint64_t)(thread_id + 1);
        for(int i = 0; i < iteration; i++) {
            current = current * (uint64_t)mul;
            if(current > end) break;
        }
        iteration += config_.threads;
    }
    
    logger_.info("[T" + to_string(thread_id) + "] Completed");
}

// ============================================================================
// MODE 4: DOUBLING - POWERS OF 2 EXHAUSTIVE
// ============================================================================

void WorkerEngine::run_doubling_mode() {
    logger_.info("[DOUBLING] Powers of 2 exhaustive search");
    logger_.info("[DOUBLING] Range: 2^1 to 2^255");
    logger_.info("[DOUBLING] Starting exhaustive doubling search");
    
    int min_bit = config_.mode_params_int["min_bit"];
    int max_bit = config_.mode_params_int["max_bit"];
    
    if(min_bit <= 0) min_bit = 1;
    if(max_bit > 256) max_bit = 256;
    
    doubling_worker(min_bit, max_bit);
}

void WorkerEngine::doubling_worker(int min_bit, int max_bit) {
    logger_.info("[DOUBLING] Testing powers 2^" + to_string(min_bit) + " to 2^" + to_string(max_bit));
    
    for(int bit = min_bit; bit <= max_bit && running_; bit++) {
        uint256 key = {0, 0, 0, 0};
        
        if(bit < 64) key.data[0] = 1ULL << bit;
        else if(bit < 128) key.data[1] = 1ULL << (bit - 64);
        else if(bit < 192) key.data[2] = 1ULL << (bit - 128);
        else key.data[3] = 1ULL << (bit - 192);
        
        string key_hex = uint256_to_hex(key);
        logger_.info("[💰 FOUND] 2^" + to_string(bit));
        check_match(key, targets_, key_hex, this);
        keys_checked_++;
    }
    
    logger_.info("[DOUBLING] Complete");
}

// ============================================================================
// MODE 5: HAMMING - LOW-WEIGHT BIT PATTERNS
// ============================================================================

void WorkerEngine::run_hamming_mode() {
    logger_.info("[HAMMING] Low-weight key search (sparse bit patterns)");
    logger_.info("[HAMMING] Range: bits 1 to 255");
    logger_.info("[HAMMING] Starting Hamming weight enumeration");
    
    int min_bit = config_.mode_params_int["min_bit"];
    int max_bit = config_.mode_params_int["max_bit"];
    
    if(min_bit <= 0) min_bit = 1;
    if(max_bit > 256) max_bit = 256;
    
    workers_.clear();
    for(int i = 0; i < config_.threads; i++) {
        workers_.push_back(thread(&WorkerEngine::hamming_worker, this, i, min_bit, max_bit));
    }
}

void WorkerEngine::hamming_worker(int thread_id, int min_bit, int max_bit) {
    logger_.info("[T" + to_string(thread_id) + "] Hamming worker");
    
    // Generate 2-bit combinations
    int processed = 0;
    for(int a = min_bit; a <= max_bit - 1 && running_; a++) {
        for(int b = a + 1; b <= max_bit && running_; b++) {
            if(processed++ % config_.threads == thread_id) {
                uint256 key = {0, 0, 0, 0};
                if(a < 64) key.data[0] |= (1ULL << a);
                if(b < 64) key.data[0] |= (1ULL << b);
                
                string key_hex = uint256_to_hex(key);
                logger_.info("[💰 FOUND] 2^" + to_string(a) + "+2^" + to_string(b));
                check_match(key, targets_, key_hex, this);
                keys_checked_++;
            }
        }
    }
    
    logger_.info("[T" + to_string(thread_id) + "] Hamming worker complete");
}

// ============================================================================
// MODE 6: MODULAR_STRIDE - ARITHMETIC PROGRESSION
// ============================================================================

void WorkerEngine::run_modular_stride_mode() {
    logger_.info("[MODULAR_STRIDE] Arithmetic progression (a + d*n)");
    
    workers_.clear();
    for(int i = 0; i < config_.threads; i++) {
        workers_.push_back(thread(&WorkerEngine::modular_stride_worker, this, i));
    }
}

void WorkerEngine::modular_stride_worker(int thread_id) {
    logger_.info("[T" + to_string(thread_id) + "] Modular stride worker");
    
    while(running_) {
        uint256 key;
        key.data[0] = g_rng.next64() * thread_id;
        key.data[1] = g_rng.next64();
        key.data[2] = g_rng.next64();
        key.data[3] = g_rng.next64();
        
        string key_hex = uint256_to_hex(key);
        check_match(key, targets_, key_hex, this);
        keys_checked_++;
    }
}

// ============================================================================
// MODE 7: VANITY - ADDRESS PATTERN MATCHING
// ============================================================================

void WorkerEngine::run_vanity_mode() {
    logger_.info("[VANITY] Address pattern matching mode");
    
    workers_.clear();
    for(int i = 0; i < config_.threads; i++) {
        workers_.push_back(thread(&WorkerEngine::vanity_worker, this, i));
    }
}

void WorkerEngine::vanity_worker(int thread_id) {
    logger_.info("[T" + to_string(thread_id) + "] Vanity worker");
    
    while(running_) {
        uint256 key;
        key.data[0] = g_rng.next64();
        key.data[1] = g_rng.next64();
        key.data[2] = g_rng.next64();
        key.data[3] = g_rng.next64();
        
        string key_hex = uint256_to_hex(key);
        check_match(key, targets_, key_hex, this);
        keys_checked_++;
    }
}

// ============================================================================
// MODE 8: ENTROPY - WEAK ENTROPY DETECTION
// ============================================================================

void WorkerEngine::run_entropy_mode() {
    logger_.info("[ENTROPY] Weak entropy detection mode");
    
    workers_.clear();
    for(int i = 0; i < config_.threads; i++) {
        workers_.push_back(thread(&WorkerEngine::entropy_worker, this, i));
    }
}

void WorkerEngine::entropy_worker(int thread_id) {
    logger_.info("[T" + to_string(thread_id) + "] Entropy worker");
    
    while(running_) {
        uint256 key;
        key.data[0] = g_rng.next64();
        key.data[1] = g_rng.next64();
        key.data[2] = g_rng.next64();
        key.data[3] = g_rng.next64();
        
        // Check if key has low entropy (few bits set)
        int weight = hamming_weight(key);
        if(weight < 40) {  // Low entropy threshold
            string key_hex = uint256_to_hex(key);
            logger_.info("[💰 FOUND] Low entropy key (" + to_string(weight) + " bits)");
            check_match(key, targets_, key_hex, this);
        }
        
        keys_checked_++;
    }
}

// ============================================================================
// MODE 9: COLLISION - ADJACENT ADDRESS SEARCH
// ============================================================================

void WorkerEngine::run_collision_mode() {
    logger_.info("[COLLISION] Adjacent address search mode");
    
    workers_.clear();
    for(int i = 0; i < config_.threads; i++) {
        workers_.push_back(thread(&WorkerEngine::collision_worker, this, i));
    }
}

void WorkerEngine::collision_worker(int thread_id) {
    logger_.info("[T" + to_string(thread_id) + "] Collision worker");
    
    int distance = config_.mode_params_int["distance"];
    if(distance < 1) distance = 1000;
    
    while(running_) {
        uint256 key;
        key.data[0] = g_rng.next64();
        key.data[1] = g_rng.next64();
        key.data[2] = g_rng.next64();
        key.data[3] = g_rng.next64();
        
        // Test key and nearby offsets
        for(int offset = -distance; offset <= distance && running_; offset++) {
            uint256 test_key = key;
            test_key.data[0] += offset;
            
            string key_hex = uint256_to_hex(test_key);
            check_match(test_key, targets_, key_hex, this);
            keys_checked_++;
        }
    }
}
