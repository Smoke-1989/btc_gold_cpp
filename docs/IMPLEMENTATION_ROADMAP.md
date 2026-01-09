# BTC GOLD v5.1 - IMPLEMENTATION ROADMAP

## 📋 TASK BREAKDOWN

### PHASE 1: CRITICAL BUG FIXES (Priority: HIGHEST)

#### Task 1.1: Fix LINEAR Mode Range Partitioning
**File:** `src/worker_v4.cpp` :: `linear_worker()`
**Issue:** Thread ranges overlap (all end at 0xffffffffffffffff)
**Fix:**
```cpp
// OLD (BROKEN):
for(uint256 key = start; key <= end; key++) { ... }

// NEW (FIXED):
uint256 chunk_size = (end - start) / threads;
for(uint256 key = my_start; key < my_end; key++) { ... }
// Ensure: start + i*chunk_size to start + (i+1)*chunk_size
// NO OVERLAP, NO GAPS
```
**Testing:**
  - Generate 8 ranges for 8 threads
  - Verify: All unions to [start, end)
  - Verify: No overlaps
  - Verify: No gaps

---

#### Task 1.2: Fix DOUBLING Mode Parameter Bug
**File:** `src/worker_v4.cpp` :: `run_doubling_mode()`
**Issue:** `min_bit` and `max_bit` parameters ignored, always tests 2^1 to 2^256
**Fix:**
```cpp
// OLD (BROKEN):
for(int bit = 1; bit <= 256; bit++) {
    test_key(2^bit);
}

// NEW (FIXED):
int min_bit = config.mode_params_int["min_bit"];
int max_bit = config.mode_params_int["max_bit"];
for(int bit = min_bit; bit <= max_bit; bit++) {
    test_key(1 << bit);
}
```
**Testing:**
  - Run: `--mode doubling --min-bit 10 --max-bit 20`
  - Verify: Exactly 11 keys tested (2^10 through 2^20)
  - Previously: 256 keys tested

---

#### Task 1.3: Fix GEOMETRIC Mode Thread Overlap
**File:** `src/worker_v4.cpp` :: `run_geometric_mode()` + `geometric_worker()`
**Issue:** All threads duplicate work on border-scan phase
**Architecture Fix:**
```cpp
// NEW DISTRIBUTION:
struct PhaseAllocation {
    Thread 0: Phase 1 (Border Scan)
             - Tests: 2^min_bit, 2^(min_bit+1), ..., 2^max_bit
             - Exclusive: Only T0 does this
    
    Threads 1-3: Phase 2 (Ceiling Ascent)
                - Divide (max_bit - min_bit) among T1, T2, T3
                - T1: 2^a * {2, 3, 4}
                - T2: 2^b * {5, 6, 7, 8}
                - T3: 2^c * {9, ...}
    
    Threads 4-7: Phase 3 (Hamming)
               - Divide hamming combinations among T4-T7
               - Use atomic counter for work stealing
};
```
**Testing:**
  - Instrument: Add unique_id to each key generation
  - Verify: Zero duplicate keys tested
  - Verify: All threads report completion

---

#### Task 1.4: Fix TERMINATOR Mode Duplicates
**File:** `src/worker_v4.cpp` :: `terminator_worker()`
**Issue:** Geometric progression creates overlaps between threads
**Algorithm Fix:**
```cpp
// SEQUENCE: start * multiplier^(thread_id + k*num_threads)
// Thread 0: start * m^0, start * m^8, start * m^16, ...
// Thread 1: start * m^1, start * m^9, start * m^17, ...
// Result: ZERO overlap guaranteed

for(int iteration = 0; ; iteration++) {
    int exponent = thread_id + iteration * num_threads;
    uint256 key = start * pow(multiplier, exponent);
    if(key > end) break;
    test_key(key);
}
```
**Testing:**
  - Set <set> to track tested keys
  - Verify: size == keys_tested
  - Verify: Zero duplicates across 8 threads

---

#### Task 1.5: Fix RANDOM Mode Performance
**File:** `src/worker_v4.cpp` :: `random_worker()`
**Issue:** Only ~91k keys/sec (should be 500k+)
**Optimization:**
```cpp
// OLD (EXPENSIVE):
for(int i = 0; i < iterations; i++) {
    uint256 key = generate_random_key();  // CSPRNG overhead
    secp256k1::point pubkey = compute_pubkey(key);  // Expensive!
    hash160_t hash = hash160(pubkey);
    check_match(hash);
}

// NEW (BATCHED):
vector<uint256> batch(100);
for(int i = 0; i < iterations; i += 100) {
    // Generate 100 random keys
    for(int j = 0; j < 100; j++) {
        batch[j] = generate_random_key();
    }
    
    // Batch compute pubkeys using Shamir's trick
    vector<secp256k1::point> pubkeys = batch_compute_pubkeys(batch);
    
    // Batch hash using SIMD
    vector<hash160_t> hashes = simd_hash160(pubkeys);
    
    // Check all matches
    for(int j = 0; j < 100; j++) {
        check_match(hashes[j]);
    }
}
```
**Expected Improvement:** 91k → 2.5M keys/sec (27x)
**Testing:** Benchmark before/after

---

### PHASE 2: NEW MODES IMPLEMENTATION (Priority: HIGH)

#### Task 2.1: Implement VANITY Mode (Mode 7)
**File:** `src/worker_v4.cpp` :: `run_vanity_mode()` + `vanity_worker()`
**Algorithm:**
```cpp
void vanity_worker(int thread_id) {
    string pattern = config.mode_params["pattern"];
    
    while(running_) {
        // Generate random private key
        uint256 privkey = random_key();
        
        // Compute public key
        secp256k1::point pubkey = privkey * G;
        
        // Hash to address
        uint160 hash160_result = hash160(pubkey);
        string address = encode_address(hash160_result);
        
        // Check if matches pattern
        if(address.find(pattern) != string::npos) {
            report_match(privkey, address);
            if(config.stop_on_find) {
                stop_workers();
            }
        }
    }
}
```
**Performance:** ~2M keys/sec (includes full hash ops)
**Testing:** Search for pattern "1Bitcoin" and verify address matches

---

#### Task 2.2: Implement ENTROPY Mode (Mode 8)
**File:** `src/worker_v4.cpp` :: `run_entropy_mode()` + `entropy_worker()`
**Entropy Tests:**
```cpp
bool is_weak_entropy(const uint256& key) {
    // Test 1: Repeating byte patterns
    if(has_repeating_pattern(key)) return true;
    
    // Test 2: Linear congruential pattern
    if(matches_lc_sequence(key)) return true;
    
    // Test 3: Low Hamming weight
    if(popcount(key) < 40) return true;
    
    // Test 4: Fibonacci-like
    if(is_fibonacci_related(key)) return true;
    
    return false;
}
```
**Performance:** ~4M keys/sec
**Output:** Store weak keys in separate file
**Testing:** Verify known weak patterns are detected

---

#### Task 2.3: Implement COLLISION Mode (Mode 9)
**File:** `src/worker_v4.cpp` :: `run_collision_mode()` + `collision_worker()`
**Algorithm:**
```cpp
void collision_worker(int thread_id) {
    int distance = config.mode_params_int["distance"];
    
    // For each target hash
    for(const auto& target : targets_) {
        // Try key +/- offset
        for(int offset = 1; offset <= distance; offset++) {
            // Check if there's a private key in database
            // that generates target hash when offset is applied
            
            uint256 test_key_1 = base_key + offset;
            uint256 test_key_2 = base_key - offset;
            
            // Expensive: iterate over all possible base keys
            // Better: use target derivatives
        }
    }
}
```
**Performance:** ~5M keys/sec
**Use Case:** Find related wallets in same derivation path
**Testing:** Verify HD wallet path relationships detected

---

### PHASE 3: INTERACTIVE MENU (Priority: HIGH)

#### Task 3.1: Complete Menu System
**Files:**
- `include/interactive_menu.hpp` ✅ (already created)
- `src/interactive_menu.cpp` ✅ (already created)

**Features Implemented:**
- [x] Mode selection with descriptions
- [x] Parameter configuration per mode
- [x] Real-time validation
- [x] Help system
- [x] Configuration preview

**Testing:**
```bash
./btc_gold  # No args = interactive mode
# Then navigate menu
```

---

#### Task 3.2: Integrate Menu with Main
**File:** `src/main_v5.cpp` ✅ (already created)
**Features:**
- [x] CLI mode auto-detection
- [x] Interactive mode entry
- [x] Config passing to WorkerEngine
- [x] Error handling

---

### PHASE 4: TESTING & VALIDATION (Priority: HIGH)

#### Task 4.1: Unit Tests
**File:** `tests/unit_tests.cpp`
```cpp
TEST(RangePartition, NoOverlapLinear) {
    // Verify [start, end) split into N non-overlapping ranges
}

TEST(GeometricPhases, NoRedundancy) {
    // Verify each key tested exactly once
}

TEST(TerminatorProgression, UniqueSequences) {
    // Verify per-thread sequences don't overlap
}

TEST(DoublingRange, RespectsBitLimits) {
    // Verify only 2^min_bit to 2^max_bit tested
}

TEST(HammingEnumeration, Completeness) {
    // Verify all C(256,2) and strategic C(256,3) tested
}
```

#### Task 4.2: Integration Tests
**Test Cases:**
1. Run each mode on test data, verify all finds detected
2. Benchmark: Keys/sec for each mode
3. Memory profiling: No leaks
4. Thread safety: Race condition tests
5. Performance regression: vs v4.0

#### Task 4.3: Validation Tests
**Test Data:** hash160p.txt (160 targets with known privkeys)
```
Expected Results:
- LINEAR: Instant find for key=1,3,7,8,...
- GEOMETRIC: Find 2^248, 2^251, combinations
- RANDOM: 0 in quick test (probabilistic)
- VANITY: N/A (requires matching address pattern)
- ENTROPY: Find low-weight keys
```

---

### PHASE 5: PERFORMANCE OPTIMIZATION (Priority: MEDIUM)

#### Task 5.1: Cache Optimization
- Precompute G, 2G, 4G, ..., 2^255G for windowing
- Align buffers to 64-byte cache lines
- NUMA-aware memory allocation

#### Task 5.2: SIMD Acceleration
- AVX2 for parallel point operations
- SIMD hash computation (64 hashes in parallel)
- Vectorized comparisons

#### Task 5.3: GPU Acceleration (Optional)
- CUDA kernels for public key generation
- GPU hash computation
- Host-device memory transfer optimization

---

## COMPILATION CHECKLIST

### Step 1: Update CMakeLists.txt
```cmake
# Add new files
set(SOURCES
    src/main_v5.cpp
    src/worker_v5.cpp       # New comprehensive worker
    src/interactive_menu.cpp
    src/config.cpp
    src/logger.cpp
    ...
)

# Add headers
include_directories(include)
```

### Step 2: Compile
```bash
cd ~/code/btc_gold_cpp
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

### Step 3: Test
```bash
# Interactive mode
./btc_gold

# CLI mode
./btc_gold --mode linear --input hash160p.txt --start-hex 1 --end-hex 1000

# Run test suite
./btc_gold_tests
```

---

## SUCCESS CRITERIA

✅ **All 10 modes fully functional**
✅ **No duplicate key testing**
✅ **Parameters properly respected**
✅ **Performance targets met:**
- LINEAR: 188M keys/sec
- RANDOM: 2.5M keys/sec  
- GEOMETRIC: 10M keys/sec
- TERMINATOR: 8M keys/sec
- DOUBLING: Instant (256 keys)
- HAMMING: 3M keys/sec
- VANITY: 2M keys/sec
- ENTROPY: 4M keys/sec
- COLLISION: 5M keys/sec

✅ **Interactive menu complete and intuitive**
✅ **All parameters validated**
✅ **Help system comprehensive**
✅ **Zero memory leaks**
✅ **Thread-safe operations**
✅ **Graceful Ctrl+C handling**
✅ **Results saved correctly**

---

## NEXT STEPS

1. **Clone/Pull latest code**
   ```bash
   cd ~/code/btc_gold_cpp
   git pull origin main
   ```

2. **Review ANALYSIS.md and FIXES_v5.1.md**
   - Understand each bug
   - Understand each fix

3. **Build v5.1**
   ```bash
   ./build_v5.sh
   ```

4. **Test thoroughly**
   - Run each mode
   - Verify no duplicates
   - Benchmark performance
   - Check parameter handling

5. **Deployment**
   - Tag as v5.1-production
   - Update documentation
   - Archive old version

---

**Estimated Time:** 4-6 hours for complete implementation
**Complexity:** Medium-High
**Risk Level:** Low (modular changes, good test coverage)
