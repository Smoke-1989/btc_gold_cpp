# BTC GOLD v4.0 - COMPILATION FIX GUIDE

## ✅ ERRORS RESOLVED

This document summarizes all compilation errors and their fixes.

---

## ERROR 1: Logger::log() - Wrong Number of Arguments

### Problem
```
error: no matching function for call to 'btc_gold::Logger::log(const char [19])'
  logger_.log("[LINEAR] Starting Linear mode");
```

### Root Cause
- `Logger::log()` expects 2 arguments: `Level` + `message`
- Code was passing only 1 argument

### Fix Applied
```cpp
// WRONG:
logger_.log("[LINEAR] Starting Linear mode");

// CORRECT:
logger_.log(Logger::Level::INFO, "[LINEAR] Starting Linear mode");
```

### Files Fixed
- `src/worker_v4.cpp` - All ~40 `logger_.log()` calls updated

---

## ERROR 2: secp256k1_ - Undeclared Member Variable

### Problem
```
error: 'secp256k1_' was not declared in this scope; did you mean 'Secp256k1'?
  secp256k1_.int_to_privkey(current, privkey);
```

### Root Cause
- `secp256k1_` member was not initialized in class
- Constructor wasn't initializing it properly

### Fix Applied
```cpp
// In worker.h:
class WorkerEngine {
private:
    Secp256k1Wrapper secp256k1_;  // Added
};

// In worker_v4.cpp constructor:
WorkerEngine::WorkerEngine(const Config& config)
    : config_(config), logger_(config.verbose), database_(config.database_file),
      secp256k1_() {}  // Initialize secp256k1_
```

### Files Fixed
- `include/worker.h` - Added member variable
- `src/worker_v4.cpp` - Initialize in constructor

---

## ERROR 3: Missing Secp256k1Wrapper Methods

### Problem
```
error: 'class btc_gold::Secp256k1Wrapper' has no member named 'int_to_privkey'
```

### Methods Added
1. **int_to_privkey()** - Convert uint64 to PrivateKey
2. **hash160()** - SHA256 + RIPEMD160 hash
3. **pubkey_tweak_add()** - EC point addition (TURBO optimization)
4. **hash160_to_address()** - Convert hash to address
5. **pubkey_to_address()** - Direct pubkey to address

### Files Fixed
- `include/secp256k1_wrapper.h` - Added method declarations
- `src/secp256k1_wrapper.cpp` - Implemented all methods

---

## ERROR 4: RIPEMD160 Deprecated (OpenSSL 3.0+)

### Problem
```
warning: 'unsigned char* RIPEMD160(...)' is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
```

### Root Cause
- OpenSSL 3.0+ deprecated direct RIPEMD160() call
- Need to use EVP interface for modern OpenSSL

### Fix Applied
```cpp
// Use EVP interface for OpenSSL 3.0+
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx, EVP_ripemd160(), NULL);
    EVP_DigestUpdate(mdctx, sha256_hash, SHA256_DIGEST_LENGTH);
    unsigned int md_len;
    EVP_DigestFinal_ex(mdctx, ripemd160_hash, &md_len);
    EVP_MD_CTX_free(mdctx);
#else
    RIPEMD160(sha256_hash, SHA256_DIGEST_LENGTH, ripemd160_hash);
#endif
```

### Files Fixed
- `src/secp256k1_wrapper.cpp` - All RIPEMD160 calls use EVP interface

---

## ERROR 5: Database::contains() - Wrong Signature

### Problem
```
error: no matching function for call to 'btc_gold::Database::contains(const Hash160&, btc_gold::Config::InputType&)'
```

### Root Cause
- Called with 2 arguments, but method expects only 1
- `Database::contains()` takes only `Hash160`

### Fix Applied
```cpp
// WRONG:
return database_.contains(hash160, config_.input_type);

// CORRECT:
return database_.contains(hash160);
```

### Files Fixed
- `src/worker_v4.cpp` - Updated check_match() method

---

## COMPILATION STEPS

### Option 1: Full Rebuild (Recommended)
```bash
cd btc_gold_cpp
chmod +x build_v4.sh
./build_v4.sh
```

### Option 2: Manual CMake
```bash
cd btc_gold_cpp
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native -flto" ..
make -j$(nproc)
```

### Option 3: Clean Incremental
```bash
cd btc_gold_cpp/build
make clean
make -j$(nproc)
```

---

## VERIFICATION

### Successful Compilation
You should see:
```
[100%] Linking CXX executable btc_gold
[100%] Built target btc_gold
```

### Test Compilation
```bash
./build/btc_gold_benchmark
```

Expected output:
```
[TEST 1] Hash160: ~833,333 k/s
[TEST 2] Random: ~36,075 k/s
```

---

## SUMMARY

| Error | Type | Fix | Files |
|-------|------|-----|-------|
| Logger::log() | Argument mismatch | Add Logger::Level parameter | worker_v4.cpp |
| secp256k1_ | Undeclared | Init in constructor | worker.h, worker_v4.cpp |
| Missing methods | API mismatch | Add 5 new methods | secp256k1_wrapper.* |
| RIPEMD160 | Deprecation warning | Use EVP interface | secp256k1_wrapper.cpp |
| Database::contains() | Signature | Remove InputType param | worker_v4.cpp |

---

## GIT COMMITS

All fixes are committed to main branch:

1. **48231291** - Fix compilation errors - Logger calls, secp256k1 initialization
2. **59be4b3f** - Fix RIPEMD160 deprecated warning, add missing methods
3. **12473f5c** - Update Secp256k1Wrapper header

---

## NEXT STEPS

After compilation succeeds:

```bash
# Test basic functionality
./build/btc_gold --mode linear --threads 4 --end 1000000

# Test DOUBLING mode (new)
./build/btc_gold --mode doubling --min-bit 66 --max-bit 76

# Test HAMMING mode (new)
./build/btc_gold --mode hamming --min-bit 1 --max-bit 100
```

---

## TROUBLESHOOTING

### Still Getting Logger Errors?
- Ensure `worker_v4.cpp` is recompiled
- Run `make clean` then `make`

### Still Getting secp256k1_ Errors?
- Check that `include/worker.h` has the member variable
- Verify `secp256k1_wrapper.h` is included in `worker.h`

### RIPEMD160 Warnings Still Appearing?
- Add to CMakeLists.txt:
  ```cmake
  add_compile_options(-Wno-deprecated-declarations)
  ```

### EVP_ripemd160 Not Found?
- Ensure OpenSSL 3.0+ is installed:
  ```bash
  apt-get install libssl-dev
  openssl version
  ```

---

## CONTACT

If compilation still fails:
1. Run `./build_v4.sh` again
2. Post full error output
3. Include: `uname -a`, `openssl version`, `cmake --version`

---

Version: v4.0 EXTERMINATOR
Date: January 5, 2026
Status: ✅ ALL ERRORS FIXED

═══════════════════════════════════════════════════════════════════════════════
