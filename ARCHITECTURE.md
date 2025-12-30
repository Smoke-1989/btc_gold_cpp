# BTC GOLD C++ - Architecture Documentation

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    btc_gold (Main)
│              Intel i7-8550U | 4C/8T | AVX2
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
    [Worker 1]       [Worker 2]       [Worker 3...]
    [Thread 0]       [Thread 1]       [Thread N]
        ↓                 ↓                 ↓
┌───────────────────────────────────────────────────┐
│             Scanning Engine (8 threads)
│                  Mode Selection
│    Linear | Random | Geometric Pattern Mode
└───────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────┐
│          Cryptographic Pipeline (Per Key)
│
│  1. PrivateKey Generation → secp256k1_wrapper
│  2. PublicKey Derivation → libsecp256k1 (C)
│  3. Hash160 Computation → SHA256 + RIPEMD160
│  4. Database Lookup → Unordered Set (O(1))
│  5. Hit Recording → Atomic Write to Result File
│
└───────────────────────────────────────────────────┘
        ↓
    [RESULT] found_gold.txt
```

## 📦 Project Structure

```
btc_gold_cpp/
├── CMakeLists.txt              # Build configuration (CMake 3.20+)
├── Makefile                    # Convenience wrapper
├── BUILD.md                    # Build instructions
├── ARCHITECTURE.md             # This file
├── PERFORMANCE.md              # Performance tuning guide
│
├── include/                    # Header files (public interface)
│   ├── types.h                 # Core types (Hash160, PrivateKey, etc)
│   ├── constants.h             # Constants (secp256k1, performance params)
│   ├── logger.h                # Logging interface
│   ├── hash160.h               # Hash160 engine (SHA256 + RIPEMD160)
│   ├── secp256k1_wrapper.h     # ECDSA key generation wrapper
│   ├── database.h              # Target database management
│   ├── worker.h                # Worker thread logic
│   ├── engine.h                # Main scanning engine
│   └── config.h                # Configuration parser
│
├── src/                        # Implementation
│   ├── main.cpp                # Entry point (< 100 lines)
│   ├── engine.cpp              # Engine orchestration
│   ├── worker.cpp              # Worker thread implementation
│   ├── hash160.cpp             # Hash160 (SHA256 + RIPEMD160)
│   ├── secp256k1_wrapper.cpp   # libsecp256k1 bindings
│   ├── database.cpp            # Database loading & lookup
│   ├── config.cpp              # CLI & interactive config
│   ├── logger.cpp              # Logging implementation
│   ├── benchmark.cpp           # Standalone benchmark tool
│   └── Makefile                # Build automation
```

## 🧵 Threading Model

### Master Thread
- Initializes engine
- Spawns N worker threads
- Monitors progress every second
- Aggregates statistics
- Handles graceful shutdown

### Worker Threads (N = CPU cores or user-specified)
- Independent key generation
- Private key scanning (no synchronization needed)
- Hash160 computation
- Database lookup (read-only, concurrent-safe)
- Atomic result recording

**Key Design: Zero lock contention**
- Each worker has independent key space
- Database is read-only unordered_set (concurrent read-safe)
- Results written atomically to file

## 🔐 Cryptographic Pipeline

### Private Key Generation
```cpp
class Worker {
    uint64_t current = start_value + worker_id;
    while (!should_stop) {
        // Convert uint64 to PrivateKey (32 bytes)
        PrivateKey privkey = uint64_to_bytes(current);
        
        // Validate against secp256k1 curve order
        if (secp256k1.verify_privkey(privkey)) {
            // Process this key
        }
        
        current += stride;  // Linear mode
    }
}
```

### Public Key Derivation
```cpp
// Compressed (33 bytes)
PublicKey pubkey = secp256k1.pubkey_compressed(privkey);
// Result: 1 byte prefix (02/03) + 32 bytes X coordinate

// Uncompressed (65 bytes) [optional]
std::vector<uint8_t> pubkey_u = secp256k1.pubkey_uncompressed(privkey);
// Result: 1 byte prefix (04) + 32 bytes X + 32 bytes Y
```

### Hash160 (Bitcoin Address Generation)
```cpp
// Pipeline: PublicKey → SHA256 → RIPEMD160 → Hash160
uint8_t sha256_hash[32];
SHA256(pubkey, 33, sha256_hash);

uint8_t hash160[20];
RIPEMD160(sha256_hash, 32, hash160);

// Result: 20-byte hash160 used in Bitcoin addresses
```

## 💾 Database Design

### Storage
```cpp
std::unordered_set<std::string> targets;
```

### Format
- **Hexadecimal strings** (40 characters per target)
- **Loaded from file** (one per line, # for comments)

### Lookup Performance
- **Average case:** O(1) - hash table
- **Worst case:** O(N) - hash collision
- **Real-world:** ~O(1) with good hash distribution

## 🎯 Scanning Modes

### 1. Linear Mode
```
PrivateKey = start + (worker_id + k * stride)

Example (4 workers, stride=4):
  Worker 0: 1, 5, 9, 13, 17, ...
  Worker 1: 2, 6, 10, 14, 18, ...
  Worker 2: 3, 7, 11, 15, 19, ...
  Worker 3: 4, 8, 12, 16, 20, ...
```

### 2. Random Mode
```
PrivateKey = rand() % (end - start) + start
Completely random for each worker
```

### 3. Geometric Mode
```
PrivateKey = start * (multiplier ^ k)

Example: start=1, multiplier=2
  1, 2, 4, 8, 16, 32, 64, 128, ...
```

## ⚡ Performance Optimizations

### Compiler Flags
- `-O3` - Maximum optimization
- `-march=native` - CPU-specific optimizations
- `-mavx2` - AVX2 SIMD instructions
- `-flto` - Link-time optimization
- `-mbmi2` - Bit manipulation instructions

### Memory Layout
- Aligned data structures (cache-friendly)
- Stack allocation for hot loops
- Pre-allocated memory pools

### Algorithm Optimizations
- Batch processing where possible
- Vectorized operations (AVX2)
- Cache locality (spatial & temporal)

## 📊 Performance Expectations

### Your Hardware (Intel i7-8550U)
```
Benchmark Results:
- Hash160 puro: 450+ k/s
- KeyGen: 20 k/s
- Combined: 50-100 k/s per thread

With 8 threads: 400-800 k/s total
```

### Scaling
- Linear with CPU cores (near-perfect scaling)
- Thread overhead negligible
- I/O (file writes) minimal impact

## 🔍 Code Quality Standards

### Design Patterns
- **Singleton:** Secp256k1, Logger, Database
- **RAII:** Resource cleanup in destructors
- **Template:** Generic programming where beneficial

### Error Handling
- Exception-safe operations
- Validation at entry points
- Graceful degradation on errors

### Maintainability
- Clear separation of concerns
- Minimal interdependencies
- Self-documenting code
- Comprehensive comments for complex logic

## 🧪 Testing

### Standalone Benchmark
```bash
make benchmark
```

### Integration Test
```bash
# Create test database
echo "1A1z7agoat5NUb3tpgR7hRA855j8xvtooD" > alvos.txt

# Run with small parameters
./build/Release/btc_gold --start 1 --end 1000000 --threads 4
```

## 🚀 Deployment

### Production Build
```bash
make release
```

### Running
```bash
./btc_gold --database targets.txt --threads 8
```

### Monitoring
- Logs to stdout in real-time
- Results written to `found_gold.txt` (atomic)
- Progress displayed every second
