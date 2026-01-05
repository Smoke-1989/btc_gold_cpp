# 🖫 BTC GOLD v4.0 - NEW MODES PRACTICAL GUIDE

## Overview

v4.0 introduces **3 new specialized attack modes** beyond the original 4. Each mode targets specific patterns or scenarios.

---

## 📘 QUICK REFERENCE

| Mode | Command | Best For | Speed |
|------|---------|----------|-------|
| **1** | `--mode linear` | Ranges (Puzzles 66-76) | 50M+ k/s |
| **2** | `--mode random` | Full 256-bit (luck) | 36k k/s |
| **3** | `--mode geometric` | Multi-phase borders | 5-50M k/s |
| **4** | `--mode terminator` | Multiplicative desc | 1-10M k/s |
| **5** | `--mode doubling` | 2^bit patterns (NEW) | 50M+ k/s |
| **6** | `--mode hamming` | 2-bit combinations (NEW) | 1M k/s |
| **7** | `--mode modular-stride` | Arithmetic prog (NEW) | 50M+ k/s |

---

# MODE 5: DOUBLING 📘

## What It Does

Searches for keys that are powers of 2 or related to exponential growth patterns.

**Mathematical Pattern:**
```
Test keys: 2^(min-1), 2^min, 2^(min+1), ..., 2^max
Range: [2^(bit-1), 2^bit)
```

## Use Cases

### Case 1: Bitcoin Puzzle #66-#76
```bash
./build/btc_gold \
    --mode doubling \
    --min-bit 66 \
    --max-bit 76 \
    --scan-mode 1 \
    --threads 8

# Tests: 2^65, 2^66, 2^67, ..., 2^76
# Speed: 50M+ k/s
# Time: <1 second for 11 keys
```

### Case 2: Search Range 2^100 to 2^120
```bash
./build/btc_gold \
    --mode doubling \
    --min-bit 100 \
    --max-bit 120 \
    --scan-mode 1

# Comprehensive coverage of powers of 2
```

### Case 3: Targeted Range (Puzzle 68)
```bash
./build/btc_gold \
    --mode doubling \
    --min-bit 68 \
    --max-bit 69 \
    --scan-mode 1

# Very fast: just 2 exponents (2^67, 2^68)
```

## Advanced: Understanding the Range

```
if user specifies: --min-bit 66 --max-bit 67

The doubling mode will test:
  - Base: 2^(66-1) = 2^65
  - Range: 2^65 to 2^67
  - Keys tested: 2^65, 2^66, 2^67
  - Scan window: 1,000,000 keys per bit
```

---

# MODE 6: HAMMING (Low-Weight Keys) 📘

## What It Does

Searches for keys with only **2 or 3 bits set** to 1.

**Mathematical Pattern:**
```
Test keys of form: 2^bit1 + 2^bit2 + (optional 2^bit3)
Total combinations: C(n, 2) + C(n, 3) where n = bit_range
```

## Why Useful?

Some people generate keys with patterns like:
- `2^66 + 2^33` (2 bits: bit 66 and bit 33 set)
- Reduced entropy but memorable

## Use Cases

### Case 1: Full Range (1-256 bits)
```bash
./build/btc_gold \
    --mode hamming \
    --min-bit 1 \
    --max-bit 256 \
    --scan-mode 1

# Warning: This tests ALL 2-bit combinations
# ~32,000 keys to check
# Speed: Fast (combinatorial, not throughput-based)
# Time: ~5-10 seconds for all combinations
```

### Case 2: Narrow Range (60-70 bits)
```bash
./build/btc_gold \
    --mode hamming \
    --min-bit 60 \
    --max-bit 70 \
    --scan-mode 1

# Test: 2^60+2^61, 2^60+2^62, ..., 2^69+2^70
# Combinations: C(10,2) = 45 keys
# Speed: Instant (~100ms)
```

### Case 3: Precision Target
```bash
./build/btc_gold \
    --mode hamming \
    --min-bit 64 \
    --max-bit 68 \
    --scan-mode 1

# Search: All combinations of 2 bits between 64-68
# Total: C(5,2) = 10 combinations
# Time: Milliseconds
```

## Hamming Weight Explained

```
A "Hamming weight" is the number of 1-bits in binary.

Example:
  0x100 (binary: 1_00000000) = Hamming weight 1 (only 1 bit set)
  0x300 (binary: 11_0000000) = Hamming weight 2 (two bits set)
  0x700 (binary: 111_000000) = Hamming weight 3 (three bits set)

Mode HAMMING searches for weight 2 keys (most common pattern).
```

---

# MODE 7: MODULAR STRIDE 📘

## What It Does

Searches for keys in an **arithmetic progression**: k, k+m, k+2m, k+3m, ...

**Mathematical Pattern:**
```
Test keys: start, start+multiplier, start+2*multiplier, ...
Condition: key <= end_value
```

## Use Cases

### Case 1: Every 7th Number
```bash
./build/btc_gold \
    --mode modular-stride \
    --start 1000 \
    --multiplier 7 \
    --end 1000000 \
    --threads 8

# Tests: 1000, 1007, 1014, 1021, 1028, ..., 1000000
# Pattern: 7-spaced numbers
# Speed: 50M+ k/s
```

### Case 2: Every Odd Number (multiplier=2)
```bash
./build/btc_gold \
    --mode modular-stride \
    --start 1 \
    --multiplier 2 \
    --end 1000000

# Tests: 1, 3, 5, 7, 9, ..., 999999
# Pattern: All odd numbers
# Speed: 50M+ k/s
```

### Case 3: Every Prime-Spaced Key
```bash
./build/btc_gold \
    --mode modular-stride \
    --start 2 \
    --multiplier 11 \
    --end 10000000

# Tests: 2, 13, 24, 35, 46, ..., 10000000
# Pattern: 11-spaced numbers (starting from 2)
# Speed: 50M+ k/s
```

### Case 4: Cover All Within Range with Threads
```bash
# Single thread covers: start, start+m, start+2m, ...
# With 8 threads:
#   Thread 0: start, start+m, start+2m, ...
#   Thread 1: start+1, start+m+1, start+2m+1, ...
#   Thread 2: start+2, start+m+2, start+2m+2, ...
#   ...
#   Thread 7: start+7, start+m+7, start+2m+7, ...

./build/btc_gold \
    --mode modular-stride \
    --start 0 \
    --multiplier 100000000 \
    --end 18446744073709551615 \
    --threads 8

# Each thread covers a different residue class
# Perfect for systematic range coverage
```

## When to Use Modular Stride

| Scenario | Config | Example |
|----------|--------|----------|
| Sequential search | multiplier=1 | Start 1M, mult 1, end 2M |
| Even numbers only | multiplier=2 | Start 0, mult 2 |
| Fibonacci-like | multiplier=φ≈1.618 | Start 1, mult 2 (approx) |
| Custom pattern | multiplier=N | Your specific spacing |

---

# 🎉 PRACTICAL RECIPES

## Recipe 1: Puzzle Chain Search (66-76)
```bash
#!/bin/bash
# Search Bitcoin Puzzles #66 through #76 sequentially

for bit in {66..76}; do
    echo "[*] Searching Puzzle #$bit (2^$((bit-1)) to 2^$bit)"
    
    ./build/btc_gold \
        --mode linear \
        --start $((2**($bit-1))) \
        --end $((2**$bit - 1)) \
        --scan-mode 1 \
        --threads 8
    
    if [ -s found_gold.txt ]; then
        echo "[+] FOUND! Check found_gold.txt"
        break
    fi
done
```

## Recipe 2: Comprehensive Bit-Weight Sweep
```bash
#!/bin/bash
# Test all low-hamming keys in a range

echo "Testing 2-bit combinations (bits 1-100)..."
./build/btc_gold --mode hamming --min-bit 1 --max-bit 100

echo "Testing 3-bit combinations (bits 50-70)..."
for bit1 in {50..70}; do
    for bit2 in $(seq $((bit1+1)) 70); do
        # Manual 3-bit testing (future: native support)
        echo "  Testing: 2^$bit1 + 2^$bit2"
    done
done
```

## Recipe 3: Multi-Mode Campaign
```bash
#!/bin/bash
# Try all modes against your target

TARGETS="alvos.txt"
LOG="campaign.log"

echo "[*] Starting multi-mode search campaign" | tee $LOG
date | tee -a $LOG

echo "\n[MODE 1] Linear" | tee -a $LOG
./build/btc_gold --mode linear --start 1 --end 1000000 --threads 8 | tee -a $LOG

echo "\n[MODE 5] Doubling (bits 60-70)" | tee -a $LOG
./build/btc_gold --mode doubling --min-bit 60 --max-bit 70 | tee -a $LOG

echo "\n[MODE 6] Hamming Weight" | tee -a $LOG
./build/btc_gold --mode hamming --min-bit 1 --max-bit 100 | tee -a $LOG

echo "\n[MODE 7] Modular Stride (every 7th)" | tee -a $LOG
./build/btc_gold --mode modular-stride --start 1 --multiplier 7 --end 1000000 | tee -a $LOG

echo "\n[*] Campaign complete" | tee -a $LOG
```

## Recipe 4: Performance Comparison
```bash
#!/bin/bash
# Benchmark all modes

echo "=== v4.0 MODE PERFORMANCE ==="

echo "[1] Linear (50M+ k/s expected)"
time ./build/btc_gold --mode linear --start 1 --end 1000000 --threads 8

echo ""
echo "[5] Doubling (50M+ k/s expected)"
time ./build/btc_gold --mode doubling --min-bit 60 --max-bit 70 --threads 8

echo ""
echo "[6] Hamming (combinatorial)"
time ./build/btc_gold --mode hamming --min-bit 1 --max-bit 100 --threads 8

echo ""
echo "[7] Modular Stride (50M+ k/s expected)"
time ./build/btc_gold --mode modular-stride --start 1 --multiplier 1000 --end 10000000 --threads 8
```

---

# 🌐 COMBINING MODES

## Strategy: Systematic Coverage

Use different modes for different assumptions about the target:

```
[Phase 1] Linear Mode
- Assumption: Sequential/obvious key
- Coverage: Range 1 to 2^66
- Time: Hours (if key nearby)

[Phase 2] Doubling Mode
- Assumption: Powers of 2 or exponent-based
- Coverage: Bits 1-256
- Time: Seconds

[Phase 3] Hamming Mode
- Assumption: Low-weight/memorable key (2-3 bits)
- Coverage: All 2-bit combinations (1-256)
- Time: Seconds

[Phase 4] Modular Stride
- Assumption: Arithmetic pattern (multiples)
- Coverage: Custom pattern
- Time: Variable
```

---

# ⚠️ IMPORTANT NOTES

1. **Database Format**: All modes expect `alvos.txt` in standard format
   - One entry per line (Address, HASH160, or PubKey)
   - Use `--input-type` flag to specify format

2. **Thread Allocation**: Each mode distributes threads optimally
   - Linear/Doubling/Modular: Full thread parallelism
   - Hamming: Single-threaded (combinatorial, not parallel)
   - Geometric: Custom per-bit distribution

3. **Performance Expectations**:
   - Linear-based (1,5,7): 50M+ k/s per thread
   - Random (2): 36k k/s baseline
   - Hamming (6): N/A (combinations, not throughput)

4. **Stop Conditions**:
   - Use `--stop-on-find` to exit immediately on first hit
   - Default: continues finding all matches

---

# 💬 EXAMPLES OUTPUT

### Linear Mode
```
[RUNNING] Speed: 45,852,123 k/s | Total: 1,000,000,000 | Time: 21s
[FOUND] 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
```

### Doubling Mode
```
[DOUBLING] Starting bit: 66 | Max bit: 76
[FOUND] 1A1z...[high-bit key]
```

### Hamming Mode
```
[HAMMING] Low-weight key search (2-3 bits)
[FOUND] 1C2...[2-bit key 2^66+2^33]
```

---

# 🚀 NEXT STEPS

1. **Build**: `chmod +x build_v4.sh && ./build_v4.sh`
2. **Try modes**: Test each with small ranges first
3. **Optimize**: Adjust `--multiplier` and range parameters
4. **Scale**: Increase thread count and range as needed
5. **Monitor**: Check `found_gold.txt` for results

---

**Questions?** See `QUICK_START.md` or `UPGRADE_NOTES_v4.0.md`
