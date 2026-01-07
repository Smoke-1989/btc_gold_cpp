#!/bin/bash

# ============================================================================
# BTC GOLD C++ v4.1 - ENTERPRISE EDITION BUILD SCRIPT
# Full 256-bit Linear Mode Support with BigInt256 Arithmetic
# ============================================================================

set -e  # Exit on error

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  🔥 BTC GOLD C++ v4.1 EXTERMINATOR - ENTERPRISE BUILD 🔥"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# Cleanup
echo "[1/5] Cleaning previous build..."
rm -rf build/
mkdir -p build

# Check for secp256k1
echo "[2/5] Checking dependencies..."
if [ ! -d "secp256k1/.libs" ]; then
    echo "⚠️  secp256k1 not built. Run ./setup.sh first."
    exit 1
fi

echo "✅ Dependencies OK"

# Compile
echo "[3/5] Compiling with full 256-bit support..."
g++ -std=c++17 -O3 -march=native -pthread \
    -Wall -Wextra \
    -Iinclude \
    -Isecp256k1/include \
    -o build/btc_gold \
    src/main.cpp \
    src/config.cpp \
    src/database.cpp \
    src/logger.cpp \
    src/secp256k1_wrapper.cpp \
    src/worker_v4.cpp \
    -Lsecp256k1/.libs -lsecp256k1 \
    -lcrypto -lssl

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed"
    exit 1
fi

echo "✅ Compilation successful"

# Verify
echo "[4/5] Verifying binary..."
if [ ! -f "build/btc_gold" ]; then
    echo "❌ Binary not found"
    exit 1
fi

chmod +x build/btc_gold
echo "✅ Binary verified"

# Summary
echo "[5/5] Build complete!"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  ✅ BTC GOLD v4.1 READY - FULL 256-BIT LINEAR MODE ENABLED"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "NEW FEATURES:"
echo "  • ✅ Full 256-bit linear range support (0x0 to 0xFFFF...FFFF)"
echo "  • ✅ BigInt256 arithmetic engine"
echo "  • ✅ Enterprise-grade TURBO mode with Point Addition"
echo "  • ✅ All 7 search modes fully operational"
echo ""
echo "USAGE:"
echo "  ./build/btc_gold --help"
echo ""
echo "EXAMPLE (256-bit range):"
echo "  ./build/btc_gold --mode linear --input hash160p.txt \\"
echo "    --start-hex 3fffffffffffffffff --end-hex 7fffffffffffffffff \\"
echo "    --threads 8 --verbose"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
