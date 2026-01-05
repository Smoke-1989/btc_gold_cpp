#!/bin/bash

###############################################################################
# BTC GOLD C++ v4.0 EXTERMINATOR BUILD SCRIPT
# Automated build with optimization verification
###############################################################################

set -e

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  🔥 BTC GOLD C++ v4.0 EXTERMINATOR - BUILD SYSTEM 🔥"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Cores for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

###############################################################################
# STEP 1: CLEANUP
###############################################################################

echo -e "${BLUE}[1/5] CLEANUP${NC}"
if [ -d "build" ]; then
    echo "  Removing old build directory..."
    rm -rf build
fi

echo -e "  ${GREEN}✓ Clean${NC}"
echo ""

###############################################################################
# STEP 2: SYSTEM CHECKS
###############################################################################

echo -e "${BLUE}[2/5] SYSTEM CHECKS${NC}"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo -e "  ${RED}✗ CMake not found. Install with: sudo apt install cmake${NC}"
    exit 1
fi
echo "  ✓ CMake $(cmake --version | head -n1 | cut -d' ' -f3)"

# Check C++ compiler
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo -e "  ${RED}✗ C++ compiler not found${NC}"
    exit 1
fi
echo "  ✓ Compiler found"

# Check OpenSSL
if ! pkg-config --exists openssl; then
    echo -e "  ${RED}✗ OpenSSL not found. Install with: sudo apt install libssl-dev${NC}"
    exit 1
fi
echo "  ✓ OpenSSL $(pkg-config --modversion openssl)"

# Check libsecp256k1
if ! pkg-config --exists libsecp256k1; then
    echo -e "  ${RED}✗ libsecp256k1 not found. Install with: sudo apt install libsecp256k1-dev${NC}"
    exit 1
fi
echo "  ✓ libsecp256k1 $(pkg-config --modversion libsecp256k1)"

# Check CPU capabilities
echo ""
echo -e "  ${YELLOW}CPU Capabilities:${NC}"
if grep -q avx2 /proc/cpuinfo; then
    echo "    ✓ AVX2 supported"
else
    echo -e "    ${YELLOW}⚠ AVX2 not detected (non-critical)${NC}"
fi

if grep -q avx512f /proc/cpuinfo; then
    echo "    ✓ AVX-512 supported"
fi

echo -e "  ${GREEN}✓ All dependencies OK${NC}"
echo ""

###############################################################################
# STEP 3: CMAKE CONFIGURATION
###############################################################################

echo -e "${BLUE}[3/5] CMAKE CONFIGURATION${NC}"

cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic"

echo -e "  ${GREEN}✓ Configuration complete${NC}"
echo ""

###############################################################################
# STEP 4: BUILD
###############################################################################

echo -e "${BLUE}[4/5] COMPILATION${NC}"

NUM_JOBS=$(nproc)
echo "  Using $NUM_JOBS CPU cores for parallel compilation..."
echo ""

if cmake --build build --config Release -j"$NUM_JOBS"; then
    echo ""
    echo -e "  ${GREEN}✓ Build successful${NC}"
else
    echo ""
    echo -e "  ${RED}✗ Build failed${NC}"
    exit 1
fi
echo ""

###############################################################################
# STEP 5: VERIFICATION
###############################################################################

echo -e "${BLUE}[5/5] VERIFICATION${NC}"

# Check binaries exist
if [ -f "build/btc_gold" ] && [ -f "build/btc_gold_benchmark" ]; then
    echo "  ✓ btc_gold binary created"
    echo "  ✓ btc_gold_benchmark binary created"
else
    echo -e "  ${RED}✗ Binary creation failed${NC}"
    exit 1
fi

# Check binary sizes
BTC_GOLD_SIZE=$(du -h build/btc_gold | cut -f1)
echo "  Binary size: $BTC_GOLD_SIZE"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "${GREEN}  ✓ BUILD COMPLETE - v4.0 READY${NC}"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

###############################################################################
# QUICK START GUIDE
###############################################################################

echo -e "${BLUE}QUICK START:${NC}"
echo ""
echo "  1. Run benchmark:"
echo -e "     ${YELLOW}./build/btc_gold_benchmark${NC}"
echo ""
echo "  2. Start scanning (Linear mode):"
echo -e "     ${YELLOW}./build/btc_gold --mode linear --start 1 --scan-mode 1 --threads 8${NC}"
echo ""
echo "  3. Try new modes:"
echo -e "     ${YELLOW}./build/btc_gold --mode doubling --min-bit 66 --max-bit 67${NC}"
echo -e "     ${YELLOW}./build/btc_gold --mode hamming --min-bit 1 --max-bit 100${NC}"
echo -e "     ${YELLOW}./build/btc_gold --mode modular-stride --start 1 --multiplier 3${NC}"
echo ""
echo "  4. View help:"
echo -e "     ${YELLOW}./build/btc_gold --help${NC}"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Optional: Run benchmark automatically
read -p "Run benchmark now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}Running benchmark...${NC}"
    echo ""
    ./build/btc_gold_benchmark
fi

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
