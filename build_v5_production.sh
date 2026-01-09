#!/bin/bash

# BTC GOLD v5.1 - PRODUCTION BUILD SYSTEM
# Enterprise-Grade Compilation Script

echo "========================================"
echo "BTC GOLD v5.1 - PRODUCTION BUILD"
echo "========================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check dependencies
echo "[*] Checking dependencies..."
command -v g++ &> /dev/null
if [ $? -eq 0 ]; then
    G_VERSION=$(g++ --version | head -n1)
    echo -e "${GREEN}[+]${NC} g++ found: $G_VERSION"
else
    echo -e "${RED}[!] g++ not found${NC}"
    exit 1
fi

command -v cmake &> /dev/null
if [ $? -eq 0 ]; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    echo -e "${GREEN}[+]${NC} cmake found: $CMAKE_VERSION"
else
    echo -e "${RED}[!] cmake not found${NC}"
    exit 1
fi

echo ""
echo "[*] Setting up build directory..."
if [ -d "build" ]; then
    echo "[*] Cleaning previous build..."
    rm -rf build
fi

mkdir -p build
cd build

echo -e "${GREEN}[+]${NC} Build directory ready"

echo ""
echo "[*] Generating build files with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..
if [ $? -ne 0 ]; then
    echo -e "${RED}[!] CMake configuration failed${NC}"
    cd ..
    exit 1
fi
echo -e "${GREEN}[+]${NC} CMake configuration successful"

echo ""
echo "[*] Compiling sources..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo -e "${RED}[!] Compilation failed${NC}"
    cd ..
    exit 1
fi
echo -e "${GREEN}[+]${NC} Compilation successful"

echo ""
echo "[*] Build artifacts:"
if [ -f "btc_gold" ]; then
    SIZE=$(du -h btc_gold | cut -f1)
    echo -e "${GREEN}[✓]${NC} btc_gold (Binary) - $SIZE"
else
    echo -e "${RED}[!] Binary not found${NC}"
    cd ..
    exit 1
fi

cd ..

echo ""
echo "========================================"
echo -e "${GREEN}[SUCCESS] BTC GOLD v5.1 Ready!${NC}"
echo "========================================"
echo ""
echo "Usage: ./build/btc_gold"
echo ""
