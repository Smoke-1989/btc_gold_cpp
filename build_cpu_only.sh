#!/bin/bash

################################################################################
# BTC GOLD CPU-ONLY BUILD SCRIPT v6.0 (FINAL FIX)
# For development machines without NVIDIA GPUs
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build_cpu_only"
INSTALL_PREFIX="."  # Will install to build_cpu_only/bin
MAX_JOBS=${1:-$(nproc)}
CLEAN=${2:-"false"}

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       BTC GOLD CPU-ONLY BUILD SYSTEM v6.0                ║${NC}"
echo -e "${BLUE}║   For development without GPU hardware                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "${YELLOW}[i]${NC} Configuration:"
echo -e "${YELLOW}[i]${NC}   Build Dir: $BUILD_DIR"
echo -e "${YELLOW}[i]${NC}   Jobs: $MAX_JOBS"
echo -e "${YELLOW}[i]${NC}   GPU Support: DISABLED"
echo ""

# Check prerequisites
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}[✗] CMake not found. Install: sudo apt install cmake${NC}"
    exit 1
fi

# Clean if requested
if [[ "$CLEAN" == "--clean" ]]; then
    echo -e "${YELLOW}[i] Cleaning...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Create and enter build dir
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo -e "${YELLOW}[*] Configuring CMake (Standard CPU Mode)...${NC}"

# We use the standard CMakeLists.txt which is CPU-only by default
# We do NOT use -f or custom file, just the standard one in root
if ! cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -std=c++17" \
    ..; then
    echo -e "${RED}[✗] Configuration failed.${NC}"
    exit 1
fi

echo -e "${YELLOW}[*] Compiling...${NC}"
if ! cmake --build . --config Release -j$MAX_JOBS; then
    echo -e "${RED}[✗] Build failed.${NC}"
    exit 1
fi

echo -e "${YELLOW}[*] Installing to bin/...${NC}"
# This moves the binary to build_cpu_only/bin/btc_gold
if ! make install; then
    echo -e "${RED}[✗] Install failed.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   BUILD COMPLETE ✓                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "${GREEN}BINARY LOCATION:${NC}"
echo -e "  ${BLUE}$(pwd)/bin/btc_gold${NC}"

echo -e "${GREEN}QUICK START:${NC}"
echo -e "  ${BLUE}./build_cpu_only/bin/btc_gold${NC}"

echo ""
echo -e "${YELLOW}NOTES:${NC}"
echo -e "  • Running in CPU-ONLY mode"
echo -e "  • Uses standard CMakeLists.txt"
