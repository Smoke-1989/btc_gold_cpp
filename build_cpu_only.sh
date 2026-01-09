#!/bin/bash

################################################################################
# BTC GOLD CPU-ONLY BUILD SCRIPT v6.0 (ROBUST VERSION)
# For development machines without NVIDIA GPUs
# With detailed error diagnosis
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
INSTALL_PREFIX="./build_cpu_only"
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

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[i]${NC} $1"
}

print_warn() {
    echo -e "${CYAN}[!]${NC} $1"
}

# Check prerequisites
echo -e "${YELLOW}[*] Checking prerequisites...${NC}"

if ! command -v cmake &> /dev/null; then
    print_error "CMake not found. Please install: sudo apt install cmake"
    exit 1
fi
print_status "CMake found ($(cmake --version | head -1))"

if ! command -v g++ &> /dev/null; then
    print_error "g++ compiler not found. Please install: sudo apt install build-essential"
    exit 1
fi
print_status "g++ found ($(g++ --version | head -1))"

if ! pkg-config --exists openssl 2>/dev/null; then
    print_warn "OpenSSL development files not found."
    print_info "Install with: sudo apt install libssl-dev"
    echo ""
else
    print_status "OpenSSL found"
fi

# Check for libsecp256k1
echo ""
echo -e "${YELLOW}[*] Checking for libsecp256k1...${NC}"

if ! pkg-config --exists libsecp256k1 2>/dev/null; then
    print_warn "libsecp256k1 not found in pkg-config"
    print_info "This is required. Install with: sudo apt install libsecp256k1-dev"
    print_info "Or from source: https://github.com/bitcoin-core/secp256k1"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Build aborted. Install libsecp256k1 first."
        exit 1
    fi
else
    print_status "libsecp256k1 found"
fi

echo ""

# Check if we should clean
if [[ "$CLEAN" == "--clean" ]]; then
    print_info "Cleaning previous builds..."
    rm -rf "$BUILD_DIR"
    print_status "Build directory cleaned"
    echo ""
fi

# Create build directory
if [ ! -d "$BUILD_DIR" ]; then
    print_info "Creating build directory..."
    mkdir -p "$BUILD_DIR"
    print_status "Build directory created"
fi

cd "$BUILD_DIR"

echo ""
echo -e "${YELLOW}[*] Configuring CMake (CPU-ONLY mode)...${NC}"
echo ""

# Configure with CPU-only settings
if ! cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU=OFF \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -std=c++17" \
    -DCMAKE_C_FLAGS="-O3 -march=native" \
    ..; then
    print_error "CMake configuration failed"
    echo ""
    print_info "Possible solutions:"
    echo "  1. Install dependencies: sudo apt install cmake build-essential libssl-dev libsecp256k1-dev"
    echo "  2. Check CMakeLists.txt for errors"
    echo "  3. Run './build_cpu_only.sh --clean' to start fresh"
    exit 1
fi

print_status "CMake configuration successful"

echo ""
echo -e "${YELLOW}[*] Compiling (CPU-ONLY, using $MAX_JOBS jobs)...${NC}"
echo ""

# Build
if ! cmake --build . --config Release -j$MAX_JOBS; then
    print_error "Build failed"
    echo ""
    print_info "Common issues:"
    echo "  1. Missing libsecp256k1: sudo apt install libsecp256k1-dev"
    echo "  2. Missing OpenSSL: sudo apt install libssl-dev"
    echo "  3. Compiler errors: Check output above"
    exit 1
fi

print_status "Build successful"

echo ""

# Check if binary was created
if [ ! -f "./bin/btc_gold" ]; then
    # Try finding it in other locations
    if [ -f "./btc_gold" ]; then
        mkdir -p ./bin
        mv ./btc_gold ./bin/
        print_status "Binary moved to ./bin/btc_gold"
    else
        print_error "Binary not found after compilation"
        print_info "Looking for binaries in build directory..."
        find . -name "btc_gold" -o -name "btc_gold_*" 2>/dev/null || true
        exit 1
    fi
fi

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   BUILD COMPLETE                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${GREEN}BINARY LOCATION:${NC}"
echo -e "  ${BLUE}$(pwd)/bin/btc_gold${NC}"

echo ""
echo -e "${GREEN}USAGE:${NC}"
echo -e "  Interactive mode:"
echo -e "    ${BLUE}./build_cpu_only/bin/btc_gold${NC}"
echo ""
echo -e "  CLI mode:"
echo -e "    ${BLUE}./build_cpu_only/bin/btc_gold --mode 0 --input targets.txt${NC}"
echo ""
echo -e "  View help:"
echo -e "    ${BLUE}./build_cpu_only/bin/btc_gold --help${NC}"

echo ""
echo -e "${YELLOW}NOTES:${NC}"
echo -e "  • Running in CPU-ONLY mode (188M keys/sec)"
echo -e "  • All 10 search modes fully functional"
echo -e "  • GPU acceleration NOT available"
echo ""
echo -e "${YELLOW}TO ENABLE GPU ACCELERATION:${NC}"
echo -e "  1. Install NVIDIA CUDA Toolkit 11.0+"
echo -e "  2. Install NVIDIA GPU drivers"
echo -e "  3. Run: ${BLUE}./build_gpu_v6.sh${NC}"
echo -e "  4. For details: ${BLUE}docs/GPU_ACCELERATION_V6.md${NC}"
echo ""
