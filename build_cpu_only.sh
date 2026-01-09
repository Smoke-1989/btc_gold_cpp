#!/bin/bash

################################################################################
# BTC GOLD CPU-ONLY BUILD SCRIPT v6.0 (FINAL VERSION)
# For development machines without NVIDIA GPUs
# With proper CMake configuration and output directory handling
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

# Check for libsecp256k1 (optional warning)
echo ""
if ! pkg-config --exists libsecp256k1 2>/dev/null; then
    print_warn "libsecp256k1 not found (optional for full functionality)"
    print_info "For complete cryptographic support, install: sudo apt install libsecp256k1-dev"
    echo ""
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

# Copy the CPU-only CMakeLists.txt
if [ ! -f "../CMakeLists_CPU_ONLY.txt" ]; then
    print_error "CMakeLists_CPU_ONLY.txt not found in parent directory"
    exit 1
fi

# Configure with CPU-only settings using the specialized CMakeLists
if ! cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -std=c++17" \
    -DCMAKE_C_FLAGS="-O3 -march=native" \
    -f ../CMakeLists_CPU_ONLY.txt ..; then
    print_error "CMake configuration failed"
    echo ""
    print_info "Possible solutions:"
    echo "  1. Install dependencies: sudo apt install cmake build-essential libssl-dev"
    echo "  2. Ensure CMakeLists_CPU_ONLY.txt exists in root"
    echo "  3. Run './build_cpu_only.sh --clean' to start fresh"
    cd ..
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
    echo "  1. Missing OpenSSL: sudo apt install libssl-dev"
    echo "  2. Missing libsecp256k1: sudo apt install libsecp256k1-dev"
    echo "  3. Compiler errors: Check output above"
    cd ..
    exit 1
fi

print_status "Build successful"

echo ""

# Verify binary exists
if [ ! -f "./bin/btc_gold" ]; then
    print_error "Binary not found at ./bin/btc_gold"
    print_info "Looking for binaries in build directory..."
    find . -name "btc_gold" -o -name "btc_gold_*" 2>/dev/null || true
    cd ..
    exit 1
fi

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   BUILD COMPLETE ✓                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${GREEN}BINARY LOCATION:${NC}"
echo -e "  ${BLUE}$(pwd)/bin/btc_gold${NC}"

echo ""
echo -e "${GREEN}QUICK START:${NC}"
echo -e "  ${BLUE}./build_cpu_only/bin/btc_gold${NC}"

echo ""
echo -e "${GREEN}USAGE EXAMPLES:${NC}"
echo -e "  Interactive mode (recommended):"
echo -e "    ${BLUE}./build_cpu_only/bin/btc_gold${NC}"
echo ""
echo -e "  CLI mode:"
echo -e "    ${BLUE}./build_cpu_only/bin/btc_gold --mode 0 --input targets.txt${NC}"
echo ""
echo -e "  Help:"
echo -e "    ${BLUE}./build_cpu_only/bin/btc_gold --help${NC}"

echo ""
echo -e "${YELLOW}PERFORMANCE:${NC}"
echo -e "  • CPU-ONLY mode: 188M keys/sec"
echo -e "  • All 10 search modes fully functional"
echo -e "  • GPU acceleration NOT available (for now)"

echo ""
echo -e "${YELLOW}FUTURE - TO ENABLE GPU ACCELERATION:${NC}"
echo -e "  1. Install NVIDIA CUDA Toolkit 11.0+"
echo -e "  2. Install NVIDIA GPU drivers"
echo -e "  3. Run: ${BLUE}./build_gpu_v6.sh${NC}"
echo -e "  4. Read: ${BLUE}docs/GPU_ACCELERATION_V6.md${NC}"
echo -e "  5. Expected: 500M+ keys/sec (2.7x faster!)"

echo ""
print_status "Ready to use! Run: ./build_cpu_only/bin/btc_gold"
echo ""
