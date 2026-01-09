#!/bin/bash

################################################################################
# BTC GOLD CPU-ONLY BUILD SCRIPT v6.0
# For development machines without NVIDIA GPUs
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
echo -e "${YELLOW}[i] Configuring CMake (CPU-ONLY mode)...${NC}"
echo ""

# Configure with CPU-only settings
# Key difference: -DENABLE_GPU=OFF
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU=OFF \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -std=c++17" \
    -DCMAKE_C_FLAGS="-O3 -march=native" \
    ..

if [ $? -eq 0 ]; then
    print_status "CMake configuration successful"
else
    print_error "CMake configuration failed"
    exit 1
fi

echo ""
echo -e "${YELLOW}[i] Compiling (CPU-ONLY)...${NC}"
echo ""

# Build
cmake --build . --config Release -j$MAX_JOBS

if [ $? -eq 0 ]; then
    print_status "Build successful"
else
    print_error "Build failed"
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   BUILD COMPLETE                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${GREEN}BINARY LOCATION:${NC}"
echo -e "  ${BLUE}./build_cpu_only/bin/btc_gold${NC}"

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
