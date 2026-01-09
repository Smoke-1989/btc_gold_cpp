#!/bin/bash

# BTC GOLD v5.1 - Build Script
# Purpose: Compile and link all components with full optimization

set -e  # Exit on error

echo "========================================"
echo "BTC GOLD v5.1 - Build System"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# Configuration
BUILD_DIR="build"
BINARY_NAME="btc_gold"
COMPILER="g++"
STANDARD="-std=c++17"
OPTIMIZATION="-O3"
WARNINGS="-Wall -Wextra -Wpedantic -Wno-unused-parameter"
DEBUG="-g"
THREADS=$(nproc)

# Check dependencies
echo -e "${YELLOW}[*] Checking dependencies...${NC}"
command -v $COMPILER >/dev/null 2>&1 || { echo -e "${RED}[!] $COMPILER not found${NC}"; exit 1; }
echo -e "${GREEN}[+] $COMPILER found${NC}"

command -v cmake >/dev/null 2>&1 && echo -e "${GREEN}[+] cmake found${NC}" || echo -e "${YELLOW}[*] cmake not found (optional)${NC}"

# Create build directory
echo ""
echo -e "${YELLOW}[*] Setting up build directory...${NC}"
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}[*] Cleaning previous build...${NC}"
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
echo -e "${GREEN}[+] Build directory ready${NC}"

# Compile sources
echo ""
echo -e "${YELLOW}[*] Compiling sources...${NC}"

SOURCES=(
    "../src/main_v5.cpp"
    "../src/worker_engine.cpp"
    "../src/worker_v5.cpp"
    "../src/interactive_menu.cpp"
    "../src/config.cpp"
    "../src/logger.cpp"
    "../src/secp256k1.cpp"
    "../src/hash_utils.cpp"
)

OBJECTS=()

for source in "${SOURCES[@]}"; do
    if [ -f "$source" ]; then
        filename=$(basename "$source")
        object="${filename%.cpp}.o"
        echo -e "${YELLOW}[*] Compiling $filename...${NC}"
        
        $COMPILER \
            $STANDARD \
            $OPTIMIZATION \
            $DEBUG \
            $WARNINGS \
            -fPIC \
            -pthread \
            -I../include \
            -c "$source" -o "$object" 2>&1 | head -20
        
        if [ -f "$object" ]; then
            echo -e "${GREEN}[+] $object created${NC}"
            OBJECTS+=("$object")
        else
            echo -e "${RED}[!] Failed to compile $source${NC}"
        fi
    else
        echo -e "${YELLOW}[*] Source not found: $source (skipping)${NC}"
    fi
done

# Link objects
echo ""
echo -e "${YELLOW}[*] Linking objects...${NC}"

if [ ${#OBJECTS[@]} -gt 0 ]; then
    $COMPILER \
        $OPTIMIZATION \
        $DEBUG \
        -pthread \
        -o "$BINARY_NAME" \
        "${OBJECTS[@]}" \
        -lm -lcrypto -lssl 2>&1
    
    if [ -f "$BINARY_NAME" ]; then
        echo -e "${GREEN}[+] Executable created: $BINARY_NAME${NC}"
        SIZE=$(du -h "$BINARY_NAME" | cut -f1)
        echo -e "${GREEN}[+] Binary size: $SIZE${NC}"
    else
        echo -e "${RED}[!] Linking failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[*] No object files to link${NC}"
    echo -e "${YELLOW}[*] Compiling single file...${NC}"
    
    $COMPILER \
        $STANDARD \
        $OPTIMIZATION \
        $DEBUG \
        $WARNINGS \
        -pthread \
        -I../include \
        -o "$BINARY_NAME" \
        ../src/main_v5.cpp \
        -lm -lcrypto -lssl 2>&1
fi

# Verify executable
echo ""
echo -e "${YELLOW}[*] Verifying executable...${NC}"
if file "$BINARY_NAME" | grep -q "ELF"; then
    echo -e "${GREEN}[+] Valid ELF binary${NC}"
else
    echo -e "${YELLOW}[*] Binary type: $(file $BINARY_NAME)${NC}"
fi

if ldd "$BINARY_NAME" 2>/dev/null | grep -q "not found"; then
    echo -e "${RED}[!] Missing dependencies!${NC}"
    ldd "$BINARY_NAME"
else
    echo -e "${GREEN}[+] All dependencies satisfied${NC}"
fi

# Create symlink in parent directory
echo ""
echo -e "${YELLOW}[*] Creating symlink...${NC}"
cd ..
ln -sf "$BUILD_DIR/$BINARY_NAME" "$BINARY_NAME"
echo -e "${GREEN}[+] Symlink created: ./$BINARY_NAME${NC}"

# Build summary
echo ""
echo "========================================"
echo -e "${GREEN}BUILD COMPLETE${NC}"
echo "========================================"
echo ""
echo "Binary: $(pwd)/$BINARY_NAME"
echo "Size: $(du -h $BINARY_NAME | cut -f1)"
echo ""
echo "Usage:"
echo "  Interactive mode: ./$BINARY_NAME"
echo "  CLI mode:         ./$BINARY_NAME --mode <0-9> --input file.txt"
echo "  Help:             ./$BINARY_NAME --help"
echo ""
echo "Next steps:"
echo "  1. Prepare your target file (hash160p.txt)"
echo "  2. Run: ./$BINARY_NAME"
echo "  3. Select mode and configure parameters"
echo "  4. Start scan"
echo ""
