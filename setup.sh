#!/bin/bash

# ============================================================================
# BTC GOLD SETUP SCRIPT
# Downloads and builds dependencies (secp256k1)
# ============================================================================

set -e

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  🛠️  BTC GOLD DEPENDENCY SETUP"
echo "═══════════════════════════════════════════════════════════════════════════"

# Clean previous installation
if [ -d "secp256k1" ]; then
    echo "[1/4] Removing existing secp256k1..."
    rm -rf secp256k1
fi

# Clone secp256k1
echo "[2/4] Cloning bitcoin-core/secp256k1..."
git clone https://github.com/bitcoin-core/secp256k1.git
cd secp256k1
git checkout ac83be33d0956faf6b7f61a60ab524ef7d6a473a  # Stable commit

# Build
echo "[3/4] Building secp256k1..."
./autogen.sh
./configure \
    --enable-experimental \
    --enable-module-recovery \
    --enable-module-ecdh \
    --enable-endomorphism \
    --with-bignum=no \
    --enable-static \
    --disable-shared \
    --disable-tests \
    --disable-benchmark

make -j$(nproc)

echo "[4/4] Dependency build complete."
cd ..

echo ""
echo "✅ Setup finished successfully."
echo "   Now run: ./build_v4.sh"
echo ""
