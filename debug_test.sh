#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              BTC GOLD C++ DEBUG TEST                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Create test targets with VALID addresses
cat > test_targets.txt << 'EOF'
# Valid Bitcoin Addresses
1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2
12c6DSiU4Rq3P4ZxziKxzrL5LmMBrzjrJX

# Hash160 Format
62e907b15cbf27d5425399ebf6f0fb50ebb88f18
EOF

echo "[*] Test 1: Check if binary exists"
if [ -f "./build/btc_gold" ]; then
    echo "    ✓ Binary exists: ./build/btc_gold"
    ls -lh ./build/btc_gold
else
    echo "    ✗ Binary NOT found!"
    exit 1
fi

echo ""
echo "[*] Test 2: Check binary permissions"
if [ -x "./build/btc_gold" ]; then
    echo "    ✓ Binary is executable"
else
    echo "    ✗ Binary is NOT executable"
    chmod +x ./build/btc_gold
    echo "    ✓ Made executable"
fi

echo ""
echo "[*] Test 3: Try running with --help"
./build/btc_gold --help 2>&1 | head -20
echo ""

echo "[*] Test 4: Run LINEAR mode with VALID addresses"
echo "─────────────────────────────────────────────────────"
timeout 5 ./build/btc_gold \
    --threads 2 \
    --mode linear \
    --start 1 \
    --end 10000 \
    --scan-mode 1 \
    --database test_targets.txt 2>&1
result=$?
echo "─────────────────────────────────────────────────────"
echo "Exit code: $result"
echo ""

if [ $result -eq 124 ]; then
    echo "\033[32m✓ Test timed out successfully (program is running)\033[0m"
elif [ $result -eq 0 ]; then
    echo "\033[32m✓ Test completed successfully\033[0m"
else
    echo "\033[31m✗ Test FAILED with exit code: $result\033[0m"
fi

echo ""
echo "[*] Check if database file exists"
ls -la test_targets.txt

echo ""
echo "[*] Cleanup"
rm -f test_targets.txt found_gold.txt
