#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              BTC GOLD C++ DEBUG TEST                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Create test targets
cat > test_targets.txt << 'EOF'
1A1z7agoat2YMSZ2qTCrni2hWVQ76i1M62
1dice8EMCQAqQSN88NYLERg7yajL87KWh
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
./build/btc_gold --help
echo ""

echo "[*] Test 4: Run LINEAR mode with verbose output"
echo "─────────────────────────────────────────────────────"
timeout 5 ./build/btc_gold \
    --threads 2 \
    --mode linear \
    --start 1 \
    --end 10000 \
    --scan-mode 1 \
    --database test_targets.txt
result=$?
echo "─────────────────────────────────────────────────────"
echo "Exit code: $result"
echo ""

if [ $result -eq 124 ]; then
    echo "✓ Test timed out successfully (program is running)"
elif [ $result -eq 0 ]; then
    echo "✓ Test completed successfully"
else
    echo "✗ Test FAILED with exit code: $result"
    echo ""
    echo "[*] Trying to get more info with strace..."
    timeout 2 strace -e trace=open,openat,read ./build/btc_gold \
        --threads 1 \
        --mode linear \
        --start 1 \
        --end 1000 \
        --scan-mode 1 \
        --database test_targets.txt 2>&1 | tail -20
fi

echo ""
echo "[*] Check if database file exists"
ls -la test_targets.txt

echo ""
echo "[*] Cleanup"
rm -f test_targets.txt found_gold.txt
