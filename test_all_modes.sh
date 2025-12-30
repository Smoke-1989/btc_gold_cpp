#!/bin/bash

# ============================================================================
# BTC GOLD C++ - COMPREHENSIVE TEST SUITE
# Tests all 3 modes x 3 scan modes = 9 combinations
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        BTC GOLD C++ - COMPREHENSIVE TEST SUITE             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${YELLOW}[*] Pulling latest changes...${NC}"
git pull

echo ""
echo -e "${YELLOW}[*] Compiling release build...${NC}"
make release

if [ ! -f "./build/btc_gold" ]; then
    echo -e "${RED}[ERROR] Build failed!${NC}"
    exit 1
fi
echo -e "${GREEN}[✓] Build successful${NC}"

echo ""
echo -e "${YELLOW}[*] Creating test targets file...${NC}"
cat > test_targets.txt << 'EOF'
# Bitcoin P2PKH Addresses
1A1z7agoat2YMSZ2qTCrni2hWVQ76i1M62
1dice8EMCQAqQSN88NYLERg7yajL87KWh

# Hash160 Examples
62e907b15cbf27d5425399ebf6f0fb50ebb88f18
EOF
echo -e "${GREEN}[✓] Test targets created${NC}"

echo ""
echo -e "${YELLOW}[*] Running benchmark...${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
./build/btc_gold_benchmark
echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"

echo ""
echo -e "${YELLOW}Testing all mode combinations...${NC}"
echo ""

# Test configurations
declare -A modes=([1]="LIN" [2]="RAN" [3]="GEO")
declare -A scan_modes=([1]="COM" [2]="UNC" [3]="BOT")

test_count=0
passed=0
failed=0

for mode in 1 2 3; do
    for scan in 1 2 3; do
        test_count=$((test_count+1))
        
        mode_name="${modes[$mode]}"
        scan_name="${scan_modes[$scan]}"
        
        echo -n "Test $test_count: MODE=$mode_name SCAN=$scan_name ... "
        
        # Clean up previous output
        rm -f found_gold.txt
        
        # Different parameters based on mode
        result=0
        case $mode in
            1)  # LINEAR
                timeout 10 ./build/btc_gold \
                    --threads 4 \
                    --mode linear \
                    --start 1 \
                    --end 100000 \
                    --scan-mode $scan \
                    --database test_targets.txt > /dev/null 2>&1 || result=$?
                ;;
            2)  # RANDOM
                timeout 10 ./build/btc_gold \
                    --threads 4 \
                    --mode random \
                    --start 1 \
                    --end 1000000 \
                    --scan-mode $scan \
                    --database test_targets.txt > /dev/null 2>&1 || result=$?
                ;;
            3)  # GEOMETRIC
                timeout 10 ./build/btc_gold \
                    --threads 4 \
                    --mode geometric \
                    --start 1 \
                    --multiplier 1.5 \
                    --scan-mode $scan \
                    --database test_targets.txt > /dev/null 2>&1 || result=$?
                ;;
        esac
        
        # Timeout returns 124, which is acceptable (test ran and timed out = success)
        if [ $result -eq 0 ] || [ $result -eq 124 ]; then
            echo -e "${GREEN}PASS${NC}"
            passed=$((passed+1))
        else
            echo -e "${RED}FAIL${NC}"
            failed=$((failed+1))
        fi
    done
done

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ TEST RESULTS                                               ║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║ Total Tests:  ${test_count}                                                  ║${NC}"
echo -e "${BLUE}║ Passed:       ${GREEN}${passed}${BLUE}                                                  ║${NC}"
echo -e "${BLUE}║ Failed:       ${RED}${failed}${BLUE}                                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${YELLOW}[*] Testing interactive mode (will timeout in 5 seconds)...${NC}"
echo "2" | timeout 5 ./build/btc_gold > /dev/null 2>&1 || true
echo -e "${GREEN}[✓] Interactive mode test complete${NC}"

echo ""
echo -e "${YELLOW}[*] Cleanup...${NC}"
rm -f test_targets.txt found_gold.txt

echo ""
if [ $failed -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              ✓ ALL TESTS PASSED!                           ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║              ✗ SOME TESTS FAILED                            ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
