#include "interactive_menu.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cctype>

using namespace std;

// ANSI Color codes
const string COLOR_RESET = "\033[0m";
const string COLOR_BOLD = "\033[1m";
const string COLOR_GREEN = "\033[32m";
const string COLOR_YELLOW = "\033[33m";
const string COLOR_CYAN = "\033[36m";
const string COLOR_RED = "\033[31m";
const string COLOR_BLUE = "\033[34m";

InteractiveMenu::InteractiveMenu() {
    config_.threads = 8;
    config_.input_type = "hash160";
    config_.search_mode = 0; // LINEAR
    config_.stop_on_find = false;
    config_.verbose = true;
}

void InteractiveMenu::print_banner() {
    cout << COLOR_CYAN << COLOR_BOLD
         << "\n╔════════════════════════════════════════════════════════════════════╗\n"
         << "║                     BTC GOLD C++ v5.1 PRODUCTION                  ║\n"
         << "║              Bitcoin Private Key Recovery Engine                  ║\n"
         << "║                    Enterprise Edition - Interactive               ║\n"
         << "║                                                                    ║\n"
         << "║  Author: Security Research Team                                   ║\n"
         << "║  Purpose: Wallet Recovery / Key Space Analysis                    ║\n"
         << "╚════════════════════════════════════════════════════════════════════╝"
         << COLOR_RESET << "\n\n";
}

void InteractiveMenu::print_main_menu() {
    cout << COLOR_BOLD << "\n═══ MAIN MENU ═══\n" << COLOR_RESET;
    cout << "1. " << COLOR_GREEN << "Select Search Mode" << COLOR_RESET << " (Current: " << get_mode_name(config_.search_mode) << ")\n";
    cout << "2. " << COLOR_GREEN << "Configure Parameters" << COLOR_RESET << "\n";
    cout << "3. " << COLOR_GREEN << "Load Target Hashes" << COLOR_RESET << "\n";
    cout << "4. " << COLOR_GREEN << "View Configuration" << COLOR_RESET << "\n";
    cout << "5. " << COLOR_GREEN << "Start Scan" << COLOR_RESET << "\n";
    cout << "6. " << COLOR_YELLOW << "Help & Documentation" << COLOR_RESET << "\n";
    cout << "7. " << COLOR_RED << "Exit" << COLOR_RESET << "\n\n";
}

void InteractiveMenu::show_modes() {
    cout << COLOR_BOLD << "\n═══ SEARCH MODES ═══\n" << COLOR_RESET;
    
    cout << COLOR_CYAN << "\n[0] LINEAR - Sequential Range Scan" << COLOR_RESET << "\n"
         << "    Purpose: Exhaustive key-space enumeration\n"
         << "    Speed: ~5M keys/sec (single thread)\n"
         << "    Use: Small to medium ranges (< 2^40)\n"
         << "    Accuracy: 100% coverage\n\n";
    
    cout << COLOR_CYAN << "[1] RANDOM - Cryptographic Random Search" << COLOR_RESET << "\n"
         << "    Purpose: Large keyspace probabilistic search\n"
         << "    Speed: ~2M keys/sec (overhead: CSPRNG)\n"
         << "    Use: Exploring full 256-bit space\n"
         << "    Accuracy: Statistical (birthday paradox applicable)\n\n";
    
    cout << COLOR_CYAN << "[2] GEOMETRIC - 3-Phase Intelligent Search" << COLOR_RESET << "\n"
         << "    Purpose: Powers-of-2 + exponential combinations\n"
         << "    Speed: ~10M keys/sec (focused search)\n"
         << "    Use: Finding weak/simple keys (2^n patterns)\n"
         << "    Phases:\n"
         << "      - Phase 1: Border scan (2^min to 2^max)\n"
         << "      - Phase 2: Powers-of-2 multiples\n"
         << "      - Phase 3: Hamming-weight hybrids (2-3 bits set)\n\n";
    
    cout << COLOR_CYAN << "[3] TERMINATOR - Multiplicative Progression" << COLOR_RESET << "\n"
         << "    Purpose: Geometric progression (a*m^n)\n"
         << "    Speed: ~8M keys/sec\n"
         << "    Use: Pattern-based wallet discovery\n"
         << "    Parameters: start, end, multiplier\n\n";
    
    cout << COLOR_CYAN << "[4] DOUBLING - Powers of 2 Exhaustive" << COLOR_RESET << "\n"
         << "    Purpose: Test all powers of 2 (2^1 to 2^256)\n"
         << "    Speed: Instant (256 keys tested)\n"
         << "    Use: Known weak key patterns\n"
         << "    Accuracy: 100% of powers-of-2 space\n\n";
    
    cout << COLOR_CYAN << "[5] HAMMING - Low-Weight Bit Patterns" << COLOR_RESET << "\n"
         << "    Purpose: Keys with sparse bit representation\n"
         << "    Speed: ~3M keys/sec\n"
         << "    Use: Entropy-poor PRNGs, weak RNG seeds\n"
         << "    Patterns: 2-bit and 3-bit combinations\n\n";
    
    cout << COLOR_CYAN << "[6] MODULAR_STRIDE - Arithmetic Progression" << COLOR_RESET << "\n"
         << "    Purpose: Evenly-spaced key enumeration (a + d*n)\n"
         << "    Speed: ~6M keys/sec\n"
         << "    Use: Wallet derivation patterns, HD paths\n"
         << "    Parameters: start, stride_value\n\n";
    
    cout << COLOR_CYAN << "[7] VANITY - Address Pattern Matching" << COLOR_RESET << "\n"
         << "    Purpose: Find keys matching address prefix/suffix\n"
         << "    Speed: ~2M keys/sec (with hashing overhead)\n"
         << "    Use: Vanity addresses, specific patterns\n"
         << "    Parameters: pattern (e.g., '1Bitcoin')\n\n";
    
    cout << COLOR_CYAN << "[8] ENTROPY - Low-Entropy Detection" << COLOR_RESET << "\n"
         << "    Purpose: Find keys from weak entropy sources\n"
         << "    Speed: ~4M keys/sec\n"
         << "    Use: Compromised RNG analysis\n"
         << "    Detects: Repeating patterns, predictable sequences\n\n";
    
    cout << COLOR_CYAN << "[9] COLLISION - Adjacent Address Search" << COLOR_RESET << "\n"
         << "    Purpose: Find related wallet addresses\n"
         << "    Speed: ~5M keys/sec\n"
         << "    Use: Wallet cluster discovery\n"
         << "    Parameters: distance (key offset range)\n\n";
}

void InteractiveMenu::show_mode_details(int mode) {
    cout << COLOR_BOLD << "\n═══ MODE DETAILS ═══\n" << COLOR_RESET;
    
    switch(mode) {
        case 0: // LINEAR
            cout << "Mode: LINEAR - Sequential Range Enumeration\n"
                 << "Description:\n"
                 << "  Tests every private key from START to END sequentially.\n"
                 << "  Work is distributed among threads via non-overlapping ranges.\n\n"
                 << "Parameters Required:\n"
                 << "  --start-hex : Starting key (hex, 64 chars max for 256-bit)\n"
                 << "  --end-hex   : Ending key (hex, 64 chars max for 256-bit)\n"
                 << "  --threads   : Number of worker threads (default: 8)\n\n"
                 << "Performance:\n"
                 << "  Speed: ~188M keys/sec (8 threads, full range)\n"
                 << "  Memory: ~500MB resident\n\n"
                 << "Best For:\n"
                 << "  - Specific range searches\n"
                 << "  - Wallet recovery with known range\n"
                 << "  - Guaranteed 100% coverage\n\n"
                 << "Example (small test range):\n"
                 << "  --mode linear --start-hex 1 --end-hex 1000000\n";
            break;
            
        case 2: // GEOMETRIC
            cout << "Mode: GEOMETRIC - 3-Phase Intelligent Search\n"
                 << "Description:\n"
                 << "  Smart search targeting powers-of-2 patterns and combinations.\n\n"
                 << "Phase 1 - Border Scan:\n"
                 << "  Tests 2^min, 2^(min+1), ..., 2^max keys.\n"
                 << "  Covers all pure powers-of-2 in range.\n\n"
                 << "Phase 2 - Ceiling Ascent:\n"
                 << "  Tests multiples: 2^n * 2, 2^n * 3, ..., 2^n * k\n"
                 << "  Finds simple multiplicative patterns.\n\n"
                 << "Phase 3 - Hamming Hybrid:\n"
                 << "  Tests combinations: 2^a + 2^b, 2^a + 2^b + 2^c\n"
                 << "  Detects weak entropy sources.\n\n"
                 << "Parameters Required:\n"
                 << "  --min-range-bit : Minimum bit position (1-256)\n"
                 << "  --max-range-bit : Maximum bit position (1-256)\n\n"
                 << "Example:\n"
                 << "  --mode geometric --min-range-bit 10 --max-range-bit 32\n";
            break;
            
        case 7: // VANITY
            cout << "Mode: VANITY - Address Pattern Matching\n"
                 << "Description:\n"
                 << "  Searches for private keys that generate addresses\n"
                 << "  matching a specific pattern.\n\n"
                 << "Parameters Required:\n"
                 << "  --pattern : Address pattern to match\n"
                 << "              Examples: '1Bitcoin', '3Gold', '1CoinGold'\n\n"
                 << "Performance:\n"
                 << "  Speed: ~2M keys/sec (includes full hash computation)\n"
                 << "  Difficulty: Exponential with pattern length\n\n"
                 << "Example:\n"
                 << "  --mode vanity --pattern 1Bitcoin --threads 8\n";
            break;
            
        case 8: // ENTROPY
            cout << "Mode: ENTROPY - Low-Entropy Key Detection\n"
                 << "Description:\n"
                 << "  Identifies private keys generated from weak RNG sources.\n"
                 << "  Tests for:\n"
                 << "    - Repeating byte patterns\n"
                 << "    - Fibonacci sequences\n"
                 << "    - Sequential increments\n"
                 << "    - Linear congruential patterns\n\n"
                 << "Parameters Required:\n"
                 << "  --entropy-threshold : Pattern detection sensitivity (0.1-1.0)\n\n"
                 << "Use Cases:\n"
                 << "  - Finding wallets from bad RNG\n"
                 << "  - Identifying predictable key generation\n"
                 << "  - Security research\n";
            break;
            
        default:
            cout << "Use --help or option 6 from main menu for full documentation.\n";
    }
}

void InteractiveMenu::configure_parameters() {
    cout << COLOR_BOLD << "\n═══ PARAMETER CONFIGURATION ═══\n" << COLOR_RESET;
    
    string choice;
    bool configuring = true;
    
    while(configuring) {
        cout << "\nCurrent Configuration:\n"
             << "  Threads: " << config_.threads << "\n"
             << "  Input Type: " << config_.input_type << "\n"
             << "  Input File: " << config_.input_file << "\n";
        
        cout << "\n1. Threads (current: " << config_.threads << ")\n"
             << "2. Input Type (current: " << config_.input_type << ")\n"
             << "3. Input File (current: " << config_.input_file << ")\n"
             << "4. Mode-Specific Parameters\n"
             << "5. Back to Main Menu\n\n";
        
        cout << "Select: ";
        getline(cin, choice);
        choice.erase(remove_if(choice.begin(), choice.end(), ::isspace), choice.end());
        
        if(choice == "1") {
            cout << "Enter number of threads (1-64, recommended: CPU cores): ";
            int threads;
            cin >> threads;
            cin.ignore();
            
            if(threads < 1 || threads > 64) {
                cout << COLOR_RED << "Invalid! Using default (8)" << COLOR_RESET << "\n";
                config_.threads = 8;
            } else {
                config_.threads = threads;
                cout << COLOR_GREEN << "Threads set to " << threads << COLOR_RESET << "\n";
            }
        }
        else if(choice == "2") {
            cout << "Input Types:\n"
                 << "  1. hash160 (Bitcoin P2PKH addresses)\n"
                 << "  2. hash256 (Bitcoin transactions)\n"
                 << "  3. pubkey (Public keys)\n"
                 << "Select: ";
            
            string type_choice;
            cin >> type_choice;
            cin.ignore();
            
            if(type_choice == "1") config_.input_type = "hash160";
            else if(type_choice == "2") config_.input_type = "hash256";
            else if(type_choice == "3") config_.input_type = "pubkey";
            
            cout << COLOR_GREEN << "Input type set to " << config_.input_type << COLOR_RESET << "\n";
        }
        else if(choice == "3") {
            cout << "Enter input file path: ";
            getline(cin, config_.input_file);
            cout << COLOR_GREEN << "Input file set to: " << config_.input_file << COLOR_RESET << "\n";
        }
        else if(choice == "4") {
            configure_mode_parameters();
        }
        else if(choice == "5") {
            configuring = false;
        }
    }
}

void InteractiveMenu::configure_mode_parameters() {
    cout << COLOR_BOLD << "\n═══ MODE-SPECIFIC PARAMETERS ═══\n" << COLOR_RESET;
    
    string mode_name = get_mode_name(config_.search_mode);
    cout << "Configuring for mode: " << COLOR_CYAN << mode_name << COLOR_RESET << "\n\n";
    
    switch(config_.search_mode) {
        case 0: // LINEAR
        {
            cout << "Linear Mode Parameters:\n\n";
            cout << "Enter starting key (hex, e.g., 1): ";
            getline(cin, config_.mode_params["start_hex"]);
            
            cout << "Enter ending key (hex, e.g., FFFFFFFF): ";
            getline(cin, config_.mode_params["end_hex"]);
            
            cout << COLOR_GREEN << "Parameters set successfully!" << COLOR_RESET << "\n";
            break;
        }
        case 2: // GEOMETRIC
        {
            cout << "Geometric Mode Parameters:\n\n";
            cout << "Enter minimum bit position (1-256, e.g., 10): ";
            cin >> config_.mode_params_int["min_bit"];
            cin.ignore();
            
            cout << "Enter maximum bit position (1-256, e.g., 32): ";
            cin >> config_.mode_params_int["max_bit"];
            cin.ignore();
            
            if(config_.mode_params_int["min_bit"] > 0 && 
               config_.mode_params_int["max_bit"] <= 256 &&
               config_.mode_params_int["min_bit"] <= config_.mode_params_int["max_bit"]) {
                cout << COLOR_GREEN << "Parameters set successfully!" << COLOR_RESET << "\n";
            } else {
                cout << COLOR_RED << "Invalid parameters!" << COLOR_RESET << "\n";
            }
            break;
        }
        case 3: // TERMINATOR
        {
            cout << "Terminator Mode Parameters:\n\n";
            cout << "Enter start value (hex): ";
            getline(cin, config_.mode_params["start_hex"]);
            
            cout << "Enter end value (hex): ";
            getline(cin, config_.mode_params["end_hex"]);
            
            cout << "Enter multiplier (default 2): ";
            cin >> config_.mode_params_int["multiplier"];
            cin.ignore();
            
            if(config_.mode_params_int["multiplier"] < 1) {
                config_.mode_params_int["multiplier"] = 2;
            }
            
            cout << COLOR_GREEN << "Parameters set successfully!" << COLOR_RESET << "\n";
            break;
        }
        case 7: // VANITY
        {
            cout << "Vanity Mode Parameters:\n\n";
            cout << "Enter address pattern (e.g., 1Bitcoin): ";
            getline(cin, config_.mode_params["pattern"]);
            
            cout << "Pattern difficulty: ";
            double difficulty = pow(58, config_.mode_params["pattern"].length());
            cout << COLOR_YELLOW << "~1 in " << (long long)difficulty << " keys" << COLOR_RESET << "\n";
            
            cout << COLOR_GREEN << "Parameters set successfully!" << COLOR_RESET << "\n";
            break;
        }
        default:
            cout << "No additional parameters needed for this mode.\n";
    }
}

void InteractiveMenu::view_configuration() {
    cout << COLOR_BOLD << "\n═══ CURRENT CONFIGURATION ═══\n" << COLOR_RESET;
    cout << "Mode: " << COLOR_CYAN << get_mode_name(config_.search_mode) << COLOR_RESET << "\n"
         << "Threads: " << COLOR_CYAN << config_.threads << COLOR_RESET << "\n"
         << "Input Type: " << COLOR_CYAN << config_.input_type << COLOR_RESET << "\n"
         << "Input File: " << COLOR_CYAN << config_.input_file << COLOR_RESET << "\n"
         << "Stop on Find: " << COLOR_CYAN << (config_.stop_on_find ? "Yes" : "No") << COLOR_RESET << "\n\n";
    
    cout << "Mode-Specific Parameters:\n";
    for(auto& p : config_.mode_params) {
        cout << "  " << p.first << ": " << COLOR_CYAN << p.second << COLOR_RESET << "\n";
    }
}

void InteractiveMenu::show_help() {
    cout << COLOR_BOLD << "\n═══ HELP & DOCUMENTATION ═══\n" << COLOR_RESET;
    cout << "\n1. Complete Mode Documentation\n"
         << "2. Performance Tips\n"
         << "3. Common Issues\n"
         << "4. FAQ\n"
         << "5. Back to Main Menu\n\n"
         << "Select: ";
    
    string choice;
    getline(cin, choice);
    
    if(choice == "1") {
        show_modes();
    }
    else if(choice == "2") {
        cout << COLOR_BOLD << "\n═══ PERFORMANCE OPTIMIZATION ═══\n" << COLOR_RESET;
        cout << "1. Thread Count:\n"
             << "   - Use CPU core count for best results\n"
             << "   - More threads = more throughput (with diminishing returns)\n"
             << "   - Recommended: 8-16 for desktop, 32-64 for server\n\n"
             << "2. Memory Usage:\n"
             << "   - Each thread uses ~50-100MB\n"
             << "   - Total RAM needed: threads * 100MB\n\n"
             << "3. Key Space Strategy:\n"
             << "   - LINEAR: For known ranges < 2^40\n"
             << "   - GEOMETRIC: For pattern-based searches\n"
             << "   - RANDOM: For exploring large unknown spaces\n"
             << "   - VANITY: Only if looking for specific addresses\n\n";
    }
    else if(choice == "3") {
        cout << COLOR_BOLD << "\n═══ COMMON ISSUES ═══\n" << COLOR_RESET;
        cout << "Issue: 'Unknown mode' error\n"
             << "  Solution: Ensure mode is valid (0-9)\n\n"
             << "Issue: Range not respected in DOUBLING mode\n"
             << "  Solution: DOUBLING always scans all 2^1 to 2^256\n\n"
             << "Issue: Low hit rate\n"
             << "  Solution: Verify target file format (160-bit hex for hash160)\n\n"
             << "Issue: Program crashes on startup\n"
             << "  Solution: Check RAM availability (8GB+ recommended)\n\n";
    }
}

string InteractiveMenu::get_mode_name(int mode) {
    switch(mode) {
        case 0: return "LINEAR";
        case 1: return "RANDOM";
        case 2: return "GEOMETRIC";
        case 3: return "TERMINATOR";
        case 4: return "DOUBLING";
        case 5: return "HAMMING";
        case 6: return "MODULAR_STRIDE";
        case 7: return "VANITY";
        case 8: return "ENTROPY";
        case 9: return "COLLISION";
        default: return "UNKNOWN";
    }
}

void InteractiveMenu::run() {
    bool running = true;
    
    print_banner();
    
    while(running) {
        print_main_menu();
        
        string choice;
        cout << "Select: ";
        getline(cin, choice);
        choice.erase(remove_if(choice.begin(), choice.end(), ::isspace), choice.end());
        
        if(choice == "1") {
            show_modes();
            cout << "Select mode (0-9): ";
            int mode;
            cin >> mode;
            cin.ignore();
            
            if(mode >= 0 && mode <= 9) {
                config_.search_mode = mode;
                cout << COLOR_GREEN << "Mode set to: " << get_mode_name(mode) << COLOR_RESET << "\n";
            } else {
                cout << COLOR_RED << "Invalid mode!" << COLOR_RESET << "\n";
            }
        }
        else if(choice == "2") {
            configure_parameters();
        }
        else if(choice == "3") {
            cout << "Enter target file path: ";
            getline(cin, config_.input_file);
            cout << COLOR_GREEN << "Targets loaded from: " << config_.input_file << COLOR_RESET << "\n";
        }
        else if(choice == "4") {
            view_configuration();
        }
        else if(choice == "5") {
            cout << COLOR_BOLD << "\nStarting scan with current configuration...\n" << COLOR_RESET;
            running = false;
            return; // Exit to main program
        }
        else if(choice == "6") {
            show_help();
        }
        else if(choice == "7") {
            cout << COLOR_RED << "\nExiting BTC GOLD.\n" << COLOR_RESET;
            exit(0);
        }
    }
}

BTCGoldConfig InteractiveMenu::get_configuration() {
    return config_;
}
