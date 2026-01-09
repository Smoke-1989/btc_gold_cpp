#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include "interactive_menu.hpp"
#include "worker_engine.hpp"
#include "config.hpp"
#include "logger.hpp"

using namespace std;

// ANSI Colors
const string COLOR_BOLD = "\033[1m";
const string COLOR_GREEN = "\033[32m";
const string COLOR_CYAN = "\033[36m";
const string COLOR_RESET = "\033[0m";

void print_version() {
    cout << COLOR_BOLD << COLOR_GREEN << "BTC GOLD C++ v5.1 PRODUCTION" << COLOR_RESET
         << " - Bitcoin Private Key Recovery Engine\n"
         << "Enterprise Edition | Interactive Mode\n\n";
}

void print_usage() {
    cout << "USAGE:\n"
         << "  Interactive Mode (RECOMMENDED):\n"
         << "    ./btc_gold\n\n"
         << "  CLI Mode (For automation):\n"
         << "    ./btc_gold --mode <mode> --input <file> [OPTIONS]\n\n"
         << "MODES:\n"
         << "  0 = LINEAR        - Sequential range enumeration\n"
         << "  1 = RANDOM        - Full 256-bit random search\n"
         << "  2 = GEOMETRIC     - 3-phase intelligent search\n"
         << "  3 = TERMINATOR    - Multiplicative progression\n"
         << "  4 = DOUBLING      - Powers of 2 exhaustive\n"
         << "  5 = HAMMING       - Low-weight bit patterns\n"
         << "  6 = MODULAR       - Arithmetic progression\n"
         << "  7 = VANITY        - Address pattern matching\n"
         << "  8 = ENTROPY       - Weak entropy detection\n"
         << "  9 = COLLISION     - Adjacent address search\n\n"
         << "OPTIONS:\n"
         << "  --threads N              Number of worker threads (default: 8)\n"
         << "  --input-type hash160     Input hash type (hash160, hash256, pubkey)\n"
         << "  --input FILE             Target file path\n"
         << "  --start-hex HEX          Start key (linear/terminator)\n"
         << "  --end-hex HEX            End key (linear/terminator)\n"
         << "  --min-range-bit N        Min bit position (geometric/doubling/hamming)\n"
         << "  --max-range-bit N        Max bit position (geometric/doubling/hamming)\n"
         << "  --multiplier N           Multiplier value (terminator)\n"
         << "  --pattern STR            Address pattern (vanity)\n"
         << "  --distance N             Offset range (collision)\n"
         << "  --stop-on-find           Exit after first match\n"
         << "  --verbose                Detailed logging\n"
         << "  --help                   Show this message\n\n";
}

bool has_flag(int argc, char* argv[], const string& flag) {
    for(int i = 1; i < argc; i++) {
        if(string(argv[i]) == flag) return true;
    }
    return false;
}

string get_value(int argc, char* argv[], const string& flag, const string& default_val = "") {
    for(int i = 1; i < argc - 1; i++) {
        if(string(argv[i]) == flag) {
            return string(argv[i + 1]);
        }
    }
    return default_val;
}

int main(int argc, char* argv[]) {
    print_version();
    
    // Check for help or no arguments
    if(argc == 1 || has_flag(argc, argv, "--help")) {
        // If no args or --help: show CLI help
        if(argc == 1) {
            cout << "No arguments provided. Starting interactive mode...\n\n";
        }
        if(has_flag(argc, argv, "--help")) {
            print_usage();
            return 0;
        }
    }
    
    // INTERACTIVE MODE
    if(argc == 1) {
        try {
            InteractiveMenu menu;
            menu.run();
            
            // Get configuration from menu
            BTCGoldConfig config = menu.get_configuration();
            
            // Convert to WorkerConfig and run
            WorkerConfig worker_config;
            worker_config.threads = config.threads;
            worker_config.search_mode = config.search_mode;
            worker_config.input_type = config.input_type;
            worker_config.input_file = config.input_file;
            
            // Copy mode-specific parameters
            for(auto& p : config.mode_params) {
                worker_config.mode_params[p.first] = p.second;
            }
            for(auto& p : config.mode_params_int) {
                worker_config.mode_params_int[p.first] = p.second;
            }
            
            // Initialize and run worker
            Logger logger;
            WorkerEngine engine(worker_config, logger);
            engine.run();
            
        } catch(const exception& e) {
            cerr << "Error: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }
    
    // CLI MODE
    try {
        WorkerConfig config;
        
        // Parse mode
        string mode_str = get_value(argc, argv, "--mode", "0");
        config.search_mode = stoi(mode_str);
        
        if(config.search_mode < 0 || config.search_mode > 9) {
            cerr << "Invalid mode: " << config.search_mode << "\n";
            print_usage();
            return 1;
        }
        
        // Parse common options
        config.threads = stoi(get_value(argc, argv, "--threads", "8"));
        config.input_type = get_value(argc, argv, "--input-type", "hash160");
        config.input_file = get_value(argc, argv, "--input", "");
        config.stop_on_find = has_flag(argc, argv, "--stop-on-find");
        config.verbose = has_flag(argc, argv, "--verbose");
        
        // Validate basic config
        if(config.input_file.empty()) {
            cerr << "Error: --input file required\n";
            print_usage();
            return 1;
        }
        
        // Parse mode-specific parameters
        switch(config.search_mode) {
            case 0: // LINEAR
                config.mode_params["start_hex"] = get_value(argc, argv, "--start-hex", "1");
                config.mode_params["end_hex"] = get_value(argc, argv, "--end-hex", 
                    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
                break;
                
            case 2: // GEOMETRIC
                config.mode_params_int["min_bit"] = stoi(get_value(argc, argv, "--min-range-bit", "1"));
                config.mode_params_int["max_bit"] = stoi(get_value(argc, argv, "--max-range-bit", "256"));
                break;
                
            case 3: // TERMINATOR
                config.mode_params["start_hex"] = get_value(argc, argv, "--start-hex", "1");
                config.mode_params["end_hex"] = get_value(argc, argv, "--end-hex",
                    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
                config.mode_params_int["multiplier"] = stoi(get_value(argc, argv, "--multiplier", "2"));
                break;
                
            case 4: // DOUBLING
                config.mode_params_int["min_bit"] = stoi(get_value(argc, argv, "--min-range-bit", "1"));
                config.mode_params_int["max_bit"] = stoi(get_value(argc, argv, "--max-range-bit", "256"));
                break;
                
            case 5: // HAMMING
                config.mode_params_int["min_bit"] = stoi(get_value(argc, argv, "--min-range-bit", "1"));
                config.mode_params_int["max_bit"] = stoi(get_value(argc, argv, "--max-range-bit", "256"));
                break;
                
            case 6: // MODULAR_STRIDE
                config.mode_params["start_value"] = get_value(argc, argv, "--start-value", "1");
                config.mode_params["stride"] = get_value(argc, argv, "--stride", "1");
                break;
                
            case 7: // VANITY
                config.mode_params["pattern"] = get_value(argc, argv, "--pattern", "1Bitcoin");
                break;
                
            case 9: // COLLISION
                config.mode_params_int["distance"] = stoi(get_value(argc, argv, "--distance", "1000"));
                break;
        }
        
        // Initialize logger
        Logger logger;
        logger.set_verbose(config.verbose);
        
        logger.info("BTC GOLD v5.1 - Starting in CLI mode");
        logger.info("Mode: " + mode_str);
        logger.info("Threads: " + to_string(config.threads));
        logger.info("Input: " + config.input_file);
        
        // Initialize and run worker
        WorkerEngine engine(config, logger);
        engine.run();
        
    } catch(const exception& e) {
        cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
