#include "config.h"
#include "logger.h"
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace btc_gold {

Config ConfigParser::parse_cli(int argc, char** argv) {
    Config config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        }
        else if (arg == "--threads" || arg == "-t") {
            if (i + 1 < argc) {
                config.num_threads = std::stoi(argv[++i]);
            }
        }
        else if (arg == "--mode" || arg == "-m") {
            if (i + 1 < argc) {
                std::string mode = argv[++i];
                if (mode == "linear" || mode == "1") {
                    config.mode = Config::Mode::LINEAR;
                } else if (mode == "random" || mode == "2") {
                    config.mode = Config::Mode::RANDOM;
                } else if (mode == "geometric" || mode == "3") {
                    config.mode = Config::Mode::GEOMETRIC;
                }
            }
        }
        else if (arg == "--scan-mode" || arg == "-s") {
            if (i + 1 < argc) {
                int mode = std::stoi(argv[++i]);
                config.scan_mode = static_cast<Config::ScanMode>(mode);
            }
        }
        else if (arg == "--database" || arg == "-d") {
            if (i + 1 < argc) {
                config.database_file = argv[++i];
            }
        }
        else if (arg == "--start") {
            if (i + 1 < argc) {
                std::string val = argv[++i];
                if (val.substr(0, 2) == "0x" || val.substr(0, 2) == "0X") {
                    config.start_value = std::stoull(val, nullptr, 16);
                } else {
                    config.start_value = std::stoull(val);
                }
            }
        }
        else if (arg == "--end") {
            if (i + 1 < argc) {
                std::string val = argv[++i];
                if (val.substr(0, 2) == "0x" || val.substr(0, 2) == "0X") {
                    config.end_value = std::stoull(val, nullptr, 16);
                } else {
                    config.end_value = std::stoull(val);
                }
            }
        }
        else if (arg == "--multiplier") {
            if (i + 1 < argc) {
                config.multiplier = std::stoull(argv[++i]);
            }
        }
        else if (arg == "--input-type") {
            if (i + 1 < argc) {
                int type = std::stoi(argv[++i]);
                config.input_type = static_cast<Config::InputType>(type);
            }
        }
    }
    
    return config;
}

Config ConfigParser::interactive_mode() {
    Config config;
    
    std::cout << "\n[Database Input Format]\n";
    std::cout << "[1] Bitcoin Address\n";
    std::cout << "[2] HASH160\n";
    std::cout << "[3] Public Key\n";
    std::cout << ">> ";
    
    int choice;
    std::cin >> choice;
    config.input_type = static_cast<Config::InputType>(choice);
    
    std::cout << "\n[Scan Mode]\n";
    std::cout << "[1] Compressed Only\n";
    std::cout << "[2] Uncompressed Only\n";
    std::cout << "[3] Both\n";
    std::cout << ">> ";
    
    std::cin >> choice;
    config.scan_mode = static_cast<Config::ScanMode>(choice);
    
    return config;
}

void ConfigParser::print_menu() {
    std::cout << "\n[BTC GOLD Menu]\n";
    std::cout << "[1] Linear Mode\n";
    std::cout << "[2] Random Mode\n";
    std::cout << "[3] Geometric Mode\n";
}

void ConfigParser::print_help() {
    std::cout << "BTC GOLD C++ - Usage\n";
    std::cout << "  btc_gold [options]\n";
    std::cout << "  btc_gold --help\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --threads, -t <N>          Number of threads (default: auto)\n";
    std::cout << "  --mode, -m <mode>          Mode: linear, random, geometric (default: random)\n";
    std::cout << "  --scan-mode, -s <N>        Scan: 1=compressed, 2=uncompressed, 3=both (default: 3)\n";
    std::cout << "  --database, -d <file>      Database file (default: alvos.txt)\n";
    std::cout << "  --start <value>            Start value (decimal or hex with 0x)\n";
    std::cout << "  --end <value>              End value (decimal or hex with 0x)\n";
    std::cout << "  --multiplier <value>       Multiplier for geometric mode (default: 2)\n";
    std::cout << "  --input-type <N>           Input: 1=address, 2=hash160, 3=pubkey (default: 1)\n";
    std::cout << "  --help, -h                 Show this help\n";
}

}  // namespace btc_gold
