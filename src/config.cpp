#include "config.h"
#include "logger.h"
#include <iostream>
#include <cstring>

namespace btc_gold {

Config ConfigParser::parse_cli(int argc, char** argv) {
    Config config;
    
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_help();
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
    std::cout >> ">> ";
    
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
}

}  // namespace btc_gold
