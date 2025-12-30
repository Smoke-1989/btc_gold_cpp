#pragma once

#include "types.h"
#include <string>
#include <iostream>

namespace btc_gold {

class ConfigParser {
public:
    static Config parse_cli(int argc, char** argv);
    static Config interactive_mode();
    
private:
    static void print_menu();
    static void print_help();
};

}  // namespace btc_gold
