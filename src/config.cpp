#include "config.hpp"
#include <algorithm>

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

void Config::parse_arguments(int argc, char* argv[]) {
    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if(arg.find("--") == 0) {
            std::string key = arg.substr(2);
            if(i + 1 < argc && argv[i + 1][0] != '-') {
                values_[key] = argv[++i];
            } else {
                values_[key] = "true";
            }
        }
    }
}

std::string Config::get(const std::string& key, const std::string& default_val) {
    auto it = values_.find(key);
    return it != values_.end() ? it->second : default_val;
}

int Config::get_int(const std::string& key, int default_val) {
    auto it = values_.find(key);
    return it != values_.end() ? std::stoi(it->second) : default_val;
}

bool Config::has(const std::string& key) {
    return values_.find(key) != values_.end();
}
