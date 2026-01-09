#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <map>

class Config {
public:
    static Config& getInstance();
    
    void parse_arguments(int argc, char* argv[]);
    std::string get(const std::string& key, const std::string& default_val = "");
    int get_int(const std::string& key, int default_val = 0);
    bool has(const std::string& key);
    
private:
    Config() = default;
    std::map<std::string, std::string> values_;
};

#endif // CONFIG_HPP
