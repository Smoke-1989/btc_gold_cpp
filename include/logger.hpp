#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

class Logger {
public:
    Logger(const std::string& log_file = "btc_gold.log");
    ~Logger();
    
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);
    void debug(const std::string& message);
    
    void set_verbose(bool verbose);
    
private:
    std::string log_file_;
    bool verbose_;
    std::ofstream log_stream_;
    
    std::string get_timestamp();
    void log(const std::string& level, const std::string& message);
};

#endif // LOGGER_HPP
