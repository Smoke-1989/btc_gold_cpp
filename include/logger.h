#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdio>
#include <cstdarg>

namespace btc_gold {

class Logger {
public:
    enum class Level { DEBUG, INFO, WARN, ERROR };
    
    static Logger& instance();
    
    void set_level(Level level) { current_level_ = level; }
    void set_verbose(bool verbose) { verbose_ = verbose; }
    
    void debug(const std::string& msg) {
        if (current_level_ <= Level::DEBUG && verbose_)
            log(Level::DEBUG, msg);
    }
    
    void info(const std::string& msg) {
        if (current_level_ <= Level::INFO)
            log(Level::INFO, msg);
    }
    
    void info(const std::string& format, uint64_t val) {
        if (current_level_ <= Level::INFO) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), format.c_str(), val);
            log(Level::INFO, std::string(buffer));
        }
    }
    
    void info(const std::string& format, double val) {
        if (current_level_ <= Level::INFO) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), format.c_str(), val);
            log(Level::INFO, std::string(buffer));
        }
    }
    
    void info(const std::string& format, int val) {
        if (current_level_ <= Level::INFO) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), format.c_str(), val);
            log(Level::INFO, std::string(buffer));
        }
    }
    
    void info(const std::string& format, long val) {
        if (current_level_ <= Level::INFO) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), format.c_str(), val);
            log(Level::INFO, std::string(buffer));
        }
    }
    
    void info(const std::string& format, uint64_t val1, const char* val2) {
        if (current_level_ <= Level::INFO) {
            char buffer[512];
            snprintf(buffer, sizeof(buffer), format.c_str(), val1, val2);
            log(Level::INFO, std::string(buffer));
        }
    }
    
    void info(const std::string& format, int val1, int val2) {
        if (current_level_ <= Level::INFO) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), format.c_str(), val1, val2);
            log(Level::INFO, std::string(buffer));
        }
    }
    
    void warn(const std::string& msg) {
        if (current_level_ <= Level::WARN)
            log(Level::WARN, msg);
    }
    
    void error(const std::string& msg) {
        log(Level::ERROR, msg);
    }
    
    void error(const std::string& format, const char* val) {
        char buffer[512];
        snprintf(buffer, sizeof(buffer), format.c_str(), val);
        log(Level::ERROR, std::string(buffer));
    }
    
private:
    Logger() = default;
    Level current_level_ = Level::INFO;
    bool verbose_ = true;
    
    void log(Level level, const std::string& msg) {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        
        std::cout << std::put_time(std::localtime(&time), "[%H:%M:%S]");
        
        switch(level) {
            case Level::DEBUG:
                std::cout << " [DEBUG] ";
                break;
            case Level::INFO:
                std::cout << " [INFO] ";
                break;
            case Level::WARN:
                std::cout << " [WARN] ";
                break;
            case Level::ERROR:
                std::cout << " [ERROR] ";
                break;
        }
        
        std::cout << msg << std::endl;
    }
};

}  // namespace btc_gold
