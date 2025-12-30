#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace btc_gold {

class Logger {
public:
    enum class Level { DEBUG, INFO, WARN, ERROR };
    
    static Logger& instance();
    
    void set_level(Level level) { current_level_ = level; }
    void set_verbose(bool verbose) { verbose_ = verbose; }
    
    template<typename... Args>
    void debug(const std::string& msg, Args... args) {
        if (current_level_ <= Level::DEBUG && verbose_)
            log(Level::DEBUG, msg, args...);
    }
    
    template<typename... Args>
    void info(const std::string& msg, Args... args) {
        if (current_level_ <= Level::INFO)
            log(Level::INFO, msg, args...);
    }
    
    template<typename... Args>
    void warn(const std::string& msg, Args... args) {
        if (current_level_ <= Level::WARN)
            log(Level::WARN, msg, args...);
    }
    
    template<typename... Args>
    void error(const std::string& msg, Args... args) {
        log(Level::ERROR, msg, args...);
    }
    
private:
    Logger() = default;
    Level current_level_ = Level::INFO;
    bool verbose_ = true;
    
    template<typename... Args>
    void log(Level level, const std::string& msg, Args... args) {
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
