#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <mutex>
#include <ctime>
#include <iomanip>

namespace btc_gold {

// ============================================================================
// LOGGER - Enterprise-grade Logging System
// ============================================================================

class Logger {
public:
    // Logging levels
    enum class Level : uint8_t {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3,
        CRITICAL = 4
    };

    // Constructor: Initialize with optional file output
    explicit Logger(const std::string& log_file = "", Level min_level = Level::INFO);
    
    // Destructor: Flush and close file
    ~Logger();
    
    // PUBLIC: Log message with level
    void log(Level level, const std::string& message);
    
    // PUBLIC: Convenience methods for each level
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);
    void critical(const std::string& message);
    
    // PUBLIC: Set minimum logging level
    void set_level(Level level) { min_level_ = level; }
    
    // PUBLIC: Set output file
    void set_file(const std::string& log_file);
    
    // PUBLIC: Enable/disable console output
    void enable_console(bool enable) { console_enabled_ = enable; }
    
    // PUBLIC: Enable/disable file output
    void enable_file(bool enable) { file_enabled_ = enable; }
    
    // PUBLIC: Get level name
    static const char* level_name(Level level);
    
private:
    // Private implementation details
    std::string log_file_path_;
    std::unique_ptr<std::ofstream> log_file_;
    mutable std::mutex mutex_;
    Level min_level_;
    bool console_enabled_;
    bool file_enabled_;
    
    // Private helper to get current timestamp
    std::string get_timestamp() const;
    
    // Private helper to format message
    std::string format_message(Level level, const std::string& message) const;
    
    // Private helper to write to all outputs
    void write(Level level, const std::string& message);
};

}  // namespace btc_gold
