#include "logger.h"
#include <iostream>

namespace btc_gold {

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

Logger::Logger(const std::string& log_file, Level min_level)
    : log_file_path_(log_file),
      min_level_(min_level),
      console_enabled_(true),
      file_enabled_(!log_file.empty()) {
    
    if (!log_file.empty()) {
        set_file(log_file);
    }
}

Logger::~Logger() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (log_file_ && log_file_->is_open()) {
        log_file_->flush();
        log_file_->close();
    }
}

// ============================================================================
// PUBLIC: Log with Level
// ============================================================================

void Logger::log(Level level, const std::string& message) {
    if (level < min_level_) {
        return;  // Skip messages below minimum level
    }
    
    std::lock_guard<std::mutex> guard(mutex_);
    write(level, message);
}

// ============================================================================
// PUBLIC: Convenience Methods
// ============================================================================

void Logger::debug(const std::string& message) {
    log(Level::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(Level::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(Level::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(Level::ERROR, message);
}

void Logger::critical(const std::string& message) {
    log(Level::CRITICAL, message);
}

// ============================================================================
// PUBLIC: Set Output File
// ============================================================================

void Logger::set_file(const std::string& log_file) {
    std::lock_guard<std::mutex> guard(mutex_);
    
    // Close existing file if open
    if (log_file_ && log_file_->is_open()) {
        log_file_->flush();
        log_file_->close();
    }
    
    log_file_path_ = log_file;
    
    if (!log_file.empty()) {
        log_file_ = std::make_unique<std::ofstream>(log_file, std::ios::app);
        file_enabled_ = log_file_->is_open();
    } else {
        log_file_.reset();
        file_enabled_ = false;
    }
}

// ============================================================================
// PUBLIC: Get Level Name
// ============================================================================

const char* Logger::level_name(Level level) {
    switch (level) {
        case Level::DEBUG:    return "DEBUG";
        case Level::INFO:     return "INFO";
        case Level::WARNING:  return "WARNING";
        case Level::ERROR:    return "ERROR";
        case Level::CRITICAL: return "CRITICAL";
        default:              return "UNKNOWN";
    }
}

// ============================================================================
// PRIVATE: Get Current Timestamp
// ============================================================================

std::string Logger::get_timestamp() const {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// ============================================================================
// PRIVATE: Format Message
// ============================================================================

std::string Logger::format_message(Level level, const std::string& message) const {
    std::ostringstream oss;
    oss << "["
        << get_timestamp()
        << "] ["
        << level_name(level)
        << "] "
        << message;
    return oss.str();
}

// ============================================================================
// PRIVATE: Write to All Enabled Outputs
// ============================================================================

void Logger::write(Level level, const std::string& message) {
    std::string formatted = format_message(level, message);
    
    // Write to console
    if (console_enabled_) {
        std::cout << formatted << std::endl;
    }
    
    // Write to file
    if (file_enabled_ && log_file_ && log_file_->is_open()) {
        *log_file_ << formatted << std::endl;
        log_file_->flush();
    }
}

}  // namespace btc_gold
