#include "logger.hpp"

Logger::Logger(const std::string& log_file) : log_file_(log_file), verbose_(true) {
    log_stream_.open(log_file_, std::ios::app);
}

Logger::~Logger() {
    if(log_stream_.is_open()) {
        log_stream_.close();
    }
}

void Logger::info(const std::string& message) {
    log("INFO", message);
}

void Logger::warning(const std::string& message) {
    log("WARNING", message);
}

void Logger::error(const std::string& message) {
    log("ERROR", message);
}

void Logger::debug(const std::string& message) {
    if(verbose_) {
        log("DEBUG", message);
    }
}

void Logger::set_verbose(bool verbose) {
    verbose_ = verbose;
}

std::string Logger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void Logger::log(const std::string& level, const std::string& message) {
    std::string formatted = "[" + get_timestamp() + "] [" + level + "] " + message;
    
    if(level == "INFO") {
        std::cout << formatted << "\n";
    } else if(level == "WARNING") {
        std::cout << "\033[33m" << formatted << "\033[0m\n";
    } else if(level == "ERROR") {
        std::cerr << "\033[31m" << formatted << "\033[0m\n";
    } else if(verbose_) {
        std::cout << formatted << "\n";
    }
    
    if(log_stream_.is_open()) {
        log_stream_ << formatted << "\n";
        log_stream_.flush();
    }
}
