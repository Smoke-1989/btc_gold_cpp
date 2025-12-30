#pragma once

#include "config.h"
#include "database.h"
#include "worker.h"
#include <vector>
#include <memory>
#include <atomic>
#include <thread> // Fix: Added missing include

namespace btc_gold {

class Engine {
public:
    Engine(const Config& config);
    ~Engine();
    
    // Fix: Updated signature to match main.cpp call and engine.cpp impl
    bool initialize(const std::string& database_file);
    bool start();
    void stop();
    
    Stats& get_stats() { return stats_; }

private:
    void print_progress();
    
    Config config_;
    Database database_;
    Stats stats_;
    
    bool running_ = false;
    
    // Fix: Renamed to match engine.cpp usage
    std::vector<std::thread> worker_threads_;
};

}  // namespace btc_gold
