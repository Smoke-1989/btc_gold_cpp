#pragma once

#include "config.h"
#include "database.h"
#include "worker.h"  // Include worker.h to get Stats definition
#include <vector>
#include <memory>
#include <atomic>

namespace btc_gold {

class Engine {
public:
    Engine(const Config& config);
    ~Engine();
    
    bool initialize();
    bool start();
    void stop();
    
    // Stats is defined in worker.h now
    Stats& get_stats() { return stats_; }

private:
    void print_progress();
    
    Config config_;
    Database database_;
    
    // Stats struct is shared between engine and workers
    Stats stats_;
    
    bool running_ = false;
    std::vector<std::thread> threads_;
};

}  // namespace btc_gold
