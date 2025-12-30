#pragma once

#include "types.h"
#include "database.h"
#include "worker.h"
#include <thread>
#include <vector>
#include <memory>

namespace btc_gold {

class Engine {
public:
    Engine(const Config& config);
    ~Engine();
    
    /**
     * Initialize engine with database
     */
    bool initialize(const std::string& database_file);
    
    /**
     * Start scanning
     */
    bool start();
    
    /**
     * Stop scanning
     */
    void stop();
    
    /**
     * Get reference to statistics
     */
    Worker::Stats& get_stats() { return stats_; }
    
private:
    Config config_;
    Database database_;
    Worker::Stats stats_;
    std::vector<std::thread> worker_threads_;
    bool running_ = false;
    
    void print_progress();
};

}  // namespace btc_gold
