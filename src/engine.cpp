#include "engine.h"
#include "logger.h"
#include <thread>
#include <chrono>

namespace btc_gold {

Engine::Engine(const Config& config)
    : config_(config) {
    if (config.num_threads <= 0) {
        config_.num_threads = std::thread::hardware_concurrency();
    }
    Logger::instance().info("Engine initialized with %d threads", config_.num_threads);
}

Engine::~Engine() {
    stop();
}

bool Engine::initialize(const std::string& database_file) {
    Logger& logger = Logger::instance();
    
    // Load database
    if (!database_.load(database_file, config_.input_type)) {
        logger.error("Failed to load database: %s", database_file.c_str());
        return false;
    }
    
    logger.info("Loaded %zu targets from %s", database_.size(), database_file.c_str());
    return true;
}

bool Engine::start() {
    Logger& logger = Logger::instance();
    running_ = true;
    
    stats_.start_time = std::chrono::system_clock::now().time_since_epoch().count();
    
    logger.info("Starting %d worker threads...", config_.num_threads);
    logger.info("Mode: %d, Scan mode: %d", 
        static_cast<int>(config_.mode),
        static_cast<int>(config_.scan_mode)
    );
    
    // Create worker threads
    for (int i = 0; i < config_.num_threads; ++i) {
        worker_threads_.emplace_back(
            [this, i]() {
                Worker worker(i, config_, database_, stats_);
                worker.run();
            }
        );
    }
    
    // Progress thread
    std::thread progress_thread([this]() {
        while (running_ && !stats_.should_stop) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            print_progress();
        }
    });
    
    // Wait for workers
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    running_ = false;
    progress_thread.join();
    
    return true;
}

void Engine::stop() {
    stats_.should_stop = true;
}

void Engine::print_progress() {
    auto elapsed = std::chrono::system_clock::now().time_since_epoch().count() - stats_.start_time;
    double seconds = elapsed / 1e9;
    double kps = (stats_.total_keys.load() / 1000.0) / (seconds > 0 ? seconds : 1);
    
    std::cout << "\r[RUNNING] Speed: " << kps << " k/s | Found: " << stats_.found_count.load();
    std::cout.flush();
}

}  // namespace btc_gold
