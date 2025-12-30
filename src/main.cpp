#include "engine.h"
#include "config.h"
#include "logger.h"
#include "constants.h"
#include <iostream>
#include <chrono>

using namespace btc_gold;

int main(int argc, char** argv) {
    std::cout << LOGO << std::endl;
    
    Logger& logger = Logger::instance();
    logger.info("BTC GOLD C++ Engine v1.0 starting...");
    
    try {
        // Parse configuration
        Config config;
        if (argc > 1) {
            config = ConfigParser::parse_cli(argc, argv);
        } else {
            config = ConfigParser::interactive_mode();
        }
        
        // Initialize engine
        Engine engine(config);
        if (!engine.initialize(config.database_file)) {
            logger.error("Failed to initialize engine");
            return 1;
        }
        
        // Start scanning
        auto start_time = std::chrono::steady_clock::now();
        
        if (!engine.start()) {
            logger.error("Failed to start engine");
            return 1;
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time
        );
        
        auto& stats = engine.get_stats();
        double kps = (stats.total_keys.load() / 1000.0) / (duration.count() > 0 ? duration.count() : 1);
        
        logger.info("Scan completed");
        logger.info("Total keys: %lu", stats.total_keys.load());
        logger.info("Found: %lu", stats.found_count.load());
        logger.info("Speed: %.1f k/s", kps);
        logger.info("Time: %ld seconds", duration.count());
        
        return 0;
    }
    catch (const std::exception& e) {
        Logger::instance().error("Exception: %s", e.what());
        return 1;
    }
}
