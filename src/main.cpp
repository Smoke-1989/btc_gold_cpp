#include "config.h"
#include "logger.h"
#include "database.h"
#include "worker.h"
#include "constants.h"
#include <iostream>
#include <csignal>
#include <atomic>
#include <chrono>

namespace btc_gold {

// Global pointer to worker for signal handling
// Using atomic pointer for thread safety, though technically signal handlers are single-threaded
static std::atomic<WorkerEngine*> g_worker{nullptr};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        WorkerEngine* worker = g_worker.load();
        if (worker) {
            // Write directly to stderr to avoid memory allocation in signal handler
            const char* msg = "\n[SIGNAL] Interrupt received. Stopping gracefully...\n";
            (void)!write(2, msg, 50); 
            worker->stop();
        }
    }
}

}  // namespace btc_gold

using namespace btc_gold;

// Need unistd.h for write()
#include <unistd.h>

int main(int argc, char** argv) {
    try {
        // Parse configuration first
        Config config;
        if (!parse_args(argc, argv, config)) {
            print_usage(argv[0]);
            return 1;
        }
        
        // Initialize logger
        Logger logger(config.log_file, 
                     config.verbose ? Logger::Level::DEBUG : Logger::Level::INFO);
        logger.enable_console(true);
        logger.enable_file(!config.log_file.empty());
        
        logger.info("=" + std::string(70, '='));
        logger.info("BTC GOLD C++ v" + std::string(VERSION));
        logger.info("Mode: " + std::to_string(static_cast<int>(config.mode)));
        logger.info("=" + std::string(70, '='));
        
        // Initialize database
        Database database;
        database.set_logger(&logger);
        
        if (!config.database_file.empty()) {
            logger.info("Loading targets from: " + config.database_file);
            
            if (!database.load(config.database_file, config.input_type)) {
                logger.error("Failed to load database from: " + config.database_file);
                return 1;
            }
            
            if (database.empty()) {
                logger.error("No valid targets loaded");
                return 1;
            }
            
            logger.info("Loaded " + std::to_string(database.size()) + " targets");
        } else {
            logger.error("No database file specified (use --input)");
            return 1;
        }
        
        // Initialize worker engine
        logger.info("Initializing worker engine...");
        WorkerEngine worker(config, logger, database);
        
        // Register global pointer for signal handler
        g_worker.store(&worker);
        
        // Install signal handlers AFTER worker is ready
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        
        // Start scanning
        auto start_time = std::chrono::high_resolution_clock::now();
        logger.info("Starting scan...");
        
        worker.run();
        
        // Unregister global pointer
        g_worker.store(nullptr);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        logger.info("=" + std::string(70, '='));
        logger.info("Scan completed");
        logger.info("Total time: " + std::to_string(duration.count()) + " seconds");
        logger.info("Results saved to: " + config.output_file);
        logger.info("=" + std::string(70, '='));
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "FATAL ERROR: Unknown exception" << std::endl;
        return 1;
    }
}
