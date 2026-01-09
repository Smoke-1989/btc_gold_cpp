#ifndef WORKER_ENGINE_HPP
#define WORKER_ENGINE_HPP

#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include "config.hpp"
#include "logger.hpp"

struct WorkerConfig {
    int threads;
    std::string input_type;
    std::string input_file;
    int search_mode;
    bool stop_on_find;
    bool verbose;
    std::map<std::string, std::string> mode_params;
    std::map<std::string, int> mode_params_int;
};

class WorkerEngine {
public:
    WorkerEngine(const WorkerConfig& config, Logger& logger);
    ~WorkerEngine();
    
    void run();
    void stop();
    
private:
    WorkerConfig config_;
    Logger& logger_;
    std::vector<std::thread> workers_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> keys_checked_;
    std::mutex results_mutex_;
    std::vector<std::string> found_keys_;
    
    void run_linear_mode();
    void run_random_mode();
    void run_geometric_mode();
    void run_terminator_mode();
    void run_doubling_mode();
    void run_hamming_mode();
    void run_modular_stride_mode();
    void run_vanity_mode();
    void run_entropy_mode();
    void run_collision_mode();
    
    void linear_worker(int thread_id);
    void random_worker(int thread_id);
    void geometric_worker(int thread_id);
    void terminator_worker(int thread_id);
    void doubling_worker();
    void hamming_worker(int thread_id);
    void modular_stride_worker(int thread_id);
    void vanity_worker(int thread_id);
    void entropy_worker(int thread_id);
    void collision_worker(int thread_id);
    
    void load_targets();
    void save_results();
};

#endif // WORKER_ENGINE_HPP
