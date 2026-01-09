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

// Simple uint256 implementation
struct uint256 {
    uint64_t data[4];
    
    uint256() : data{0, 0, 0, 0} {}
    uint256(uint64_t a, uint64_t b, uint64_t c, uint64_t d) : data{a, b, c, d} {}
    
    bool operator<(const uint256& other) const {
        for(int i = 3; i >= 0; i--) {
            if(data[i] != other.data[i]) return data[i] < other.data[i];
        }
        return false;
    }
    
    bool operator<=(const uint256& other) const;
    bool operator>(const uint256& other) const;
    bool operator==(const uint256& other) const;
    uint256 operator+(const uint256& other) const;
    uint256 operator*(uint64_t mul) const;
};

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
    
    // Public for callbacks
    std::vector<std::string> found_keys_;
    std::mutex results_mutex_;
    std::vector<std::string> targets_;
    Logger& logger_;
    std::atomic<uint64_t> keys_checked_;
    std::atomic<bool> running_;
    
private:
    WorkerConfig config_;
    std::vector<std::thread> workers_;
    
    // Mode runners
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
    
    // Worker threads
    void linear_worker(int thread_id, uint256 start, uint256 end);
    void random_worker(int thread_id);
    void geometric_worker(int thread_id, int min_bit, int max_bit);
    void terminator_worker(int thread_id, uint256 start, uint256 end, int mul);
    void doubling_worker(int min_bit, int max_bit);
    void hamming_worker(int thread_id, int min_bit, int max_bit);
    void modular_stride_worker(int thread_id);
    void vanity_worker(int thread_id);
    void entropy_worker(int thread_id);
    void collision_worker(int thread_id);
    
    // Utilities
    void load_targets();
    void save_results();
};

#endif // WORKER_ENGINE_HPP
