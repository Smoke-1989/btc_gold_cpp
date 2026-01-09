/**
 * @file gpu_engine.h
 * @brief GPU Engine - Main CUDA Orchestrator
 * 
 * CLASSIFICATION: ENTERPRISE PRODUCTION
 * SECURITY LEVEL: Governmental Grade (Level 5)
 * 
 * Core GPU engine for managing CUDA devices, kernel launches,
 * memory management, and multi-GPU coordination.
 * 
 * @author BTC GOLD Development Team
 * @version 6.0
 * @date January 2026
 */

#ifndef GPU_ENGINE_H
#define GPU_ENGINE_H

#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <cuda_runtime.h>
#include "gpu_config.h"
#include "logger.h"

namespace BTCGold {
namespace GPU {

/**
 * @class GPUEngine
 * @brief Main GPU acceleration engine
 */
class GPUEngine {
public:
    /**
     * @brief Constructor
     * @param logger Logger instance
     */
    GPUEngine(Logger& logger);
    
    /**
     * @brief Destructor - cleanup resources
     */
    ~GPUEngine();
    
    // ========================================================================
    // DEVICE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Initialize GPU devices
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Query available GPU devices
     * @return Number of available CUDA devices
     */
    int query_devices();
    
    /**
     * @brief Get information about a specific device
     * @param device_id Device ID
     * @param info Output device information
     * @return true if successful
     */
    bool get_device_info(int device_id, DeviceInfo& info);
    
    /**
     * @brief Select GPU device for computation
     * @param device_id Device ID
     * @return true if successful
     */
    bool set_active_device(int device_id);
    
    /**
     * @brief Configure multi-GPU setup
     * @param config Multi-GPU configuration
     * @return true if successful
     */
    bool configure_multi_gpu(const MultiGPUConfig& config);
    
    // ========================================================================
    // KERNEL EXECUTION
    // ========================================================================
    
    /**
     * @brief Launch ECDSA kernel (point multiplication)
     * @param batch_size Number of keys to process
     * @param input_keys Input private keys (device memory)
     * @param output_pubkeys Output public keys (device memory)
     * @param config Kernel configuration
     * @return true if successful
     */
    bool launch_ecdsa_kernel(size_t batch_size, void* input_keys,
                             void* output_pubkeys,
                             const GPUKernelConfig& config);
    
    /**
     * @brief Launch Hash160 kernel
     * @param batch_size Number of hashes to compute
     * @param input_data Input data (device memory)
     * @param output_hash Hash160 results (device memory)
     * @param config Kernel configuration
     * @return true if successful
     */
    bool launch_hash160_kernel(size_t batch_size, void* input_data,
                               void* output_hash,
                               const GPUKernelConfig& config);
    
    /**
     * @brief Launch database matching kernel
     * @param batch_size Number of hashes to check
     * @param input_hashes Hashes to check (device memory)
     * @param output_matches Match results (device memory)
     * @param config Kernel configuration
     * @return true if successful
     */
    bool launch_database_kernel(size_t batch_size, void* input_hashes,
                                void* output_matches,
                                const GPUKernelConfig& config);
    
    /**
     * @brief Launch mode-specific kernel
     * @param mode Search mode
     * @param batch_size Keys to process
     * @param input Input data
     * @param output Output results
     * @param config Kernel config
     * @return true if successful
     */
    bool launch_mode_kernel(int mode, size_t batch_size,
                            void* input, void* output,
                            const GPUKernelConfig& config);
    
    // ========================================================================
    // MEMORY MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Allocate device memory
     * @param size Size in bytes
     * @param purpose Purpose description (for audit trail)
     * @param pinned Use page-locked memory
     * @return Device pointer (nullptr on failure)
     */
    void* allocate_device_memory(size_t size, const std::string& purpose,
                                 bool pinned = false);
    
    /**
     * @brief Free device memory
     * @param ptr Device pointer
     * @return true if successful
     */
    bool free_device_memory(void* ptr);
    
    /**
     * @brief Copy data from host to device
     * @param dst Destination (device)
     * @param src Source (host)
     * @param size Size in bytes
     * @param async Use async copy (default: false)
     * @return true if successful
     */
    bool copy_to_device(void* dst, const void* src, size_t size,
                       bool async = false);
    
    /**
     * @brief Copy data from device to host
     * @param dst Destination (host)
     * @param src Source (device)
     * @param size Size in bytes
     * @param async Use async copy
     * @return true if successful
     */
    bool copy_from_device(void* dst, const void* src, size_t size,
                          bool async = false);
    
    /**
     * @brief Get memory usage statistics
     * @return Pair of (used_bytes, total_bytes)
     */
    std::pair<size_t, size_t> get_memory_usage();
    
    /**
     * @brief Optimize memory layout (defragmentation)
     * @return true if successful
     */
    bool optimize_memory();
    
    // ========================================================================
    // SYNCHRONIZATION
    // ========================================================================
    
    /**
     * @brief Synchronize all GPU operations
     * @return true if successful
     */
    bool synchronize();
    
    /**
     * @brief Synchronize specific stream
     * @param stream CUDA stream
     * @return true if successful
     */
    bool synchronize_stream(cudaStream_t stream);
    
    /**
     * @brief Create CUDA stream for async operations
     * @param priority Stream priority
     * @return Stream handle (nullptr on failure)
     */
    cudaStream_t create_stream(int priority = 0);
    
    /**
     * @brief Destroy CUDA stream
     * @param stream Stream to destroy
     * @return true if successful
     */
    bool destroy_stream(cudaStream_t stream);
    
    // ========================================================================
    // ERROR HANDLING & DIAGNOSTICS
    // ========================================================================
    
    /**
     * @brief Get last CUDA error
     * @return Error structure
     */
    GPUError get_last_error();
    
    /**
     * @brief Check GPU health (diagnostics)
     * @return true if healthy
     */
    bool check_gpu_health();
    
    /**
     * @brief Generate diagnostic report
     * @return Diagnostic report string
     */
    std::string generate_diagnostic_report();
    
    /**
     * @brief Enable/disable debug logging
     * @param enable true to enable
     */
    void set_debug_logging(bool enable);
    
    // ========================================================================
    // PERFORMANCE MONITORING
    // ========================================================================
    
    /**
     * @brief Get kernel execution time (microseconds)
     * @return Execution time in microseconds
     */
    uint64_t get_kernel_time();
    
    /**
     * @brief Get memory bandwidth utilization
     * @return Utilization percentage (0-100)
     */
    float get_bandwidth_utilization();
    
    /**
     * @brief Get SM occupancy
     * @return Occupancy percentage (0-100)
     */
    float get_sm_occupancy();
    
    /**
     * @brief Get throughput (keys/sec)
     * @return Throughput in keys/second
     */
    uint64_t get_throughput();
    
    // ========================================================================
    // CONFIGURATION
    // ========================================================================
    
    /**
     * @brief Set GPU compute configuration
     * @param config Compute configuration
     */
    void set_compute_config(const GPUComputeConfig& config);
    
    /**
     * @brief Get current compute configuration
     * @return Current configuration
     */
    GPUComputeConfig get_compute_config() const;
    
    // ========================================================================
    // CLEANUP
    // ========================================================================
    
    /**
     * @brief Shutdown GPU engine and free resources
     */
    void shutdown();
    
private:
    Logger& logger_;                        ///< Logger instance
    
    std::vector<DeviceInfo> devices_;       ///< Detected GPU devices
    int active_device_;                     ///< Currently active device
    
    std::vector<GPUAllocation> allocations_;  ///< Memory allocation tracking
    std::atomic<uint64_t> allocation_counter_; ///< Allocation ID counter
    std::mutex memory_mutex_;               ///< Protect memory allocations
    
    GPUComputeConfig compute_config_;       ///< Current compute config
    MultiGPUConfig multi_gpu_config_;       ///< Multi-GPU config
    
    std::vector<cudaStream_t> streams_;     ///< Active CUDA streams
    std::mutex stream_mutex_;               ///< Protect stream list
    
    bool initialized_;                      ///< Initialization flag
    bool debug_logging_;                    ///< Debug logging enabled
    
    // Performance metrics
    std::atomic<uint64_t> kernel_time_us_;  ///< Last kernel time (us)
    std::atomic<uint64_t> keys_processed_;  ///< Keys processed since last reset
    std::atomic<uint64_t> throughput_kps_;  ///< Current throughput (keys/sec)
    
    // Private methods
    bool init_device(int device_id);
    bool check_cuda_error(cudaError_t error, const std::string& context);
    void audit_memory_operation(const std::string& operation, void* ptr, size_t size);
};

} // namespace GPU
} // namespace BTCGold

#endif // GPU_ENGINE_H
