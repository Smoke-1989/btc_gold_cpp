/**
 * @file gpu_config.h
 * @brief GPU Configuration & Device Management
 * 
 * CLASSIFICATION: ENTERPRISE PRODUCTION
 * SECURITY LEVEL: Governmental Grade (Level 5)
 * 
 * This header defines GPU configuration structures and device management
 * for enterprise-grade CUDA acceleration.
 * 
 * @author BTC GOLD Development Team
 * @version 6.0
 * @date January 2026
 */

#ifndef GPU_CONFIG_H
#define GPU_CONFIG_H

#include <cstdint>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "types.h"

/**
 * @namespace BTCGold::GPU
 * @brief GPU acceleration namespace
 */
namespace BTCGold {
namespace GPU {

// ============================================================================
// CUDA DEVICE CONFIGURATION
// ============================================================================

/**
 * @struct DeviceInfo
 * @brief Information about a single GPU device
 */
struct DeviceInfo {
    int device_id;                          ///< CUDA device ID
    std::string device_name;                ///< Device name (e.g., "Tesla V100")
    cudaDeviceProp properties;              ///< Full CUDA device properties
    
    size_t global_memory_bytes;             ///< Total global memory
    size_t available_memory_bytes;          ///< Available memory
    int compute_capability_major;           ///< Compute capability (e.g., 7)
    int compute_capability_minor;           ///< Compute capability (e.g., 0)
    int max_threads_per_block;              ///< Max threads/block
    int max_blocks_per_grid;                ///< Max blocks in grid
    int warp_size;                          ///< Warp size (usually 32)
    int sm_count;                           ///< Multiprocessor count
    int max_warps_per_sm;                   ///< Max warps per SM
    
    bool supports_unified_memory;           ///< Unified memory support
    bool supports_concurrent_kernels;       ///< Concurrent kernel execution
    bool supports_managed_memory;           ///< Managed memory support
    
    // Performance metrics
    double theoretical_peak_flops;          ///< Peak FLOPS (single precision)
    double theoretical_peak_memory_bw;      ///< Peak memory bandwidth (GB/s)
    
    // Constructor
    DeviceInfo() : device_id(-1), compute_capability_major(0),
                   compute_capability_minor(0), max_threads_per_block(0),
                   supports_unified_memory(false) {}
};

/**
 * @struct GPUAllocation
 * @brief Memory allocation tracker for audit trail
 */
struct GPUAllocation {
    uint64_t allocation_id;                 ///< Unique allocation ID
    void* device_ptr;                       ///< Device pointer
    size_t size_bytes;                      ///< Allocation size
    std::string purpose;                    ///< Allocation purpose (audit)
    bool is_pinned;                         ///< Is page-locked
    bool is_unified;                        ///< Uses unified memory
    uint64_t timestamp_created;             ///< Creation timestamp
    uint64_t timestamp_freed;               ///< Freed timestamp (0 if active)
};

/**
 * @struct GPUKernelConfig
 * @brief Configuration for CUDA kernel launches
 */
struct GPUKernelConfig {
    // Thread block organization
    dim3 threads_per_block;                 ///< Threads per block (default: 256)
    dim3 blocks_per_grid;                   ///< Blocks per grid
    
    // Memory configuration
    size_t shared_memory_bytes;             ///< Shared memory per block
    size_t global_memory_bytes;             ///< Global memory for kernel
    
    // Stream configuration
    cudaStream_t stream;                    ///< CUDA stream for async execution
    int priority;                           ///< Stream priority (high/low)
    
    // Execution policy
    int grid_size;                          ///< Number of thread blocks
    int block_size;                         ///< Threads per block
    
    // Performance tuning
    bool use_cooperative_groups;            ///< Enable cooperative groups
    bool use_dynamic_parallelism;           ///< Enable dynamic parallelism
    int occupancy_target_percent;           ///< Target SM occupancy (%)
    
    GPUKernelConfig() : threads_per_block(256, 1, 1),
                        blocks_per_grid(1, 1, 1),
                        shared_memory_bytes(0),
                        stream(nullptr),
                        priority(0),
                        grid_size(1),
                        block_size(256),
                        use_cooperative_groups(false),
                        use_dynamic_parallelism(false),
                        occupancy_target_percent(100) {}
};

/**
 * @struct GPUComputeConfig
 * @brief CUDA computation configuration
 */
struct GPUComputeConfig {
    // Mode selection
    int search_mode;                        ///< Search mode (0-9)
    int batch_size;                         ///< Keys per batch
    int num_batches_per_kernel;             ///< Batches per kernel launch
    
    // Pipeline configuration
    bool enable_pipelining;                 ///< Enable GPU-CPU overlap
    int pipeline_stages;                    ///< Number of pipeline stages
    
    // Memory optimization
    bool use_page_locked_memory;            ///< Use pinned (page-locked) memory
    bool use_unified_memory;                ///< Use CUDA unified memory
    bool use_texture_memory;                ///< Use texture cache for database
    
    // Synchronization
    bool use_event_timing;                  ///< Time kernel execution
    bool use_stream_barriers;               ///< Synchronize streams
    
    // Data transfer
    size_t host_device_xfer_size;           ///< Size of H2D transfers
    size_t device_host_xfer_size;           ///< Size of D2H transfers
    
    // Error handling & audit
    bool enable_error_checking;             ///< Check every CUDA call
    bool enable_debug_logging;              ///< Detailed logging
    bool enable_memory_audit;               ///< Track all allocations
    
    GPUComputeConfig() : search_mode(0), batch_size(1024),
                         num_batches_per_kernel(10),
                         enable_pipelining(true),
                         pipeline_stages(3),
                         use_page_locked_memory(true),
                         use_unified_memory(false),
                         use_texture_memory(true),
                         use_event_timing(true),
                         enable_error_checking(true),
                         enable_debug_logging(false),
                         enable_memory_audit(true) {}
};

/**
 * @struct MultiGPUConfig
 * @brief Configuration for multi-GPU execution
 */
struct MultiGPUConfig {
    std::vector<int> device_ids;            ///< GPU device IDs to use
    bool enable_p2p;                        ///< Peer-to-peer access
    bool enable_nvlink;                     ///< NVLink if available
    
    // Load balancing
    enum LoadBalanceStrategy {
        STATIC,                             ///< Equal work per GPU
        DYNAMIC,                            ///< Work stealing
        ADAPTIVE                            ///< Performance-based
    };
    LoadBalanceStrategy strategy;           ///< Load balance strategy
    
    // Synchronization
    bool enable_concurrent_execution;       ///< Run kernels concurrently
    bool enable_gpu_affinity;               ///< CPU-GPU NUMA affinity
    
    MultiGPUConfig() : enable_p2p(true),
                       enable_nvlink(true),
                       strategy(ADAPTIVE),
                       enable_concurrent_execution(true),
                       enable_gpu_affinity(true) {}
};

} // namespace GPU
} // namespace BTCGold

#endif // GPU_CONFIG_H
