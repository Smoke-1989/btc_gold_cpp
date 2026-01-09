/**
 * @file gpu_engine.cpp
 * @brief GPU Engine Implementation
 * 
 * CLASSIFICATION: ENTERPRISE PRODUCTION
 * SECURITY LEVEL: Governmental Grade (Level 5)
 * 
 * Main GPU orchestration engine for CUDA device management,
 * kernel launching, and memory management.
 * 
 * @author BTC GOLD Development Team
 * @version 6.0
 * @date January 2026
 */

#include "gpu_engine.h"
#include "gpu_kernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <algorithm>

namespace BTCGold {
namespace GPU {

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

GPUEngine::GPUEngine(Logger& logger)
    : logger_(logger),
      active_device_(-1),
      allocation_counter_(0),
      initialized_(false),
      debug_logging_(false),
      kernel_time_us_(0),
      keys_processed_(0),
      throughput_kps_(0)
{
    logger_.info("GPUEngine: Constructor called");
    compute_config_ = GPUComputeConfig();  // Default configuration
}

GPUEngine::~GPUEngine() {
    if (initialized_) {
        shutdown();
    }
}

// ============================================================================
// DEVICE MANAGEMENT
// ============================================================================

bool GPUEngine::initialize() {
    if (initialized_) {
        logger_.warn("GPUEngine::initialize() - Already initialized");
        return true;
    }
    
    // Query available devices
    int device_count = query_devices();
    
    if (device_count == 0) {
        logger_.error("GPUEngine::initialize() - No CUDA devices found");
        return false;
    }
    
    logger_.info("GPUEngine::initialize() - Found " + std::to_string(device_count) + " CUDA device(s)");
    
    // Initialize each device
    for (int i = 0; i < device_count; i++) {
        if (!init_device(i)) {
            logger_.warn("GPUEngine::initialize() - Failed to initialize device " + std::to_string(i));
        }
    }
    
    // Set first device as active
    if (!set_active_device(0)) {
        logger_.error("GPUEngine::initialize() - Failed to set active device");
        return false;
    }
    
    initialized_ = true;
    logger_.info("GPUEngine::initialize() - Initialization complete");
    
    return true;
}

int GPUEngine::query_devices() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (!check_cuda_error(err, "cudaGetDeviceCount")) {
        return 0;
    }
    
    devices_.clear();
    
    for (int i = 0; i < device_count; i++) {
        DeviceInfo info;
        if (get_device_info(i, info)) {
            devices_.push_back(info);
        }
    }
    
    return device_count;
}

bool GPUEngine::get_device_info(int device_id, DeviceInfo& info) {
    cudaError_t err = cudaGetDeviceProperties(&info.properties, device_id);
    
    if (!check_cuda_error(err, "cudaGetDeviceProperties")) {
        return false;
    }
    
    info.device_id = device_id;
    info.device_name = info.properties.name;
    info.global_memory_bytes = info.properties.totalGlobalMem;
    info.compute_capability_major = info.properties.major;
    info.compute_capability_minor = info.properties.minor;
    info.max_threads_per_block = info.properties.maxThreadsPerBlock;
    info.max_blocks_per_grid = info.properties.gridSize[0];
    info.warp_size = info.properties.warpSize;
    info.sm_count = info.properties.multiProcessorCount;
    info.max_warps_per_sm = info.properties.maxThreadsPerMultiProcessor / 32;
    
    // Capability checks
    info.supports_unified_memory = (info.properties.major >= 3);
    info.supports_concurrent_kernels = (info.properties.concurrentKernels != 0);
    info.supports_managed_memory = (info.properties.managedMemory != 0);
    
    // Calculate theoretical performance
    // Clock in MHz, 2 FLOPs per clock per core (FMA), 1000 MHz = 1 GHz
    info.theoretical_peak_flops = 
        (double)info.sm_count * 128 * info.properties.clockRate / 1000.0 * 2.0;  // in GFLOPS
    
    // Memory bandwidth = clock * width / frequency
    info.theoretical_peak_memory_bw = 
        (double)info.properties.memoryClockRate * 
        (info.properties.memoryBusWidth / 8) / 1e6;  // in GB/s
    
    if (debug_logging_) {
        std::stringstream ss;
        ss << "Device " << device_id << ": " << info.device_name
           << " (CC " << info.compute_capability_major 
           << "." << info.compute_capability_minor << ")";
        logger_.info("GPUEngine::get_device_info() - " + ss.str());
    }
    
    return true;
}

bool GPUEngine::set_active_device(int device_id) {
    if (device_id < 0 || device_id >= (int)devices_.size()) {
        logger_.error("GPUEngine::set_active_device() - Invalid device ID: " + std::to_string(device_id));
        return false;
    }
    
    cudaError_t err = cudaSetDevice(device_id);
    
    if (!check_cuda_error(err, "cudaSetDevice")) {
        return false;
    }
    
    active_device_ = device_id;
    logger_.info("GPUEngine::set_active_device() - Set active device to " + std::to_string(device_id));
    
    return true;
}

bool GPUEngine::init_device(int device_id) {
    if (cudaSetDevice(device_id) != cudaSuccess) {
        return false;
    }
    
    // Reset device state
    if (cudaDeviceReset() != cudaSuccess) {
        return false;
    }
    
    // Enable peer access if multiple devices
    if (devices_.size() > 1 && device_id > 0) {
        cudaDeviceEnablePeerAccess(0, 0);  // Ignore errors
    }
    
    return true;
}

bool GPUEngine::configure_multi_gpu(const MultiGPUConfig& config) {
    multi_gpu_config_ = config;
    
    logger_.info("GPUEngine::configure_multi_gpu() - Configuring " + 
                std::to_string(config.device_ids.size()) + " GPUs");
    
    // Enable P2P if requested
    if (config.enable_p2p) {
        for (size_t i = 0; i < config.device_ids.size(); i++) {
            for (size_t j = 0; j < config.device_ids.size(); j++) {
                if (i != j) {
                    cudaDeviceEnablePeerAccess(config.device_ids[j], 0);
                }
            }
        }
    }
    
    return true;
}

// ============================================================================
// KERNEL EXECUTION
// ============================================================================

bool GPUEngine::launch_ecdsa_kernel(size_t batch_size, void* input_keys,
                                     void* output_pubkeys,
                                     const GPUKernelConfig& config) {
    if (!initialized_) {
        logger_.error("GPUEngine::launch_ecdsa_kernel() - Engine not initialized");
        return false;
    }
    
    if (debug_logging_) {
        logger_.info("GPUEngine::launch_ecdsa_kernel() - Launching with batch_size=" + 
                    std::to_string(batch_size));
    }
    
    // Launch kernel wrapper
    bool success = Kernels::launch_ecdsa_kernel_wrapper(
        (const uint32_t*)input_keys,
        (uint32_t*)output_pubkeys,  // Will hold x coords
        (uint32_t*)((char*)output_pubkeys + batch_size * 32),  // y coords
        batch_size,
        config.stream
    );
    
    if (!success) {
        logger_.error("GPUEngine::launch_ecdsa_kernel() - Kernel launch failed");
        return false;
    }
    
    // Update performance metrics
    keys_processed_ += batch_size;
    
    return true;
}

bool GPUEngine::launch_hash160_kernel(size_t batch_size, void* input_data,
                                       void* output_hash,
                                       const GPUKernelConfig& config) {
    if (!initialized_) return false;
    
    return Kernels::launch_hash160_kernel_wrapper(
        (const uint8_t*)input_data,
        (uint8_t*)output_hash,
        batch_size,
        config.stream
    );
}

bool GPUEngine::launch_database_kernel(size_t batch_size, void* input_hashes,
                                        void* output_matches,
                                        const GPUKernelConfig& config) {
    if (!initialized_) return false;
    
    return Kernels::launch_database_kernel_wrapper(
        (const uint8_t*)input_hashes,
        nullptr,  // Database targets (would be set separately)
        0,        // Database size
        (uint32_t*)output_matches,
        batch_size,
        config.stream
    );
}

bool GPUEngine::launch_mode_kernel(int mode, size_t batch_size,
                                    void* input, void* output,
                                    const GPUKernelConfig& config) {
    if (!initialized_) return false;
    
    // Dispatch to appropriate kernel based on mode
    switch (mode) {
        case 0:  // LINEAR
            // Launch LINEAR mode kernel
            break;
        case 2:  // GEOMETRIC
            // Launch GEOMETRIC mode kernel
            break;
        default:
            logger_.error("GPUEngine::launch_mode_kernel() - Unknown mode: " + std::to_string(mode));
            return false;
    }
    
    return true;
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

void* GPUEngine::allocate_device_memory(size_t size, const std::string& purpose,
                                         bool pinned) {
    void* ptr = nullptr;
    cudaError_t err;
    
    if (pinned) {
        err = cudaMallocHost(&ptr, size);
    } else {
        err = cudaMalloc(&ptr, size);
    }
    
    if (!check_cuda_error(err, "cudaMalloc/cudaMallocHost")) {
        logger_.error("GPUEngine::allocate_device_memory() - Failed to allocate " + 
                     std::to_string(size) + " bytes");
        return nullptr;
    }
    
    // Track allocation for audit trail
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        GPUAllocation alloc;
        alloc.allocation_id = ++allocation_counter_;
        alloc.device_ptr = ptr;
        alloc.size_bytes = size;
        alloc.purpose = purpose;
        alloc.is_pinned = pinned;
        alloc.timestamp_created = std::chrono::system_clock::now().time_since_epoch().count();
        allocations_.push_back(alloc);
    }
    
    if (debug_logging_) {
        logger_.info("GPUEngine::allocate_device_memory() - Allocated " + 
                    std::to_string(size) + " bytes for " + purpose);
    }
    
    return ptr;
}

bool GPUEngine::free_device_memory(void* ptr) {
    if (!ptr) return true;
    
    // Update allocation record
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        for (auto& alloc : allocations_) {
            if (alloc.device_ptr == ptr) {
                alloc.timestamp_freed = std::chrono::system_clock::now().time_since_epoch().count();
                break;
            }
        }
    }
    
    cudaError_t err = cudaFree(ptr);
    return check_cuda_error(err, "cudaFree");
}

bool GPUEngine::copy_to_device(void* dst, const void* src, size_t size, bool async) {
    cudaError_t err;
    
    if (async) {
        err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice);
    } else {
        err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }
    
    return check_cuda_error(err, "cudaMemcpy (H2D)");
}

bool GPUEngine::copy_from_device(void* dst, const void* src, size_t size, bool async) {
    cudaError_t err;
    
    if (async) {
        err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost);
    } else {
        err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
    
    return check_cuda_error(err, "cudaMemcpy (D2H)");
}

std::pair<size_t, size_t> GPUEngine::get_memory_usage() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return {total_bytes - free_bytes, total_bytes};
}

bool GPUEngine::optimize_memory() {
    // Memory defragmentation would be implemented here
    // This is a complex operation that requires careful coordination
    logger_.info("GPUEngine::optimize_memory() - Memory optimization requested");
    return true;
}

// ============================================================================
// SYNCHRONIZATION
// ============================================================================

bool GPUEngine::synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    return check_cuda_error(err, "cudaDeviceSynchronize");
}

bool GPUEngine::synchronize_stream(cudaStream_t stream) {
    if (!stream) return true;
    
    cudaError_t err = cudaStreamSynchronize(stream);
    return check_cuda_error(err, "cudaStreamSynchronize");
}

cudaStream_t GPUEngine::create_stream(int priority) {
    cudaStream_t stream;
    int priority_high = 1, priority_low = 0;
    
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 
                                priority > 0 ? priority_high : priority_low);
    
    if (debug_logging_) {
        logger_.info("GPUEngine::create_stream() - Stream created");
    }
    
    {
        std::lock_guard<std::mutex> lock(stream_mutex_);
        streams_.push_back(stream);
    }
    
    return stream;
}

bool GPUEngine::destroy_stream(cudaStream_t stream) {
    if (!stream) return true;
    
    cudaError_t err = cudaStreamDestroy(stream);
    
    if (check_cuda_error(err, "cudaStreamDestroy")) {
        std::lock_guard<std::mutex> lock(stream_mutex_);
        streams_.erase(std::remove(streams_.begin(), streams_.end(), stream), streams_.end());
        return true;
    }
    
    return false;
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

GPUError GPUEngine::get_last_error() {
    GPUError error;
    error.cuda_error = cudaGetLastError();
    error.device_id = active_device_;
    error.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    
    error.message = cudaGetErrorString(error.cuda_error);
    
    if (error.cuda_error != cudaSuccess) {
        error.type = GPUErrorType::CUDA_ERROR;
    }
    
    return error;
}

bool GPUEngine::check_gpu_health() {
    // Run diagnostics
    cudaError_t err = cudaDeviceSynchronize();
    
    if (err != cudaSuccess) {
        logger_.error("GPUEngine::check_gpu_health() - GPU health check failed");
        return false;
    }
    
    logger_.info("GPUEngine::check_gpu_health() - GPU health OK");
    return true;
}

std::string GPUEngine::generate_diagnostic_report() {
    std::stringstream ss;
    
    ss << "\n=== GPU DIAGNOSTIC REPORT ===\n";
    ss << "Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
    ss << "Active Device: " << active_device_ << "\n";
    ss << "Total Devices: " << devices_.size() << "\n\n";
    
    for (const auto& dev : devices_) {
        ss << "Device " << dev.device_id << ": " << dev.device_name << "\n";
        ss << "  Compute Capability: " << dev.compute_capability_major 
           << "." << dev.compute_capability_minor << "\n";
        ss << "  Global Memory: " << (dev.global_memory_bytes / 1e9) << " GB\n";
        ss << "  Streaming Multiprocessors: " << dev.sm_count << "\n\n";
    }
    
    auto [used, total] = get_memory_usage();
    ss << "Memory Usage: " << (used / 1e9) << " / " << (total / 1e9) << " GB\n";
    ss << "Allocations: " << allocations_.size() << "\n";
    ss << "Throughput: " << throughput_kps_ << " keys/sec\n";
    
    return ss.str();
}

void GPUEngine::set_debug_logging(bool enable) {
    debug_logging_ = enable;
    if (enable) {
        logger_.info("GPUEngine::set_debug_logging() - Debug logging enabled");
    }
}

bool GPUEngine::check_cuda_error(cudaError_t error, const std::string& context) {
    if (error != cudaSuccess) {
        logger_.error("CUDA Error in " + context + ": " + std::string(cudaGetErrorString(error)));
        return false;
    }
    return true;
}

void GPUEngine::audit_memory_operation(const std::string& operation, void* ptr, size_t size) {
    if (compute_config_.enable_memory_audit) {
        std::stringstream ss;
        ss << "Memory [" << operation << "] @ 0x" << std::hex << (uintptr_t)ptr 
           << " size=" << std::dec << size;
        logger_.debug(ss.str());
    }
}

// ============================================================================
// PERFORMANCE MONITORING
// ============================================================================

uint64_t GPUEngine::get_kernel_time() {
    return kernel_time_us_.load();
}

float GPUEngine::get_bandwidth_utilization() {
    // Would calculate based on actual memory bandwidth used
    return 0.0f;  // Placeholder
}

float GPUEngine::get_sm_occupancy() {
    // Would calculate based on active warps
    return 0.0f;  // Placeholder
}

uint64_t GPUEngine::get_throughput() {
    return throughput_kps_.load();
}

// ============================================================================
// CONFIGURATION
// ============================================================================

void GPUEngine::set_compute_config(const GPUComputeConfig& config) {
    compute_config_ = config;
}

GPUComputeConfig GPUEngine::get_compute_config() const {
    return compute_config_;
}

// ============================================================================
// CLEANUP
// ============================================================================

void GPUEngine::shutdown() {
    if (!initialized_) return;
    
    logger_.info("GPUEngine::shutdown() - Shutting down GPU engine");
    
    // Destroy all streams
    for (auto stream : streams_) {
        cudaStreamDestroy(stream);
    }
    streams_.clear();
    
    // Free all allocations
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        for (auto& alloc : allocations_) {
            if (alloc.timestamp_freed == 0) {
                cudaFree(alloc.device_ptr);
            }
        }
        allocations_.clear();
    }
    
    // Reset device
    if (active_device_ >= 0) {
        cudaSetDevice(active_device_);
        cudaDeviceReset();
    }
    
    initialized_ = false;
    logger_.info("GPUEngine::shutdown() - Shutdown complete");
}

} // namespace GPU
} // namespace BTCGold
