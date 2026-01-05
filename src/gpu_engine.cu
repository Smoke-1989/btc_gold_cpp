#include <cuda_runtime.h>
#include <iostream>
#include "logger.h"

// Explicitly ensure valid CUDA kernel to prevent empty object file
__global__ void dummy_kernel() {}

namespace btc_gold {

class GpuEngine {
public:
    GpuEngine(Logger& logger) : logger_(logger) {
        logger_.info("[GPU] Initializing CUDA engine...");
    }

    void run() {
        int deviceCount;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            logger_.warning("[GPU] No CUDA devices found or driver error!");
            return;
        }

        logger_.info("[GPU] Found " + std::to_string(deviceCount) + " CUDA devices");
        
        // Force kernel instantiation to fix linker error
        dummy_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

private:
    Logger& logger_;
};

} // namespace btc_gold
