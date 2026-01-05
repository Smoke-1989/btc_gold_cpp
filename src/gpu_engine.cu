#include <cuda_runtime.h>
#include <iostream>
#include "logger.h"

namespace btc_gold {

class GpuEngine {
public:
    GpuEngine(Logger& logger) : logger_(logger) {
        logger_.info("[GPU] Initializing CUDA engine...");
    }

    void run() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            logger_.warning("[GPU] No CUDA devices found!");
            return;
        }

        logger_.info("[GPU] Found " + std::to_string(deviceCount) + " CUDA devices");
    }

private:
    Logger& logger_;
};

} // namespace btc_gold
