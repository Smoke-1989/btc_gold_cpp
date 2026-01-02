#include "gpu_engine.h"
#include "logger.h"
#include <cuda_runtime.h>
#include <iostream>

namespace btc_gold {

// Kernel Placeholder - A implementação matemática completa virá nos próximos headers
__global__ void find_key_kernel(uint64_t start_key, uint64_t end_key, int stride) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t key = start_key + idx * stride;
    
    // TODO: Injetar Secp256k1 + SHA256 + RIPEMD160 aqui
    // Este kernel será expandido com as bibliotecas math.cuh
}

struct GPUEngine::Impl {
    int device_id = 0;
    cudaStream_t stream;
    
    // Buffers
    unsigned char* d_targets = nullptr;
    unsigned char* d_result = nullptr;
};

GPUEngine::~GPUEngine() {
    stop();
    delete impl_;
}

GPUEngine& GPUEngine::instance() {
    static GPUEngine instance;
    return instance;
}

bool GPUEngine::check_device() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    char msg[256];
    snprintf(msg, sizeof(msg), "GPU Found: %s (CC %d.%d)", 
             deviceProp.name, deviceProp.major, deviceProp.minor);
    Logger::instance().info(msg);

    return true;
}

bool GPUEngine::initialize(const Config& config, const std::vector<unsigned char>& target_hash160s) {
    if (!check_device()) return false;
    
    impl_ = new Impl();
    
    cudaError_t err = cudaSetDevice(impl_->device_id);
    if (err != cudaSuccess) return false;
    
    // Criar stream para execução assíncrona
    cudaStreamCreate(&impl_->stream);
    
    // Alocar memória para targets na GPU
    size_t targets_size = target_hash160s.size();
    if (targets_size > 0) {
        cudaMalloc(&impl_->d_targets, targets_size);
        cudaMemcpy(impl_->d_targets, target_hash160s.data(), targets_size, cudaMemcpyHostToDevice);
    }
    
    return true;
}

void GPUEngine::start_search() {
    if (!impl_) return;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = 1024; // Ajustável dinamicamente
    
    // Lançar Kernel
    find_key_kernel<<<blocksPerGrid, threadsPerBlock, 0, impl_->stream>>>(1, 1000000, 1);
    
    Logger::instance().info("[GPU] Kernel launched with %d threads", threadsPerBlock * blocksPerGrid);
}

void GPUEngine::stop() {
    if (impl_) {
        if (impl_->d_targets) cudaFree(impl_->d_targets);
        if (impl_->d_result) cudaFree(impl_->d_result);
        cudaStreamDestroy(impl_->stream);
        delete impl_;
        impl_ = nullptr;
    }
}

} // namespace btc_gold
