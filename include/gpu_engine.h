#pragma once

#include "config.h"
#include <vector>
#include <string>
#include <atomic>

namespace btc_gold {

class GPUEngine {
public:
    static GPUEngine& instance();

    // Detecta se há GPU compatível disponível
    bool check_device();
    
    // Inicializa a engine GPU (aloca memória, prepara kernels)
    bool initialize(const Config& config, const std::vector<unsigned char>& target_hash160s);
    
    // Dispara a busca em GPU (não bloqueante)
    void start_search();
    
    // Para a busca
    void stop();

private:
    GPUEngine() = default;
    ~GPUEngine();
    
    GPUEngine(const GPUEngine&) = delete;
    GPUEngine& operator=(const GPUEngine&) = delete;

    struct Impl;
    Impl* impl_ = nullptr; // Pimpl idiom para esconder tipos CUDA do compilador C++
};

} // namespace btc_gold
