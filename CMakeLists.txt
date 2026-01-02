cmake_minimum_required(VERSION 3.20)
project(btc_gold_cpp VERSION 3.0 LANGUAGES CXX C)

# --- 1. Configurações "Exterminador" ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Flags de Otimização Agressiva CPU
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -O3 
        -march=native
        -mtune=native
        -funroll-loops
        -fomit-frame-pointer
        -ffast-math
        -flto
    )
endif()

# --- 2. Detecção de CUDA (GPU) ---
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS ">>> GPU DETECTED: CUDA Hybrid Mode ENABLED <<<")
    
    # Configuração CUDA Enterprise
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES "native") # Otimiza para a placa local se existir
    
    # Flags nvcc para performance máxima
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math -Xptxas -v")
    
    add_compile_definitions(BTC_GOLD_GPU)
    
    set(GPU_SOURCES 
        src/gpu_engine.cu
    )
else()
    message(WARNING ">>> NO GPU DETECTED: Building CPU-Only Version <<<")
    set(GPU_SOURCES "")
endif()

# --- 3. Dependências ---
find_package(OpenSSL 3.0 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SECP256K1 REQUIRED libsecp256k1)

# --- 4. Fontes ---
set(SOURCES
    src/main.cpp
    src/engine.cpp
    src/worker.cpp
    src/hash160.cpp
    src/secp256k1_wrapper.cpp
    src/database.cpp
    src/config.cpp
    src/logger.cpp
    ${GPU_SOURCES}
)

# --- 5. Executável Principal ---
add_executable(btc_gold ${SOURCES})

target_include_directories(btc_gold PRIVATE 
    include 
    src/cuda # Include path para headers internos CUDA
    ${OPENSSL_INCLUDE_DIR}
    ${SECP256K1_INCLUDE_DIRS}
)

target_link_libraries(btc_gold PRIVATE 
    OpenSSL::Crypto
    ${SECP256K1_LIBRARIES}
    pthread
)

if(CMAKE_CUDA_COMPILER)
    set_target_properties(btc_gold PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# --- 6. Benchmark Tool ---
add_executable(btc_gold_benchmark src/benchmark.cpp src/hash160.cpp src/secp256k1_wrapper.cpp src/logger.cpp src/config.cpp src/database.cpp)
target_include_directories(btc_gold_benchmark PRIVATE include ${OPENSSL_INCLUDE_DIR} ${SECP256K1_INCLUDE_DIRS})
target_link_libraries(btc_gold_benchmark PRIVATE OpenSSL::Crypto ${SECP256K1_LIBRARIES} pthread)

message(STATUS ">>> BUILD ENTERPRISE v3.0: READY <<<")
