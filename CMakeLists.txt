cmake_minimum_required(VERSION 3.20)
project(btc_gold_cpp VERSION 4.0 LANGUAGES CXX C)

# --- 1. Configurações de Build v4.0 Exterminator ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Flags de Otimização Agressiva
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -O3 
        -march=native
        -mtune=native
        -funroll-loops
        -fomit-frame-pointer
        -ffast-math
        # -flto  <-- REMOVED: LTO breaks CUDA fatbin linking in GCC
        -Wall                  # v4.0: Mostrar todos warnings
        -Wextra                # v4.0: Extra warnings
        -Wpedantic             # v4.0: Strict C++ compliance
        -Wno-unused-parameter  # Suprimir false positives
    )
    
    # v4.0: Habilitar warnings de segurança
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(
            -Wformat-security
            -Wstack-protector
        )
    endif()
endif()

# --- 2. Detecção de CUDA (GPU) ---
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS ">>> GPU DETECTED: CUDA Hybrid Mode ENABLED <<<")
    
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native") 
    endif()
    
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math")
    add_compile_definitions(BTC_GOLD_GPU)
    
    set(GPU_SOURCES src/gpu_engine.cu)
else()
    message(WARNING ">>> NO GPU DETECTED: Building CPU-Only Version <<<")
    set(GPU_SOURCES "")
endif()

# --- 3. Dependências ---
find_package(OpenSSL 3.0 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SECP256K1 REQUIRED libsecp256k1)

# --- 4. Fontes v4.0 (Nova implementação) ---
set(SOURCES
    src/main.cpp
    src/worker_v4.cpp          # v4.0: Nova implementação (7 modos completos)
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
    ${OPENSSL_INCLUDE_DIR}
    ${SECP256K1_INCLUDE_DIRS}
)

target_link_libraries(btc_gold PRIVATE 
    OpenSSL::Crypto
    ${SECP256K1_LIBRARIES}
    pthread
)

if(CMAKE_CUDA_COMPILER)
    set_target_properties(btc_gold PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# --- 6. Benchmark Tool ---
add_executable(btc_gold_benchmark 
    src/benchmark.cpp 
    src/hash160.cpp 
    src/secp256k1_wrapper.cpp 
    src/logger.cpp 
    src/config.cpp 
    src/database.cpp
)

target_include_directories(btc_gold_benchmark PRIVATE 
    include 
    ${OPENSSL_INCLUDE_DIR} 
    ${SECP256K1_INCLUDE_DIRS}
)

target_link_libraries(btc_gold_benchmark PRIVATE 
    OpenSSL::Crypto 
    ${SECP256K1_LIBRARIES} 
    pthread
)

# --- 7. Custom Targets ---
# Build and run in one command
add_custom_target(run 
    COMMAND ${CMAKE_BUILD_TYPE}/btc_gold
    DEPENDS btc_gold
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Clean build target
add_custom_target(clean-build
    COMMAND rm -rf build/
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

message(STATUS ">>> BUILD v4.0 EXTERMINATOR: READY <<<")
message(STATUS ">>> CPU Flags: -march=native + AVX2/AVX512 (LTO Disabled) <<<")
message(STATUS ">>> New Modes: DOUBLING (5) + HAMMING (6) + MODULAR_STRIDE (7) <<<")
message(STATUS ">>> Performance: Linear Mode Turbo + Batch Write <<<")
