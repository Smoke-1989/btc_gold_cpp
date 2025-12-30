cmake_minimum_required(VERSION 3.20)
project(btc_gold_cpp VERSION 2.5 LANGUAGES CXX C)

# --- 1. Configurações "Exterminador" ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Flags de Otimização Agressiva (Intel/AMD Modernos)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -O3 
        -march=native          # Usa instruções específicas da sua CPU (AVX2/AVX512)
        -mtune=native
        -funroll-loops         # Desenrola loops para menos jumps
        -fomit-frame-pointer   # Libera registrador extra
        -ffast-math            # Relaxa precisão de float (não afeta crypto integer)
        -flto                  # Link Time Optimization (Crucial!)
    )
endif()

# --- 2. Dependências ---
find_package(OpenSSL 3.0 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SECP256K1 REQUIRED libsecp256k1)

# --- 3. Fontes ---
set(SOURCES
    src/main.cpp
    src/engine.cpp
    src/worker.cpp
    src/hash160.cpp
    src/secp256k1_wrapper.cpp   # FIXED: added underscore
    src/database.cpp
    src/config.cpp
    src/logger.cpp
)

# --- 4. Executável Principal ---
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

# --- 5. Benchmark Tool ---
add_executable(btc_gold_benchmark src/benchmark.cpp src/hash160.cpp src/secp256k1_wrapper.cpp src/logger.cpp src/config.cpp src/database.cpp)
target_include_directories(btc_gold_benchmark PRIVATE include ${OPENSSL_INCLUDE_DIR} ${SECP256K1_INCLUDE_DIRS})
target_link_libraries(btc_gold_benchmark PRIVATE OpenSSL::Crypto ${SECP256K1_LIBRARIES} pthread)

message(STATUS ">>> BUILD EXTERMINATOR MODE: ON <<<")
message(STATUS ">>> CPU Flags: -march=native enabled <<<")
