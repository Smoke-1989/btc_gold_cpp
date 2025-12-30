cmake_minimum_required(VERSION 3.20)
project(btc_gold_cpp
    VERSION 1.0.0
    DESCRIPTION "BTC GOLD C++ - High-Performance Bitcoin Key Recovery"
    LANGUAGES CXX C
)

# ============================================================================
# COMPILER SETTINGS & OPTIMIZATION
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /arch:AVX2")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mbmi2 -mlzcnt")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto")
endif()

# ============================================================================
# DEPENDENCIES
# ============================================================================

find_package(OpenSSL 3.0 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBSECP256K1 REQUIRED libsecp256k1)

message(STATUS "OpenSSL: ${OPENSSL_VERSION}")
message(STATUS "libsecp256k1: FOUND")

# ============================================================================
# SOURCE FILES
# ============================================================================

set(SOURCES
    src/main.cpp
    src/engine.cpp
    src/worker.cpp
    src/hash160.cpp
    src/secp256k1_wrapper.cpp
    src/database.cpp
    src/logger.cpp
    src/config.cpp
)

# ============================================================================
# MAIN EXECUTABLE
# ============================================================================

add_executable(btc_gold ${SOURCES})

target_include_directories(btc_gold PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENSSL_INCLUDE_DIR}
    ${LIBSECP256K1_INCLUDE_DIRS}
)

target_link_libraries(btc_gold PRIVATE
    ${OPENSSL_LIBRARIES}
    ${LIBSECP256K1_LIBRARIES}
    pthread
)

# ============================================================================
# BENCHMARK EXECUTABLE
# ============================================================================

add_executable(btc_gold_benchmark
    src/benchmark.cpp
    src/hash160.cpp
    src/secp256k1_wrapper.cpp
    src/logger.cpp
)

target_include_directories(btc_gold_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENSSL_INCLUDE_DIR}
    ${LIBSECP256K1_INCLUDE_DIRS}
)

target_link_libraries(btc_gold_benchmark PRIVATE
    ${OPENSSL_LIBRARIES}
    ${LIBSECP256K1_LIBRARIES}
    pthread
)

# ============================================================================
# BUILD INFO
# ============================================================================

message(STATUS "")
message(STATUS "=== BTC GOLD C++ (Enterprise Edition) ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "AVX2: ENABLED")
message(STATUS "LTO: ENABLED")
message(STATUS "")
