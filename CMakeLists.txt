cmake_minimum_required(VERSION 3.10)
project(btc_gold_v5 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -pthread")

# Find OpenSSL
find_package(OpenSSL REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OPENSSL_INCLUDE_DIR})

# Source files
set(SOURCES
    src/main_v5.cpp
    src/worker_engine.cpp
    src/interactive_menu.cpp
    src/config.cpp
    src/logger.cpp
    src/secp256k1_wrapper.cpp
    src/hash160.cpp
)

# Create executable
add_executable(btc_gold ${SOURCES})

# Link libraries
# Note: User must have libsecp256k1 installed on their system
target_link_libraries(btc_gold pthread ${OPENSSL_CRYPTO_LIBRARY} secp256k1)

# Enable testing
enable_testing()

# Install target
install(TARGETS btc_gold DESTINATION bin)
