cmake_minimum_required(VERSION 3.10)
project(btc_gold_v5 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -pthread")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main_v5.cpp
    src/worker_engine.cpp
    src/interactive_menu.cpp
    src/config.cpp
    src/logger.cpp
)

# Create executable
add_executable(btc_gold ${SOURCES})

# Link pthread for threading
target_link_libraries(btc_gold pthread)

# Enable testing
enable_testing()

# Install target
install(TARGETS btc_gold DESTINATION bin)
