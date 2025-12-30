.PHONY: all build release debug clean install benchmark help

BUILD_DIR := build
RELEASE_BUILD := $(BUILD_DIR)/Release
DEBUG_BUILD := $(BUILD_DIR)/Debug

all: release

build:
	@mkdir -p $(BUILD_DIR)

 release: build
	@echo "Building Release..."
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release ..
	@cmake --build $(BUILD_DIR) --config Release -j$(shell nproc)
	@echo "Build complete: ./build/Release/btc_gold"

debug: build
	@echo "Building Debug..."
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug ..
	@cmake --build $(BUILD_DIR) --config Debug -j$(shell nproc)

clean:
	@rm -rf $(BUILD_DIR)
	@echo "Cleaned"

benchmark: release
	@echo "Running benchmark..."
	@./$(RELEASE_BUILD)/btc_gold_benchmark

install: release
	@cmake --install $(BUILD_DIR)
	help:
	@echo "BTC GOLD C++ Build System"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build release (default)"
	@echo "  release    - Build optimized release"
	@echo "  debug      - Build debug version"
	@echo "  benchmark  - Run benchmark"
	@echo "  clean      - Remove build directory"
	@echo "  install    - Install binaries"
