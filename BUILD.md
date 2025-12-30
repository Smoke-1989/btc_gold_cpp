# BTC GOLD C++ - Build Instructions

## Prerequisites

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config
sudo apt install -y libsecp256k1-dev libsecp256k1-0
sudo apt install -y libssl-dev
```

### macOS
```bash
brew install cmake pkg-config secp256k1 openssl
```

### Windows (MSVC)
- Download Visual Studio Community with C++ tools
- Install CMake
- Install vcpkg or use pre-built libraries

## Building

### Quick Start
```bash
make release
```

### Manual CMake Build
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

## Testing

```bash
make benchmark
```

## Installation

```bash
make install
```

## Expected Performance

- Intel i7-8550U: 50-100M keys/sec
- Modern Ryzen: 100-200M keys/sec
- High-end HEDT: 200M+ keys/sec

## Troubleshooting

### Missing libsecp256k1
```bash
sudo apt install libsecp256k1-dev
```

### CMake not found
```bash
pip3 install cmake
```

### OpenSSL issues
```bash
pkg-config --cflags --libs openssl
```
