# BTC GOLD C++ - Deployment Guide

## 📋 Pre-Deployment Checklist

- [ ] Built with Release configuration
- [ ] Benchmark shows expected performance (50+ k/s)
- [ ] Database file properly formatted
- [ ] Output directory writable
- [ ] Sufficient disk space (1GB minimum)
- [ ] CPU frequency scaling disabled (optional but recommended)
- [ ] All dependencies installed

## 🚀 Quick Start

### 1. Build Release Binary
```bash
make release
```

### 2. Prepare Database
```bash
# Copy target file
cp /path/to/targets.txt ./alvos.txt

# Verify format (one target per line)
head -5 alvos.txt
```

### 3. Run Scanner
```bash
# Interactive mode (prompts for configuration)
./build/Release/btc_gold

# Or with explicit parameters
./build/Release/btc_gold --threads 8 --start 1 --end 1000000
```

### 4. Monitor Progress
```bash
# In another terminal
tail -f found_gold.txt

# Or monitor resource usage
watch -n 1 'ps aux | grep btc_gold'
```

## 📁 Directory Structure (Deployment)

```
/opt/btc_gold/
├── btc_gold              # Main executable
├── btc_gold_benchmark    # Benchmark tool
├── alvos.txt             # Target database (input)
├── found_gold.txt        # Results (output)
├── config.json           # Configuration (optional)
└── logs/                 # Log directory
    ├── run_001.log
    ├── run_002.log
    └── latest.log
```

## ⚙️ Configuration Options

### Interactive Mode
```
[Database Input Format]
[1] Bitcoin Address
[2] HASH160
[3] Public Key
>> 1

[Scan Mode]
[1] Compressed Only
[2] Uncompressed Only
[3] Both
>> 3

[Scanning Mode]
[1] Linear
[2] Random
[3] Geometric
>> 1

[Number of Threads]
>> 8

[Start Value]
>> 1

[End Value]
>> 0xFFFFFFFFFFFFFFFF
```

### Command Line
```bash
./btc_gold \
  --database alvos.txt \
  --output found_gold.txt \
  --threads 8 \
  --mode linear \
  --scan-mode 3 \
  --start 1 \
  --end 1000000
```

## 🔍 Input File Formats

### Bitcoin Addresses
```
# alvos.txt (one address per line)
1A1z7agoat5NUb3tpgR7hRA855j8xvtooD
1dice8EMCQAqQxMnUHcgqeSmMBpGtxPfgd
17SkEw2md5avw4QPCxJAkJmGQN6tXVqgaY
# Comments supported with #
```

### HASH160 Format
```
# alvos.txt (one hash per line, lowercase hex)
62e907b15cbf27d5425399ebf6f0fb50ebb88f18
65a4e8e2d95f1dd53d1ed6a2c6b15b3c7e1a3a5
```

### Public Key Format
```
# alvos.txt (compressed or uncompressed, lowercase hex)
02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7d91d92e47816a7b8
```

## 📊 Output Format

### Results File (found_gold.txt)
```
[FOUND] Hash160: 62e907b15cbf27d5425399ebf6f0fb50ebb88f18 PrivKey: 0000000000000000000000000000000000000000000000000000000000000001
[FOUND] Hash160: 65a4e8e2d95f1dd53d1ed6a2c6b15b3c7e1a3a5 PrivKey: 0000000000000000000000000000000000000000000000000000000000000002
```

## 🔐 Security Considerations

### File Permissions
```bash
# Restrict access to results (sensitive)
chmod 600 found_gold.txt

# Set up isolated user
sudo useradd -r -s /bin/false btc_gold
sudo chown btc_gold:btc_gold /opt/btc_gold
```

### Private Key Handling
```bash
# NEVER:
# - commit found_gold.txt to git
# - push to cloud storage
# - leave on shared systems

# DO:
# - encrypt at rest
# - secure transport (SSH, SFTP)
# - verify integrity (SHA256)
sha256sum found_gold.txt
```

## 🔧 System Tuning

### Linux Performance Tuning

```bash
# 1. Disable power saving
sudo cpupower frequency-set -g performance

# 2. Disable CPU frequency scaling
sudo echo "0" > /sys/devices/system/cpu/intel_pstate/no_turbo

# 3. Increase process priority
nice -n -20 ./btc_gold

# 4. Allocate more stack (if needed)
ulimit -s unlimited

# 5. Increase file descriptors
ulimit -n 65536
```

### Process Monitoring
```bash
# Monitor in real-time
watch -n 1 'ps aux | grep btc_gold | grep -v grep'

# Or use htop
htop -p $(pgrep btc_gold)

# Check thermal status
watch -n 1 'sensors'
```

## 📈 Scaling to Multiple Machines

### Distributed Scanning
```bash
# Machine 1: Lower half of keyspace
./btc_gold --start 1 --end 0x8000000000000000 --output results_m1.txt

# Machine 2: Upper half
./btc_gold --start 0x8000000001000000 --end 0xFFFFFFFFFFFFFFFF --output results_m2.txt

# Combine results
cat results_m*.txt > combined_results.txt
```

## 🐳 Docker Deployment (Optional)

### Dockerfile
```dockerfile
FROM ubuntu:22.04

RUN apt update && apt install -y \
    build-essential cmake git pkg-config \
    libsecp256k1-dev libssl-dev

WORKDIR /app
COPY . .

RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . -j$(nproc)

ENTRYPOINT ["./build/Release/btc_gold"]
```

### Build and Run
```bash
# Build image
docker build -t btc_gold:latest .

# Run container
docker run -v $(pwd)/data:/data btc_gold:latest \
  --database /data/alvos.txt --output /data/found_gold.txt
```

## 📝 Logging and Monitoring

### Enable Verbose Logging
```bash
./btc_gold --verbose > btc_gold.log 2>&1 &
```

### Monitor with Tail
```bash
# Real-time progress
tail -f btc_gold.log

# Extract performance metrics
grep "Speed:" btc_gold.log
grep "Found:" btc_gold.log
```

### Systemd Service (Optional)

```ini
# /etc/systemd/system/btc_gold.service
[Unit]
Description=BTC GOLD Key Scanner
After=network.target

[Service]
Type=simple
User=btc_gold
WorkingDirectory=/opt/btc_gold
ExecStart=/opt/btc_gold/btc_gold --database alvos.txt
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable btc_gold
sudo systemctl start btc_gold
sudo systemctl status btc_gold
```

## 🐛 Troubleshooting

### Seg Fault / Crash
```bash
# Run with debug info
GDB=1 ./build/Release/btc_gold

# Or use valgrind
valgrind --track-origins=yes ./build/Release/btc_gold
```

### Low Performance
```bash
# Check CPU frequency
cat /proc/cpuinfo | grep "cpu MHz"

# Check thermal throttling
grep -i "thermal" /var/log/kern.log

# Profile with perf
perf stat ./btc_gold --end 1000000
```

### Out of Memory
```bash
# Reduce database size
head -100 alvos.txt > alvos_small.txt
./btc_gold --database alvos_small.txt

# Or reduce thread count
./btc_gold --threads 2
```

## ✅ Verification

### Benchmark Verification
```bash
./build/Release/btc_gold_benchmark
```

Expected output:
- Hash160: 450+ k/s
- Full Pipeline: 50-100 k/s

### Functional Test
```bash
# Create small test database
echo "1A1z7agoat5NUb3tpgR7hRA855j8xvtooD" > test.txt

# Run scanner
./build/Release/btc_gold --database test.txt --start 1 --end 1000000

# Check for any matches (unlikely but validates engine)
```

## 🎯 Production Best Practices

1. **Always use Release build**
   ```bash
   make release
   ```

2. **Test with small dataset first**
   ```bash
   head -10 alvos.txt > test.txt
   ./btc_gold --database test.txt --start 1 --end 1000
   ```

3. **Monitor system resources**
   - CPU: Should stay near 100%
   - Memory: Stable, not growing
   - Disk: Writing results only

4. **Keep logs of all runs**
   ```bash
   ./btc_gold >> logs/run_$(date +%s).log 2>&1 &
   ```

5. **Verify results integrity**
   ```bash
   sha256sum found_gold.txt > found_gold.txt.sha256
   ```
